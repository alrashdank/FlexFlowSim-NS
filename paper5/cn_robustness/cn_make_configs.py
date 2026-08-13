"""
Generate FlexFlowSim-NS JSON configs from Carlier-Neron instances.

The overlay is deterministic and fully disclosed; every rule below is
stated in the manuscript's Section 4.6 and must not be changed between
runs without changing that text.

OVERLAY SPECIFICATION (frozen)
------------------------------
Taken from the instance:  k, machines per stage M_i, and the per-stage
mean processing time mu_i (mean over the instance's jobs).

1. Time scale. Stage means are rescaled by a single factor s so that
   the across-stage mean service time equals 15.0 timesteps, keeping
   the 480-step shift and the MTTR of 20 in the same regime as the
   paper's testbeds. s = 15.0 / mean_i(mu_i).

2. Server heterogeneity. CN machines are identical; the paper's
   routing problem requires a cost-speed trade-off. Server m of M_i
   (m = 1..M_i) receives service-time multiplier g_m, the m-th point
   of linspace(0.80, 1.20, M_i); M_i = 1 gives g = 1.0. Service times
   are normal with mean mu_i * s * g_m, std 0.30 * mean, min 0.5,
   matching the electronics testbed's dispersion.

3. Costs. processing_cost_m proportional to 1/g_m, normalised so the
   stage mean is 1.0 (slower servers are cheaper, as in both paper
   testbeds). idle_cost 0.4 and waiting_cost 0.1 uniformly.

4. Arrivals. Poisson; the rate lambda is set so that the bottleneck
   stage's offered load is 0.90 in the stationary configuration:
   lambda = 0.90 * min_i sum_m 1/serviceMean_{i,m}. This matches the
   electronics testbed's measured bottleneck utilisation (~0.91).
   Breakdown configurations keep the stationary lambda, so severity
   raises effective load exactly as in the paper.

5. Breakdowns. Identical to the paper: per-server, pre-empt-resume,
   TTR lognormal with mean 20 and std 6.67, TTF exponential with
   MTBF = MTTR * A / (1 - A), all servers sharing availability A.
   Configurations: stationary, A = 0.90, A = 0.70.

6. Episode: max_time 480, dt 1, max_queue 50, norm_constants [1,1,1]
   (reward normalisation is irrelevant to the training-free methods).

NOTE ON THE BREAKDOWN BLOCK KEY NAMES
-------------------------------------
The "breakdowns" block below uses descriptive keys. If FlexFlowSim-NS
expects different key names, edit ns_adapt() HERE, in one place, and
regenerate. The stationary configs contain "breakdowns": {"enabled":
false} and run unmodified on base FlexFlowSim.
"""

from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np

from cn_instances import CNInstance, load_instances

SEVERITIES = {"stationary": None, "A090": 0.90, "A070": 0.70}
MTTR_MEAN = 20.0
MTTR_STD = 6.67
TARGET_MEAN_SERVICE = 15.0
BOTTLENECK_RHO = 0.90
SERVICE_CV = 0.30
IDLE_COST = 0.4
WAITING_COST = 0.1


def _lognormal_params(mean, std):
    """Convert desired arithmetic mean/std to numpy lognormal mu/sigma."""
    var = std ** 2
    sigma2 = math.log(1.0 + var / mean ** 2)
    mu = math.log(mean) - 0.5 * sigma2
    return mu, math.sqrt(sigma2)


def build_config(inst: CNInstance, availability):
    s = TARGET_MEAN_SERVICE / float(np.mean(inst.stage_means))
    stages = []
    stage_capacity = []
    for i, (M, mu_raw) in enumerate(zip(inst.machines, inst.stage_means)):
        g = np.linspace(0.80, 1.20, M) if M > 1 else np.array([1.0])
        inv = 1.0 / g
        costs = inv / inv.mean()          # stage mean cost = 1.0
        servers = []
        cap = 0.0
        for m in range(M):
            mean_t = mu_raw * s * float(g[m])
            cap += 1.0 / mean_t
            servers.append({
                "name": f"S{i+1}M{m+1}",
                "service_time": {
                    "distribution": "normal",
                    "mean": round(mean_t, 4),
                    "std": round(SERVICE_CV * mean_t, 4),
                    "min": 0.5,
                },
                "processing_cost": round(float(costs[m]), 4),
                "idle_cost": IDLE_COST,
            })
        stage_capacity.append(cap)
        stages.append({"name": f"Stage{i+1}", "servers": servers})

    lam = BOTTLENECK_RHO * min(stage_capacity)
    cfg = {
        "_instance": inst.name,
        "_instance_source": inst.source,
        "_overlay": "CN-overlay v1 (see cn_make_configs.py header)",
        "stages": stages,
        "arrival": {"distribution": "exponential",
                    "mean": round(1.0 / lam, 4)},
        "waiting_cost": WAITING_COST,
        "max_time": 480,
        "dt": 1,
        "max_queue": 50,
        "norm_constants": [1.0, 1.0, 1.0],
    }

    if availability is None:
        cfg["breakdowns"] = {"enabled": False}
    else:
        A = float(availability)
        mtbf = MTTR_MEAN * A / (1.0 - A)
        cfg["_availability_target"] = A
        cfg["breakdowns"] = {
            "enabled": True,
            "default": {
                "ttf": {"distribution": "exponential",
                        "mean": round(mtbf, 4)},
                "ttr": {"distribution": "lognormal",
                        "mean": round(MTTR_MEAN, 4), "std": MTTR_STD},
            },
        }
    return ns_adapt(cfg)


def ns_adapt(cfg):
    """Single seam for FlexFlowSim-NS schema differences.

    If the NS repo names the breakdown keys differently (for example
    'breakdown' singular, or 'availability_target'), remap here and
    nowhere else, then regenerate all configs.
    """
    return cfg


def main(instance_dir, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for inst in load_instances(instance_dir):
        for sev_name, A in SEVERITIES.items():
            cfg = build_config(inst, A)
            fn = out / f"{inst.name}__{sev_name}.json"
            fn.write_text(json.dumps(cfg, indent=2))
            manifest.append({
                "instance": inst.name, "severity": sev_name,
                "config": fn.name, "k": inst.k,
                "machines": list(inst.machines),
                "n_actions": inst.n_actions,
                "interarrival_mu": cfg["arrival"]["mean"],
            })
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(manifest)} configs to {out}/ "
          f"({len(manifest)//len(SEVERITIES)} instances x "
          f"{len(SEVERITIES)} severities)")


if __name__ == "__main__":
    import sys
    inst_dir = sys.argv[1] if len(sys.argv) > 1 else "instances"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "cn_configs"
    main(inst_dir, out_dir)
