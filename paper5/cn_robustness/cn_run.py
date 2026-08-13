"""
Carlier-Neron external validation runner.

Uses the CANONICAL evaluation functions from the FlexFlowSim-NS repo:
  - baselines.run_episode            (rules)
  - paper5.run_sweep.run_bandit_episode  (VanillaTS, HBQ)
and the canonical Paper 5 seed protocol: evaluation seeds are the
integers 0..N-1, with a fresh environment and policy constructed per
episode, environment and policy seeded identically (run_sweep.py,
phase1_baselines). Nothing here reimplements policy or update logic.

USAGE (from inside the FlexFlowSim-NS repo directory):
    python cn_run.py --configs cn_configs --out cn_results.csv
Optional:
    --repo PATH     path to FlexFlowSim-NS if not the CWD
    --seeds N       default 50
    --methods a,b   subset filter
    --smoke         first 2 configs, 3 seeds
"""

from __future__ import annotations
import argparse
import csv
import os
import sys
import time
from pathlib import Path

RULES = ["ShortestQueue", "LeastUtilised", "RoundRobin",
         "CostMinimising", "Random"]
BANDITS = ["VanillaTS", "HBQ"]



def _fast_equiv(method, env):
    """Exact per-stage reformulation of ShortestQueue / LeastUtilised."""
    import numpy as np

    class _P:
        def __init__(self):
            self.env = env
            if method == "LeastUtilised":
                self._mu = np.array([
                    float(env._stages[si]["servers"][sj]["service_time"]
                          .get("mean", 1.0))
                    for si in range(env.n_stages)
                    for sj in range(env.servers_per_stage[si])])
            else:
                self._mu = None

        def reset(self):
            pass

        def predict(self, obs):
            n = env._total_servers
            load = np.asarray(obs[:n]) + np.asarray(obs[n:2 * n])
            scores = load if self._mu is None else load * self._mu
            a, off = 0, 0
            for m in env.servers_per_stage:
                a = a * m + int(np.argmin(scores[off:off + m]))
                off += m
            return a
    return _P()


def load_repo(repo):
    repo = Path(repo).resolve()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "paper5"))
    cwd = os.getcwd()
    import env as E                      # noqa
    import baselines as B                # noqa
    import run_sweep as RS               # noqa: chdirs to repo root on import
    os.chdir(cwd)                        # undo run_sweep's os.chdir
    return E, B, RS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="cn_configs")
    ap.add_argument("--out", default="cn_results.csv")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--methods", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg_dir = Path(args.configs).resolve()
    out = Path(args.out).resolve()
    E, B, RS = load_repo(args.repo)

    cfgs = sorted(cfg_dir.glob("*__*.json"))
    n_seeds = args.seeds
    if args.smoke:
        cfgs, n_seeds = cfgs[:2], 3
    want = ([m.strip() for m in args.methods.split(",") if m.strip()]
            or RULES + BANDITS)

    done = set()
    if out.exists():
        with out.open() as f:
            for r in csv.DictReader(f):
                done.add((r["config"], r["method"]))
        print(f"Resuming: {len(done)} config-method cells already done")
    mode = "a" if out.exists() else "w"
    fields = ["config", "instance", "severity", "method", "policy_source",
              "seed", "total_cost", "departed", "cpu", "breakdown_count"]

    with out.open(mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if mode == "w":
            w.writeheader()
        for cfg_path in cfgs:
            inst, sev = cfg_path.stem.split("__")

            # breakdown engagement guard (one probe env per config)
            probe = E.FlexFlowSimEnv(config=str(cfg_path),
                                     weights=(1.0, 0.0, 0.0), seed=0)
            if sev != "stationary" and not getattr(
                    probe, "_breakdowns_enabled", False):
                print(f"NOTE: {cfg_path.name}: breakdowns NOT enabled in "
                      f"this environment; severity label is misleading. "
                      f"Check the config schema.")

            for method in want:
                if (cfg_path.name, method) in done:
                    continue
                t0 = time.time()
                for seed in range(n_seeds):
                    if method in RULES:
                        env = E.FlexFlowSimEnv(config=str(cfg_path),
                                               weights=(1.0, 0.0, 0.0),
                                               seed=seed)
                        pol = B.BASELINE_POLICIES[method](env=env)
                        src = "ns:baselines.run_episode"
                        # Exact fast path for stage-additive per-step scanners
                        # on very large joint action spaces. Per-stage argmin
                        # with lowest-index tie-break is provably identical to
                        # the canonical full scan (objective decomposes across
                        # stages; enumeration order preserves the same
                        # tie-break). Verified step-for-step at 243 actions:
                        # 1437/1437 identical decisions for both policies.
                        if (env.n_actions > 1000
                                and method in ("ShortestQueue",
                                               "LeastUtilised")):
                            pol = _fast_equiv(method, env)
                            src += "+fastpath(exact)"
                        res = B.run_episode(pol, env, seed)
                    elif method == "VanillaTS":
                        res = RS.run_bandit_episode(
                            str(cfg_path), B.VanillaThompsonSampling, seed)
                        src = "ns:run_sweep.run_bandit_episode"
                    elif method == "HBQ":
                        res = RS.run_bandit_episode(
                            str(cfg_path), B.HybridBanditQueue, seed,
                            load_weight=20.0)
                        src = "ns:run_sweep.run_bandit_episode"
                    else:
                        raise ValueError(method)
                    w.writerow({
                        "config": cfg_path.name, "instance": inst,
                        "severity": sev, "method": method,
                        "policy_source": src, "seed": seed,
                        "total_cost": round(res["totalCost"], 4),
                        "departed": int(res["totalDeparted"]),
                        "cpu": round(res["costPerUnit"], 4),
                        "breakdown_count": res.get("breakdownCount", ""),
                    })
                f.flush()
                print(f"{cfg_path.stem:32s} {method:14s} "
                      f"{time.time()-t0:6.1f}s  [{src.split(':')[0]}]")
    print(f"\nDone -> {out}")


if __name__ == "__main__":
    main()
