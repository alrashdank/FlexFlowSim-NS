"""
Carlier-Neron instance handling for the FlexFlowSim-NS external validation.

Parses hybrid flow shop instances in the de facto standard distribution
format and reduces each to the triple this study consumes:
    k          number of stages
    M[i]       identical machines at stage i
    mu[i]      mean processing time at stage i (mean over jobs)

Two accepted text formats (auto-detected):

  FORMAT A (layout header):
      n k
      M_1 M_2 ... M_k
      p_{1,1} ... p_{1,k}          (n rows of k processing times)
      ...

  FORMAT B (Neron per-job stage pairs):
      n k
      then per job: k pairs "M_i p_{j,i}" on one line
      (machine counts repeated per job; must agree across jobs)

Anything else fails loudly. Do not silently coerce.

SYNTHETIC MODE is for pipeline smoke-testing only. It follows the
published Carlier-Neron generation recipe (stages in {5,10}, machines
per stage in {1,2,3} with a designated one-machine bottleneck stage in
the 'b'/'c' style variants, processing times uniform integers in
[3,20]) but the files it writes are NOT the benchmark and are named
SYNTH_* to make that impossible to miss. Results on synthetic
instances must never appear in the paper.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CNInstance:
    name: str
    n_jobs: int
    k: int
    machines: tuple      # M_i per stage
    stage_means: tuple   # mean processing time per stage (raw units)
    source: str          # 'file' or 'synthetic'

    @property
    def n_actions(self) -> int:
        out = 1
        for m in self.machines:
            out *= m
        return out


def _tokens(path: Path):
    txt = Path(path).read_text()
    return [t for t in re.split(r"[\s,;]+", txt.strip()) if t]


def parse_instance(path) -> CNInstance:
    path = Path(path)
    tok = _tokens(path)
    ints = []
    for t in tok:
        try:
            ints.append(int(float(t)))
        except ValueError:
            raise ValueError(f"{path.name}: non-numeric token {t!r}")
    if len(ints) < 4:
        raise ValueError(f"{path.name}: too short to be an instance file")
    n, k = ints[0], ints[1]
    body = ints[2:]

    # FORMAT A: k machine counts then n*k processing times.
    # Orientation detected from line structure: the Carlier-Neron
    # distribution is stage-major (k rows of n job times); job-major
    # (n rows of k) is also accepted.
    if len(body) == k + n * k:
        M = body[:k]
        lines = [l.split() for l in Path(path).read_text().strip().splitlines()]
        data_rows = [l for l in lines[2:] if l]
        vals = np.asarray(body[k:], dtype=float)
        if len(data_rows) == k and all(len(r) == n for r in data_rows):
            P = vals.reshape(k, n).T
        elif len(data_rows) == n and all(len(r) == k for r in data_rows):
            P = vals.reshape(n, k)
        else:
            raise ValueError(
                f"{path.name}: ambiguous matrix orientation "
                f"({len(data_rows)} data rows for n={n}, k={k})")
    # FORMAT B: per job, k pairs (machine count, time)
    elif len(body) == n * 2 * k:
        arr = np.asarray(body, dtype=float).reshape(n, k, 2)
        M_rows = arr[:, :, 0].astype(int)
        if not (M_rows == M_rows[0]).all():
            raise ValueError(f"{path.name}: machine counts differ across jobs")
        M = M_rows[0].tolist()
        P = arr[:, :, 1]
    else:
        raise ValueError(
            f"{path.name}: token count {len(body)} matches neither "
            f"format A ({k + n * k}) nor format B ({n * 2 * k}) "
            f"for n={n}, k={k}"
        )

    if any(m < 1 for m in M):
        raise ValueError(f"{path.name}: stage with zero machines")
    if (P <= 0).any():
        raise ValueError(f"{path.name}: non-positive processing time")

    return CNInstance(
        name=path.stem,
        n_jobs=n,
        k=k,
        machines=tuple(int(m) for m in M),
        stage_means=tuple(float(x) for x in P.mean(axis=0)),
        source="file",
    )


def load_instances(folder) -> list:
    folder = Path(folder)
    files = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in {".txt", ".dat", ""} and p.is_file()
    )
    if not files:
        raise FileNotFoundError(f"No instance files in {folder}")
    out = []
    for p in files:
        try:
            out.append(parse_instance(p))
        except ValueError as e:
            print(f"SKIP  {e}")
    if not out:
        raise ValueError("No parsable instances found")
    return out


# ------------------------------------------------------------------
# SYNTHETIC (smoke-test only)
# ------------------------------------------------------------------

_SYNTH_LAYOUTS = [
    # (stages, machines per stage) in the CN style: 'a' balanced,
    # 'b' one-machine bottleneck mid-stage, 'd' three everywhere.
    (5,  (3, 3, 3, 3, 3)),
    (5,  (2, 2, 1, 2, 2)),
    (5,  (3, 2, 1, 2, 3)),
    (5,  (1, 2, 3, 2, 1)),
    (5,  (2, 3, 3, 3, 2)),
    (10, (3,) * 10),
    (10, (2, 2, 2, 1, 2, 2, 1, 2, 2, 2)),
    (10, (3, 2, 3, 2, 1, 1, 2, 3, 2, 3)),
    (10, (1, 3, 3, 3, 3, 3, 3, 3, 3, 1)),
    (10, (2,) * 10),
]


def write_synthetic(folder, seed=20260801):
    """Write 10 SYNTH_* instances for pipeline smoke tests only."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    names = []
    for idx, (k, M) in enumerate(_SYNTH_LAYOUTS):
        n = 10 if k == 5 else 15
        P = rng.integers(3, 21, size=(n, k))
        name = f"SYNTH_j{n}c{k}v{idx}"
        lines = [f"{n} {k}", " ".join(str(m) for m in M)]
        lines += [" ".join(str(int(x)) for x in row) for row in P]
        (folder / f"{name}.txt").write_text("\n".join(lines) + "\n")
        names.append(name)
    return names


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if args and args[0] == "--make-synth":
        tgt = args[1] if len(args) > 1 else "instances_synth"
        ns = write_synthetic(tgt)
        print(f"Wrote {len(ns)} SYNTHETIC smoke-test instances to {tgt}/")
    else:
        tgt = args[0] if args else "instances"
        insts = load_instances(tgt)
        print(f"Parsed {len(insts)} instance(s) from {tgt}/")
        for inst in insts:
            tag = "  [SYNTHETIC - not for the paper]" if inst.name.startswith("SYNTH_") else ""
            print(f"  {inst.name}: k={inst.k}, M={inst.machines}, "
                  f"stage means={[round(m,1) for m in inst.stage_means]}{tag}")
