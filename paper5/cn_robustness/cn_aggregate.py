"""
Aggregate cn_results.csv into the numbers Section 4.6 and the response
letter need.

Outputs:
  cn_summary.csv     mean/sd CpU per instance x severity x method
  cn_table.md        compact markdown table (one row per instance x
                     severity: leader, ShortestQueue rank, margin of
                     ShortestQueue over the best other method)
  stdout             headline counts for the letter

Usage:  python cn_aggregate.py cn_results.csv
"""

from __future__ import annotations
import csv
import sys
from collections import defaultdict
from statistics import mean, stdev


def main(path="cn_results.csv"):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit("empty results file")
    srcs = {r["policy_source"] for r in rows}
    if any(s == "reference" for s in srcs):
        print("WARNING: results include policy_source=reference; "
              "not usable for the paper.\n")

    cell = defaultdict(list)
    for r in rows:
        cell[(r["instance"], r["severity"], r["method"])].append(
            float(r["cpu"]))

    with open("cn_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance", "severity", "method",
                    "cpu_mean", "cpu_sd", "episodes"])
        for (inst, sev, m), v in sorted(cell.items()):
            sd = stdev(v) if len(v) > 1 else 0.0
            w.writerow([inst, sev, m,
                        round(mean(v), 2), round(sd, 2), len(v)])

    combos = sorted({(i, s) for (i, s, _) in cell})
    lead_count = defaultdict(int)
    sq_leads = 0
    sq_within = 0
    lines = ["| Instance | Severity | Leader | SQ rank | "
             "SQ margin vs best other |",
             "|---|---|---|---|---|"]
    for inst, sev in combos:
        methods = {m: mean(v) for (i, s, m), v in cell.items()
                   if i == inst and s == sev}
        order = sorted(methods, key=methods.get)
        leader = order[0]
        lead_count[leader] += 1
        sq = methods.get("ShortestQueue")
        rank = order.index("ShortestQueue") + 1
        others = [v for m, v in methods.items() if m != "ShortestQueue"]
        margin = (min(others) - sq) / sq * 100.0
        if rank == 1:
            sq_leads += 1
        if margin >= -3.0:
            sq_within += 1
        lines.append(f"| {inst} | {sev} | {leader} | {rank} | "
                     f"{margin:+.1f}% |")
    open("cn_table.md", "w").write("\n".join(lines) + "\n")

    n = len(combos)
    print(f"instance x severity cells: {n}")
    print(f"ShortestQueue strict leader: {sq_leads}/{n}")
    print(f"ShortestQueue within 3% of leader: {sq_within}/{n}")
    print("leader counts:", dict(sorted(lead_count.items(),
                                        key=lambda kv: -kv[1])))
    print("wrote cn_summary.csv and cn_table.md")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cn_results.csv")
