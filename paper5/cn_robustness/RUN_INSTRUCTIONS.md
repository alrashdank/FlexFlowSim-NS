# Carlier-Neron external validation: how to run

Everything below happens inside your local FlexFlowSim-NS repository
(the one at commit 1f7a1c4, with the breakdown mechanisms). Copy the
four cn_*.py files into that folder first.

## 1. Obtain the instances (5 minutes)

Download the Carlier & Neron (2000) hybrid flow shop instance files
(the j10c5* / j15c5* / j15c10* set). Sources, in order of preference:
the scheduling benchmark collection you can reach, any HFS paper's
supplementary material, or a colleague's copy; the set is small and
very widely redistributed. Put the raw text files in a folder named
`instances/`.

Then check they parse:

    python cn_instances.py instances

If a file fails, the error names it and says which of the two accepted
formats it does not match. Send me one failing file and I will extend
the parser; do not edit the instance files.

Pick the 10 for the paper: aim for coverage of 5-stage and 10-stage
layouts and different machine patterns. Delete or move the rest, or
keep all and we subset at aggregation.

## 2. Generate configs (1 minute)

    python cn_make_configs.py instances cn_configs

30 JSON files (10 instances x stationary/A090/A070) plus manifest.json.
The overlay rules are frozen in the header of cn_make_configs.py; the
same text goes in Section 4.6.

If FlexFlowSim-NS uses different key names for the breakdown block,
edit ns_adapt() in cn_make_configs.py (one function, clearly marked)
and regenerate. The stationary configs run on base FlexFlowSim
unchanged.

## 3. Nothing to configure

The runner imports the canonical evaluation functions directly from
your repo: baselines.run_episode for the rules and
paper5/run_sweep.run_bandit_episode for VanillaTS and HBQ, with the
Paper 5 seed protocol (evaluation seeds 0..49, fresh environment and
policy per episode). Every CSV row records the source (policy_source
column, all ns:).

## 4. Smoke test (2 minutes)

    python cn_run.py --configs cn_configs --out smoke.csv --smoke

Expect: no "environment exposes no breakdown mechanism" NOTE lines
(if you see them on A090/A070 configs, the runner is not talking to
the NS env), and policy_source starting ns: on every row.

## 5. Full run (roughly 30-60 minutes)

    python cn_run.py --configs cn_configs --out cn_results.csv

10,500 episodes, training-free, single process. The CSV is resumable:
rerunning skips completed config-method cells, so interruptions cost
nothing.

## 6. Aggregate and send

    python cn_aggregate.py cn_results.csv

Send me cn_results.csv (or just cn_summary.csv + cn_table.md). I will
draft Section 4.6, the new table, and the Editor Q1 response around
whatever the numbers say. Both outcomes are usable: if ShortestQueue's
lead weakens off the paper's saturated regime, that is a boundary
condition and it goes in, consistent with the severity-bounded framing
the revision already uses.

## Protocol notes (match the paper exactly)

- EVAL_SEED = 42; episode seeds drawn as default_rng(42).integers(0,
  2**31, 50), identical to step3_eval_all_seeds.py.
- Metrics: total_cost, total_departed, CpU = cost / max(departed, 1).
- 50 episodes per (instance, severity, method) cell.
- Methods: ShortestQueue, LeastUtilised, RoundRobin, CostMinimising,
  Random, VanillaTS, HBQ. No PPO in this experiment.
