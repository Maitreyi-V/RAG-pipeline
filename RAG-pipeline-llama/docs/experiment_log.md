# Experiment log

One row per run. Commit = the code the run used. Results are copied from the
script output named in "config"; never typed by hand.

| date | commit | config | result | takeaway |
|---|---|---|---|---|
| 2026-10-16 | a1818c4 | locked Step 4 (`step4_viterbi.py`), tuning LOLO | KN strict 0.938, EN strict 0.680 | system locked on 9 tuning lectures |
| 2026-10-18 | fb35bdf → e2d3c00 | frozen single run: `freeze_grounding.py --split test` | KN 0.817 / 0.894 / 0.899; EN 0.600 / 0.680 / 0.684 | **baseline, tag `baseline-frozen-v1`** |
| 2026-10-18 | e2d3c00 | `freeze_grounding.py --split heldout` | new KN 0.696, new EN 0.798 strict | carry baselines competitive on unseen speakers |
| 2026-10-18 | e2d3c00 | `freeze_e4.py`, `freeze_e6.py`, `freeze_e7.py --index all --questions test` | E4 KN 0.848 (= full); E7 Hit@1 0.880, P@5 0.725, 21/25 abstained | see results/freeze_*.txt |
| 2026-10-19 | d8ec5c8 | LLM pilot, claude.ai chat, 3 test lectures (`llm_pilot.py`) | EN pooled strict 0.744 vs ours 0.521 | frontier LLM stronger, mainly on digressions |
| 2026-10-20 | (uncommitted) | GPT-5 API, video_02 (`llm_pilot.py api --model gpt-5`) | strict 0.699 vs ours 0.627 | 24,770 in / 5,664 out tokens |
| 2026-10-20 | (this commit) | Phase 0: `make_folds.py` → `data/folds_v1.csv` (seed 2026) | 5 folds, 10–11 lectures each | folds for cross-fitting and CV |
