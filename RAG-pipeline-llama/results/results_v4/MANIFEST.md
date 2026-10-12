# results_v4 — frozen 2026-10-12

Code at commit `de33ec0` plus `scripts/heldout_eval.py` as committed with this folder.
Supersedes `results_v3`, which is kept unchanged.

## What changed since v3

- Gold: the 6 unlabelled English units are labelled and the corrupt tier on `en_video_09_0019` is fixed. All 1,390 original units are now annotated.
- New-speaker held-out set added: 12 lectures, 494 units, 4 speakers (`lectures_new.csv`, `units_heldout_v1.csv`).
- GPT-4o-mini run on every unit in both languages (v3 had a 302-unit Kannada subset and no English).
- New systems: embeddings with expanded verse documents and a margin criterion; gated propagation; unbounded-carry, monotonic and oracle rows.
- Every cell has P / R / F1, per-tier hit/n, and a lecture-bootstrap 95% CI where a group has at least 5 lectures.

## Corpus

| set | language | speaker (tradition) | lectures | units | empty units |
|---|---|---|---|---|---|
| original test | kn | Dvaita speaker | 23 | 340 | 20 |
| original test | en | Advaita speaker | 7 | 676 | 328 |
| held-out | kn | Sumith (Dvaita) | 8 | 88 | 4 |
| held-out | kn | Vidwan Brahmanyacharya (Dvaita) | 1 | 44 | 0 |
| held-out | en | HG Gaurmandal Das (Gaudiya Vaishnava) | 1 | 215 | 16 |
| held-out | en | Swami Aparajitananda (Advaita) | 2 | 147 | 1 |

English uses Sarvam transcripts throughout. Dev lectures (6 kn, 3 en) are used only to choose settings.

## Protocol

Settings are the v3 values (string-matching thresholds, `max_gap` kn 3 / en 50) or are chosen on the original dev lectures (embedding margin threshold, gate tau). Nothing is tuned on the original test lectures or on the held-out speakers. GPT-4o-mini has no settings.

## F1 (unit-level localisation)

| system | kn test | en test | Sumith | Brahmanyacharya | Gaurmandal Das | Aparajitananda |
|---|---|---|---|---|---|---|
| string matching | 0.714 | 0.158 | 0.458 | 0.741 | 0.451 | 0.120 |
| embeddings (expanded, margin) | 0.557 | 0.194 | 0.435 | 0.389 | 0.379 | 0.210 |
| GPT-4o-mini | 0.360 | 0.275 | 0.423 | 0.379 | 0.411 | 0.254 |
| carry, v3 max_gap | 0.749 | 0.417 | 0.674 | 0.853 | 0.766 | 0.833 |
| carry, unbounded | 0.757 | 0.394 | 0.667 | 0.879 | 0.766 | 0.833 |
| gated, drop | 0.788 | 0.476 | 0.696 | 0.841 | 0.762 | 0.747 |
| gated, resume | 0.759 | 0.502 | 0.673 | 0.879 | 0.749 | 0.596 |
| monotonic, unbounded | 0.514 | 0.335 | 0.645 | 0.745 | 0.640 | 0.833 |
| oracle gate (resume) | 0.834 | 0.879 | 0.780 | 0.885 | 0.844 | 0.890 |
| oracle anchors (T1+T2) | 0.924 | 0.549 | 0.755 | 0.923 | 0.853 | 0.914 |

Full rows with P, R, CI and per-tier hit/n: `table_kn.txt`, `table_en.txt`, `heldout_kn.csv`, `heldout_en.csv`.

## Findings these numbers support

- No single-passage method recovers paraphrase. T4 recall is at most 14% for string matching and GPT-4o-mini on every set, and at most 33% for embeddings (Sumith, 19/58).
- Carrying the active verse forward lifts every speaker, new speakers included.
- The embedding gate improves on plain carry only for the original English speaker, where 328 of 676 units (48.5%) are empty. On the new speakers, who have almost no empty units, it is equal or worse.
- The monotonic-order prior is not a useful baseline.

## Known limitations

- One annotator per unit; the blind second pass (`data/iaa_assignment.csv`) has not been run. All 494 held-out units were labelled by one annotator.
- Three of the four new speakers have 1 or 2 lectures, so they have no confidence interval.
- The four gate variants (drop / resume, two verse representations) were all inspected on the original test set during development. tau itself was chosen on dev.
- Verse coverage differs by speaker (original English: 2.1 to 2.29).
- Unit timestamps are empty; transcripts carry no timing.

## Files

Inputs: `mention_spans.csv`, `units_v2.csv`, `units_heldout_v1.csv`, `lectures_new.csv`, `*_dev.txt`, `*_test.txt`.
Predictions: `final_l1_*.csv`, `heldout_l1_*.csv` (string matching), `llm_kn.csv`, `llm_en.csv`, `heldout_llm.csv` (GPT-4o-mini), `embed_scores*.npz` (unit x verse similarities).
Curves: `carry_curve_*_test.csv` (max_gap sweep), `gated_*_curve.csv` (tau sweep), `run_length_*.csv`.
