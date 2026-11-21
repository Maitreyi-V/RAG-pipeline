# results_v3 — frozen 2026-10-07

Commit: `117fbd4` · Python 3.13.14 · see requirements.txt for pinned deps

## Corpus
39 lectures · 1,384 annotated units · 1,038 verse mentions
- Kannada (Dvaita, 29 lectures): 422 units, 479 mentions, all 72 verses of Ch.2
- English (Advaita, 10 lectures): 962/968 units, 559 mentions, 32 verses

## Protocol
Hyperparameters chosen on dev lectures, each system scored ONCE on test.
Dev/test split by lecture, fixed before results were seen (see *_dev.txt / *_test.txt).
- Kannada: 6 dev / 23 test (390 test mentions)
- English:  3 dev /  7 test (369 test mentions)

## Test-set results (unit-level localisation)
| system | corpus | config | P | R | F1 |
|---|---|---|---|---|---|
| string matching | kn | match_threshold=45 | 0.779 | 0.659 | 0.714 |
| embeddings      | kn | BGE-M3 top-1, θ=0.45 | 0.359 | 0.313 | 0.334 |
| GPT-4o-mini     | kn | 302-unit subset | 0.411 | 0.348 | 0.377 |
| string matching | en | romanised, match_threshold=55 | 0.717 | 0.089 | 0.159 |
| embeddings      | en | BGE-M3 top-1, θ=0.55 | 0.116 | 0.171 | 0.138 |
| GPT-4o-mini     | en | — | pending API credit | | |

## Per-tier recall
| tier | kn string | kn embed | en string | en embed |
|---|---|---|---|---|
| T1 verbatim   | 0.968 | 0.306 | 0.833 | 0.278 |
| T2 partial    | 0.684 | 0.368 | 0.117 | 0.267 |
| T3 translation| 0.500 | 0.545 | 0.208 | 0.250 |
| T4 paraphrase | 0.106 | 0.121 | 0.025 | 0.145 |
| T5 allusion   | 0.000 | 0.000 | 0.000 | 0.031 |

## Secondary findings
- Attractors are embedding-specific. BG 2.39 attracts in BOTH languages (kn 32 FPs, en 60).
  kn embed: 2.49 (37), 2.39 (32), 2.62 (23). en embed: 2.45 (77), 2.39 (60), 2.13 (38).
  String matching shows no attractors (max 7 kn, 3 en).
- Out-of-scope Sanskrit (Kena/Katha/Mundaka etc., 141 flagged units) does NOT
  elevate false positives: spurious rates on OOS units match baseline.
  Embeddings fire indiscriminately (69-77%) regardless of content.
- Open 7-8B models (llama3.1, qwen2.5) fail entirely: 0/3 on Kannada recitations,
  1/3 on English. Degenerate verse enumeration on Kannada script.

## Known limitations
- Single annotator per unit; inter-annotator agreement not measured.
- One speaker per language; generalisation across speakers not demonstrated.
- 6 English units unannotated (video_02 x2, video_08 x3, video_09 x1).
- English LLM row missing (API credit).
- Kannada vs English confounds language with tradition (Dvaita vs Advaita).
