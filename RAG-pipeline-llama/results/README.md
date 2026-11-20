# results/ — what to look at

Splits: **tuning** = 9 lectures (6 KN, 3 EN) used for design · **E2 test** = 30 lectures (23 KN, 7 EN) ·
**E3** = 12 new-speaker lectures (9 KN, 3 EN) · **new 4** = en_chinmaya_day06–09 (annotated Nov 2026).
"The 42" = E2 + E3. Metrics: strict / ±1 / theme F1, micro over (passage, verse) pairs.

## 1. Main leaderboard (end-to-end systems on the 42)

| File | What |
|---|---|
| `llm_frontier_gpt61sol.txt` | GPT-6.1 Sol (API, high) vs Opus, Cue, Local, Plain Carry |
| `llm_frontier_opus55.txt` | Claude Opus 5.5 (claude.ai chat, high, 1 run) |
| `llm_frontier_gemini31pro_api_low.txt` | Gemini 3.1 Pro (API, low), run 1 |
| `gemini_runs/run1_earlier_low_55.txt` | Gemini 3.1 Pro low, **run 1**, all 55 lectures, per lecture |
| `gemini_runs/run2_rerun_low_55.txt` | Gemini 3.1 Pro low, **run 2** (re-run 16 Nov), all 55, per lecture |
| `llm_e2e_gpt-5-mini.txt` | gpt-5-mini (API, low) |
| `freeze_v2_test.txt`, `freeze_v2_heldout.txt` | GitaGround-Cue (v2) and Local (v1), E2 and E3 |
| `freeze_grounding_test.txt`, `freeze_grounding_heldout.txt` | GitaGround-Local (v1) frozen run, baselines |

## 2. Human agreement
| `exp_e1_iaa.txt` | blind second pass, 209 passages: κ and F1 per language |

## 3. Cascade / hybrid
| File | What |
|---|---|
| `cascade4_test.txt` | 4-layer cascade, layers frozen on tuning, on the 42 (main cascade test) |
| `cascade3_test.txt` | 3-layer cascade (no gpt-5-mini layer) on the 42 |
| `cascade4_cv.txt` | 4-layer cascade, keep rule by CV over 39 lectures (secondary) |
| `cascade3_leftover_test.txt` | 3-layer with Gemini, **real** leftover-only calls on the 42 (post-hoc) |
| `all51_gemini3l.txt` | Gemini vs Cue vs 3-layer, 9 tuning + 42 test: F1, digressions, cost |
| `all51_gemini3l_local.txt` | same with GitaGround-Local instead of Cue |
| `new4_gemini3l.txt` | Gemini vs Cue vs 3-layer on the 4 new Chinmaya lectures |
| `v3_hybrid_test.txt` | v3 hybrid (post-hoc), one run on the 42 |
| `cascade*_tuning*.txt`, `combine_gemini_cue_tuning.txt`, `headroom_*_tuning.txt` | tuning-only design steps |

## 4. Fine-tuning (secondary protocol)
| `cv_ft_vs_noft.txt`, `cv_ft/`, `cv_noft/` | 5-fold CV with fine-tuned BGE reranker, and without |

## 5. Analysis
| `error_decomp_v2_en_test.txt` | error classes (FALSE-VERSE, FALSE-OFF, …), English test |
| `recall_by_type.txt` | recall by mention type (T1–T5) |
| `exp_e6_ablation.txt`, `freeze_e4_*.txt`, `freeze_e6_test.txt`, `freeze_e7_*.txt` | ablations, E4/E6/E7 |
| `teacher_agreement_*.txt`, `teacher_v3_tuning.txt` | gpt-5-mini off-verse teacher checks |

## Not results — inputs the code reads (do not move)
`v2_teacher145.pkl` (frozen Cue), `step4_locked.pkl` (frozen Local), `step3_scorers.pkl`, `step2_features.npz`,
`unit_embeddings*.npz`, `embed_scores*.npz`, `silver/`, `student/`, `results_v3/` (split files), `results_v4/`
(frozen bundle + MANIFEST), `llm_frontier_*.csv`, `llm_e2e_*.csv` (LLM labels per passage), `figures/`.

## Other
- `old/` — early-pipeline files (layer sweeps, old embeddings, gating curves, old evals, results_v2/) that no
  script reads and no document cites; moved here on 16 Nov 2026 with `git mv`, nothing deleted.
- Not committed on purpose: `llm_e2e_gemma3-12b.*` (Gemma 3 12B, Kaggle) and `llm_frontier_gemini31pro_42.*`
  (Gemini AI Studio run, file-name problem unresolved).
