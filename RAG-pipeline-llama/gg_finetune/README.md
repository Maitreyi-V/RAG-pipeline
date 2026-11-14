# GitaGround fine-tuning (secondary protocol)

Fine-tunes `BAAI/bge-reranker-v2-m3` twice per run:
- **verse scorer**: (passage, verse text) → logit. Becomes the 9th feature of your verse LR.
- **off-verse classifier**: (passage, neighbours) → logit. A new P_off variant next to the cue.

The frozen 9-lecture protocol stays the **primary** result. This is a separately declared experiment.

## 0. Protocol (decided before any training)

- 5 lecture-level folds over the 39 course lectures (tune + e2), stratified by language, seed 42.
- Run `cvk`: test = fold k, calibration = fold (k+1) mod 5, train = the other 3 folds.
  Every course lecture is tested exactly once.
- Run `e3`: train = folds 1–4, calibration = fold 0, test = the 12 unseen-speaker lectures (never trained on).
- Calibration lectures pick the LR weights, P_off variant, γ, β_off, τ. Test lectures are decoded once.
- **Go/no-go (Nov 21):** CV system ≥ frozen GitaGround-Cue + 0.03 Strict F1 on the **30 E2 test lectures**
  in at least one language. (Compare on the 30, not 39: the frozen baselines are in-sample on the 9 tuning lectures.)

## 1. Export your data (one adapter script on your side)

`data/passages.jsonl`, one line per passage, all 51 lectures (silver lectures optional, see §6):

```json
{"lecture_id": "video_02", "idx": 0, "lang": "en", "split": "e2", "speaker": "orig_en",
 "text": "<passage text as used by the pipeline>", "gold": [47, 48]}
```

- `idx`: 0..T−1 in lecture order, no gaps. `lang`: `kn` or `en`. `split`: `tune`, `e2`, `e3` or `silver`.
- `gold`: verse numbers 1..72 of chapter 2, **primary first**, then secondaries. `[]` = off-verse.

`data/verses.json`: list of 72 objects:

```json
{"v": 47, "kn": "<verse in Kannada script>", "roman": "<IAST>", "trans": "<English translation>"}
```

Kannada lectures are paired with `kn`, English lectures with `roman`; both get `trans`.

Optional `data/prior_scores/<lecture_id>.npy`, shape `[T, 72]`: used only to mine hard negatives on
training lectures. Use the **unsupervised** BGE-M3 profile cosine (cos_α), not the LR scorer
(the LR was fit on tuning gold, which is test in some folds).

## 2. Split and log (once)

```bash
python make_folds.py        # writes folds.json, prints the log entry
```

Paste the printed entry into the experiment log, commit `folds.json` **before any training**.
Check: 39 course lectures; English 2 per fold.

## 3. Train and score on Kaggle (GPU T4 ×2, Internet on)

Upload this folder + `data/` as a Kaggle dataset, copy to `/kaggle/working`, then per run:

```bash
for RUN in cv0 cv1 cv2 cv3 cv4 e3; do
  python build_examples.py --run $RUN
  python train.py --run $RUN --task verse
  python train.py --run $RUN --task off
  python score.py --run $RUN
  mkdir -p meta && cp models/${RUN}_verse/train_meta.json meta/${RUN}_verse.json \
                && cp models/${RUN}_off/train_meta.json meta/${RUN}_off.json
  rm -rf models/${RUN}_*        # each checkpoint is ~2.3 GB; Kaggle output limit is 20 GB
done
python calibrate_eval.py        # Platt-calibrates P_off on calib lectures, prints diagnostics
```

- **Do `cv0` alone first.** Check the loss falls between epoch 0 and 1, note the time, then run the rest.
- Two GPUs: run two runs at once with `CUDA_VISIBLE_DEVICES=0 …` and `CUDA_VISIBLE_DEVICES=1 …` in separate cells.
- Out of memory: `--bs 4 --accum 8` (same effective batch), or `--max_len 320`.
- Kaggle sessions stop at 12 h: `scores/` and `meta/` persist in `/kaggle/working`; restart from the next run.
- Settings are fixed: 2 epochs, lr 2e-5, batch 8 × accum 4, max_len 384, seed 42. Don't change them after seeing results.

Outputs per lecture: `scores/<run>/<lecture>_cal.npz` with `verse_feature [T,72]` (z-scored logit)
and `p_off_ft [T]` (calibrated). `calibrate_eval.py` prints verse top-1 and off-verse AUC by language:
diagnostics only, not the paper metric.

## 4. Plug into your decoder (your pipeline), per run

1. **Transitions and buckets:** count from the gold of that run's **train + calib** lectures, not from the
   9 tuning lectures (some tuning lectures are test lectures in some folds).
2. **Verse LR:** your 8 features + `verse_feature[t, v−1]`; fit on the **calib** lectures' gold.
3. **P_off variants:** (A) cue as now, (B) `p_off_ft`, (C) mean of cue and `p_off_ft`.
4. **Select** variant, γ, β_off, τ on calib lectures (same grid as before) by Strict F1.
5. **Decode the test lectures once** with the chosen settings; save predictions.

Pool test predictions over cv0–cv4 (each of the 39 lectures decoded once). `e3` gives the 12 unseen-speaker lectures.

## 5. Report back

| Lectures | CV system | Frozen GitaGround-Cue | Sol | Opus |
|---|---|---|---|---|
| E2 Kannada test (23) | | 0.820 | 0.934 | 0.911 |
| E2 English test (7) | | 0.631 | 0.792 | 0.758 |
| E3 new Kannada (9) | | 0.665 | 0.929 | 0.901 |
| E3 new English (3) | | 0.814 | 0.910 | 0.831 |

Plus: paired lecture bootstrap (10,000) for CV − Cue and CV − Sol; the chosen P_off variant per run;
the `calibrate_eval.py` printout; train times from `meta/`.

## 6. Silver data (only after the Nov 21 go/no-go passes)

- New Chapter 2 lectures only; **no lecture by an E3 speaker** (Sumith, Brahmanyacharya, Gaurmandal Das,
  Aparajitananda), none of the 51 annotated lectures.
- One Sol run, logged first, same frozen prompt. Keep a passage if Sol says NONE (→ `gold: []`), or Sol's verse
  agrees with a string anchor or recitation memory (→ `gold: [Sol's verses]`). Drop the rest.
- Add them to `passages.jsonl` with `"split": "silver"`, rerun `make_folds.py` and confirm `fold_of` is unchanged
  (silver never enters calib or test). `build_examples.py --silver_weight 0.4` (default).
- Ablation on the same folds: gold-only vs gold + silver (use `--out_dir`/`--out_root` to keep both).
  Keep silver only if it helps on calib **and** test.

## Tested

End-to-end on synthetic data with a tiny stand-in model (CPU): fold split, example building, training,
scoring, calibration, leakage checks (no lecture in two roles; each course lecture tested once).
Not tested here: the real `bge-reranker-v2-m3` on GPU, which needs Kaggle.
