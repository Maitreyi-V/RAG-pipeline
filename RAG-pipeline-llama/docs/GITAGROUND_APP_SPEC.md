# GitaGround app — build specification

For whoever builds the user-facing app. Everything below is already true in the
repo (branch `Implementation`) unless marked **TO BUILD**.

---

## 1. What the app does

A user types one of three things:

1. a **pasted shloka** (Devanagari, Kannada script or Roman letters, spelling can be noisy),
2. a **pasted explanation** (a few sentences from a book or a talk),
3. a **question** ("Why should a wise person not grieve?"), or simply a verse number ("2.47").

The app answers with the **verse(s) of Bhagavad Gita Chapter 2** it is about, the
verse's **theme**, which **speakers and traditions explain it and how deeply**, and
the **actual lecture passages**, with where in the lecture they occur.

The intelligence is already built and evaluated: it lives in **`gitaground.py`**.
The app is a presentation layer on top of it.

---

## 2. Rules you must not break

These are what the paper's numbers (Experiment E7) were measured on.

1. **Never re-implement retrieval or verse matching.** Call `GitaGround` methods.
   Do not change the constants or the ranking/abstention logic in `gitaground.py`
   (`K`, `N_VECTOR`, `MIN_LETTERS`, `VERSE_REF`, `ABSTAIN`, `retrieve()`,
   `rank_verses()`, `abstains()`, `answer()`). If the UI needs more data, add a
   *new* method; do not alter existing behaviour.
2. **The LLM never chooses a verse or a topic.** Verses come only from
   `answer()`. The LLM is used for one optional thing: summarising the 5 passages
   already retrieved (`llm_summary()`), which must keep its abstention instruction.
3. **Abstention is shown, never hidden.** If `answer()` returns `verse=None` with a
   note, show the note and nothing else under it.
4. **Never silently substitute a verse.** If `answer()` returns a `note`, show it
   prominently (e.g. "BG 2.30 is not discussed in any indexed lecture; the
   passages below are for BG 2.20").
5. **No invented numbers.** Do not show raw cosine scores and do not invent
   "High / Medium / Low" confidence bands (they were never calibrated). Positions
   are approximate and must be labelled so (§6.4).

---

## 3. Setup

```bash
cd RAG-pipeline/RAG-pipeline-llama
../.venv/bin/python -m pip install -r requirements.txt      # once; Python venv at repo root
# .env at the REPO ROOT must contain OPENAI_API_KEY=...       (only for the LLM summary)
../.venv/bin/python -m streamlit run gitaground_app.py --server.fileWatcherType none
```

- Always use `../.venv/bin/python` (a conda base install can shadow the venv).
- First start downloads **BAAI/bge-m3** (~2.3 GB) via sentence-transformers and
  loads it (Apple GPU "mps" if present, else CPU). Expect ~1 minute to start;
  afterwards a query takes about a second.
- **Embedding caches**: `GitaGround` needs `results/unit_embeddings.npz` and
  `results/unit_embeddings_units_heldout_v1.npz`. Ask Maitreyi for them (they are
  not committed). Do **not** regenerate them with `scripts/embed_scores.py`
  without `--out` pointing somewhere else: that script also overwrites
  `results/embed_scores.npz`, which the frozen baselines use.

---

## 4. Files the app reads (all already in the repo)

| file | what it is |
|---|---|
| `gitaground.py` | the engine (§5) |
| `results/step5_index_all.csv` | one row per passage, 1,884 passages, 51 lectures (§4.1) |
| `results/step5_segments_all.csv` | one row per contiguous run of one verse (§4.2) |
| `results/step2_features.npz` | string-match detections, used to build verse "meaning profiles" |
| `results/e7_abstain.json` | the locked abstention rule (do not edit) |
| `data/units_v2.csv`, `data/units_heldout_v1.csv` | passage **text** (`unit_id`, `text`, `n_words`) |
| `data/verse_themes.csv` | the 9 themes (T01–T09) per verse, made by the domain expert |
| `verse_database.py`, `verse_translations.py` | verse text (Devanagari, Kannada script) and English translations |
| `data/lectures_new.csv` | speaker, organisation, tradition, **YouTube id** for the 12 new-speaker lectures |

### 4.1 `step5_index_all.csv` columns
`unit_id, video_file, language (kn/en), speaker, tradition (Dvaita / Advaita / Gaudiya Vaishnava),
unit_index (order in the lecture), primary_verse ("BG 2.x" or "OFF"), p_primary,
p_off, theme_id, theme_name, secondary_verses ("BG 2.3;BG 2.7" or empty), segment_id,
word_start, word_end, lecture_words, pos_start, pos_end` (positions are 0–1 shares of the lecture).

- `primary_verse` = the verse this passage **explains**; `OFF` = a story/digression.
- `secondary_verses` = verses only **mentioned in passing** on this passage.
- Speaker is `original Kannada speaker` / `original English speaker` for the 39 original
  lectures (names to come, §6.4).

### 4.2 `step5_segments_all.csv` columns
`segment_id, video_file, primary_verse, theme_id, first_unit, last_unit, dwell
(number of passages = coverage depth), effective_dwell (dwell weighted by
confidence), pos_start, pos_end`.

---

## 5. The engine: `GitaGround` API

Create **once** per process (it loads the model and the index):

```python
from gitaground import GitaGround, ABSTAIN
g = GitaGround("all")          # "all" = every lecture. ("dev" = the 9 tuning lectures only)
```

| call | returns |
|---|---|
| `g.answer(text)` | **the main call**: `dict(ranked, verse, passages, note, query_for_summary)` |
| `g.rank_verses(text)` | all 72 verses as `[(ref, how), ...]`, `how` ∈ `"verse number"`, `"shloka match"`, `"meaning profile"` |
| `g.related(ref)` | verses that occur with `ref` on the same passages: `[(ref, count), ...]` |
| `g.coverage(ref)` | `{(speaker, tradition): {"segments": [segment rows, deepest first], "passing": n}}` |
| `g.extractive(text, ks, n=3)` | the n sentences of passages `ks` closest to `text`: `[(k, sentence), ...]` (no LLM) |
| `g.llm_summary(text, ks)` | GPT-4o-mini summary restricted to passages `ks`; may return the `ABSTAIN` sentence |
| `g.index[k]` | passage row `k` (a dict with the §4.1 columns) |
| `g.text[unit_id]` | passage text |
| `g.verses[ref]` | `{"ref", "devanagari", "kannada", ...}` |
| `g.translations[ref]` | list of English translations |
| `g.themes[ref]` | `(theme_id, theme_name)` |
| `g.by_verse[ref]` | indices `k` of passages whose primary verse is `ref` |

`answer(text)` fields:
- `ranked` — as `rank_verses`.
- `verse` — the verse Levels 2–3 are about, or `None` (abstained, or a named verse no lecture covers).
- `passages` — list of `k` (indices into `g.index`), at most 5, best first; `[]` when nothing honest can be shown.
- `note` — a message for the user, or `""`.
- `query_for_summary` — the text to feed `extractive()` / `llm_summary()` (for a bare verse number it is the verse's translation).

What `answer()` already handles (do not duplicate):
- verse numbers: "2.45", "BG 2.45", "Gita 2.45", "verse 45";
- shlokas in all three scripts (string matching against the verified verse text);
- abstention for off-topic input ("hello", "how does TCP work") → `note == ABSTAIN`;
- "named verse not covered by any lecture" → `verse=None` + note.

---

## 6. Screens and behaviour

### 6.1 Input
- One text box ("Paste a shloka, an explanation, or ask a question") + Submit.
- A toggle **Summary: Extractive (no AI) | AI summary (GPT-4o-mini)**. Default: Extractive.
- Optional filters (TO BUILD, apply only to what is *displayed*, never to `answer()`):
  tradition (Dvaita / Advaita / Gaudiya Vaishnava), language (Kannada / English).

### 6.2 Flow
```
a = g.answer(text)
if a["note"]:            show a["note"] (warning box)
if a["verse"] is None:   stop after Level 1 (if any verse was named) or after the note
else:                    Level 1 → Level 2 → Level 3
```
If the input named a verse or matched a shloka, Level 1 shows **only those verses**
(`[x for x in a["ranked"] if x[1] != "meaning profile"]`); otherwise the top 3 of `ranked`.

### 6.3 Level 1 — Overview (one card per matched verse)
- Verse reference and **how it was matched** (`verse number` / `shloka match` / `meaning profile`).
- Theme: `g.themes[ref]` → "T02 · Immortality and Eternal Nature of the Self".
- Devanagari (`g.verses[ref]["devanagari"]`), optional Kannada script, one translation.
- **Related verses**: `g.related(ref)[:5]`, each clickable (re-runs with that verse number).

### 6.4 Level 2 — Who explains it (multi-tradition comparison)
- `cov = g.coverage(a["verse"])`. Group by **tradition first, then speaker**.
- Under each speaker, one row per segment, deepest first:
  lecture · **dwell** ("explained over N passages") · effective dwell · position.
- Underneath: "Mentioned in passing in N other passages" (`passing`).
- "In depth" vs "passing" = segment (primary verse) vs secondary mention. **Do not add a
  dwell threshold** to call something "in depth"; none was validated.
- **Position / time**: until lecture durations exist show `~59%–100% of the lecture`.
  **TO BUILD**: `data/lecture_meta.csv` with `video_file, speaker, organisation,
  tradition, youtube_id, duration_sec` (Maitreyi will supply). Then show
  `~MM:SS – MM:SS (approximate)` with `seconds = pos × duration_sec`, and a YouTube
  link `https://www.youtube.com/watch?v=<id>&t=<seconds>s`. The estimate assumes a
  constant speaking rate — always label it "approximate". Use the same file to
  replace the two placeholder speaker names.

### 6.5 Level 3 — Evidence
- Summary block:
  - Extractive: `g.extractive(a["query_for_summary"], a["passages"])` → quote each sentence with speaker, lecture, position.
  - AI summary: `g.llm_summary(...)` — label it "AI summary of the 5 passages below"; if it returns `ABSTAIN`, show that.
- Then the passages `a["passages"]`, each expandable:
  header `speaker · lecture · passage unit_index · position · primary_verse`;
  if `primary_verse != a["verse"]` add "found by text similarity" (the engine mixes 2
  plain-search passages with 3 verse passages — this must be visible);
  body = `g.text[unit_id]`; footer = theme, "also mentions …" (`secondary_verses`).
- **TO BUILD (plan: ±1 context)**: a "show surrounding passages" link that shows the
  previous and next passage of the same lecture (same `video_file`, `unit_index ± 1`).

### 6.6 Browse mode (TO BUILD; plan's Course → Theme → Verse → Segment)
- 9 themes from `data/verse_themes.csv` → verses in the theme → for each verse the
  Level 2 view → click a segment → its passages in order (`first_unit` … `last_unit`).
- Verses with no segment: "not explained in the indexed lectures".

---

## 7. Choice of technology

**Recommended (deadline 10 Nov): keep Streamlit** and extend `gitaground_app.py`
(a working prototype of Levels 1–3 already exists and already uses `GitaGround("all")`).
- Keep `@st.cache_resource` around the `GitaGround` constructor (load once).

**Alternative: FastAPI + React** (only if there is time):
- Load `GitaGround("all")` once at startup (module-level singleton); it is read-only,
  so concurrent requests are safe; run model calls in the threadpool (`def`, not `async def`).
- Endpoints (JSON):
  - `POST /api/answer` `{text, summary: "extractive"|"llm"}` → `{note, verse, matched:[{ref, how, theme_id, theme_name, devanagari, translation, related:[...]}], coverage:[{tradition, speaker, segments:[...], passing}], summary:[...], passages:[{unit_id, video_file, speaker, unit_index, pos_start, pos_end, primary_verse, secondary_verses, theme_id, text, by_text_similarity}]}`
  - `GET /api/themes`, `GET /api/verse/{ref}`, `GET /api/segment/{segment_id}`, `GET /api/passage/{unit_id}?context=1`
- Never send `p_primary`/`p_off` as "confidence %".

---

## 8. Acceptance tests (must all pass)

| input | expected |
|---|---|
| `2.47` | Level 1 shows only BG 2.47 (theme T06); Levels 2–3 for 2.47; no other verses listed |
| `verse 47`, `BG 2.47`, `Gita 2.47` | same as above |
| `hello` | the abstention sentence, nothing else |
| `How does TCP congestion control work?` | abstention |
| `नैनं छिन्दन्ति शस्त्राणि नैनं दहति पावकः` | BG 2.23, matched by "shloka match" |
| the same line in Kannada script | BG 2.23 |
| the same line romanised (`nainam chindanti shastrani nainam dahati pavakah`) | BG 2.23 first; the Roman matcher also lists similar verses (2.29, 2.60, 2.19) — expected |
| `Why should a wise person not grieve when the body dies?` | a Chapter 2 verse via "meaning profile", 5 passages, any non-verse passage marked "found by text similarity" |
| AI summary toggle on a question | summary cites [1]–[5] or returns the abstention sentence |

Headless check (no browser):
```python
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("gitaground_app.py", default_timeout=600); at.run()
at.text_area[0].input("2.47").run()
assert not at.exception
```

---

## 9. Known limitations (show in an "About" page)

- 51 lectures, 6 speakers, Chapter 2 only.
- Verse labels on passages are automatic (measured strict F1: 0.82 Kannada test,
  0.60 English test lectures); stories inside long explanations are often still
  labelled with the verse.
- Positions/timestamps are estimates from word counts.
- Abstention catches most off-topic questions (21 of 25 in the test set), not all.

---

## 10. Suggested order of work

1. Run the prototype and the §8 tests (they pass on 18 Oct).
2. Level 2 grouping by tradition → speaker; clickable related verses.
3. `data/lecture_meta.csv` → real speaker names, approximate times, YouTube links.
4. ±1 context; browse-by-theme mode.
5. About page (§9); polish; screenshots for the report.
