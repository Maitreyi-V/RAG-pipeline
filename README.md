# GitaGround

Verse grounding in oral discourse: Bhagavad Gita, Chapter 2.

Paste a shloka (Devanagari, Kannada script or Roman letters), an explanation, a question, or a verse
number. GitaGround returns the Chapter 2 verse it is about, the verse's theme, which speakers and
traditions explain it and over how many passages, and the lecture passages themselves with approximate
positions in the lecture.

The corpus is 51 lectures from 6 speakers across three traditions: Dvaita, Advaita and Gaudiya Vaishnava.

## How it works

- **Verse matching:** by verse number, by string matching against the verified verse text (all three
  scripts), or by similarity to a meaning profile of each verse (BGE-M3 embeddings).
- **Coverage:** every lecture is split into 100-word passages, each labelled with the verse it explains.
  Consecutive passages on one verse form a segment, so "explained over N passages" is a measure of depth.
- **Abstention:** off-topic input gets a single "could not find sufficient support" message.
- **Optional AI summary:** GPT-4o-mini rewrites only the five passages already retrieved. It never chooses
  a verse or a topic.
- The app shows no similarity scores or confidence bands, and lecture positions are labelled approximate.

## Setup

Python 3.12 or newer.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Mac/Linux: source .venv/bin/activate
pip install -r requirements-app.txt
```

- Create a `.env` at the repo root with `OPENAI_API_KEY=...` (only needed for the AI summary).
- Place the embedding caches `results/unit_embeddings.npz` and
  `results/unit_embeddings_units_heldout_v1.npz` in `results/`. They are large and not committed.
- The first start downloads `BAAI/bge-m3` (about 2.3 GB).

## Run

```bash
cd RAG-pipeline-llama
python -m streamlit run gitaground_app.py --server.fileWatcherType none
```

Views:

- **Ask:** matched verses, who explains the verse (by tradition, then speaker, with related verses you can
  click), and the evidence passages with an optional surrounding-passage view.
- **Browse by theme:** nine themes, then a verse, then a segment, then its passages in order.
- **About.**

The sidebar filters (tradition, language) only hide results; they never change what is found.

Try: `2.47`, `nainam chindanti shastrani nainam dahati pavakah`,
`Why should a wise person not grieve when the body dies?`

## Tests

```bash
python test_gitaground_app.py
```

Runs the app headless with stand-in embeddings, so no model is needed.

## Lecture metadata

`data/lecture_meta.csv` has the columns `video_file, speaker, organisation, tradition, youtube_id, duration_sec`.
It supplies speaker names, MM:SS positions and YouTube links. With `duration_sec` blank, positions show as a
share of the lecture (for example `~59%-100% of the lecture (approximate)`) and links carry no time.

## Repository layout

| path | contents |
|---|---|
| `gitaground.py` | the engine: matching, ranking, abstention, coverage |
| `gitaground_app.py`, `gitaground_ui.py` | Streamlit app and its display helpers |
| `test_gitaground_app.py` | headless app tests |
| `scripts/` | experiment pipeline: units, detection, propagation, index building, agreement, frozen runs |
| `results/` | experiment outputs, `step5_index_all.csv`, `step5_segments_all.csv`, `e7_abstain.json` |
| `data/` | passages, transcripts, annotations, `verse_themes.csv`, `lecture_meta.csv` |
| `docs/` | annotation guideline |
| `annotate_app.py` | blind annotation tool |
| `shloka_detector.py`, `roman_detector.py`, `layer3_semantic.py`, `layer4_llm_detector.py`, `embeddings.py`, `verse_database.py`, `verse_translations.py`, `config.py` | detector and data modules used by the engine and scripts |

## Licence

See `LICENSE`.
