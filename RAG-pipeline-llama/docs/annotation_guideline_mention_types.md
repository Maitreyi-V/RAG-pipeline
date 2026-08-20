# Mention-Type Annotation Guideline (T1–T5)

**Project:** Verse Grounding in Oral Discourse — Bhagavad Gita Chapter 2
**Purpose of this document:** Every gold verse mention in our corpus must be tagged with *how* the
verse appears in the speech. This "mention type" is the axis that turns our results from a flat
"detection is hard (F1 ≈ 0.5)" into an explanatory finding: string matching solves verbatim
recitation, embeddings handle quotes and translations, and even LLMs collapse on paraphrase and
allusion. The taxonomy, its guideline, and the inter-annotator agreement on it all go **into the
paper**, so annotate carefully and consistently.

Read this whole page once before annotating. Keep it open while you work.

---

## What you are tagging

A **mention** = one span of the transcript where a specific canonical verse (e.g. BG 2.11) is being
recited, quoted, translated, explained, or alluded to. You already have gold mentions:

- **Kannada / Dvaita:** rows in `data/annotations.csv` (each row = one verse, with `sanskrit_shloka`,
  `explanation_start`, `explanation_end`).
- **English / Advaita:** verses in `advaita_ground_truth.py` and the spans behind
  `advaita_test_questions.json`, located in the `video_XX.json` transcripts.

For **each** gold mention, assign exactly **one** mention type T1–T5 (the *dominant* one — see
the tie-break rules). Add it in the `mention_type` column (see **Output format** below).

---

## The five mention types

| Tag | Name | One-line test | Which paradigm should solve it |
|-----|------|---------------|-------------------------------|
| **T1** | Verbatim recitation | The **full** Sanskrit verse is spoken/recited (even if ASR mangles the spelling). | Layer 1 — string / fuzzy matching |
| **T2** | Partial quote / key phrase | A **recognizable Sanskrit fragment** of the verse is spoken, not the whole thing. | Layer 1 (partial) → Layer 3 |
| **T3** | Direct translation | The verse's meaning is given as a **faithful line-by-line translation** into English/Kannada, clearly presented *as* the verse. | Layer 3 — embedding similarity |
| **T4** | Paraphrase / explanation | The verse's content is **explained in the speaker's own words**, not quoted or literally translated. | Hard — Layer 4 LLM, degraded |
| **T5** | Allusion | The verse's *theme/idea* is discussed **without** signalling that a specific verse is the source. | Open problem — all layers fail |

The tiers are a **difficulty ladder**: T1 is nearly solved, T5 is the open research problem. If a
mention is genuinely ambiguous between two adjacent tiers, that ambiguity is itself signal — record
it (see tie-breaks) rather than agonising.

---

### T1 — Verbatim recitation
**Definition.** The speaker recites the complete verse (or a complete pāda/line sequence covering the
whole verse) in Sanskrit. ASR transliteration errors do **not** disqualify T1 — what matters is that
the *speaker* recited the full verse.

**Diagnostic:** Could you, in principle, line up the spoken words against the canonical Devanāgarī
verse token-for-token? → T1.

**Real examples from our data:**
- *Kannada, `annotations.csv`, gita-1st_video, BG 2.1:*
  `ತಂ ತಥಾ ಕೃಪಯಾವಿಷ್ಟಂ ಅಶ್ರುಪೂರ್ಣಾಕುಲೇಕ್ಷಣಂ ...` — full śloka recited before explanation. This is
  the classic Dvaita case and why Kannada detection scores F1 ≈ 0.92.
- *English, `video_02.json` [~290s], BG 2.11:* speaker says "Please repeat after me" then recites
  *"Asocyananvasocastvam prajnavadanshchabharsase / Gatasunagatasunstha / Nanusocanti panditah."*
  Full verse → **T1**, even though it appears inside an English lecture and Whisper mangled it.

---

### T2 — Partial quote / key Sanskrit phrase
**Definition.** The speaker quotes a **fragment** of the verse in Sanskrit — a pāda, a compound, or a
signature phrase — but not the whole verse.

**Diagnostic:** Is there Sanskrit-origin text that maps to *part* of the verse, but you could not
reconstruct the full verse from what was said? → T2.

**Real examples:**
- *`video_02.json` [~352s], BG 2.11:* after translating, the speaker isolates and repeats the closing
  pāda *"Nanusocanti panditah"* ("the wise do not grieve"). Fragment of the verse → **T2**.
- *`video_02.json` [~301s]:* an isolated line *"Gatasunagatasunstha"* spoken on its own would be T2;
  spoken as part of the full recitation it folds into the T1 above (dominant-span rule).
- The canonical `sthitaprajñasya kā bhāṣā` (BG 2.54) when quoted as a phrase → **T2**.

---

### T3 — Direct translation
**Definition.** The speaker renders the verse's meaning as a **faithful translation**, clearly framed
as *the verse itself* ("Krishna says…", "the verse means…", "he says…"), staying close to the literal
content rather than expanding on it.

**Diagnostic:** Is this a line you could put next to a published translation and call it "a
translation of BG 2.x," AND is it framed as the verse speaking (not the speaker's commentary)? → T3.

**Real examples:**
- *`video_02.json` [~315–326s], BG 2.11:* "So the blessed Lord said, you have been sorrowing for those
  who should not be grieved for, and yet you are talking learned words… The learned grieve neither for
  the dead nor for the living." A direct English translation of the verse → **T3**.
- *`video_01.json` (BG 2.3), [~2921s]:* "Krishna says you have to stand up, shake off this meanness,
  this smallness." Close translation of *kṣudraṃ hṛdaya-daurbalyaṃ tyaktvottiṣṭha* → **T3** (borderline
  T4; see tie-breaks).

---

### T4 — Paraphrase / explanation without quoting
**Definition.** The speaker conveys the verse's teaching **in their own words** — expanding,
illustrating, restating — while still clearly discussing *that* verse. No Sanskrit, no literal
translation.

**Diagnostic:** Is the verse being *explained* rather than *quoted or translated*, and is it clear
which verse is being explained (position in the sequence, "this verse", the surrounding recitation)?
→ T4.

**Real examples:**
- *`video_01.json` [~3572–3595s], BG 2.7:* "…when they are in confusion, unhappy, they want help. They
  repeat this verse in a prayerful attitude to God before going to sleep…" — commentary *about* the
  verse's content and use, own words → **T4**.
- Any stretch where the speaker walks through "what this verse is really telling us is…" → **T4**.
  This is where F1 starts collapsing.

---

### T5 — Allusion
**Definition.** The speaker discusses the **idea, theme, or teaching** of a verse **without any signal
that a specific canonical verse is the source** — no verse number, no "this verse", no recitation
nearby. The link to the verse exists only in the annotator's/gold knowledge.

**Diagnostic:** Strip away your prior knowledge of the corpus — would a listener know a *specific
verse* is being referenced? If **no**, but the gold says a verse is grounded here → **T5**.

**Real examples:**
- *`video_01.json` [~160–207s]:* the long passage on "our spiritual battle… each one of us has to
  fight this battle of life daily" evokes the framing of Chapter 2 without citing any verse → **T5** if
  gold-linked to a verse; otherwise not a mention at all.
- Discussing the *qualities of the sthitaprajña* (steady-minded person) generally, without pointing to
  BG 2.54–2.72 → **T5**. These broad-theme passages are exactly what get absorbed by "attractor"
  verses (BG 2.39, 2.45) in the semantic layer.

---

## ⚠️ ADDENDUM (2026-08-20) — T5 in the exhaustive workflow

**Read this. It overrides the T5 wording above.**

The sections above were written for a *verification* workflow: you already had a gold answer key
(`annotations.csv`, `advaita_ground_truth.py`) and only had to tag *how* each known mention appeared.
That is why the T5 test says "if **the gold** says a verse is grounded here."

**We no longer work that way.** You now annotate every unit in `data/units_v1.csv` from scratch and
decide yourself whether a verse is grounded. There is no answer key to consult — **you are writing
it.** So the old T5 test is circular, and left as-is two annotators will split on the same passage:

> *"A wise person isn't shaken by pleasure or pain; he stays steady whatever happens."*
> — Annotator A: that's the sthitaprajña, → T5 on BG 2.56.
> — Annotator B: no verse is signalled, → ∅ No verse here.

Both follow the old text. Both are "right". That disagreement is concentrated in T5 (the fuzziest
tier), so left unfixed it lands directly on our reported Cohen's κ.

### The operative rule

> **Tag T5 only if you can name ONE specific verse** (or a tight 2–3 verse set) whose content this
> passage is explicating. Your confidence must come from either:
> - **(a) content** — the passage matches that verse's specific content closely, or
> - **(b) discourse context** — the speaker has been working through that verse and is still
>   elaborating it without re-citing.
>
> If the most you can honestly say is *"this is generally about Chapter 2's themes"* without landing
> on a specific verse → **∅ No verse here**, **not** T5.
>
> If you are genuinely torn, tag T5 **and tick `uncertain`**. Those rows get adjudicated, and the
> count of them is a dataset statistic we report.

### Applying it to the two examples above

- The *"spiritual battle of life"* passage in `video_01.json`: no specific verse is identifiable from
  content or context → **∅**, not T5.
- The *sthitaprajña* passage: **T5 on the specific verse** if the speaker has been walking through,
  say, BG 2.56 and is still unpacking it. If it is a free-floating remark about staying calm → **∅**.

**"No identifiable verse" is never automatically T5.** T5 means a specific verse *is* grounded here
and the speaker simply gave no signal about which one.

---

## Decision procedure (apply top-down, stop at first match)

```
1. Is the FULL verse spoken in Sanskrit (ASR errors OK)? ............ YES → T1
2. Is a Sanskrit FRAGMENT of the verse spoken? ..................... YES → T2
3. Is there a faithful translation, framed as the verse? .......... YES → T3
4. Is the verse explained in the speaker's own words,
   with the specific verse still identifiable? .................... YES → T4
5. Only the theme/idea is present, no signal of a specific verse,
   BUT you can still name ONE specific verse it explicates
   (from content or discourse context — see ADDENDUM) ............. YES → T5
6. Otherwise — narration, story, logistics, a digression, or a
   general theme you cannot pin to a specific verse ............... → ∅ No verse here
```

Step 6 is a **normal, frequent outcome** — most units in a lecture are not verse mentions. Do not
force a verse onto a passage that does not carry one; that corrupts the gold data far more than a
missed mention does.

---

## Tie-break & edge-case rules

1. **One dominant tag per mention.** A single passage often climbs the ladder (recite → translate →
   explain), as BG 2.11 does in video_02. Tag the mention by its **highest-fidelity** form present
   (T1 > T2 > T3 > T4 > T5). Rationale: if the verse is recited *somewhere* in the span, a detector can
   catch it there, so the mention's difficulty is set by its easiest evidence.
   - *Exception for per-type scoring later:* if a span clearly contains **separable** sub-mentions
     (e.g. a T1 recitation at 0:05 and an unrelated T4 explanation of the *same* verse 3 minutes later
     with other verses in between), you may record two rows. When in doubt, keep it one row.

2. **ASR garble never downgrades T1/T2.** Judge by what the speaker said, not what Whisper wrote.
   Sanskrit-term ASR errors are a known confound and are being measured separately.

3. **T3 vs T4 (the frequent hard call).** If it reads like *a translation of the verse* → T3. If it
   reads like *the speaker teaching around the verse* → T4. The "Krishna says stand up, shake off this
   meanness" case (video_01, BG 2.3) sits on this line — default to **T3** when it tracks the verse
   line-by-line, **T4** when it adds illustration or application.

4. **T4 vs T5 (the other hard call).** The deciding question is *identifiability*: in T4 you can tell
   **which** verse; in T5 you only have a theme. "This verse says…" or a verse number nearby → T4.
   Pure theme with no anchor → T5.

5. **Reference-by-number is not its own tier.** "The 11th verse" / "second chapter, verse 11" is a
   *locator*, not a mention type. Tag the tier by *how the content is delivered* in that span. Note the
   presence of an explicit number in the `has_explicit_ref` flag (below) — useful for analysis.

6. **Not-a-mention.** If, on review, no specific verse is actually being grounded (pure narrative,
   logistics, "which edition to buy"), it is **not** a gold mention. Flag it for removal from gold
   rather than forcing a T5.

7. **When you cannot decide between two adjacent tiers,** pick the higher-fidelity one and set
   `uncertain = 1`. Do **not** leave blank. The `uncertain` flag lets us report annotation confidence
   and feeds the disagreement-adjudication pass.

---

## Output format

Add these columns to the gold. For Kannada, extend `data/annotations.csv`; for English, add to the
Advaita gold table (or a parallel `data/mention_types.csv` keyed by `video_file` + `verse_ref`):

| Column | Values | Meaning |
|--------|--------|---------|
| `mention_type` | `T1`–`T5` | The dominant tier (required). |
| `has_explicit_ref` | `0` / `1` | Was the verse named by number/position in the span? |
| `uncertain` | `0` / `1` | Set to 1 when torn between two adjacent tiers. |
| `annotator` | initials | Who tagged it (needed for the κ pass). |
| `notes` | free text | The phrase you used to decide, edge cases, adjudication notes. |

---

## Inter-annotator agreement protocol (do this, reviewers will ask)

1. **Two annotators independently** tag the **same 4–5 videos** (mix of Dvaita and Advaita), using
   only this guideline — no discussion during tagging.
2. Compute **Cohen's κ** on `mention_type` over the shared mentions (a small helper script will be
   added: `scripts/compute_iaa.py`).
3. **Adjudicate** every disagreement together; record the resolution in `notes`. Adjudicated labels
   become gold for those videos.
4. Report κ (and per-tier confusion) in the dataset section. κ on T3/T4 and T4/T5 boundaries will
   likely be the lowest — that is expected and worth discussing, not hiding.

---

## Why this matters (keep in mind while tagging)

Once every mention is typed, we re-score Layers 1/3/4 **broken down by tier**. The expected headline:

> String matching solves **T1** (F1 ≈ 0.92). Embeddings recover **T2–T3**. Even GPT-4o-mini degrades on
> **T4** and collapses on **T5** — paraphrase and allusion in free-flowing oral discourse are the open
> problem.

Your tags are what make that sentence provable. Tag honestly; the hard-to-call cases are the paper.
