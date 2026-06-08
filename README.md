# The Unofficial Guide — Project 1

A retrieval-augmented question-answering system over crowd-sourced student
reviews of how difficult courses are at UC Riverside. Ask "how hard is BIOL005B
and why?" or "who's the easiest professor for CS008?" and get an answer grounded
**only** in real student reviews, with the source reviews cited.

## Pipeline at a glance

```
Google Sheet ──► build_chunks.py ──► embed_index.py ──► retrieve.py ──► generate.py ──► app.py
 (raw reviews)    clean + chunk       embed + store      top-k search    grounded LLM     Gradio UI
                  data/chunks.jsonl   ChromaDB           (+class filter)  (Groq llama-3.3)
```

| Stage | File | What it does |
|-------|------|--------------|
| Ingest + chunk | [src/build_chunks.py](src/build_chunks.py) | Download the sheet, clean it, emit `data/chunks.jsonl` |
| Embed + store | [src/embed_index.py](src/embed_index.py) | Embed chunks with MiniLM, store in a persistent ChromaDB collection |
| Retrieve | [src/retrieve.py](src/retrieve.py) | Embed the query, optional class-code filter, return top-6 chunks |
| Generate | [src/generate.py](src/generate.py) | Grounded answer via Groq + programmatic source attribution |
| Interface | [src/app.py](src/app.py) | Gradio web UI |

### Run it

```bash
pip install -r requirements.txt
cp .env.example .env          # then put your real GROQ_API_KEY in .env
python src/build_chunks.py    # Google Sheet -> data/chunks.jsonl
python src/embed_index.py     # build the ChromaDB vector store
python src/app.py             # launch the UI at http://127.0.0.1:7860
```

---

## Domain

Course difficulty at UC Riverside. Official course catalogs describe *what* a
class covers but say nothing about how hard it actually is to pass — exam style,
workload, weeder reputation, which professor to take or avoid, or how a class has
changed year to year. That knowledge lives in informal student lore. This system
turns a crowd-sourced spreadsheet of first-hand student reviews into something
queryable in plain English, so a student can get the candid "should I take this?"
read that the catalog will never give them.

---

## Document Sources

The corpus is a single, large, continuously crowd-sourced Google Sheet of
student-submitted course reviews (one row per review, contributed via a public
Google Form). After cleaning it yields **3,558 reviews across 818 distinct UCR
courses**, spanning roughly **2017–2025**. Each review carries a class code, the
class's average difficulty, the reviewer's own 0–10 difficulty rating, the free-
text review, and a date.

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | UCR Course Difficulty Database (crowd-sourced spreadsheet) | Google Sheet (CSV export) | https://docs.google.com/spreadsheets/d/1qiy_Oi8aFiPmL4QSTR3zHe74kmvc6e_159L1mAUUlU0 |

Although it is one file, it aggregates thousands of independent contributors and
covers the full spread of difficulty (the same class often has ratings from 1 to
10), which is exactly the variety of perspective the system needs.

---

## Chunking Strategy

**Chunk size:** One chunk per review (row-based, not fixed-length), capped at
**256 tokens** measured with the embedding model's own tokenizer (≈115 tokens on
average; the cap is rarely hit). Each chunk is prefixed with a metadata header —
`[Class: CS010A | Difficulty rating: 5/10 | Class average: 3.38/10 | Date: ...]`
— so a retrieved chunk carries its class, rating, and date with it. One extra
**summary chunk per class** (818 of them) carries the class code, average
difficulty, review count, and rating range so numeric/aggregate questions
("what's the average difficulty of CS010A?") retrieve cleanly.

**Overlap:** Zero between separate reviews — they are independent observations,
and overlap would bleed one class's review into another's. A ~**40-token
overlap** is applied **only** when a single oversized review must be split, so a
thought spanning the split stays retrievable in at least one piece.

**Why these choices fit your documents:** The corpus is a structured set of short,
independent opinions, each tied to one class with a numeric rating and date — so
the review is the natural atomic unit. Fixed-length chunking would both fragment
long reviews and merge unrelated short ones from different classes. The 256-token
cap matches all-MiniLM-L6-v2's max sequence length (it silently truncates beyond
256 tokens), so larger chunks would lose text at embedding time anyway. Splitting
uses the model tokenizer's character offsets to slice the *original* text, so
casing and punctuation are preserved (decoding token IDs back to text would have
lowercased everything).

**Preprocessing before chunking** (see [src/build_chunks.py](src/build_chunks.py)):
- **Forward-fill class codes** onto continuation rows — only the first review per
  class carries the code in the sheet; the rest are blank and inherit it.
- **Forward-fill the per-class average difficulty**, which similarly lives only on
  each class's header row.
- **Drop junk**: the announcement banner row, plus 46 rows with no review text
  (which include 22 `See OTHER` cross-reference pointers).
- **Normalize fields**: coerce ratings/averages to numbers (`"Github"`, `"See
  CLA017"` → null), collapse whitespace, parse dates and fix an obvious far-future
  typo year.
- **Fold 6 `Note: Formerly <CODE>` rename annotations** (not real reviews) into
  the relevant class's summary chunk instead of embedding them as content-free
  review chunks.

**Final chunk count:** **4,536 chunks** — 3,718 review chunks (309 of which came
from splitting oversized reviews) + 818 per-class summary chunks.

---

## Embedding Model

**Model used:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, normalized
vectors; cosine distance in ChromaDB). Chosen because it runs locally with no API
cost or key, is fast, and 384-dim vectors are cheap to store and search for a
corpus this size. It is English-only and truncates at 256 tokens — both fine
here, since the reviews are English and chunks are capped at 256 tokens.

**Production tradeoff reflection:** If deploying for real users with cost no
object, I would weigh:
- **Context length** — a longer-context model (e.g. OpenAI `text-embedding-3-large`,
  ~8k tokens) would not truncate the essay-length reviews MiniLM cuts at 256, so
  I could drop the splitting step entirely.
- **Domain/accuracy** — reviews are informal, sarcastic, and slang-heavy ("weeder
  class," "got Jeff'd," "destroyed my GPA"). A larger model captures connotation
  and sarcasm better, which directly improves retrieval on opinion text. This is
  the most relevant gain for this corpus.
- **Multilingual** — not needed; the corpus is English, so I'd avoid multilingual
  models that trade English accuracy for languages I don't use.
- **Latency vs. cost** — MiniLM is instant and free locally; API models add
  network latency and per-call cost. For a class project the quality gain doesn't
  justify it, but at scale I'd accept the latency for the accuracy on sarcasm.

---

## Grounded Generation

The generator is **Groq `llama-3.3-70b-versatile`** (free-tier, OpenAI-compatible)
called from [src/generate.py](src/generate.py). Grounding is *enforced*, not
suggested, through both the prompt and the code.

**System prompt grounding instruction** (verbatim, abridged):

> Use ONLY the facts found in the CONTEXT. Never use prior knowledge, training
> data, or assumptions about how courses or professors 'usually' work. If the
> CONTEXT does not contain enough information to answer, reply with EXACTLY:
> *"I don't have enough information on that."* … Reviews for one class often
> disagree; report the range/spread rather than presenting one review as the
> single truth. Do not invent professor names, ratings, or dates not present in
> the CONTEXT.

Structural choices that back this up:
- The question and the retrieved chunks are sent together; the chunks are labeled
  `[S1]…[S6]` and each carries its class/rating/date metadata header, so the model
  has concrete, attributable text to ground in.
- If retrieval returns **zero** chunks, the code returns the refusal string
  **without ever calling the LLM**.
- `temperature=0` for deterministic, conservative answers.

**How source attribution is surfaced in the response:** It is **programmatically
guaranteed**, not left to the LLM. After generation, the code appends a `Sources:`
block built directly from the retrieved chunks (their class, type, date, and chunk
id) — so the source list cannot be hallucinated or omitted even if the model
forgets to cite. The model is *additionally* asked to cite `[S#]` labels inline,
and those labels line up exactly with the appended list. Example:

```
Reviewers repeatedly warn against taking CHEM001A with Professor Bartels [S1], [S2]...

Sources:
  [S1] CHEM001A — student review (2020-12-11)  (chunk CHEM001A__r600)
  [S2] CHEM001A — student review (2018-12-17)  (chunk CHEM001A__r582)
  ...
```

---

## Evaluation Report

Run with the full pipeline (retrieval top-k=6 + class-code filter + Groq
generation). Responses summarized; full text is reproducible via
`python src/generate.py "<question>"`.

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | Average difficulty of CS010A? | 3.38 | "The average difficulty of CS010A is **3.38/10**," cites the summary chunk. | Relevant | Accurate |
| 2 | Average difficulty of BIOL005B and why hard? | 7.59; memorization-heavy weeder, animal/plant diversity | "**7.59/10**; memorization-heavy, lots of information, hard tests," notes the 3–10 spread and that a fair TA/prof can ease it. | Relevant | Accurate |
| 3 | Which professor to avoid for CHEM001A? | Ludwig Bartels | "Reviewers warn against **Professor Bartels** ('avoid Bartels at all costs')," cites [S1][S2]. | Partially relevant | Accurate |
| 4 | What makes CS061 hard to pass? | Weeder; LC-3 assembly; ≥70% on final or fail; heavy labs | "Fast pace, heavy programming, must get **70%+ on the final or fail**, LC-3 limitations, memorization for the final." | Relevant | Accurate |
| 5 | Easiest professor for CS008? | Toby Gustafson | "**Professor Gustafson (Toby Gustafson)** — class is easy/a breeze," flags the one dissenting reviewer. | Relevant | Accurate |

**Retrieval quality:** Relevant for 4/5; Partially relevant for Q3 (see below).
**Response accuracy:** Accurate for all 5.

A separate grounding probe — *"What is the capital of France?"* — correctly
returns *"I don't have enough information on that."* even though the answer is
trivially in the LLM's training data, confirming answers are confined to the
retrieved corpus.

---

## Failure Case Analysis

**Question that failed (retrieval, partially):** *"Which professor do reviewers
repeatedly warn against taking for CHEM001A?"*

**What the system returned:** The correct answer (Professor Bartels) at ranks #1
and #2 — but ranks #3–#6 were **glowing reviews of *other* CHEM001A professors**
(Generoux, Hartman, Fokwa), the opposite of a warning.

**Root cause (retrieval / embedding stage):** The class-code filter correctly
kept every chunk on CHEM001A, so this is not a wrong-source failure — it is a
*semantic* one. In MiniLM's vector space, "avoid this professor at all costs" and
"best professor ever" sit close together because both are dense professor-
evaluation language; the model encodes the *topic* (a professor's teaching) more
strongly than the *polarity* (warning vs. praise). So a "warn against" query pulls
in praise too. The cosine distances reflect this — the warnings sat around
0.39–0.40 and the praise only slightly farther at 0.45–0.46.

**Why it still answered correctly:** the two warnings ranked highest, and the
grounding prompt tells the model to read all six and report which professor is
*warned against*, so it singled out Bartels despite the off-polarity neighbors.

**What I would change to fix it:** (1) Use an embedding model that better captures
sentiment/sarcasm on opinion text (the production tradeoff above). (2) Add a light
re-ranking or sentiment filter that boosts chunks containing negative cues
("avoid," "do not take," "worst") for "who should I avoid" style queries. (3)
Retrieve a few more chunks (k=8–10) for whole-class professor questions so the
warnings aren't crowded out, accepting a slightly larger prompt.

---

## Spec Reflection

**One way the spec helped you during implementation:** The "Anticipated
Challenges" section of `planning.md` named the blank-class-code problem before I
wrote any code — only the first review per class carries its code in the sheet,
the rest are blank. Because the spec called it out, the forward-fill was the first
thing the ingestion step did, so no review ever lost its class identity. The same
section's note on contradictory within-class reviews directly justified retrieving
top-k=6 and prompting the model to report the spread, which is exactly what made
the BIOL005B and CS008 answers honest about disagreement.

**One way your implementation diverged from the spec, and why:** The architecture
diagram specified **FAISS** for the vector store, but the implementation uses
**ChromaDB**. The project's `requirements.txt` ships ChromaDB, and Chroma gives
persistence and first-class metadata filtering (the `where={"class_code": ...}`
filter that fixes cross-class contamination) out of the box, whereas with FAISS
I'd have had to bolt on a separate metadata sidecar. The concept is identical — an
ANN index over the chunk vectors with cosine distance — so the change is purely in
the tool, not the approach. I also added a per-class **summary chunk** type that
the original chunking note hinted at but didn't fully specify, because numeric
questions ("what's the average difficulty?") retrieve far more reliably against a
single summary than against scattered individual reviews.

---

## AI Usage

**Instance 1 — Cleaning + chunking**

- *What I gave the AI:* The raw sheet schema, my Chunking Strategy section, and the
  "blank class codes" challenge from `planning.md`.
- *What it produced:* A pandas script that forward-fills class codes, drops junk
  rows, normalizes fields, and chunks one-per-review with the 256-token cap and
  40-token overlap on oversized reviews.
- *What I changed or overrode:* I directed it to (a) **also** forward-fill the
  per-class average (it initially only filled the class code), and (b) split
  oversized reviews by slicing the *original* text via the tokenizer's character
  offsets rather than decoding token IDs, because decoding lowercased everything
  and mangled apostrophes ("anna ' s archive"). I also had it fold the six
  `Note: Formerly <CODE>` annotations into summaries after auditing random chunks
  and finding they were editorial notes, not reviews.

**Instance 2 — Grounded generation + interface**

- *What I gave the AI:* My grounding requirement (answer from retrieved context
  only, with source attribution), the desired output format (answer + source
  list), and a request to wire up a Gradio UI on top of the retrieval function.
- *What it produced:* A Groq-backed generator with a grounding system prompt and a
  Gradio app.
- *What I changed or overrode:* I rejected leaving source attribution to the LLM
  and required it to be **programmatic** — the `Sources:` block is built in code
  from the retrieved chunks so it can't be hallucinated. I also hardened the UI
  after finding the multi-line input box swallowed the Enter key (so it felt like
  it ignored input): I switched it to a single-line submit-on-Enter box with the
  events wired to both Enter and the Ask button.
