# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

The domain I choose is course difficulty at UCR. This kind of thing is hard to come by because it may be hard to get a good sense of how difficult a class is based on the universities description, having accounts from people who actaully took the class is more valueable.

---

## Documents

https://docs.google.com/spreadsheets/d/1qiy_Oi8aFiPmL4QSTR3zHe74kmvc6e_159L1mAUUlU0/edit?usp=sharing

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |
| 7 | | | |
| 8 | | | |
| 9 | | | |
| 10 | | | |

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:**
One review per chunk (row-based, not fixed-length).
- Cap each chunk at ~256 tokens (≈1,000 chars) to match all-MiniLM-L6-v2's max
  sequence length (it truncates beyond 256 tokens).
- Split only the rare essay-length reviews that exceed the cap; use ~40-token
  overlap within those splits.

**Overlap:**
 Zero overlap between separate reviews — they're independent and overlap would just bleed one class's review into another's. Use ~40-token overlap only when you're forced to split one oversized review, so a thought spanning the split stays retrievable in at least one piece.

**Reasoning:**
The corpus is a structured set of independent reviews, each tied to a single class with a numeric difficulty and date. Each review is the natural atomic unit, so I chunk per-review rather than by fixed length. Fixed-length chunking would both fragment long reviews and conflate unrelated short ones from different classes. The ~256-token cap matches the embedding model's max sequence length (all-MiniLM-L6-v2 truncates at 256 tokens), so larger chunks would silently lose text anyway. Overlap is unnecessary between independent reviews and only applied within split long reviews
---

## Retrieval Approach

**Embedding model:** all-MiniLM-L6-v2 via sentence-transformers. Chosen because
it runs locally (no API cost/key), is fast, and 384-dim vectors are cheap to
store/search for a corpus this size. It's English-only and truncates at 256
tokens — acceptable since reviews are English and chunks are capped at 256.

**top-k:** 6. A single class can have many reviews spanning the full 0–10 range
(e.g. BIOL005A, CHEM001A), so one or two chunks would capture only a slice of a
divided opinion. ~6 gives the LLM enough perspectives to summarize consensus and
spread. Too few → misses contradicting reviews, answer looks falsely confident.
Too many (e.g. 20) → dilutes with off-topic/other-class reviews, bloats the
prompt, and can bury the relevant chunk. I'll also filter by class-code metadata
when the query names a specific class, so top-k pulls from the right class
instead of semantically-similar reviews of other classes.

**If deploying for real (cost no object):** I'd weigh:
- *Context length* — a longer-context model (e.g. OpenAI text-embedding-3-large,
  ~8k tokens) wouldn't truncate the essay-length reviews MiniLM cuts at 256.
- *Domain/accuracy* — reviews are informal, sarcastic, slang-heavy ("weeder
  class," "got Jeff'd," "destroyed my GPA"). A larger model captures connotation
  and sarcasm better, improving retrieval on opinion text.
- *Multilingual* — not needed here; the corpus is English. I'd skip multilingual
  models since they trade some English accuracy for languages I don't use.
- *Latency/cost* — MiniLM is instant and free locally; API models add network
  latency and per-call cost. For a class project the quality gain doesn't justify
  it, but at scale I'd accept latency for accuracy.

**Why semantic search works without shared words:** embeddings map *meaning* to
nearby vectors, so a query "which classes are hard" matches a review that says
"brutal, everyone fails" even with zero shared words — the model learned those
phrases occupy similar regions of vector space.
**Production tradeoff reflection:**

---

## Evaluation Plan

1. Q: "What is the average difficulty rating of CS010A?"
   Expected: 3.38 (tests the per-class summary chunk + numeric retrieval).

2. Q: "What is the average difficulty of BIOL005B, and why do students consider
   it hard?"
   Expected: 7.59; heavy memorization, animal/plant diversity, known weeder
   course, several reviews say to avoid Prof. Chappell.

3. Q: "Which professor do reviewers repeatedly warn against taking for CHEM001A?"
   Expected: Ludwig Bartels — multiple reviews explicitly say "AVOID BARTELS"
   (free-response exams, teaches beyond intro scope, harsh grading).

4. Q: "According to reviews, what makes CS061 difficult to pass?"
   Expected: it's a weeder course; LC-3 assembly, data paths, register transfer
   notation; must score ≥70% on the final or fail; time-consuming labs/projects.

5. Q: "Which professor is recommended as the easiest for CS008?"
   Expected: Toby Gustafson — reviews say he posts lectures, tells you what's on
   the exam, attendance optional, easy A.

## Anticipated Challenges

1. **Blank class codes on continuation rows.** In the CSV, only the first review
   for each class has the class code; subsequent reviews leave that column empty
   and inherit it from the row above. If I chunk naively per row, every empty-code
   review loses its class identity and becomes unretrievable/misattributed.
   Mitigation: forward-fill the class code down empty rows during preprocessing.

2. **Contradictory reviews within one class.** A single class can have ratings
   from 1 to 10 and opposite opinions (often about different professors or years).
   The LLM may present one cherry-picked review as fact. Mitigation: retrieve
   several chunks (top-k 6) and prompt the model to report the range/consensus and
   note disagreement rather than asserting a single answer.

3. **Cross-class contamination.** Generic phrases like "avoid this professor,
   weeder class" appear across many classes, so semantic search can pull reviews
   from the wrong class. Mitigation: metadata-filter by class code when the query
   names one.

4. **Truncation of long reviews.** Essay-length reviews exceed MiniLM's 256-token
   limit and get silently cut, losing later content. Mitigation: split oversized
   reviews into capped sub-chunks with small overlap.

5. **Stale information.** Reviews span 2017–2025; professors leave and courses
   change. Including the date in each chunk lets the model weight or caveat
   recency.

## Architecture
┌─────────────────────────────────────────────────────────────────────┐
│ 1. DOCUMENT INGESTION                                  pandas         │
│    Read UCR class-difficulty CSV                                      │
│    • forward-fill blank class codes onto continuation rows           │
│    • drop empty rows                                                  │
│    → records: {class_code, avg_difficulty, rating, review, date}     │
└───────────────────────────────┬─────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. CHUNKING                                            custom Python  │
│    One chunk per review (row-based, not fixed-length)                │
│    • prefix each chunk w/ metadata (class, rating, avg, date)        │
│    • cap ~256 tokens; split oversized reviews w/ ~40-tok overlap     │
│    • + one summary chunk per class (code + avg + review count)       │
└───────────────────────────────┬─────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. EMBEDDING + VECTOR STORE      sentence-transformers (MiniLM-L6-v2) │
│                                                          + FAISS      │
│    • embed each chunk → 384-dim vector                               │
│    • store vectors + metadata in FAISS index                        │
└───────────────────────────────┬─────────────────────────────────────┘
│
┌─────────────────┘
│  user query
▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. RETRIEVAL                     sentence-transformers + FAISS        │
│    • embed query with same model                                    │
│    • optional metadata filter by class_code if query names one      │
│    • return top-k = 6 nearest chunks                                 │
└───────────────────────────────┬─────────────────────────────────────┘
│  query + retrieved chunks
▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. GENERATION                                          LLM (Claude)   │
│    • build prompt: question + retrieved chunks                      │
│    • instruct: cite class codes, report opinion spread/range,       │
│      don't assert a single review as fact                           │
│    → grounded answer                                                 │
└─────────────────────────────────────────────────────────────────────┘

---

## AI Tool Plan

- **Preprocessing / data cleaning:** I'll give Claude the raw CSV schema and the
  "blank class codes" challenge from this doc, and ask it to write a pandas script
  that forward-fills the class code, strips empty rows, and outputs clean records
  of {class_code, avg_difficulty, rating, review_text, date}.

- **Chunking:** I'll give Claude my Chunking Strategy section and ask it to
  implement chunk_text() — one chunk per review with the metadata prefix, the
  256-token cap with ~40-token overlap on oversized reviews, plus a function that
  emits one summary chunk per class.

- **Embedding + index:** I'll give Claude my Retrieval Approach section and ask it
  to write the code that embeds chunks with all-MiniLM-L6-v2 (sentence-
  transformers) and builds a searchable vector index (e.g. FAISS), including the
  class-code metadata filter.

- **Retrieval + generation:** I'll ask Claude to write the query function that
  embeds the question, retrieves top-k 6 (filtered by class if one is named), and
  builds the LLM prompt, instructing it to cite class codes and report opinion
  spread rather than a single review.

- **Evaluation:** I'll give Claude my 5 eval questions + expected answers and ask
  it to write a harness that runs each question through the pipeline and prints
  the retrieved chunks alongside the generated answer so I can grade correctness.
**Milestone 3 — Ingestion and chunking:**

**Milestone 4 — Embedding and retrieval:**

**Milestone 5 — Generation and interface:**
