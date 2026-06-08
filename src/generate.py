"""Grounded answer generation over the retrieved chunks (Architecture step 5).

planning.md step 5 (Generation): build a prompt from the question + retrieved
chunks, instruct the model to cite class codes and report the opinion
spread/range rather than asserting a single review as fact.

Grounding is enforced two ways, not merely suggested:

  1. System prompt — the model is told to answer ONLY from the provided context,
     to never use outside knowledge, and to return the exact fallback string
     "I don't have enough information on that." when the context is insufficient.

  2. Source attribution is PROGRAMMATIC — the "Sources" block appended to every
     grounded answer is built in code from the retrieved chunks, so attribution
     is guaranteed by the pipeline and cannot be hallucinated or omitted by the
     LLM. (The model is also asked to cite [S#] inline, but that is a bonus on
     top of the code-built list.)

LLM: Groq llama-3.3-70b-versatile (free-tier, OpenAI-compatible).

Run:  python src/generate.py "why is BIOL005B so hard?"
"""

from __future__ import annotations

import argparse
import os
import sys
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

import retrieve as R

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

LLM_MODEL = "llama-3.3-70b-versatile"
NO_INFO = "I don't have enough information on that."

SYSTEM_PROMPT = (
    "You are a factual assistant that answers questions about how difficult "
    "courses are at UC Riverside, using ONLY the student reviews supplied to you "
    "as CONTEXT.\n\n"
    "Hard rules:\n"
    "1. Use ONLY facts found in the CONTEXT. Never use prior knowledge, training "
    "data, or assumptions about how courses or professors 'usually' work.\n"
    f'2. If the CONTEXT does not contain enough information to answer, reply with '
    f'EXACTLY this and nothing else: "{NO_INFO}"\n'
    "3. Support each claim with the source label(s) in brackets, e.g. [S1], [S3], "
    "drawn from the CONTEXT.\n"
    "4. Reviews for one class often disagree (ratings can span the whole scale). "
    "Report the range/spread and note disagreement rather than presenting one "
    "review as the single truth.\n"
    "5. Do not invent professor names, ratings, dates, or any detail that is not "
    "present in the CONTEXT.\n"
    "6. Refer to classes by their class code."
)


@lru_cache(maxsize=1)
def _client() -> Groq:
    key = os.getenv("GROQ_API_KEY")
    if not key or key == "your_key_here":
        raise RuntimeError("GROQ_API_KEY is missing — set it in .env")
    return Groq(api_key=key)


def _source_label(meta: dict) -> str:
    """Human-readable source descriptor built from chunk metadata (code-side)."""
    code = meta.get("class_code", "?")
    kind = "class summary" if meta.get("type") == "summary" else "student review"
    date = meta.get("date")
    return f"{code} — {kind}" + (f" ({date})" if date else "")


def build_context(results: list[dict]) -> tuple[str, list[dict]]:
    """Render the labeled CONTEXT block and the parallel source list.

    Both are derived from the same retrieved chunks, so the [S#] labels in the
    prompt line up exactly with the programmatically appended Sources block.
    """
    blocks, sources = [], []
    for i, r in enumerate(results, 1):
        label = f"S{i}"
        blocks.append(f"[{label}] {_source_label(r['metadata'])}\n{r['text']}")
        sources.append({
            "label": label,
            "id": r["id"],
            "descriptor": _source_label(r["metadata"]),
            "distance": r["distance"],
        })
    return "\n\n".join(blocks), sources


def format_sources(sources: list[dict]) -> str:
    """The Sources block that is appended in code (the attribution guarantee)."""
    lines = ["Sources:"]
    for s in sources:
        lines.append(f"  [{s['label']}] {s['descriptor']}  (chunk {s['id']})")
    return "\n".join(lines)


def generate(query: str, k: int = R.TOP_K, use_class_filter: bool = True,
             temperature: float = 0.0) -> dict:
    """Retrieve, generate a grounded answer, and attach code-built sources."""
    retrieval = R.retrieve(query, k=k, use_class_filter=use_class_filter)
    results = retrieval["results"]

    # No context retrieved -> refuse without ever calling the LLM.
    if not results:
        return {
            "query": query,
            "answer": NO_INFO,
            "answer_with_sources": NO_INFO,
            "sources": [],
            "class_filter": retrieval["class_filter"],
            "results": results,
        }

    context, sources = build_context(results)
    user_msg = (
        f"CONTEXT (student reviews; cite by label):\n\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer using only the CONTEXT above, citing source labels."
    )

    resp = _client().chat.completions.create(
        model=LLM_MODEL,
        temperature=temperature,
        max_tokens=600,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    answer = resp.choices[0].message.content.strip()

    # Programmatic source attribution: append the code-built Sources block to
    # every grounded answer. Skipped only when the model correctly refuses.
    if answer.rstrip(' .') == NO_INFO.rstrip('.'):
        answer_with_sources = NO_INFO
    else:
        answer_with_sources = f"{answer}\n\n{format_sources(sources)}"

    return {
        "query": query,
        "answer": answer,
        "answer_with_sources": answer_with_sources,
        "sources": sources,
        "class_filter": retrieval["class_filter"],
        "results": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=R.TOP_K)
    ap.add_argument("--no-filter", action="store_true")
    args = ap.parse_args()

    out = generate(args.query, k=args.k, use_class_filter=not args.no_filter)
    print(f"Q: {out['query']}")
    print(f"(class filter: {out['class_filter'] or 'none'})\n")
    print(out["answer_with_sources"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
