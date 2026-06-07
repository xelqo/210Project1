"""Retrieve the most relevant chunks for a query from the Chroma index.

Implements Milestone 4, step 4 of planning.md (Retrieval):

  - embed the query with the *same* model used for indexing (all-MiniLM-L6-v2)
  - if the query names a specific class, metadata-filter by that class_code so
    top-k pulls from the right class instead of semantically-similar reviews of
    other classes (planning: cross-class contamination, challenge #3)
  - return top-k = 6 nearest chunks (planning: enough perspectives to capture a
    divided opinion without diluting the prompt)

Run:  python src/retrieve.py "why is BIOL005B so hard?"
      python src/retrieve.py "which classes are easy" --k 6 --no-filter
"""

from __future__ import annotations

import argparse
import re
import sys
from functools import lru_cache

import embed_index as ix

TOP_K = 6  # planning.md Retrieval Approach

# Course codes look like CS010A, BIOL005B, CHEM01LB, MATH010A, STAT100A:
# letters, then digits, with optional trailing letters (and the odd letter in
# the numeric block). Used only to *find candidates* in the query; every
# candidate is verified against the real class codes in the index.
_CODE_RE = re.compile(r"[A-Za-z]{2,5}\s?\d{1,3}[A-Za-z]{0,3}")


def _normalize_code(code: str) -> str:
    """Canonical form for matching: upper, no spaces, no leading zeros in the
    digit run — so 'cs10a', 'CS 010A' and 'CS010A' all collapse together."""
    code = re.sub(r"\s+", "", code).upper()
    return re.sub(r"\d+", lambda m: str(int(m.group())), code)


@lru_cache(maxsize=1)
def _code_index() -> dict[str, str]:
    """Map normalized code -> real class_code, drawn from the summary chunks."""
    collection = ix.get_collection(ix.get_client())
    metas = collection.get(where={"type": "summary"}, include=["metadatas"])["metadatas"]
    return {_normalize_code(m["class_code"]): m["class_code"] for m in metas}


def detect_class_code(query: str) -> str | None:
    """Return the real class_code named in the query, or None."""
    index = _code_index()
    for cand in _CODE_RE.findall(query):
        real = index.get(_normalize_code(cand))
        if real:
            return real
    return None


@lru_cache(maxsize=1)
def _model():
    return ix.get_embedder()


def retrieve(query: str, k: int = TOP_K, use_class_filter: bool = True) -> dict:
    """Embed `query` and return up to `k` nearest chunks.

    Returns {"query", "class_filter", "results": [{id, text, metadata,
    distance}, ...]}. If a class filter is applied but yields nothing (e.g. a
    class with no reviews), we fall back to an unfiltered search.
    """
    collection = ix.get_collection(ix.get_client())

    code = detect_class_code(query) if use_class_filter else None
    q_emb = _model().encode([query], normalize_embeddings=True).tolist()

    def _run(where):
        res = collection.query(
            query_embeddings=q_emb,
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            out.append({"id": cid, "text": doc, "metadata": meta, "distance": dist})
        return out

    where = {"class_code": code} if code else None
    results = _run(where)
    if where and not results:  # class named but nothing matched -> unfiltered
        code = None
        results = _run(None)

    return {"query": query, "class_filter": code, "results": results}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("query", help="natural-language question")
    ap.add_argument("--k", type=int, default=TOP_K, help="number of chunks (default 6)")
    ap.add_argument("--no-filter", action="store_true",
                    help="disable the class-code metadata filter")
    args = ap.parse_args()

    out = retrieve(args.query, k=args.k, use_class_filter=not args.no_filter)
    print(f"Query: {out['query']}")
    print(f"Class filter: {out['class_filter'] or '(none)'}")
    print(f"Top {len(out['results'])} chunks:\n")
    for i, r in enumerate(out["results"], 1):
        m = r["metadata"]
        print(f"--- #{i}  {r['id']}  [{m.get('type')}]  cos_dist={r['distance']:.3f} ---")
        print(r["text"])
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
