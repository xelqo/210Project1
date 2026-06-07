"""Embed the chunks and store them in a persistent ChromaDB collection.

Implements Milestone 4, step 3 of planning.md (Embedding + Vector store):

  - embed each chunk with all-MiniLM-L6-v2 (sentence-transformers) -> 384-dim
    normalised vector
  - store vectors + text + metadata in a persistent Chroma collection, using
    cosine distance (planning names FAISS; the project ships ChromaDB, so we use
    Chroma — same idea: an ANN index over the chunk vectors)

The model and Chroma handles defined here are imported by retrieve.py so the
query side embeds with the *same* model and reads the *same* collection.

Run:  python src/embed_index.py            # build from data/chunks.jsonl
      python src/embed_index.py --rebuild  # drop and rebuild the collection
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"
CHROMA_DIR = ROOT / "chroma_db"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION = "ucr_reviews"
# Cosine matches the normalised MiniLM vectors (planning: "384-dim vectors are
# cheap to store/search").
DISTANCE_SPACE = "cosine"

ADD_BATCH = 1000

# Chroma metadata values must be str/int/float/bool — None is rejected.
_META_KEYS = ("type", "class_code", "rating", "avg_difficulty", "date", "part", "n_parts")


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run src/build_chunks.py first")
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def get_embedder() -> SentenceTransformer:
    """The single embedding model, shared by indexing and retrieval."""
    return SentenceTransformer(EMBED_MODEL)


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(client: chromadb.ClientAPI):
    """Open the existing collection (read side). Raises if it isn't built yet."""
    return client.get_collection(COLLECTION, embedding_function=None)


def _clean_meta(chunk: dict) -> dict:
    """Keep only Chroma-safe, non-null metadata fields."""
    meta = {}
    for k in _META_KEYS:
        v = chunk.get(k)
        if v is None:
            continue
        if isinstance(v, float) and v != v:  # NaN
            continue
        meta[k] = v
    return meta


def build(rebuild: bool = False) -> None:
    chunks = load_chunks()
    ids = [c["id"] for c in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate chunk ids — cannot index")
    print(f"Loaded {len(chunks)} chunks")

    print(f"Loading embedding model ({EMBED_MODEL}) ...")
    model = get_embedder()
    print(f"Embedding {len(chunks)} chunks ...")
    embeddings = model.encode(
        [c["text"] for c in chunks],
        batch_size=64,
        normalize_embeddings=True,   # unit vectors -> cosine distance
        show_progress_bar=True,
    )
    print(f"Embeddings: {embeddings.shape}")

    client = get_client()
    if rebuild:
        try:
            client.delete_collection(COLLECTION)
            print(f"Dropped existing collection '{COLLECTION}'")
        except Exception:
            pass
    collection = client.get_or_create_collection(
        COLLECTION,
        metadata={"hnsw:space": DISTANCE_SPACE},
        embedding_function=None,     # we always supply our own vectors
    )

    metadatas = [_clean_meta(c) for c in chunks]
    documents = [c["text"] for c in chunks]
    for i in range(0, len(chunks), ADD_BATCH):
        sl = slice(i, i + ADD_BATCH)
        collection.add(
            ids=ids[sl],
            embeddings=embeddings[sl].tolist(),
            documents=documents[sl],
            metadatas=metadatas[sl],
        )
        print(f"  added {min(i + ADD_BATCH, len(chunks))}/{len(chunks)}")

    print(f"Done. Collection '{COLLECTION}' now holds {collection.count()} vectors "
          f"at {CHROMA_DIR}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true",
                    help="drop the collection and re-add every chunk")
    args = ap.parse_args()
    build(rebuild=args.rebuild)
    return 0


if __name__ == "__main__":
    sys.exit(main())
