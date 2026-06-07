"""Ingest the UCR course-difficulty Google Sheet, clean it, and emit chunks.

Implements Milestone 3 of planning.md (Ingestion + Chunking):

  1. DOCUMENT INGESTION
     - read the sheet (live CSV export, or a local cached copy)
     - forward-fill blank class codes onto continuation rows
     - forward-fill the per-class "Average Difficulty" onto continuation rows
     - drop the announcement junk row, "See OTHER" pointer rows, and rows
       with no review text
     - normalise the per-review rating, average, and date
     -> clean records: {class_code, avg_difficulty, rating, review, date}

  2. CHUNKING
     - one chunk per review (row-based, not fixed-length)
     - each chunk is prefixed with a metadata header (class / rating / avg / date)
     - the full chunk (prefix + review) is capped at ~256 tokens to match
       all-MiniLM-L6-v2's max sequence length; oversized reviews are split into
       sub-chunks with ~40-token overlap
     - one summary chunk per class (code + avg + review count + rating range)

Outputs (under data/):
  - records.csv  : the cleaned, one-row-per-review table (for inspection)
  - chunks.jsonl : the chunks, one JSON object per line, ready for embedding

Run:  python src/build_chunks.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

SHEET_ID = "1qiy_Oi8aFiPmL4QSTR3zHe74kmvc6e_159L1mAUUlU0"
SHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# all-MiniLM-L6-v2 truncates at 256 tokens. Reserve 2 slots for the [CLS]/[SEP]
# special tokens that sentence-transformers adds, so the *embedded* chunk never
# silently overflows.
MAX_TOKENS = 256
SPECIAL_TOKEN_BUDGET = 2
TOKEN_CAP = MAX_TOKENS - SPECIAL_TOKEN_BUDGET  # usable tokens per chunk
SPLIT_OVERLAP = 40                              # token overlap for split reviews

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = ROOT / "documents" / "ucr_raw.csv"
OUT_DIR = ROOT / "data"

# Raw column names in the sheet.
COL_CLASS = "Class"
COL_AVG = "Average Difficulty"
COL_REVIEW = "Additional Comments"
COL_RATING = "Difficulty"
COL_DATE = "Date"


# --------------------------------------------------------------------------- #
# 1. Ingestion + cleaning
# --------------------------------------------------------------------------- #

def load_raw(raw_path: Path, refresh: bool) -> pd.DataFrame:
    """Load the sheet as a DataFrame, downloading a fresh CSV if asked/needed."""
    if refresh or not raw_path.exists():
        print(f"Downloading sheet -> {raw_path}")
        df = pd.read_csv(SHEET_CSV_URL, dtype=str)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_path, index=False)
    else:
        print(f"Reading cached CSV <- {raw_path}")
        df = pd.read_csv(raw_path, dtype=str)
    return df


def _to_float(value) -> float | None:
    """Parse a numeric cell; return None for blanks/non-numeric (e.g. 'Github')."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    try:
        return float(s)
    except ValueError:
        return None


def _clean_date(value) -> str | None:
    """Return a normalised date string, fixing the obvious 3022 -> 2022 typo."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    parsed = pd.to_datetime(s, errors="coerce")
    if pd.isna(parsed):
        return s  # keep the raw token if unparseable rather than dropping info
    if parsed.year > 2026:  # typo'd far-future year (e.g. 3022)
        parsed = parsed.replace(year=parsed.year - 1000)
    return parsed.strftime("%Y-%m-%d")


def _norm_ws(text: str) -> str:
    """Collapse runs of whitespace (incl. embedded newlines) to single spaces."""
    return re.sub(r"\s+", " ", str(text)).strip()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill, drop junk, and normalise into one clean row per review."""
    df = df.copy()

    # Challenge #1: only the first review per class carries the class code; the
    # per-class average sits on that same header row. Forward-fill both down the
    # continuation rows so every review keeps its class identity and class avg.
    df[COL_CLASS] = df[COL_CLASS].ffill()
    df[COL_AVG] = df[COL_AVG].ffill()

    # Drop the leading announcement row (no class code above it to inherit).
    df = df[df[COL_CLASS].notna()].copy()
    df[COL_CLASS] = df[COL_CLASS].str.strip()

    # Normalise fields.
    df["class_code"] = df[COL_CLASS]
    df["avg_difficulty"] = df[COL_AVG].map(_to_float)   # "See OTHER" -> None
    df["rating"] = df[COL_RATING].map(_to_float)        # "Github" -> None
    df["date"] = df[COL_DATE].map(_clean_date)
    df["review"] = df[COL_REVIEW].fillna("").map(_norm_ws)

    # A review chunk needs review text. Rows with no comment are class headers
    # with no submission yet or "See OTHER" cross-reference pointers -> drop.
    df = df[df["review"].str.len() > 0].copy()

    return df[["class_code", "avg_difficulty", "rating", "review", "date"]].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Chunking
# --------------------------------------------------------------------------- #

def _fmt_num(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "N/A"
    return f"{x:g}"


def _fmt_date(d) -> str:
    if d is None or (isinstance(d, float) and pd.isna(d)):
        return "N/A"
    return str(d)


def review_prefix(rec) -> str:
    """Metadata header prepended to every review chunk."""
    return (
        f"[Class: {rec.class_code} | "
        f"Difficulty rating: {_fmt_num(rec.rating)}/10 | "
        f"Class average: {_fmt_num(rec.avg_difficulty)}/10 | "
        f"Date: {_fmt_date(rec.date)}]"
    )


def make_review_chunks(rec, tokenizer) -> list[dict]:
    """One chunk per review, splitting oversized reviews with token overlap."""
    prefix = review_prefix(rec)
    prefix_tokens = len(tokenizer.encode(prefix + "\n", add_special_tokens=False))
    body_budget = TOKEN_CAP - prefix_tokens

    # Tokenise with offset mapping so we can slice the *original* review text
    # (preserving casing/punctuation) rather than decoding token ids back.
    enc = tokenizer(rec.review, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    n_tok = len(offsets)

    # Common case: the whole review fits under the cap -> single chunk.
    if n_tok <= body_budget:
        windows = [(0, n_tok)]
    else:
        # Challenge #4: split essay-length reviews so nothing is silently
        # truncated; keep ~40-token overlap so a thought spanning the cut stays
        # retrievable in at least one piece.
        step = max(1, body_budget - SPLIT_OVERLAP)
        windows = [(i, min(i + body_budget, n_tok)) for i in range(0, n_tok, step)]
        # drop a trailing window fully covered by the previous window's overlap
        if len(windows) >= 2 and windows[-1][1] - windows[-1][0] <= SPLIT_OVERLAP:
            windows.pop()

    chunks = []
    n = len(windows)
    for i, (start, end) in enumerate(windows):
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        body = rec.review[char_start:char_end].strip()
        text = f"{prefix}\n{body}"
        chunks.append({
            "id": f"{rec.class_code}__r{rec.Index}" + (f"__p{i}" if n > 1 else ""),
            "type": "review",
            "text": text,
            "class_code": rec.class_code,
            "avg_difficulty": rec.avg_difficulty,
            "rating": rec.rating,
            "date": rec.date,
            "part": i if n > 1 else None,
            "n_parts": n,
            "n_tokens": prefix_tokens + (end - start),
        })
    return chunks


def make_summary_chunks(records: pd.DataFrame) -> list[dict]:
    """One summary chunk per class: code + avg + review count + rating range."""
    chunks = []
    for code, grp in records.groupby("class_code", sort=True):
        ratings = grp["rating"].dropna()
        avg = grp["avg_difficulty"].dropna()
        avg_val = float(avg.iloc[0]) if len(avg) else (float(ratings.mean()) if len(ratings) else None)
        count = len(grp)

        prefix = (
            f"[Class: {code} | Class average difficulty: {_fmt_num(avg_val)}/10 | "
            f"Reviews: {count}]"
        )
        if len(ratings):
            body = (
                f"Summary for {code}: {count} student review(s), "
                f"average difficulty {_fmt_num(avg_val)} out of 10. "
                f"Individual difficulty ratings range from {_fmt_num(ratings.min())} "
                f"to {_fmt_num(ratings.max())} out of 10."
            )
        else:
            body = (
                f"Summary for {code}: {count} student review(s), "
                f"average difficulty {_fmt_num(avg_val)} out of 10."
            )
        chunks.append({
            "id": f"{code}__summary",
            "type": "summary",
            "text": f"{prefix}\n{body}",
            "class_code": code,
            "avg_difficulty": avg_val,
            "rating": None,
            "date": None,
            "part": None,
            "n_parts": 1,
            "n_tokens": None,
        })
    return chunks


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW,
                    help="path to the cached raw CSV")
    ap.add_argument("--refresh", action="store_true",
                    help="re-download the sheet even if a cached CSV exists")
    args = ap.parse_args()

    df_raw = load_raw(args.raw, args.refresh)
    print(f"Raw rows: {len(df_raw)}")

    records = clean(df_raw)
    print(f"Clean review records: {len(records)} across {records['class_code'].nunique()} classes")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records_path = OUT_DIR / "records.csv"
    records.to_csv(records_path, index=False)
    print(f"Wrote {records_path}")

    print(f"Loading tokenizer ({EMBED_MODEL}) ...")
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)

    review_chunks: list[dict] = []
    for rec in records.itertuples():
        review_chunks.extend(make_review_chunks(rec, tokenizer))
    summary_chunks = make_summary_chunks(records)
    all_chunks = review_chunks + summary_chunks

    chunks_path = OUT_DIR / "chunks.jsonl"
    with chunks_path.open("w") as fh:
        for c in all_chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    n_split = sum(1 for c in review_chunks if c["n_parts"] > 1)
    tok_counts = [c["n_tokens"] for c in review_chunks if c["n_tokens"]]
    print(
        f"Chunks: {len(all_chunks)} total "
        f"({len(review_chunks)} review + {len(summary_chunks)} summary); "
        f"{n_split} from split oversized reviews"
    )
    if tok_counts:
        print(f"Review chunk tokens: max={max(tok_counts)} (cap {TOKEN_CAP}), "
              f"mean={sum(tok_counts) / len(tok_counts):.0f}")
    over = [t for t in tok_counts if t > TOKEN_CAP]
    print(f"Chunks over cap: {len(over)}")
    print(f"Wrote {chunks_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
