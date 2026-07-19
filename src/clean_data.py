"""
clean_data.py
=============
Turn RAW Reddit data into one tidy "posts" table that the rest of the project
understands. This is the only step that knows about messy file formats.

It can read:
  - .zst       Pushshift / torrent dumps (compressed, streamed line by line)
  - .ndjson / .jsonl   one JSON post per line (uncompressed)
  - .csv       a flat table
  - .parquet   a flat table

It always outputs the SAME columns, so everything downstream is identical no
matter where the data came from:

    id, date, author, score, subreddit, title, selftext, num_comments

You control it with three simple filters (all optional):
  - subreddits : keep only these forums   (empty list = keep all)
  - start_date : keep posts on/after this date  (e.g. "2021-01-01")
  - end_date   : keep posts BEFORE this date     (exclusive)

The .zst reading is STREAMED, meaning we read one line at a time and never load
the whole (possibly huge) file into memory - so the full Reddit torrent is fine.
"""

import os
import io
import json
import glob
import datetime

import pandas as pd
import zstandard


# The columns every cleaned file will have, in this order.
# 'source' says where a row came from: 'reddit' here; X (Twitter) rows get
# source='x' via src/x_data.py + ingestion/add_x_data.py.
OUTPUT_COLUMNS = ["id", "date", "author", "score", "subreddit", "title", "selftext", "num_comments", "source"]


# ----------------------------------------------------------------------
# Finding and reading raw files
# ----------------------------------------------------------------------
def find_input_files(path):
    """'path' can be a single file OR a folder. Return the list of data files."""
    if os.path.isfile(path):
        return [path]
    files = []
    for pattern in ("*.zst", "*.ndjson", "*.jsonl", "*.json", "*.csv", "*.parquet"):
        files.extend(glob.glob(os.path.join(path, pattern)))
    return sorted(files)


def read_json_lines(filepath):
    """
    Yield one parsed record (a dict) at a time from a JSON-lines file.
    Handles .zst by streaming-decompressing it.
    """
    if filepath.endswith(".zst"):
        with open(filepath, "rb") as raw_file:
            # Big window size because Pushshift uses long-distance compression.
            decompressor = zstandard.ZstdDecompressor(max_window_size=2 ** 31)
            stream = decompressor.stream_reader(raw_file)
            text = io.TextIOWrapper(stream, encoding="utf-8", errors="ignore")
            for line in text:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
    else:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as text_file:
            for line in text_file:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


# ----------------------------------------------------------------------
# Turning one raw record into our standard shape
# ----------------------------------------------------------------------
def normalise(record):
    """
    Map one raw Reddit record to our standard dict.

    Submissions have 'title' + 'selftext'. Comments have 'body'. We put a
    comment's body into the 'selftext' field so the text is never lost.
    """
    created = record.get("created_utc", 0)
    created = int(created) if str(created).isdigit() else 0
    if created == 0:
        date_string = ""
    else:
        date_string = datetime.datetime.utcfromtimestamp(created).strftime("%Y-%m-%d")

    title = record.get("title", "") or ""
    selftext = record.get("selftext", "") or ""
    body = record.get("body", "") or ""        # comments use 'body'
    if not selftext and body:
        selftext = body

    return {
        "id": record.get("id", ""),
        "date": date_string,
        "author": record.get("author", "") or "",
        "score": int(record.get("score", 0) or 0),
        "subreddit": (record.get("subreddit", "") or "").lower(),
        "title": title,
        "selftext": selftext,
        "num_comments": int(record.get("num_comments", 0) or 0),
        "source": "reddit",
    }


def keep_this_post(post, wanted_subreddits, start_date, end_date):
    """Apply the subreddit and date filters. Return True to keep the post."""
    if not post["date"]:
        return False
    if wanted_subreddits and post["subreddit"] not in wanted_subreddits:
        return False
    if start_date and post["date"] < start_date:     # string dates compare correctly as YYYY-MM-DD
        return False
    if end_date and post["date"] >= end_date:         # end_date is exclusive
        return False
    return True


# ----------------------------------------------------------------------
# The main entry point used by the notebook
# ----------------------------------------------------------------------
def clean(input_path, output_path, subreddits=None, start_date=None, end_date=None):
    """
    Read raw data from input_path, filter it, and write a tidy posts file.

    input_path  : a file or a folder of raw files
    output_path : where to save (.parquet recommended, .csv also works)
    subreddits  : list like ["wallstreetbets", "stocks"]  (None/[] = all)
    start_date  : "YYYY-MM-DD" inclusive, or None
    end_date    : "YYYY-MM-DD" exclusive, or None

    Returns the number of posts written.
    """
    wanted = set(s.lower() for s in (subreddits or []))
    files = find_input_files(input_path)
    print("Found", len(files), "raw file(s).")

    # We read records one at a time, but only KEEP the ones that pass the
    # filters. Because you normally filter by subreddit and/or date, the kept
    # list stays small even when the raw files are enormous.
    kept_rows = []
    seen_subreddits = set()

    for filepath in files:
        name = os.path.basename(filepath)
        print("  reading", name, "...")

        # CSV / parquet inputs are already tables - read with pandas.
        if filepath.endswith(".csv") or filepath.endswith(".parquet"):
            table = pd.read_csv(filepath) if filepath.endswith(".csv") else pd.read_parquet(filepath)
            records = table.to_dict("records")
        else:
            # JSON-lines / .zst inputs - streamed record by record.
            records = read_json_lines(filepath)

        for record in records:
            post = normalise(record)
            seen_subreddits.add(post["subreddit"])
            if keep_this_post(post, wanted, start_date, end_date):
                kept_rows.append(post)

    # Build one tidy table and save it in the format implied by the file name.
    df = pd.DataFrame(kept_rows, columns=OUTPUT_COLUMNS)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if output_path.endswith(".csv"):
        df.to_csv(output_path, index=False)
    else:
        df.to_parquet(output_path, index=False)

    print("\nKept", len(df), "posts ->", output_path)
    if wanted:
        print("Subreddit filter:", sorted(wanted))
    print("Subreddits seen in the raw data:", sorted(s for s in seen_subreddits if s)[:20])
    return len(df)
