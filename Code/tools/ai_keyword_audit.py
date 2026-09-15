"""LLM-proposed keyword-map maintenance, applied only after human approval.

The theme-to-keyword map (``config/theme_keywords.csv``) rots quietly:
new slang appears, companies get renamed, a keyword drifts to the wrong
theme. This tool has the LLM audit the map against what the crowd
actually says and writes its proposals as a reviewable diff::

    Data/reference/keyword_suggestions/keyword_suggestions_<date>.csv
        action (add|move|remove), theme, keyword, reason, approved

The ``approved`` column ships empty. Review the file (a spreadsheet is
fine), put ``YES`` on the rows you accept, then apply exactly those
rows::

    python Code/tools/ai_keyword_audit.py            # audit -> suggestions csv
    python Code/tools/ai_keyword_audit.py --apply Data/reference/keyword_suggestions/keyword_suggestions_<date>.csv

The apply step edits ``config/theme_keywords.csv`` (adds, moves,
removes), prints the diff it made, and never touches a row that was not
approved. The config stays the single human-owned source of truth; the
model is a research assistant with no write access to it.

The model sees the current map, the top unmapped high-frequency terms
from the crowd's own text (``daily_term_counts`` against the map), and
the theme list. After a full update, a weekly audit is plenty.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from src import ai                                     # noqa: E402
from src.config import PROCESSED_DIR                   # noqa: E402
from src.themes import THEME_KEYWORDS                  # noqa: E402
from src.config import DATA_DIR  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KW_CSV = os.path.join(ROOT, "config", "theme_keywords.csv")

# The proposals and notes printed below are model text and crowd terms,
# which are not ASCII. A console encoding that cannot hold one of those
# characters would end the audit on a print.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _unmapped_terms(top_n: int = 80) -> list[dict]:
    """Return high-frequency crowd terms (trailing 60d) not in any keyword list.

    Args:
        top_n: Maximum number of terms to return, by mention volume.

    Returns:
        List of ``{"term", "mentions_60d"}`` dicts; empty when the term
        counts file is absent.
    """
    p = os.path.join(PROCESSED_DIR, "daily_term_counts.parquet")
    if not os.path.exists(p):
        return []
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    w = df[df["date"] > df["date"].max() - pd.Timedelta(days=60)]
    col = "term" if "term" in w.columns else w.columns[1]
    known = {k.lower() for ks in THEME_KEYWORDS.values() for k in ks}
    agg = (w.groupby(col)["mention_count"].sum()
           .sort_values(ascending=False))
    out = []
    for term, n in agg.items():
        t = str(term).lower()
        if t in known or len(t) < 3:
            continue
        out.append({"term": str(term), "mentions_60d": int(n)})
        if len(out) >= top_n:
            break
    return out


_SYSTEM = (
    "You maintain the theme->keyword map of a retail-chatter monitoring "
    "system. Keywords are matched case-insensitively: single words as "
    "whole tokens, multi-word phrases as substrings. HARD RULES: never "
    "propose a bare ticker symbol (C, O, AI as a symbol) - short "
    "uppercase symbols match ordinary prose; company names as words are "
    "good; propose ONLY changes you can justify from the given data or "
    "well-known facts; when unsure, propose nothing - an empty audit is "
    "a fine audit. Return ONLY the JSON object.")


def audit() -> str:
    """Ask the LLM to audit the map and write the suggestions CSV.

    Returns:
        Path of the suggestions file written.

    Raises:
        SystemExit: When the LLM gateway is unavailable.
    """
    if not ai.available():
        raise SystemExit(f"[SKIP] LLM unavailable: "
                         f"{ai.explain_unavailable()}")
    unmapped = _unmapped_terms()
    prompt = (
        "CURRENT MAP (theme -> keywords):\n"
        + json.dumps({t: ks for t, ks in THEME_KEYWORDS.items()},
                     indent=0)
        + "\n\nTOP UNMAPPED HIGH-FREQUENCY TERMS from the crowd's own "
          "text, trailing 60d (candidates to add, if and only if they "
          "clearly belong to one of the themes above):\n"
        + json.dumps(unmapped, indent=0)
        + "\n\nAudit the map. Return ONE JSON object:\n"
          '{"additions": [{"theme","keyword","reason"}],\n'
          ' "moves":     [{"keyword","from_theme","to_theme","reason"}],\n'
          ' "removals":  [{"theme","keyword","reason"}],\n'
          ' "notes": "<=60 words"}\n'
          "Be conservative: additions only for terms with real volume "
          "and an unambiguous home; moves/removals only for clear "
          "errors.")
    try:
        res = ai.chat(prompt, system=_SYSTEM, want_json=True,
                      max_tokens=3200)
    except RuntimeError as exc:
        # An unreachable gateway, a spent budget or an unparseable answer.
        # None of them is a fault in the map, so say so in one line.
        raise SystemExit(f"[SKIP] LLM call failed: {exc}")
    out_dir = os.path.join(DATA_DIR, "reference", "keyword_suggestions")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"keyword_suggestions_{date.today()}.csv")
    # Staged and renamed, as the apply path below is: a second run on the
    # same day writes over the file the reviewer is annotating, and a
    # half-written proposal list reads as a complete one.
    out_tmp = out_path + ".tmp"
    with open(out_tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["action", "theme", "keyword", "reason", "approved"])
        for r in res.get("additions", []):
            w.writerow(["add", r.get("theme", ""), r.get("keyword", ""),
                        r.get("reason", ""), ""])
        for r in res.get("moves", []):
            w.writerow(["move",
                        f"{r.get('from_theme', '')}->"
                        f"{r.get('to_theme', '')}",
                        r.get("keyword", ""), r.get("reason", ""), ""])
        for r in res.get("removals", []):
            w.writerow(["remove", r.get("theme", ""),
                        r.get("keyword", ""), r.get("reason", ""), ""])
    os.replace(out_tmp, out_path)
    n = (len(res.get("additions", [])) + len(res.get("moves", []))
         + len(res.get("removals", [])))
    print(f"[OK] {n} suggestions -> {os.path.relpath(out_path, ROOT)}")
    print(f"     notes: {res.get('notes', '')}")
    print("     review the file, set approved=YES on rows you accept, "
          "then rerun with --apply <file>")
    return out_path


def apply(path: str) -> None:
    """Apply the approved rows of a suggestions CSV to the keyword map.

    Args:
        path: A ``keyword_suggestions_<date>.csv`` with ``approved`` set
            to ``YES`` (or ``Y``/``TRUE``/``1``) on the rows to apply.

    Raises:
        SystemExit: When the file is absent, no row is approved, or the
            keyword map has no header row.
    """
    if not os.path.exists(path):
        raise SystemExit(f"[SKIP] no suggestions file at {path}")
    rows = list(csv.DictReader(open(path, newline="",
                                    encoding="utf-8-sig")))
    approved = [r for r in rows
                if (r.get("approved") or "").strip().upper()
                in ("YES", "Y", "TRUE", "1")]
    if not approved:
        raise SystemExit("[SKIP] no approved rows (set approved=YES on "
                         "the ones you accept)")
    kw = list(csv.reader(open(KW_CSV, newline="", encoding="utf-8-sig")))
    if not kw:
        raise SystemExit(f"[SKIP] {os.path.relpath(KW_CSV, ROOT)} is empty; "
                         "restore it before applying anything to it")
    hdr, body = kw[0], kw[1:]
    changed = []
    for r in approved:
        act = (r.get("action") or "").strip()
        theme = (r.get("theme") or "").strip()
        word = (r.get("keyword") or "").strip()
        if not word:
            continue
        if act == "add":
            if not any(b[0] == theme and b[1].lower() == word.lower()
                       for b in body):
                body.append([theme, word])
                changed.append(f"+ {theme}: {word}")
        elif act == "remove":
            before = len(body)
            body = [b for b in body
                    if not (b[0] == theme
                            and b[1].lower() == word.lower())]
            if len(body) < before:
                changed.append(f"- {theme}: {word}")
        elif act == "move" and "->" in theme:
            frm, to = (s.strip() for s in theme.split("->", 1))
            body = [b for b in body
                    if not (b[0] == frm
                            and b[1].lower() == word.lower())]
            if not any(b[0] == to and b[1].lower() == word.lower()
                       for b in body):
                body.append([to, word])
            changed.append(f"~ {word}: {frm} -> {to}")
    if not changed:
        print(f"[SKIP] the approved rows leave "
              f"{os.path.relpath(KW_CSV, ROOT)} exactly as it is; "
              "nothing written.")
        return
    # Through a temp file in the same directory, then one rename. The
    # keyword map is the human-owned source of truth for every count in
    # the project, and a rewrite killed halfway would truncate it.
    tmp = KW_CSV + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(body)
    os.replace(tmp, KW_CSV)
    print(f"[OK] applied {len(changed)} change(s) to "
          f"{os.path.relpath(KW_CSV, ROOT)}:")
    for c in changed:
        print("  " + c)
    print("rerun the FULL rebuild for the new keywords to reach the "
          "aggregates (they apply at ingestion time).")


if __name__ == "__main__":
    if "--apply" in sys.argv:
        _at = sys.argv.index("--apply") + 1
        if _at >= len(sys.argv):
            raise SystemExit("--apply takes the path of a suggestions CSV, "
                             "e.g. --apply Data/reference/"
                             "keyword_suggestions/keyword_suggestions_"
                             "<date>.csv")
        apply(sys.argv[_at])
    else:
        audit()
