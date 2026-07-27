"""Prove that a notebook edit touched PROSE ONLY.

WHY THIS EXISTS
---------------
The notebooks are the research record. When we rewrite them for readability
there is exactly one unacceptable outcome: the *numbers* change because the
*code* was quietly edited while we were moving paragraphs around. Re-running
and eyeballing the output does not catch that - a plausible-looking wrong
number looks exactly like a plausible-looking right one.

So: hash the code cells alone, before and after. Same hash => the analysis is
byte-identical and any output change is impossible. Different hash => the diff
is printed and has to be justified deliberately.

USAGE
-----
    python tools/nb_codehash.py snapshot notebooks/*.py      # before editing
    python tools/nb_codehash.py check    notebooks/*.py      # after editing

The snapshot lives in .nb_codehash.json at the repo root and is not tracked -
it is a scratch guard for one editing session, not a project artefact.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / ".nb_codehash.json"


def code_only(path: Path) -> str:
    """Every CODE line of a jupytext percent file, markdown discarded.

    A percent-format file marks cells with `# %%` (code) and `# %% [markdown]`
    (prose). Prose lines are all commented, so a naive "drop comments" filter
    would also drop genuine code comments - which we DO want to compare, since
    a comment change is a documentation change we should be able to see. Hence
    the explicit cell-boundary walk.
    """
    keep, in_code = [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# %%"):
            in_code = "[markdown]" not in line and "[raw]" not in line
            continue
        if in_code:
            keep.append(line)
    return "\n".join(keep)


def digest(path: Path) -> str:
    return hashlib.sha256(code_only(path).encode("utf-8")).hexdigest()[:16]


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    # resolve() so the tool works from any cwd and with relative
    # globs - relative_to(ROOT) below needs absolute paths.
    mode = argv[1]
    files = [Path(p).resolve() for p in argv[2:]]
    if mode == "snapshot":
        STORE.write_text(json.dumps(
            {str(p.relative_to(ROOT)): digest(p) for p in files}, indent=1))
        print(f"snapshotted {len(files)} notebooks -> "
              f"{STORE.relative_to(ROOT)}")
        return 0
    if mode == "check":
        old = json.loads(STORE.read_text()) if STORE.exists() else {}
        bad = 0
        for p in files:
            k = str(p.relative_to(ROOT))
            now, was = digest(p), old.get(k)
            if was is None:
                print(f"?  {k}: no snapshot")
            elif now == was:
                print(f"OK {k}: code unchanged ({now})")
            else:
                print(f"!! {k}: CODE CHANGED {was} -> {now}")
                bad += 1
        print("PROSE-ONLY" if not bad else f"{bad} notebook(s) changed code")
        return 1 if bad else 0
    print(f"unknown mode {mode!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
