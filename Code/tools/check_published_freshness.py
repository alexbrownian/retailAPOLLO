#!/usr/bin/env python
"""Is the PUBLISHED dashboard bundle still current?

    python Code/tools/check_published_freshness.py
    python Code/tools/check_published_freshness.py --max-age-days 2
    python Code/tools/check_published_freshness.py --quiet

This reads the two small files that travel WITH the committed bundle,
so it answers a different question from the run log: not "did the
pipeline finish" but "is the hosted dashboard still being fed, and by a
healthy run". A watcher outside the machine runs it against a fresh
checkout, which is the only way to notice the two failures the pipeline
cannot report on its own - a run that never started, and a publish that
never reached the remote.

The manifest gives the published data day, which measures both of those.
The run status gives the verdict the run reached on itself, which
measures what a fresh data day cannot: a stage that failed while the run
carried on, a paid dependency out of credit or refused, a store that
stopped advancing. A bundle that carries no run status is read as
silence rather than as a failure.

Exit codes: 0 current, and the run that published it raised nothing;
1 stale, or a manifest or run status that does not parse; 2 no bundle at
all; 3 the bundle is current and the run that published it raised at
least one condition.
"""

import argparse
import datetime
import json
import os
import sys

# The index spells this folder whichever way it was first committed, and
# a checkout on a case-sensitive filesystem gets that spelling verbatim.
# Both are looked for so the check reads the same on a Linux runner as
# on the machine that writes it.
BUNDLE_DIRS = ("Data/dashboard", "data/dashboard")
MANIFEST = "publish_manifest.json"
STATUS = "run_status.json"


def find_manifest(root="."):
    """Path of the committed publish manifest, or ``None``."""
    for d in BUNDLE_DIRS:
        p = os.path.join(root, d, MANIFEST)
        if os.path.isfile(p):
            return p
    return None


def find_status(manifest_path):
    """Path of the run status beside ``manifest_path``, or ``None``.

    Taken from the manifest's own folder rather than searched for again,
    so both files are always read out of the same bundle.
    """
    p = os.path.join(os.path.dirname(manifest_path), STATUS)
    return p if os.path.isfile(p) else None


def read_status(path):
    """``(finished, exit_code, conditions)`` from one run status file.

    ``conditions`` is a list of ``(tag, reason)`` pairs and is empty when
    the run reported none. An entry the writer shaped differently is
    still reported, under whatever it does carry: a condition that
    cannot be described is worth more than a condition dropped.
    """
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    rows = []
    for item in doc.get("conditions") or []:
        if isinstance(item, dict):
            rows.append((str(item.get("tag") or "condition"),
                         str(item.get("reason") or "")))
        else:
            rows.append((str(item), ""))
    return (doc.get("finished") or "at an unrecorded time",
            doc.get("exit_code"), rows)


def read_manifest(path):
    """``(data_through, published_at)`` as dates, either possibly ``None``.

    A manifest that does not parse is reported by the caller rather than
    raised through: an unreadable manifest is itself a publishing
    failure, and the watcher exists to say so in words.
    """
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    through = doc.get("data_through") or None
    at = doc.get("published_at") or None
    if through:
        through = datetime.date.fromisoformat(str(through)[:10])
    if at:
        at = datetime.date.fromisoformat(str(at)[:10])
    return through, at, doc


def main():
    p = argparse.ArgumentParser(
        description="Check that the published bundle is still advancing.")
    p.add_argument("--root", default=".",
                   help="checkout to inspect (default: the current folder)")
    p.add_argument("--max-age-days", type=int, default=3,
                   help="how many days the newest published data day may "
                        "fall behind today before this is a failure "
                        "(default: 3, which absorbs a weekend)")
    p.add_argument("--quiet", action="store_true",
                   help="print only on failure")
    args = p.parse_args()

    path = find_manifest(args.root)
    if path is None:
        print(f"NO BUNDLE: no {MANIFEST} under "
              + " or ".join(BUNDLE_DIRS)
              + " in this checkout, so the hosted dashboard has no data "
                "to serve.")
        return 2

    try:
        through, at, doc = read_manifest(path)
    except (ValueError, OSError) as exc:
        print(f"UNREADABLE: {path} does not parse ({type(exc).__name__}: "
              f"{exc}), so the publish that wrote it did not complete.")
        return 1

    if through is None:
        print(f"UNREADABLE: {path} carries no data_through, so there is "
              "nothing to check the bundle's age against.")
        return 1

    today = datetime.date.today()
    age = (today - through).days
    head = (f"published bundle: data through {through} ({age} day(s) behind "
            f"{today}), published {at or 'at an unrecorded time'}, "
            f"{doc.get('files', '?')} file(s)")

    # The run status is read before the age is judged so that a stale
    # bundle can report both halves at once: the data has stopped, and
    # here is what the run that last published said about itself.
    finished, code, fired = "at an unrecorded time", None, []
    status_path = find_status(path)
    if status_path is not None:
        try:
            finished, code, fired = read_status(status_path)
        except (ValueError, OSError) as exc:
            print(f"UNREADABLE: {status_path} does not parse "
                  f"({type(exc).__name__}: {exc}), so the publish that "
                  "wrote it did not complete.")
            return 1

    def _say_conditions(indent):
        print(f"{indent}the run that published this bundle finished "
              f"{finished}, exit code "
              f"{code if code is not None else 'unrecorded'}, and raised "
              f"{len(fired)} condition(s):")
        for tag, reason in fired:
            print(f"{indent}  {tag}" + (f" - {reason}" if reason else ""))

    if age > args.max_age_days:
        print("STALE: " + head)
        print(f"       the newest published data day is {age} days behind, "
              f"past the {args.max_age_days}-day allowance. Either the daily "
              "run is not firing, or it is running and its publish is not "
              "reaching this remote.")
        print("       Check the machine's scheduled task, then the "
              "GIT PUBLISH line in the newest file under Reports/logs/.")
        if fired:
            _say_conditions("       ")
        return 1

    if fired:
        # The data is current, so the publishing works and the fault is
        # inside the run. Its own exit code is usually 0 in this state -
        # that is the whole reason the conditions have to travel.
        print("RUN RAISED: " + head)
        _say_conditions("       ")
        print("       The published data is current, so this is a fault "
              "in the run rather than in the publishing. Read the ALERT "
              "block in the newest file under Reports/logs/ on the "
              "machine that runs the pipeline.")
        return 3

    if not args.quiet:
        print("CURRENT: " + head)
        if status_path is not None:
            print(f"         the run that published it finished {finished}, "
                  f"exit code {code if code is not None else 'unrecorded'}, "
                  "and raised nothing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
