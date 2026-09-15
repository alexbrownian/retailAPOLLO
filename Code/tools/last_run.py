#!/usr/bin/env python
"""How is the daily refresh doing?

    python Code/tools/last_run.py
    python Code/tools/last_run.py --log        # also print the run's ALERT block
    python Code/tools/last_run.py --tail 40    # and the last 40 log lines

The daily refresh leaves its account of itself in four places: the
scheduled task knows whether it fired, the run log holds what happened,
``Reports/run_state.json`` holds the day the data reached, and the
published bundle carries the verdict the run reached on itself. This
reads all four and prints one report, so "is it working" is one command
rather than four.

It only reads. Exit code 0 when the last run looks healthy, 1 when
something in it wants attention, so it is usable from a script as well
as by eye.
"""

import argparse
import datetime
import glob
import json
import os
import re
import subprocess
import sys

THIS = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(THIS)
PROJECT = os.path.dirname(CODE)
sys.path.insert(0, CODE)

from src.config import LOG_DIR, REPORTS_DIR                # noqa: E402

TASK_NAME = "retailAPOLLO daily refresh"
BUNDLE_DIRS = ("Data/dashboard", "data/dashboard")


def _read_json(path):
    """Parsed JSON, or ``None`` when the file is absent or unreadable."""
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def newest_log():
    """Path of the most recently written run log, or ``None``."""
    logs = glob.glob(os.path.join(LOG_DIR, "run_*.log"))
    return max(logs, key=os.path.getmtime) if logs else None


def bundle_file(name):
    """Path of ``name`` inside whichever bundle folder this copy has."""
    for d in BUNDLE_DIRS:
        p = os.path.join(PROJECT, d, name)
        if os.path.isfile(p):
            return p
    return None


def task_state():
    """``(last_run, last_result, next_run)`` from the scheduled task.

    Every value is ``None`` off Windows, or where no task of this name is
    registered - neither is a fault, so the caller reports rather than
    fails on it.
    """
    if os.name != "nt":
        return None, None, None
    try:
        r = subprocess.run(["schtasks", "/Query", "/TN", TASK_NAME,
                            "/V", "/FO", "LIST"],
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None, None, None
    if r.returncode != 0:
        return None, None, None

    def field(label):
        m = re.search(rf"^{label}:\s*(.+)$", r.stdout, re.MULTILINE)
        return m.group(1).strip() if m else None

    return field("Last Run Time"), field("Last Result"), field("Next Run Time")


def alert_block(log_path):
    """The ALERT block from a run log as a list of lines, or ``[]``."""
    if not log_path:
        return []
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return []
    starts = [i for i, ln in enumerate(lines) if ln.rstrip().endswith("ALERT")]
    if not starts:
        # A run that raised nothing says so on one line instead.
        return [ln for ln in lines if "ALERT:" in ln][-1:]
    out = lines[starts[-1] - 1:]
    return out[:40]


def main():
    p = argparse.ArgumentParser(
        description="Report on the most recent daily refresh.")
    p.add_argument("--log", action="store_true",
                   help="also print the run's ALERT block")
    p.add_argument("--tail", type=int, default=0, metavar="N",
                   help="also print the last N lines of the run log")
    args = p.parse_args()

    state = _read_json(os.path.join(REPORTS_DIR, "run_state.json")) or {}
    status = _read_json(bundle_file("run_status.json") or "") or {}
    manifest = _read_json(bundle_file("publish_manifest.json") or "") or {}
    log_path = newest_log()
    last_run, last_result, next_run = task_state()
    today = datetime.date.today()
    trouble = []

    print("=" * 64)
    print("DAILY REFRESH")

    # ---- the scheduled task: did it fire at all ----
    if last_run is None:
        print("  scheduled   : no task registered on this machine, or this "
              "is not Windows")
        print("                (register one with tools/setup_schedule.py)")
    else:
        print(f"  scheduled   : last fired {last_run}, next {next_run}")
        # schtasks reports the LAST TASK RESULT: 0 succeeded, 267011 means
        # the task has never run yet, anything else is the exit code the
        # command returned.
        if last_result not in (None, "0", "267011"):
            print(f"                last result {last_result} - the command "
                  "exited non-zero")
            trouble.append("the scheduled task's last result was not 0")

    # ---- the run itself ----
    if not state:
        print("  last run    : no Reports/run_state.json yet - no run has "
              "finished on this copy")
        trouble.append("no run has finished yet")
    else:
        code = state.get("exit_code")
        print(f"  last run    : finished {state.get('finished', '?')}, "
              f"exit code {code if code is not None else '?'}")
        if code not in (0, None):
            trouble.append(f"the last run exited {code}")

    # ---- has the data actually moved ----
    day = state.get("newest_data_day")
    if day:
        try:
            age = (today - datetime.date.fromisoformat(day)).days
        except ValueError:
            age = None
        since = state.get("newest_data_day_since", "?")
        print(f"  newest data : {day}"
              + (f" ({age} day(s) behind {today})" if age is not None else "")
              + f", unchanged since {since}")
        if age is not None and age > 3:
            trouble.append(f"the newest data day is {age} days behind today")

    # ---- what the run said about itself ----
    fired = status.get("conditions") or []
    if not status:
        print("  run raised  : no run_status.json in the bundle yet - it is "
              "written by the next full run")
    elif not fired:
        print("  run raised  : nothing - the run was clean")
    else:
        print(f"  run raised  : {len(fired)} condition(s)")
        for item in fired:
            tag = item.get("tag", "condition") if isinstance(item, dict) \
                else str(item)
            why = item.get("reason", "") if isinstance(item, dict) else ""
            print(f"                - {tag}" + (f": {why}" if why else ""))
        trouble.append(f"the last run raised {len(fired)} condition(s)")

    # ---- what the hosted copy is being served ----
    if manifest:
        print(f"  published   : data through "
              f"{manifest.get('data_through', '?')}, "
              f"{manifest.get('files', '?')} file(s), at "
              f"{manifest.get('published_at', 'an unrecorded time')}")
    else:
        print("  published   : no publish manifest - nothing has been "
              "staged for the hosted copy")

    print(f"  log         : {log_path or 'none yet'}")
    print("=" * 64)

    if args.log:
        block = alert_block(log_path)
        if block:
            print()
            for ln in block:
                print(ln)

    if args.tail and log_path:
        print()
        try:
            with open(log_path, encoding="utf-8", errors="replace") as fh:
                for ln in fh.read().splitlines()[-args.tail:]:
                    print(ln)
        except OSError as exc:
            print(f"(could not read the log: {exc})")

    if trouble:
        print()
        print("WANTS ATTENTION:")
        for t in trouble:
            print(f"  - {t}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
