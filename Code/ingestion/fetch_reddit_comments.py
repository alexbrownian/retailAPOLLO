"""Reddit comment ingestion from the Arctic Shift public API.

Comments are the data source for the influence tracker (per-author calls
plus the reply graph that maps who responds to whom). Each comment carries
``author``, ``body``, ``created_utc``, ``link_id`` (the post it belongs to)
and ``parent_id`` (what it replies to); the last two are the edges of the
social interaction graph.

Usage::

    python Code/ingestion/fetch_reddit_comments.py            # live, derived window
    python Code/ingestion/fetch_reddit_comments.py --lookback-days 30
    python Code/ingestion/fetch_reddit_comments.py --max-pages 465    # budgeted
    python Code/ingestion/fetch_reddit_comments.py --backfill 2021-01-01 2021-06-30
    python Code/ingestion/fetch_reddit_comments.py --test                # one page

Page budget: ``update_data.py`` computes how many API pages a run may
spend (the pipeline's ~10-minute ceiling minus what this copy measurably
spends on everything else) and passes it here as ``--max-pages`` via
``fetch_all.py``. The allowance is shared out across the panel in
proportion to what each subreddit owes, before the first request, so two
runs are comparable. Hitting a cap is a deferral: the watermark does not
advance and the next run resumes on the same ground. Without
``--max-pages`` the crawl is unbudgeted (backfills, catch-ups). The
arithmetic lives in ``src/pipeline_budget.py``.

Data boundary: the raw comment files stay local (gitignored, like all raw
text), but the influence store derived from them (calls, scores, edges;
text-free and pseudonymous) is committed and shared. Raw text never
crosses git; metadata does.

Scope: comments run live-first. ``fetch_all`` calls this script on every
live pass (watermarked, incremental), and the recommended one-off backfill
is the current year only. At the API's polite rate (1s/page, 100/page)
busy subreddits cost hours per half-year, and the influence tracker's
value is who is right now.

Time parameters: ``after``/``before`` are normalised to epoch seconds
before the first request and stay epoch for every page. Mixing an ISO
date with an epoch cursor in one request draws HTTP 422 from the API, so
one format is chosen once and used everywhere.

Output: ``Data/raw/RedditComments/comments_<range>.jsonl.zst``, one raw
JSON comment per line, reduced to the ``KEEP`` fields. Dedup is by comment
id via a rolling seen-file; a per-subreddit watermark makes repeat live
runs incremental exactly like the post fetcher. A single-instance lock
prevents two crawls from interleaving zstd frames into one output file.
"""

import argparse
import datetime
import json
import os
import sys
import time

import requests
import zstandard

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)
from src.config import DATA_DIR  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OUT_DIR = os.path.join(DATA_DIR, "raw", "RedditComments")
SEEN_FILE = os.path.join(DATA_DIR, "reference",
                         "reddit_comments_seen.json")
WM_FILE = os.path.join(DATA_DIR, "reference",
                       "reddit_comments_watermark.json")
# Forum panel: config/forums.csv via src.settings.load_forums().
# single-instance guard - two crawls writing one output path interleave
# their zstd frames and corrupt the file
LOCK_FILE = os.path.join(DATA_DIR, "reference",
                         "reddit_comments_fetch.lock")
API = "https://arctic-shift.photon-reddit.com/api/comments/search"
PAGE = 100
PAUSE_S = 1.0
# Consecutive full pages yielding ZERO new comments before a crawl is
# called finished. Three rather than one because the API can return a
# page of already-seen ids in the middle of a live band (deletions,
# edits, and the 1-day watermark overlap all produce short all-seen
# runs); three in a row is the crowd being genuinely exhausted, not a
# gap. Without the stop a run can spend ~95 pages on nothing new.
DRY_PAGES_STOP = 3
MAX_SEEN = 200_000
# the fields the influence tracker needs - dropping the rest keeps the raw
# files a fraction of full-comment size
KEEP = ("id", "author", "body", "created_utc", "subreddit",
        "link_id", "parent_id", "score")


class Pacer:
    """Hold the crawl to one request per ``period_s`` seconds, and no slower.

    The sleep is a remainder, not a flat pause after each round-trip: a
    flat sleep makes the true period RTT + period and gives away roughly
    a quarter of the crawl as dead time. The request rate itself is
    unchanged at ``COMMENT_RATE_PER_S``; only the idle gap between the
    response arriving and the next request leaving is removed. Raising the
    rate, or splitting the panel across parallel workers, would break the
    politeness contract of a free public API (``src/pipeline_budget.py``
    explains the rejection), so removing dead time is the one legitimate
    speedup.
    """

    def __init__(self, period_s=PAUSE_S):
        self.period = float(period_s)
        self._next = 0.0

    def wait(self):
        """Block until the next request slot, then reserve the one after it."""
        now = time.monotonic()
        if self._next and now < self._next:
            time.sleep(self._next - now)
        self._next = max(now, self._next) + self.period


def default_lookback_days() -> int:
    """Return the live window in days, derived from the measured run cadence.

    The window is ``ceil(measured cadence) + LATE_ARRIVAL_DAYS`` rather
    than a typed-in constant, so it moves when the cadence moves: a copy
    whose runs are spaced further apart reaches back further on its own
    instead of quietly under-covering.

    Falls back to a fixed 3 days if ``src.pipeline_budget`` cannot be
    imported, so this file still runs standalone without the package on
    its path. Also read by the dashboard's comment catch-up estimate.
    """
    try:
        from src import pipeline_budget
        return int(pipeline_budget.live_lookback_days())
    except Exception:                                 # noqa: BLE001
        return 3


def read_subreddits():
    """Return the enabled forums from ``config/forums.csv``, in file order."""
    from src.settings import load_forums
    return list(load_forums())


def _load(path):
    """Read a JSON file, returning ``{}`` when absent or unreadable."""
    if os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except (ValueError, OSError):
            pass
    return {}


def _save(path, obj):
    """Write ``obj`` as JSON atomically (write beside, then replace)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(path + ".tmp", path)


def _free_path(path: str) -> str:
    """Return ``path``, or a ``..._2``/``..._3`` sibling if it already exists.

    A re-run of the same date range only writes ids the seen-file has not
    got, so the earlier file is still wanted and must never be clobbered.
    """
    if not os.path.exists(path):
        return path
    stem = path[:-len(".jsonl.zst")]
    n = 2
    while os.path.exists(f"{stem}_{n}.jsonl.zst"):
        n += 1
    return f"{stem}_{n}.jsonl.zst"


def _promote(tmp_path: str, out_path: str, tries: int = 5):
    """Rename the temporary file to its final name, retrying briefly.

    On Windows an indexer or sync client can hold a just-written file
    open for a moment, so a ``PermissionError`` is retried with a growing
    pause before it is treated as fatal.
    """
    for i in range(tries):
        try:
            os.replace(tmp_path, out_path)
            return
        except PermissionError:
            if i == tries - 1:
                print(f"could not rename {os.path.basename(tmp_path)} - the "
                      f"data is intact, rename it to "
                      f"{os.path.basename(out_path)} by hand", flush=True)
                raise
            time.sleep(2 * (i + 1))


def _pid_alive(pid: int) -> bool:
    """Return True when the process with ``pid`` is still running.

    On Windows the kernel is asked directly via ``OpenProcess`` /
    ``GetExitCodeProcess``: ``os.kill(pid, 0)`` is not a probe there. It
    raises WinError 87 through a CPython path that surfaces as
    ``SystemError``, which is not caught by ``except OSError`` and would
    kill the whole fetch whenever a stale lock exists.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes                                    # noqa: PLC0415
        _STILL_ACTIVE = 259
        _Q = 0x1000                # PROCESS_QUERY_LIMITED_INFORMATION
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(_Q, False, int(pid))
        if not h:
            return False           # no such process
        try:
            code = ctypes.c_ulong()
            ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
            return bool(ok) and code.value == _STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True          # exists, owned by someone else
    except OSError:
        return False
    return True


def acquire_lock(label: str):
    """Take the single-instance lock, or explain who holds it.

    A lock whose PID is gone is stale (a crashed run) and is taken over.

    Args:
        label: Human-readable tag for this crawl, recorded in the lock.

    Returns:
        True when the lock was taken; False when another live crawl
        holds it.
    """
    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
    held = _load(LOCK_FILE)
    if held and _pid_alive(int(held.get("pid", 0))):
        print(f"another comment fetch is already running "
              f"(pid {held['pid']}, label {held.get('label')}, started "
              f"{held.get('started')}).\nTwo crawls writing the same output "
              f"corrupt it - wait for that one, or kill it first:\n"
              f"  taskkill /PID {held['pid']} /F", flush=True)
        return False
    if held:
        print(f"clearing stale lock from dead pid {held.get('pid')}",
              flush=True)
    _save(LOCK_FILE, {"pid": os.getpid(), "label": label,
                      "started": datetime.datetime.now().isoformat(
                          timespec="seconds")})
    return True


def release_lock():
    """Remove the single-instance lock file if present."""
    try:
        os.remove(LOCK_FILE)
    except OSError:
        pass


def to_epoch(v) -> int:
    """Convert a date to epoch seconds, the one time format the crawl uses.

    Called once on the CLI bounds, so every request the crawl makes (the
    first page and every cursor page after it) uses the same format;
    mixing an ISO ``after`` with an epoch ``before`` cursor draws HTTP 422
    from the API.

    Args:
        v: ``YYYY-MM-DD`` (or any ISO date/datetime, naive values taken
            as UTC), or a string of digits that is already epoch seconds.

    Returns:
        Epoch seconds as an int.
    """
    s = str(v)
    if s.isdigit():
        return int(s)
    d = datetime.datetime.fromisoformat(s)
    if d.tzinfo is None:
        d = d.replace(tzinfo=datetime.timezone.utc)
    return int(d.timestamp())


def fetch_page(sub, after, before, retries=4):
    """Fetch one API page of comments.

    The retry policy depends on the failure type. A 429, a 5xx, a
    "slow down" 422 or a network drop is transient, so the call waits and
    retries (20s, 40s, 60s...). Any other 4xx means the request itself is
    malformed; retrying the identical request cannot help, so the
    server's explanation is printed and the call stops immediately.

    Args:
        sub: Subreddit name.
        after: Window start in epoch seconds.
        before: Window end in epoch seconds (exclusive).
        retries: Attempts before giving up on transient failures.

    Returns:
        The list of comment records (possibly empty), or ``None`` when the
        page could not be fetched.
    """
    for attempt in range(retries):
        try:
            r = requests.get(API, params={"subreddit": sub,
                                          "after": int(after),
                                          "before": int(before),
                                          "limit": PAGE},
                             timeout=(10, 60))
            if r.status_code == 200:
                return r.json().get("data", [])
            # A 422 is normally a malformed range and must not be
            # retried. But this API also answers a too-fast crawl with 422
            # and the body {"error": "Timeout. Maybe slow down a bit"},
            # which is a rate limit wearing a client-error status code.
            # The body is what separates the two cases, so the body is
            # what decides.
            _slow = (r.status_code == 422
                     and "slow down" in r.text.lower())
            if r.status_code == 429 or r.status_code >= 500 or _slow:
                print(f"    HTTP {r.status_code}"
                      f"{' (rate limit)' if _slow else ''} - backing off "
                      f"{20*(attempt+1)}s", flush=True)
            else:
                print(f"    HTTP {r.status_code} (client error - not "
                      f"retrying): {r.text[:200]}", flush=True)
                return None
        except requests.RequestException as e:
            print(f"    network problem ({type(e).__name__}) - retrying in "
                  f"{20*(attempt+1)}s. (network drop? safe to Ctrl-C and "
                  "re-run later: the seen-file dedups everything already "
                  "saved)", flush=True)
        time.sleep(20 * (attempt + 1))
    print(f"    r/{sub}: giving up this run", flush=True)
    return None


def main():
    """Crawl every configured subreddit and write one raw comment file.

    Returns:
        ``0`` on success, ``1`` when another crawl holds the lock, and
        ``130`` when the crawl was interrupted or crashed after saving
        what it had fetched.
    """
    p = argparse.ArgumentParser(description="Arctic Shift comment ingestion "
                                            "(influence tracker source)")
    p.add_argument("--lookback-days", type=int, default=None,
                   help="live window in days. Default is DERIVED from this "
                        "machine's measured run cadence "
                        "(ceil(cadence) + LATE_ARRIVAL_DAYS, currently "
                        f"{default_lookback_days()}d) rather than typed in - "
                        "the per-subreddit watermark makes every later run "
                        "incremental, so only the FIRST live run pays for "
                        "the whole window")
    p.add_argument("--backfill", nargs=2, metavar=("START", "END"),
                   help="historical range YYYY-MM-DD YYYY-MM-DD (end excl); "
                        "live scope is the current year only, e.g. "
                        "2026-01-01 <today>")
    p.add_argument("--max-pages", type=int, default=None,
                   help="TOTAL API pages this run may spend across the whole "
                        "panel (the budgeted path - update_data.py computes "
                        "it from the pipeline's runtime ceiling and passes it "
                        "through fetch_all.py). Omitted = UNBUDGETED: crawl "
                        "until the data runs out, which is what "
                        "update_comments.py does for backfills and "
                        "catch-ups. A subreddit that hits its share keeps "
                        "its old watermark, so nothing is lost - it is "
                        "deferred to the next run and printed as such")
    p.add_argument("--test", action="store_true")
    args = p.parse_args()
    if args.lookback_days is None:
        args.lookback_days = default_lookback_days()

    subs = read_subreddits()
    if args.backfill:
        label = f"{args.backfill[0]}_{args.backfill[1]}"
        # ONE format for the whole crawl - see to_epoch's docstring
        after = to_epoch(args.backfill[0])
        before = to_epoch(args.backfill[1])
        incremental = False
    else:
        today = datetime.date.today()
        after = to_epoch((today - datetime.timedelta(
            days=args.lookback_days)).isoformat())
        before = to_epoch((today + datetime.timedelta(days=1)).isoformat())
        label = today.isoformat()
        incremental = True

    if args.test:
        rows = fetch_page(subs[0], after, before) or []
        print(f"TEST: r/{subs[0]} returned {len(rows)} comments; sample:",
              flush=True)
        for rec in rows[:3]:
            print(f"  u/{rec.get('author')}: {str(rec.get('body',''))[:60]}",
                  flush=True)
        return 0

    seen_obj = _load(SEEN_FILE)
    seen_list = seen_obj.get("ids", [])
    seen = set(seen_list)
    marks = _load(WM_FILE)

    # ---- THE PAGE ALLOWANCE, SHARED OUT BEFORE A SINGLE REQUEST IS MADE ----
    # Allocating up front - rather than crawling until a clock runs out - is
    # what makes two runs comparable: every subreddit knows its share before
    # the first request, so what gets collected does not depend on network
    # luck or on the order the panel happens to be listed in.
    #
    # A subreddit's share is proportional to what it OWES: the days since its
    # watermark, times its own measured pages/day.  A subreddit crawled an
    # hour ago owes almost nothing; one that has not been reached for a week
    # owes a week.  See src/pipeline_budget.allocate for the one-page floor
    # and why surplus is deliberately not redistributed mid-run.
    plan, budget = {}, None
    if args.max_pages:
        try:
            from src import pipeline_budget
            now_s = time.time()
            span_d = max(0.0, (before - after) / 86400.0)
            owed = {}
            for s in subs:
                wm = marks.get(s)
                if incremental and wm:
                    owed[s] = min(span_d,
                                  max(0.0, (now_s - float(wm)) / 86400.0))
                else:
                    # never crawled (or a backfill): it owes the whole window
                    owed[s] = span_d
            plan = pipeline_budget.allocate(owed, int(args.max_pages))
            budget = int(args.max_pages)
            print(f"page budget {budget} across {len(subs)} subreddits "
                  f"(allocated {sum(plan.values())}): "
                  + ", ".join(
                      f"r/{s} {plan[s]}"
                      for s in sorted(plan, key=lambda k: -plan[k])[:5])
                  + (" ..." if len(plan) > 5 else ""), flush=True)
        except Exception as e:                        # noqa: BLE001
            # A BUDGET FAILING MUST NOT STOP AN INGESTION.  If the allocator
            # cannot be imported or the ledger is unreadable, crawl unbudgeted
            # and say so, rather than fetching nothing because the accounting
            # broke.
            print(f"  page budget unavailable ({type(e).__name__}: {e}) - "
                  "crawling unbudgeted", flush=True)
            plan, budget = {}, None

    # ONE PACER FOR THE WHOLE CRAWL, not one per subreddit: the contract is a
    # rate for this process, so the clock must not reset at every panel member.
    pacer = Pacer(PAUSE_S)
    deferred = []

    if not acquire_lock(label):
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = _free_path(os.path.join(OUT_DIR,
                                       f"comments_{label}.jsonl.zst"))
    # the .tmp name carries our PID: even if the lock is ever defeated,
    # two runs cannot share one output stream
    tmp_path = f"{out_path}.{os.getpid()}.tmp"
    raw = open(tmp_path, "wb")
    writer = zstandard.ZstdCompressor().stream_writer(raw)
    total = 0
    interrupted = False
    try:
        for sub in subs:
            print(f"  r/{sub:<24} starting crawl", flush=True)
            sub_after = after                 # already epoch (see to_epoch)
            wm = marks.get(sub)
            if incremental and wm:
                # start from the watermark (minus a 1-day overlap for late
                # arrivals) if that is LATER than the lookback window start
                sub_after = max(after, int(wm) - 86400)
            got, newest, completed = 0, int(wm) if wm else 0, True
            cursor = before                   # epoch, stays epoch
            page_num = 0
            pages_used = 0
            dry_pages = 0                     # see DRY_PAGES_STOP
            last_got = 0
            oldest_ts = None                  # how far back this crawl got
            cap = plan.get(sub) if budget is not None else None
            while True:
                # THE CAP IS A DEFERRAL, NOT A TRUNCATION.  Breaking here
                # leaves `completed` False, so the watermark below does NOT
                # advance and the next run resumes on exactly this ground.
                # Without that, a capped run would jump its watermark past
                # days it never crawled - silent data loss, and the one
                # failure mode nothing downstream could detect.
                if cap is not None and page_num >= cap:
                    completed = False
                    deferred.append(sub)
                    print(f"    page budget reached ({cap} page"
                          f"{'' if cap == 1 else 's'}) - the rest of "
                          f"r/{sub} is deferred to the next run", flush=True)
                    break
                page_num += 1
                print(f"    fetching page {page_num}...", flush=True)
                pacer.wait()
                rows = fetch_page(sub, sub_after, cursor)
                if rows is None:
                    completed = False
                    break
                pages_used += 1
                if not rows:
                    print(f"    page {page_num}: no rows", flush=True)
                    break
                for rec in rows:
                    cid = str(rec.get("id", ""))
                    if not cid or cid in seen:
                        continue
                    seen.add(cid)
                    seen_list.append(cid)
                    slim = {k: rec.get(k) for k in KEEP}
                    newest = max(newest, int(rec.get("created_utc", 0) or 0))
                    writer.write((json.dumps(slim) + "\n").encode("utf-8"))
                    got += 1
                # ---- HAVE WE RUN DRY? -------------------------------
                # The crawl walks NEWEST-FIRST, so every new comment is at
                # the front. Once the pages stop yielding anything the
                # crawl has re-entered ground an earlier run already
                # covered, and every further page is a full 100 rows of
                # comments already held. Left running, those dead pages
                # come out of the same page budget that then defers busier
                # subreddits, and because a capped run keeps its
                # watermark, the next run starts in the same place and
                # buys the same dead pages again.
                #
                # Breaking here leaves `completed` True on purpose, so the
                # watermark advances. That is safe: `sub_after` is
                # `max(after, watermark - 1 day)`, so the crawl can never
                # reach further back than a day before the watermark
                # anyway. Refusing to advance buys no extra history.
                if got == last_got:
                    dry_pages += 1
                    if dry_pages >= DRY_PAGES_STOP:
                        print(f"    caught up - {DRY_PAGES_STOP} pages "
                              "with nothing new, so r/" + sub + " is "
                              "fully collected. Stopping here and "
                              "advancing the watermark.", flush=True)
                        break
                else:
                    dry_pages = 0
                last_got = got
                oldest = min(int(r["created_utc"]) for r in rows)
                oldest_ts = oldest if oldest_ts is None else min(oldest_ts,
                                                                 oldest)
                print(f"    page {page_num}: +{got} new comments so far "
                      f"({len(rows)} rows)", flush=True)
                if len(rows) < PAGE:
                    break
                cursor = oldest               # epoch int, same as page 1
                # No sleep here: the Pacer above already spends whatever is
                # left of the second before the NEXT request goes out; a
                # flat sleep here would be dead time on top of the RTT.
            print(f"  r/{sub:<24} {got:>6} new comments "
                  f"({pages_used} page{'' if pages_used == 1 else 's'})",
                  flush=True)
            if incremental and completed and newest:
                marks[sub] = newest
            total += got
            # WHAT THIS SUBREDDIT ACTUALLY COST, fed straight back into the
            # ledger that plans the next run. This is the loop that lets the
            # budget stop being a guess after one run: pages and comments are
            # observed here, not assumed anywhere.
            #
            # MEASURE THE SPAN THE CRAWL COVERED, NOT THE SPAN IT ASKED FOR.
            # A subreddit that hit its cap after 3 pages covered ~half a day,
            # not the whole 5-day window - dividing its pages by the window
            # would book it as CHEAP precisely because it ran out of budget,
            # and the allocator would then hand it even fewer pages next
            # time. That is a spiral, not a measurement, so a deferred crawl
            # is measured against how far back it actually reached.
            _cov = (max(0.0, (before - sub_after) / 86400.0) if completed
                    else (max(0.0, (before - oldest_ts) / 86400.0)
                          if oldest_ts else 0.0))
            try:
                from src import pipeline_budget
                pipeline_budget.record_pages(sub, pages_used, got, _cov)
            except Exception:                     # noqa: BLE001
                pass          # a cost ledger is an optimisation, never a
                              # blocker - a run that fetched data has already
                              # done its job
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted - keeping the comments fetched so far",
              flush=True)
    except Exception as e:                    # noqa: BLE001 - see below
        # ANY crash mid-crawl still promotes what was fetched; an exception
        # escaping before the rename would leave hours of good data in an
        # orphaned .tmp the influence ingester cannot see.
        interrupted = True
        print(f"\ncrawl failed ({type(e).__name__}: {e}) - keeping the "
              f"comments fetched so far", flush=True)
    finally:
        # close BOTH layers before any rename: the zstd stream flushes its
        # final frame, then the OS handle is released (a still-open handle
        # is what makes os.replace fail with WinError 32 on Windows)
        try:
            writer.close()
        finally:
            raw.close()
        release_lock()

    # EVERY DEFERRAL IS PRINTED. The pipeline never silently collects less
    # than it claims: if the allowance ran out, the run says which
    # communities it stopped short on and what closes the gap. A budget the
    # operator cannot see is indistinguishable from a bug.
    if deferred:
        print(f"deferred (page budget): {len(deferred)} subreddit"
              f"{'' if len(deferred) == 1 else 's'} - "
              + ", ".join("r/" + s for s in deferred), flush=True)
        print("  their watermarks did NOT advance: the next run resumes "
              "exactly here. Run more often, or use "
              "'python Code/ingestion/update_comments.py' to catch up unbudgeted.",
              flush=True)

    if total == 0:
        os.remove(tmp_path)
        print("no new comments this run", flush=True)
        return 0
    _promote(tmp_path, out_path)
    seen_obj["ids"] = seen_list[-MAX_SEEN:]
    _save(SEEN_FILE, seen_obj)
    # the watermark only advanced for subreddits that finished, so a
    # partial run is always safe to simply re-run - nothing duplicates
    if incremental:
        _save(WM_FILE, marks)
    print(f"comments: {total:,} new -> {os.path.basename(out_path)}",
          flush=True)
    print("next:  cd Code && python -m src.analytics.influence --update", flush=True)
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
