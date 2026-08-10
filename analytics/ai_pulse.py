"""
ai_pulse.py — the LLM writes the qualitative read of the market.
================================================================

The AI Pulse tab carried hand-written SAMPLE text since it shipped; this
module produces the real thing.  It runs at the end of every update_data
pass (and on demand from the dashboard), builds a compact EVIDENCE PACK
from the same aggregates every other tab reads, hands the freshest raw
posts to the firm's LLM through `src/ai.py` (the Apollo gateway), and
saves one JSON the dashboard renders verbatim:

    data/processed/ai_pulse.json
        as_of, generated_at, model, evidence (the numbers used),
        market_pulse, talk_of_the_town,
        theme_briefs[], catalyst_watch[], divergences[],
        agentic{digest, asks[], actions[], risk_note}

DESIGN RULES
  * NUMBERS COME FROM THE STORES, WORDS COME FROM THE MODEL.  The LLM
    never invents a statistic: every figure it may cite is handed to it
    in the evidence pack, and the pack itself is saved alongside the
    prose so any sentence can be audited against the inputs.
  * FOUR STAGES PER RUN (desk request 2026-08-04: "longer and much
    more detailed"): (1) the whole-market read, (2) one brief per theme
    - BATCHED, so every theme the crowd discusses gets one and the
    dropdown is never truncated by a token ceiling, (3) catalysts and
    divergences, (4) the agentic digest.
    Splitting keeps each response inside the deployment's output
    ceiling; budget-capped by AI_MAX_CALLS regardless.
  * PARAPHRASE, NEVER QUOTE.  Raw post text goes TO the model; only
    model-written summaries and paraphrases come back and are stored -
    no verbatim crowd text, no usernames, same text-free boundary as
    the committed aggregates.
  * DEGRADE, NEVER CRASH.  No VPN / no dimsum_lite / budget spent ->
    generate() returns (False, reason); update_data logs it and moves
    on; the dashboard keeps showing the last pulse (or the samples)
    with an honest banner.

CLI:
    python -m analytics.ai_pulse           # generate now
    python -m analytics.ai_pulse --dry     # print the evidence pack only
"""

from __future__ import annotations

import io
import re as _re
import json
import os
import re
from datetime import datetime, timezone

import pandas as pd

from src import ai
from src.config import PROCESSED_DIR
from src.themes import THEME_ETFS, themes_in_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(PROCESSED_DIR, "ai_pulse.json")
RAW_DIR = os.path.join(ROOT, "data", "raw", "RedditComments")
POST_SAMPLE_N = 320         # posts handed to the model for the market read
POST_CLIP = 360             # chars per post - mood, not essays
HARVEST_MAX = 60_000        # recent posts held in memory for sampling
POSTS_PER_THEME = 18        # posts behind each per-theme brief
THEME_BRIEF_MIN_SHARE = 0.001   # a theme needs at least this share of the
                            # week's mentions to earn a brief - it must have
                            # enough posts for the brief to be written FROM
                            # something. LOWERED from 0.004 on 2026-08-04:
                            # the old floor silently dropped 8 tradeable
                            # themes, including europe_defense (0.27% of
                            # mentions but the highest-conviction theme on
                            # the board, euphoria 90) - a share floor alone
                            # cannot see that. 0.1% keeps 33 of the 34
                            # tradeable themes; the one it drops has 0.02%
                            # and genuinely has nothing to read.
MAX_THEME_BRIEFS = 40       # i.e. no practical cap - every tradeable theme
                            # gets a brief. It used to be 18, which quietly
                            # truncated the dropdown even when more themes
                            # qualified. The briefs are BATCHED below so the
                            # count is no longer limited by one response's
                            # token ceiling.
THEMES_PER_CALL = 12        # briefs per gateway call. 12 x ~110 words sits
                            # comfortably inside the 4k-token response
                            # ceiling with room for the model to run long;
                            # more themes simply means more calls.


# THE AS-OF DATE. None = today, which is every normal run.
#
# Desk 2026-08-05: "can we make it so we can dial back to a specific day
# and then re run the pulse?" Set this and the whole pulse is rebuilt as
# if that were the newest day on record - useful for reading back into an
# episode, and for showing what the page WOULD have said the week before
# a top.
#
# It is applied HERE, in the one function every store passes through,
# rather than threaded into a dozen call sites. Everything downstream
# derives "now" from `.max()` on the frame it is handed, so clipping the
# frame moves the whole clock at once and no section can be left behind
# reading a different day. That property is the reason for the design:
# a pulse where one paragraph is dated differently from another would be
# worse than no back-dating at all.
AS_OF: "pd.Timestamp | None" = None


def _read(name: str) -> pd.DataFrame | None:
    p = os.path.join(PROCESSED_DIR, name)
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        if AS_OF is not None:
            df = df[df["date"] <= AS_OF]
    return df


def _evidence() -> dict:
    """The numbers the model is allowed to use - nothing else."""
    ev: dict = {}
    tc = _read("daily_theme_counts.parquet")
    if tc is not None and len(tc):
        tc = tc[tc["theme"].isin(THEME_ETFS)]
        hi = tc["date"].max()
        w = tc[tc["date"] > hi - pd.Timedelta(days=7)]
        prev = tc[(tc["date"] <= hi - pd.Timedelta(days=7))
                  & (tc["date"] > hi - pd.Timedelta(days=35))]
        cur = w.groupby("theme")["mention_count"].sum()
        base = prev.groupby("theme")["mention_count"].sum() / 4.0
        share = (cur / cur.sum()).sort_values(ascending=False)
        ev["as_of"] = str(hi.date())
        # every theme with a MATERIAL share, not just the top 10: the page
        # now offers a per-theme dropdown, so the model has to be able to
        # speak about anything the desk can select
        share = share[share >= THEME_BRIEF_MIN_SHARE].head(MAX_THEME_BRIEFS)
        ev["theme_mention_share_7d"] = {
            t: {"share": round(float(s), 4),
                "vs_4w_avg": (round(float(cur.get(t, 0)
                                          / base.get(t)), 2)
                              if base.get(t) else None)}
            for t, s in share.items()}
    cz = _read("daily_theme_conviction.parquet")
    if cz is not None and len(cz):
        last = cz[cz["date"] == cz["date"].max()]
        last = last[last["theme"].isin(THEME_ETFS)]
        top = last.nlargest(5, "conviction_z")
        bot = last.nsmallest(3, "conviction_z")
        ev["conviction_z"] = {
            r.theme: round(float(r.conviction_z), 2)
            for r in pd.concat([top, bot]).itertuples()}
    dk = _read("euphoria_desk.parquet")
    if dk is not None and len(dk):
        hi = dk["date"].max()
        recent = dk[dk["date"] > hi - pd.Timedelta(days=21)]
        ev["live_flags"] = (
            [{"name": r.name, "flag": "GET OUT",
              "date": str(r.date.date())}
             for r in recent[recent["get_out"].astype(bool)].itertuples()]
            + [{"name": r.name, "flag": "GET IN",
                "date": str(r.date.date())}
               for r in recent[recent["get_in"].astype(bool)]
               .itertuples()])
    eu = _read("euphoria_levels.parquet")
    if eu is not None and len(eu):
        last = eu[eu["date"] == eu["date"].max()]
        ev["euphoria_top5"] = {
            r.name: round(float(r.level))
            for r in last.nlargest(5, "level").itertuples()}
    # THE WORDS THE CROWD HAS JUST STARTED USING.
    #
    # Added 2026-08-05. The pulse could name themes and tickers but had
    # no way to say "everyone is suddenly talking about tariffs" - the
    # topical phrase that spreads through a forum in a week and is often
    # the actual subject, ahead of any ticker. Measured as this week
    # against its own 4-week average, so a permanently common word never
    # qualifies and a genuinely new one always does.
    tm = _read("daily_term_counts.parquet")
    if tm is not None and len(tm):
        hi = tm["date"].max()
        cur = (tm[tm["date"] > hi - pd.Timedelta(days=7)]
               .groupby("term")["mention_count"].sum())
        prev = (tm[(tm["date"] <= hi - pd.Timedelta(days=7))
                   & (tm["date"] > hi - pd.Timedelta(days=35))]
                .groupby("term")["mention_count"].sum() / 4.0)
        spikes = {t: round(float(cur[t] / prev[t]), 2) for t in cur.index
                  if prev.get(t, 0) >= 3 and cur[t] >= 15}
        top = sorted(spikes.items(), key=lambda kv: -kv[1])[:15]
        if top:
            ev["emerging_terms_7d"] = {
                t: {"vs_4w_avg": r, "mentions_7d": int(cur[t])}
                for t, r in top}

    # WHICH FORUMS are carrying the conversation - so the market read can
    # say "r/investing vs r/wallstreetbets" instead of "the crowd"
    sub = _read("daily_ticker_counts_by_subreddit.parquet")
    if sub is not None and len(sub) and "subreddit" in sub.columns:
        hi = sub["date"].max()
        w = sub[sub["date"] > hi - pd.Timedelta(days=7)]
        vol = w.groupby("subreddit")["mention_count"].sum()
        if vol.sum():
            ev["forums_7d"] = {
                str(k): {"share": round(float(v) / float(vol.sum()), 3)}
                for k, v in vol.nlargest(10).items()}
    ag = _read("daily_agentic_counts.parquet")
    if ag is not None and len(ag):
        hi = ag["date"].max()
        w = ag[ag["date"] > hi - pd.Timedelta(days=28)]
        tot = w[w["category"] == "_total_posts"]["mention_count"].sum()
        sig = w[w["category"] != "_total_posts"]
        ev["agentic_28d"] = {
            "per_10k_posts": (round(float(sig["mention_count"].sum())
                              / tot * 10000, 1) if tot else None),
            "by_category": {k: int(v) for k, v in
                            sig.groupby("category")["mention_count"]
                            .sum().items()},
            "top_themes": {k: int(v) for k, v in
                           sig[sig["theme"] != ""]
                           .groupby("theme")["mention_count"].sum()
                           .nlargest(5).items()},
        }
    return ev


_HARVEST: list[dict] | None = None
_HARVEST_KEY = None


def _harvest(max_posts: int = HARVEST_MAX) -> list[dict]:
    """Every recent raw post, theme-tagged and engagement-stamped, held
    once per process.  The samplers below all slice from this, so the
    archives are decompressed a single time however many sections the
    pulse writes."""
    global _HARVEST, _HARVEST_KEY
    if _HARVEST is not None and _HARVEST_KEY == AS_OF:
        return _HARVEST
    # a back-dated run must not be served the live cache, and vice versa
    _HARVEST_KEY = AS_OF
    # string compare on YYYY-MM-DD: same ordering as dates, and it runs
    # once per post over 300k+ posts, where a Timestamp round-trip is not
    _AS_OF_STR = f"{AS_OF:%Y-%m-%d}" if AS_OF is not None else ""
    import zstandard
    files = [f for f in os.listdir(RAW_DIR)
             if f.endswith(".jsonl.zst") and ".tmp" not in f
             and "_salvaged" not in f]

    # READ THE ARCHIVES MOST LIKELY TO CONTAIN THE TARGET DATE FIRST.
    #
    # Newest-mtime order is right for a live run and wrong for a
    # back-dated one. The filenames carry their date range, so for an
    # as-of run the file whose range CONTAINS that date is opened first
    # and the rest follow backwards. Without this, one 43 MB archive
    # filled the cap, the loop broke, and archives holding the target
    # week were never opened at all - which is why as-of 2021-02-01
    # returned nothing while the 2021 archive sat on disk.
    def _span(fname):
        ds = _re.findall(r"(\d{4}-\d{2}-\d{2})", fname)
        return (ds[0], ds[-1]) if ds else ("", "")

    def _rank(fname):
        lo, hi = _span(fname)
        if AS_OF is None:
            return (0, -os.path.getmtime(os.path.join(RAW_DIR, fname)))
        if lo and hi and lo <= _AS_OF_STR <= hi:
            return (0, 0)               # contains the day - read first
        if hi and hi <= _AS_OF_STR:
            return (1, -_ord(hi))       # before it, newest first
        return (2, 0)                   # entirely after it - last resort

    def _ord(d):
        return int(d.replace("-", ""))

    files.sort(key=_rank)
    rows: list[dict] = []
    for fname in files:
        with open(os.path.join(RAW_DIR, fname), "rb") as fh:
            t = io.TextIOWrapper(
                zstandard.ZstdDecompressor().stream_reader(fh),
                encoding="utf-8", errors="replace")
            for line in t:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                body = str(d.get("body") or "")
                if len(body) < 40:      # one-word replies carry no read
                    continue
                try:
                    day = datetime.fromtimestamp(
                        int(float(d.get("created_utc") or 0)),
                        tz=timezone.utc).strftime("%Y-%m-%d")
                except (ValueError, TypeError, OSError):
                    day = str(d.get("created_at", ""))[:10]
                try:
                    score = int(float(d.get("score") or 0))
                except (ValueError, TypeError):
                    score = 0
                # THE AS-OF FILTER APPLIES *HERE*, NOT AFTER THE CAP.
                #
                # It used to run once at the end, over whatever the
                # newest-first walk had already collected. For a live run
                # that is identical. For a BACK-DATED one it was fatal:
                # the cap filled with recent posts, the filter then threw
                # nearly all of them away, and the model was handed
                # either a stale sample or nothing at all. Measured
                # before this fix - as-of 2026-07-20 returned posts whose
                # newest was 30 June, a twenty-day hole, and as-of
                # 2021-02-01 returned ZERO despite the 2021 archive
                # sitting right there on disk.
                #
                # Filtering inside the loop means the cap fills with
                # posts that are actually ELIGIBLE, so the walk keeps
                # reading older archives until it has enough of them.
                # That is what makes "drag the slider back and have the
                # model read that week" work at all.
                if AS_OF is not None and day and day > _AS_OF_STR:
                    continue
                rows.append({"day": day,
                             "sub": str(d.get("subreddit", "")),
                             "score": score,
                             "themes": sorted(themes_in_text(body))[:3],
                             "text": body[:POST_CLIP]})
                if len(rows) >= max_posts:
                    break               # the cap was only checked BETWEEN
                                        # files, so a single large archive
                                        # blew past it by 5x and starved
                                        # every archive after it
        if len(rows) >= max_posts:
            break
    _HARVEST = rows
    return rows


def _fresh_posts(n: int = POST_SAMPLE_N) -> list[dict]:
    """The model's ears on the WHOLE market, not just the newest corner.

    Stratified deliberately: the sample is spread across the most recent
    days and across forums, and inside each bucket it is ranked by the
    crowd's OWN measure of what mattered (score).  A plain "newest N"
    sample - what this used to be - reads whichever subreddit happened to
    be awake in the last hour and calls it the market."""
    rows = _harvest()
    if not rows:
        return []
    days = sorted({r["day"] for r in rows if r["day"]}, reverse=True)[:7]
    per_day = max(1, n // max(1, len(days)))
    out: list[dict] = []
    for day in days:
        pool = [r for r in rows if r["day"] == day]
        by_sub: dict[str, list[dict]] = {}
        for r in sorted(pool, key=lambda r: -r["score"]):
            by_sub.setdefault(r["sub"], []).append(r)
        # round-robin the forums so one loud board cannot own the sample
        picked, i = [], 0
        while len(picked) < per_day and any(by_sub.values()):
            for sub in list(by_sub):
                if by_sub[sub] and len(picked) < per_day:
                    picked.append(by_sub[sub].pop(0))
            i += 1
            if i > per_day:
                break
        out += picked
    return [{k: r[k] for k in ("day", "sub", "themes", "text")}
            for r in out[:n]]


def _posts_by_theme(themes: list[str],
                    per_theme: int = POSTS_PER_THEME) -> dict:
    """The most-engaged recent posts for each theme, so the per-theme
    briefs are written from that theme's OWN conversation instead of the
    model's memory of what people usually say about it."""
    rows = _harvest()
    out: dict[str, list[str]] = {}
    for th in themes:
        pool = [r for r in rows if th in r["themes"]]
        pool.sort(key=lambda r: (r["day"], r["score"]), reverse=True)
        if pool:
            out[th] = [r["text"] for r in pool[:per_theme]]
    return out


_PULSE_SYSTEM = (
    "You write the daily qualitative read of retail-investor chatter for "
    "a professional trading desk. Rules, strictly: cite only numbers "
    "present in the EVIDENCE block; when you characterise the crowd's "
    "words, PARAPHRASE - never quote verbatim, never name users; be "
    "concrete and falsifiable, never vague; mark genuine uncertainty "
    "plainly; no investment advice language, this is a description of "
    "the crowd, not a recommendation.\n"
    "THE NO-FILLER RULE, which overrides every length target below: a "
    "desk reads this page for the most interesting, most discussed and "
    "most recent things happening in the crowd. NEVER write that "
    "something is quiet, minimal, unremarkable, 'not much discussed' or "
    "'nothing notable' - if an item would say that, DELETE THE ITEM and "
    "return a shorter list. Returning three excellent entries beats six "
    "padded ones, and an empty list is a perfectly good answer. The one "
    "exception: a silence that is genuinely surprising given the "
    "numbers is itself interesting, and you should say WHY it is "
    "surprising rather than merely noting it.\n"
    "Answer ONLY with the requested JSON object, no prose around it.")


def _market_prompt(ev: dict, posts: list[dict]) -> str:
    """Call 1 - THE WHOLE MARKET.  Desk instruction 2026-08-04: section 1
    is 'general sentiment / feelings / vibes' as bullets plus one line
    that represents how the whole market feels; section 2 is the deep
    read of 'what all the forums are saying as a whole'.  Neither is a
    trending-topics list - the loudest theme is an input here, not the
    subject."""
    seg = {
        # LENGTHENED 2026-08-05, desk: "i just want like that but more
        # words". Section 1 stays BULLETS - the format was right - but
        # 10-20 words could only carry a mood adjective. 30-45 gives room
        # for the observation AND what it implies, which is the part a
        # desk can act on.
        "market_vibe": "object {bullets: list of 5-8 lines (30-45 "
                       "words each) describing how the market FEELS "
                       "right now as a whole - mood, confidence, "
                       "frustration, greed, boredom, fatigue, who is "
                       "winning arguments, what the emotional register "
                       "actually is. This is sentiment across ALL the "
                       "chatter, NOT a list of trending tickers or "
                       "themes; a bullet naming one stock is only "
                       "allowed if that stock IS the market's mood. "
                       "one_liner: a single sentence, 6-20 words, "
                       "written in the crowd's own voice and register, "
                       "that captures how the whole market feels this "
                       "week - the kind of line that would get 2k "
                       "upvotes ('I lost money and I don't want to "
                       "trade anymore' is the register). It MUST be a "
                       "PARAPHRASE you compose, never a real post "
                       "copied. one_liner_why: 15-30 words on what in "
                       "the chatter that line is distilling}",
        # (mood_gauge {score, why} was removed from this spec on desk
        # instruction 2026-08-07 - "remove the bullish score number and
        # all the associated code". The vibe bullets + one_liner carry
        # the mood; no numeric mood score is requested or displayed.)
        # THE REGISTER, desk 2026-08-05, quoting the output they want:
        # "users on WSB are really talking a lot about this stock xx
        # because of this but many are worried about y. lots of them
        # talking about situational awareness. sentiment is a little
        # worried due to losses or sentiment super bullish as everyone
        # is posting that they are making money."
        #
        # Three things in that sentence, and the spec below is built to
        # force all three: a NAMED forum and a NAMED instrument with the
        # REASON attached; the counter-worry alongside it; and - the one
        # that matters most - sentiment expressed as the BEHAVIOUR that
        # reveals it ("everyone is posting their gains") rather than as
        # an adjective ("sentiment is bullish"). An adjective is the
        # model's conclusion; the behaviour is the evidence, and a desk
        # can judge evidence.
        "market_pulse": "4-5 substantial paragraphs (450-550 words). "
                        "REGISTER, and follow it closely: write the way "
                        "a colleague who reads these boards all day "
                        "would brief you. Name the forum, name the "
                        "instrument, and give the REASON in the same "
                        "sentence - 'r/wallstreetbets is all over NVDA "
                        "because of the HBM supply headlines, though a "
                        "lot of them are worried about the valuation'. "
                        "Say what people are DOING that shows the mood, "
                        "never just label the mood: 'half the front "
                        "page is gain screenshots' and 'the loss posts "
                        "are back' are worth more than 'sentiment is "
                        "positive'. Use emerging_terms_7d to say what "
                        "phrase the crowd has suddenly picked up this "
                        "week and what they mean by it. "
                        "This is what ALL the forums are saying, taken "
                        "as a whole. Paragraph 1 - the state of the "
                        "market "
                        "conversation overall and how it changed this "
                        "week. Paragraph 2 - WHAT EACH BOARD IS ACTUALLY "
                        "SAYING: for the two or three most active forums "
                        "in forums_7d, give the SPECIFIC claims, trades "
                        "and arguments appearing there this week, "
                        "paraphrased from the posts. NEVER describe what "
                        "a forum IS or what it is generally about - the "
                        "desk knows that r/investing skews long-term and "
                        "r/wallstreetbets skews speculative, and a "
                        "sentence spent on it is a sentence wasted. Write "
                        "'r/X is arguing that <claim>' and 'the case "
                        "being made on r/Y is <argument>', never "
                        "'r/X is a long-term-oriented community'. If two "
                        "boards disagree about the same name, say what "
                        "each one thinks and which is louder. "
                        "Paragraph 3 - "
                        "the rotation: where attention came FROM and "
                        "went TO, citing share-vs-4-week numbers. "
                        "Paragraph 4 - positioning and conviction: what "
                        "the crowd is DOING vs merely discussing, and "
                        "where bulls and bears actually argue. "
                        "Paragraph 5 - what is surprisingly ABSENT "
                        "given the numbers, and why that matters",
        "talk_of_the_town": "2 paragraphs (150-220 words): the specific "
                            "topics, threads, arguments and running "
                            "jokes the crowd keeps returning to - "
                            "concrete, never generic; name the recurring "
                            "arguments and who is winning them",
    }
    return (f"EVIDENCE (the only numbers you may cite):\n"
            f"{json.dumps(ev, indent=1)}\n\n"
            f"POSTS (a sample spread across the last days and across "
            f"forums, ranked by engagement inside each; `sub` is the "
            f"forum):\n{json.dumps(posts, indent=0)}\n\n"
            f"Write the whole-market read, at full depth. Return ONE "
            f"JSON object with exactly these keys:\n"
            f"{json.dumps(seg, indent=1)}")


def _themes_prompt(ev: dict, by_theme: dict) -> str:
    """Call 2 - one brief per theme the desk can select in the dropdown,
    each written from THAT theme's own posts."""
    # LENGTH RAISED 80-120 -> 220-300 WORDS, desk 2026-08-05: "i want
    # LONGER thoughts about a theme please". The extra words are spent on
    # SUBSTANCE, not on padding, so the structure below is prescriptive:
    # four things to cover, in order. Without that a longer target just
    # produces the same brief with more adjectives.
    seg = {"theme_briefs":
           "list of objects {theme (exactly as given), brief: 220-300 "
           "words, written as 2-3 paragraphs} - ONE for each theme in "
           "THEME POSTS below, in the same order. Cover, in this order: "
           "(1) THE ARGUMENT - the specific case the crowd is making "
           "for or against this theme right now, paraphrased with "
           "enough detail that a PM could repeat it; name the "
           "instruments and the reasoning, not just the mood. "
           "(2) THE EVIDENCE THEY CITE - what facts, numbers, "
           "catalysts, earnings or events the posts point to, and "
           "whether they are being read correctly. (3) THE DISSENT - "
           "who is arguing the other side, what their strongest point "
           "is, and how seriously it is being taken; if there is no "
           "real dissent, say that the conversation is one-sided and "
           "what that implies. (4) WHAT CHANGED - the shift against "
           "the 4-week baseline in EVIDENCE, and what a desk should "
           "watch next. Write about what THESE posts say, never about "
           "the theme in general or what people usually think about "
           "it. If a theme's posts genuinely contain nothing worth a "
           "desk's attention, OMIT that theme entirely rather than "
           "writing that it is quiet."}
    return (f"EVIDENCE (the only numbers you may cite):\n"
            f"{json.dumps(ev, indent=1)}\n\n"
            f"THEME POSTS (the most-engaged recent posts for each "
            f"theme):\n{json.dumps(by_theme, indent=0)}\n\n"
            f"Return ONE JSON object with exactly this key:\n"
            f"{json.dumps(seg, indent=1)}")


def _watch_prompt(ev: dict, posts: list[dict]) -> str:
    """Call 4 - catalysts and divergences."""
    seg = {
        "catalyst_watch": "list of 3-6 objects {event, themes[], "
                          "chatter: 30-50 words - how the crowd is "
                          "positioning for it, which side is louder, "
                          "and any date they cite}. Only events the "
                          "posts actually discuss.",
        "divergences": "list of 3-5 objects {name, story: 50-70 words - "
                       "what the crowd SAYS vs what the measured "
                       "numbers in EVIDENCE show, and which one has "
                       "been right lately}. A divergence is only worth "
                       "a row if the two sides genuinely disagree.",
    }
    return (f"EVIDENCE (the only numbers you may cite):\n"
            f"{json.dumps(ev, indent=1)}\n\n"
            f"POSTS:\n{json.dumps(posts, indent=0)}\n\n"
            f"Return ONE JSON object with exactly these keys:\n"
            f"{json.dumps(seg, indent=1)}")


_AGENTIC_SYSTEM = _PULSE_SYSTEM


def _agentic_prompt(ev: dict, samples: list[dict]) -> str:
    return (
        "These are posts where retail traders describe USING AI to "
        "trade (asking models for picks, running AI agents/bots, "
        "building AI strategies, or mocking those who do). Counts for "
        f"context: {json.dumps(ev.get('agentic_28d', {}))}\n\n"
        f"POSTS:\n{json.dumps(samples, indent=0)}\n\n"
        "Return ONE JSON object: {digest: <=120 words - what retail is "
        "prompting AIs for and what the AIs appear to be telling/doing "
        "for them, as observed in these posts; asks: list of <=5 short "
        "strings - the typical prompts/questions; actions: list of <=5 "
        "short strings - what the AI reportedly said/did; risk_note: "
        "<=40 words - herding/correlation risk if many follow the same "
        "model outputs. PARAPHRASE only, no verbatim quotes, no "
        "usernames.}")


# The no-filler rule, enforced twice: the model is told (see
# _PULSE_SYSTEM) and then checked here.  Instructions get followed most of
# the time; a desk page that promises "never useless information" needs
# the other times covered too.
_FILLER_RE = re.compile(
    r"\b("
    r"(?:no|not|little|minimal|limited|insufficient|hardly any|barely any)"
    r"\s+(?:meaningful\s+|significant\s+|notable\s+|substantial\s+|real\s+)?"
    r"(?:discussion|chatter|mention|activity|conversation|data|"
    r"information|signal|evidence)"
    r"|nothing\s+(?:notable|noteworthy|significant|of\s+note|much)"
    r"|not\s+(?:much|enough)\s+(?:to\s+say|discussed|being\s+said)"
    r"|(?:remains?|stays?|is)\s+(?:very\s+)?quiet\s*[.;]?\s*$"
    r"|no\s+(?:clear\s+)?(?:trend|pattern|view)\s+(?:emerges|is\s+visible)"
    r"|(?:sparse|scant|thin)\s+(?:discussion|coverage|chatter)"
    r"|too\s+(?:few|little)\s+posts?"
    r")\b", re.I)


def _is_filler(text: str) -> bool:
    """True for a sentence whose entire content is 'there is nothing
    here'.  Short items are judged whole; long ones are spared, because a
    200-word brief that happens to note a silence is doing real work."""
    t = str(text or "").strip()
    if not t:
        return True
    if len(t.split()) > 45:
        return False
    return bool(_FILLER_RE.search(t))


def _drop_filler(doc: dict) -> dict:
    """Strip empty-calorie entries from a finished pulse."""
    for key, field in (("theme_briefs", "brief"),
                       ("catalyst_watch", "chatter"),
                       ("divergences", "story")):
        items = doc.get(key)
        if isinstance(items, list):
            doc[key] = [it for it in items
                        if isinstance(it, dict)
                        and not _is_filler(it.get(field, ""))]
    vibe = doc.get("market_vibe")
    if isinstance(vibe, dict) and isinstance(vibe.get("bullets"), list):
        vibe["bullets"] = [b for b in vibe["bullets"] if not _is_filler(b)]
    return doc


def generate(log=print, as_of=None) -> tuple[bool, str]:
    """Build the evidence, call the model, write ai_pulse.json.

    `as_of` (a date or YYYY-MM-DD string) rebuilds the pulse as if that
    were the newest day on record. A back-dated run writes to
    `ai_pulse_<date>.json` and NEVER touches the live file - reading
    history must not be able to overwrite today's page.

    Returns (ok, message) - never raises for gateway problems."""
    global AS_OF
    AS_OF = pd.Timestamp(as_of).normalize() if as_of else None
    out_path = (OUT_PATH if AS_OF is None else
                os.path.join(PROCESSED_DIR,
                             f"ai_pulse_{AS_OF:%Y-%m-%d}.json"))
    if AS_OF is not None:
        log(f"AI PULSE: BACK-DATED to {AS_OF:%Y-%m-%d} - every store and "
            "every post is clipped to that day")
    log("AI PULSE: building evidence pack")
    ev = _evidence()
    if not ev:
        return False, "no aggregates on disk yet - run update_data first"
    if not ai.available():
        return False, f"LLM unavailable: {ai.explain_unavailable()}"
    posts = _fresh_posts()
    theme_list = list((ev.get("theme_mention_share_7d") or {}).keys())
    by_theme = _posts_by_theme(theme_list)
    log(f"AI PULSE: {len(posts)} posts across "
        f"{len({p['sub'] for p in posts})} forums, "
        f"{len(by_theme)} themes; "
        f"model {ai.MODEL}, generating (4 calls)")
    try:
        log("AI PULSE: call 1 - the whole-market read (vibe, mood, "
            "forums)")
        pulse = ai.chat(_market_prompt(ev, posts), system=_PULSE_SYSTEM,
                        want_json=True, max_tokens=4000)
        # THEME BRIEFS, BATCHED. One call per THEMES_PER_CALL themes, so
        # the number of themes on the dropdown is set by how many themes
        # the crowd is actually discussing - never by how much text fits
        # in a single response.
        _keys = list(by_theme)
        _batches = [_keys[i:i + THEMES_PER_CALL]
                    for i in range(0, len(_keys), THEMES_PER_CALL)] or [[]]
        briefs = []
        for _bi, _batch in enumerate(_batches, 1):
            if not _batch:
                continue
            log(f"AI PULSE: call 2.{_bi}/{len(_batches)} - theme briefs "
                f"({len(_batch)} themes)")
            _part = ai.chat(
                _themes_prompt(ev, {k: by_theme[k] for k in _batch}),
                system=_PULSE_SYSTEM, want_json=True, max_tokens=4000)
            if isinstance(_part, dict):
                briefs += list(_part.get("theme_briefs") or [])
        themes = {"theme_briefs": briefs}
        log("AI PULSE: call 3 - catalysts and divergences")
        watch = ai.chat(_watch_prompt(ev, posts), system=_PULSE_SYSTEM,
                        want_json=True, max_tokens=2000)
        from src.agentic_watch import recent_samples
        log("AI PULSE: call 4 - the agentic digest")
        agentic = ai.chat(
            _agentic_prompt(ev, recent_samples(per_cat=8)),
            system=_AGENTIC_SYSTEM, want_json=True, max_tokens=900)
        log(f"AI PULSE: done ({2 + len(_batches)} calls)")
    except (RuntimeError, ValueError) as e:
        return False, f"generation failed: {e}"
    doc = {
        "as_of": ev.get("as_of"),
        "generated_at": datetime.now(timezone.utc)
        .strftime("%Y-%m-%d %H:%M UTC"),
        "model": ai.MODEL,
        "mock": ai.MOCK,
        "evidence": ev,
    }
    for part in (pulse, themes, watch):
        doc.update(part if isinstance(part, dict) else {})
    doc["agentic"] = agentic if isinstance(agentic, dict) else {}
    doc = _drop_filler(doc)
    if AS_OF is not None:
        doc["as_of_override"] = f"{AS_OF:%Y-%m-%d}"
    json.dump(doc, open(out_path, "w", encoding="utf-8"), indent=1)
    log(f"AI PULSE: saved -> {os.path.relpath(out_path, ROOT)} "
        f"({len(doc.get('theme_briefs') or [])} theme briefs)")
    return True, "ok"


def load() -> dict | None:
    if not os.path.exists(OUT_PATH):
        return None
    try:
        return json.load(open(OUT_PATH, encoding="utf-8"))
    except ValueError:
        return None


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    _ap.add_argument("--dry", action="store_true",
                     help="print the evidence pack and exit, no LLM call")
    _ap.add_argument("--as-of", dest="as_of", metavar="YYYY-MM-DD",
                     help="rebuild the pulse as if this were the newest "
                          "day on record. Writes ai_pulse_<date>.json and "
                          "leaves the live file alone.")
    _a = _ap.parse_args()
    if _a.dry:
        AS_OF = (pd.Timestamp(_a.as_of).normalize() if _a.as_of else None)
        print(json.dumps(_evidence(), indent=1))
    else:
        ok, msg = generate(as_of=_a.as_of)
        print(f"[{'OK' if ok else 'SKIP'}] {msg}")
        raise SystemExit(0 if ok else 1)
