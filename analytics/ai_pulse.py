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
        market_pulse, talk_of_the_town, mood_gauge{score,why},
        theme_briefs[], rally_watch[], catalyst_watch[], divergences[],
        agentic{digest, asks[], actions[], risk_note}

DESIGN RULES
  * NUMBERS COME FROM THE STORES, WORDS COME FROM THE MODEL.  The LLM
    never invents a statistic: every figure it may cite is handed to it
    in the evidence pack, and the pack itself is saved alongside the
    prose so any sentence can be audited against the inputs.
  * THREE CALLS PER RUN (desk request 2026-08-04: "longer and much
    more detailed"): (1) the market read - a proper multi-paragraph
    brief plus deep per-theme sections, (2) the watchlists - rallying,
    catalysts, divergences at forensic length, (3) the agentic digest.
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
RALLY_POSTS = 40            # mobilising posts read for the rally section
THEME_BRIEF_MIN_SHARE = 0.004   # a theme needs at least this share of the
                            # week's mentions to earn a brief. Below ~0.4%
                            # there is nothing to say that is not padding,
                            # and the desk's standing rule (2026-08-04) is
                            # that a section with nothing to say is omitted,
                            # never filled.
MAX_THEME_BRIEFS = 18       # ceiling on the dropdown, loudest first


def _read(name: str) -> pd.DataFrame | None:
    p = os.path.join(PROCESSED_DIR, name)
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
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
    # MOBILISATION: measured over every post by src/rally_watch.py, not
    # judged by the model. The model reads the matched posts and explains
    # what these numbers are pointing at.
    try:
        from src.rally_watch import (load_series as _rl, top_rallies,
                                     COUNTER_CATEGORY)
        rs = _rl()
        if rs is not None and len(rs):
            hi = rs["date"].max()
            w = rs[rs["date"] > hi - pd.Timedelta(days=7)]
            tot = w[w["category"] == "_total_posts"]["mention_count"].sum()
            sig = w[(w["kind"] == "_all")
                    & (~w["category"].str.startswith("_"))
                    & (w["category"] != COUNTER_CATEGORY)]
            ev["rally"] = {
                "pct_of_posts_mobilising": (
                    round(float(sig["mention_count"].sum())
                          / float(tot) * 100, 2) if tot else None),
                "reference_points": {"June 2021 meme summer": 3.41,
                                     "quiet 2026 week": 1.0},
                "by_category_per_1k_posts": {
                    k: round(float(v) / float(tot) * 1000, 1)
                    for k, v in sig.groupby("category")["mention_count"]
                    .sum().sort_values(ascending=False).items()} if tot
                else {},
                "pushback_hits": int(
                    w[(w["kind"] == "_all")
                      & (w["category"] == COUNTER_CATEGORY)]
                    ["mention_count"].sum()),
                "most_mobilised_themes": top_rallies("theme", n=8),
                "most_mobilised_tickers": top_rallies("ticker", n=8),
            }
    except Exception:                                    # noqa: BLE001
        pass                       # detector not scanned yet - not fatal
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


def _harvest(max_posts: int = HARVEST_MAX) -> list[dict]:
    """Every recent raw post, theme-tagged and engagement-stamped, held
    once per process.  The samplers below all slice from this, so the
    archives are decompressed a single time however many sections the
    pulse writes."""
    global _HARVEST
    if _HARVEST is not None:
        return _HARVEST
    import zstandard
    files = sorted((f for f in os.listdir(RAW_DIR)
                    if f.endswith(".jsonl.zst") and ".tmp" not in f
                    and "_salvaged" not in f),
                   key=lambda f: os.path.getmtime(
                       os.path.join(RAW_DIR, f)), reverse=True)
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
                rows.append({"day": day,
                             "sub": str(d.get("subreddit", "")),
                             "score": score,
                             "themes": sorted(themes_in_text(body))[:3],
                             "text": body[:POST_CLIP]})
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
        "market_vibe": "object {bullets: list of 5-8 SHORT lines (10-20 "
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
        "mood_gauge": "object {score: 0-100 int (0 fear, 100 greed), "
                      "why: 40-60 words - the two or three observations "
                      "that set the score, with the strongest "
                      "counter-signal acknowledged}",
        "market_pulse": "5-6 substantial paragraphs (500-650 words): "
                        "what ALL the forums are saying, taken as a "
                        "whole. Paragraph 1 - the state of the market "
                        "conversation overall and how it changed this "
                        "week. Paragraph 2 - THE FORUMS THEMSELVES: use "
                        "forums_7d and the `sub` field on the posts to "
                        "contrast what the different boards are doing - "
                        "the speculative boards vs the index/dividend/"
                        "personal-finance boards vs the research-minded "
                        "ones. Who is greedy, who is scared, who is "
                        "bored, where the newcomers are. Paragraph 3 - "
                        "the rotation: where attention came FROM and "
                        "went TO, citing share-vs-4-week numbers. "
                        "Paragraph 4 - positioning and conviction: what "
                        "the crowd is DOING vs merely discussing, and "
                        "where bulls and bears actually argue. "
                        "Paragraph 5 - the mobilisation reading: what "
                        "the `rally` numbers say about whether this "
                        "crowd is being organised or merely talking, "
                        "against the reference points given. Paragraph "
                        "6 - what is surprisingly ABSENT given the "
                        "numbers, and why that matters",
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
    seg = {"theme_briefs":
           "list of objects {theme (exactly as given), brief: 80-120 "
           "words} - ONE for each theme in THEME POSTS below, in the "
           "same order. Each brief: the tone and emotional register of "
           "that theme's own conversation, the dominant framing, the "
           "actual ARGUMENTS being made (paraphrased), where the "
           "dissent is and how serious it sounds, and any change "
           "against the 4-week baseline in EVIDENCE. Write about what "
           "these specific posts say, not about the theme in general. "
           "If a theme's posts genuinely contain nothing worth a "
           "desk's attention, OMIT that theme entirely rather than "
           "writing that it is quiet."}
    return (f"EVIDENCE (the only numbers you may cite):\n"
            f"{json.dumps(ev, indent=1)}\n\n"
            f"THEME POSTS (the most-engaged recent posts for each "
            f"theme):\n{json.dumps(by_theme, indent=0)}\n\n"
            f"Return ONE JSON object with exactly this key:\n"
            f"{json.dumps(seg, indent=1)}")


def _rally_prompt(ev: dict, hits: list[dict]) -> str:
    """Call 3 - the rally watch, ORGANISED BY THEME and anchored to the
    detector.  The model no longer decides WHETHER something is being
    rallied - src/rally_watch.py measured that over every post - it
    explains WHAT the mobilising posts are actually doing."""
    seg = {"rally_watch":
           "list of objects {theme: the theme or name this entry is "
           "about (use the names given in rally.most_mobilised_themes / "
           "most_mobilised_tickers), verdict: one of 'organised push' | "
           "'building' | 'ambient hype' | 'pushback winning', why: "
           "90-140 words - forensic: WHAT KIND of mobilising language "
           "these posts contain (recruiting, squeeze mechanics, "
           "refusal-to-sell pledges, rescue-the-company framing, "
           "coordinated timing, extreme-outcome claims), whether it "
           "reads organic or organised, who it is aimed at, how "
           "objections get handled, and whether the pushback counts "
           "(`counter`) say the crowd is policing itself, example: ONE "
           "paraphrase prefixed 'paraphrase - ' that captures the "
           "register}. Order by how interesting the entry is to a desk. "
           "Only include a theme or name where the posts actually show "
           "you something; omit the rest. If nothing in the sample is "
           "genuinely mobilised, return an empty list - the numbers "
           "already say so and the page will show them."}
    return (
        "The desk runs a lexical detector over EVERY post - recruiting, "
        "squeeze mechanics, hold-the-line pledges, save-the-company "
        "framing, coordinated timing, extreme-outcome claims - and the "
        "counts are in EVIDENCE under `rally` (`share` = the fraction "
        "of that name's own chatter that is mobilising, `z` = how "
        "unusual that is against its own history, `counter` = posts "
        "calling it a pump). YOUR JOB IS NOT TO RE-JUDGE WHETHER "
        "MOBILISATION EXISTS - it is to read the matched posts and "
        "explain what is actually going on.\n\n"
        f"EVIDENCE:\n{json.dumps(ev, indent=1)}\n\n"
        f"THE MATCHED POSTS (what the detector caught, most-engaged "
        f"first; `category` is which pattern matched):\n"
        f"{json.dumps(hits, indent=0)}\n\n"
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
    for key, field in (("theme_briefs", "brief"), ("rally_watch", "why"),
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


def generate(log=print) -> tuple[bool, str]:
    """Build the evidence, call the model, write ai_pulse.json.
    Returns (ok, message) - never raises for gateway problems."""
    log("AI PULSE: building evidence pack")
    ev = _evidence()
    if not ev:
        return False, "no aggregates on disk yet - run update_data first"
    if not ai.available():
        return False, f"LLM unavailable: {ai.explain_unavailable()}"
    posts = _fresh_posts()
    theme_list = list((ev.get("theme_mention_share_7d") or {}).keys())
    by_theme = _posts_by_theme(theme_list)
    try:
        from src.rally_watch import recent_samples as _rally_samples
        hits = [{"category": h.get("category"),
                 "themes": h.get("themes"), "tickers": h.get("tickers"),
                 "text": str(h.get("text", ""))[:POST_CLIP]}
                for h in _rally_samples(per_cat=RALLY_POSTS // 6)]
    except Exception:                                    # noqa: BLE001
        hits = []
    log(f"AI PULSE: {len(posts)} posts across "
        f"{len({p['sub'] for p in posts})} forums, "
        f"{len(by_theme)} themes, {len(hits)} mobilising posts; "
        f"model {ai.MODEL}, generating (5 calls)")
    try:
        log("AI PULSE: call 1/5 - the whole-market read (vibe, mood, "
            "forums)")
        pulse = ai.chat(_market_prompt(ev, posts), system=_PULSE_SYSTEM,
                        want_json=True, max_tokens=4000)
        log("AI PULSE: call 2/5 - theme briefs "
            f"({len(by_theme)} themes)")
        themes = ai.chat(_themes_prompt(ev, by_theme),
                         system=_PULSE_SYSTEM,
                         want_json=True, max_tokens=4000)
        log("AI PULSE: call 3/5 - the rally watch")
        rally = ai.chat(_rally_prompt(ev, hits), system=_PULSE_SYSTEM,
                        want_json=True, max_tokens=2600)
        log("AI PULSE: call 4/5 - catalysts and divergences")
        watch = ai.chat(_watch_prompt(ev, posts), system=_PULSE_SYSTEM,
                        want_json=True, max_tokens=2000)
        from src.agentic_watch import recent_samples
        log("AI PULSE: call 5/5 - the agentic digest")
        agentic = ai.chat(
            _agentic_prompt(ev, recent_samples(per_cat=8)),
            system=_AGENTIC_SYSTEM, want_json=True, max_tokens=900)
        log("AI PULSE: 5/5 calls done")
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
    for part in (pulse, themes, rally, watch):
        doc.update(part if isinstance(part, dict) else {})
    doc["agentic"] = agentic if isinstance(agentic, dict) else {}
    doc = _drop_filler(doc)
    json.dump(doc, open(OUT_PATH, "w", encoding="utf-8"), indent=1)
    log(f"AI PULSE: saved -> {os.path.relpath(OUT_PATH, ROOT)} "
        f"({len(doc.get('theme_briefs') or [])} theme briefs, "
        f"{len(doc.get('rally_watch') or [])} rally entries)")
    return True, "ok"


def load() -> dict | None:
    if not os.path.exists(OUT_PATH):
        return None
    try:
        return json.load(open(OUT_PATH, encoding="utf-8"))
    except ValueError:
        return None


if __name__ == "__main__":
    import sys
    if "--dry" in sys.argv:
        print(json.dumps(_evidence(), indent=1))
    else:
        ok, msg = generate()
        print(f"[{'OK' if ok else 'SKIP'}] {msg}")
        raise SystemExit(0 if ok else 1)
