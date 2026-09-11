"""Bot screen: each signal fires on its target and not on ordinary posts.

The corpus below is synthetic but shaped like the real store: varied
human posts on many names, one copy-paste promoter using several
accounts, one burst account, one disclosed bot, one templated account,
and short exclamations that must never count as duplicates.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion import bot_screen as bs                       # noqa: E402

pytest.importorskip("datasketch")

HUMAN = [
    "Thinking about adding to my NVDA position after earnings, the data center numbers looked strong but the guide was soft.",
    "Anyone else holding GLD through this? Real yields are falling and I think the trade has legs into year end.",
    "Sold half my SMH today. The semis run has been incredible but the crowd is getting loud and I want to lock some in.",
    "Long-term holder of AMD here. Not selling on this dip, the MI300 ramp is the story and it hasn't played out yet.",
    "Why is everyone on this sub so bearish on XBI? Biotech has been dead money for two years, feels like capitulation.",
    "Bought some URA on the pullback. Uranium supply story is real and the utilities are finally signing contracts.",
    "Is IGV worth it versus just owning the big names directly? Software ETF feels diluted with mediocre companies.",
    "My take on ARKK: it is a leveraged bet on rates. If cuts come it rips, if not it bleeds. Sized accordingly.",
    "Been DCAing into VOO for six years. Boring but it works. Not touching the meme stuff anymore after 2021.",
    "Earnings season thoughts: MU should benefit from HBM demand but the stock already ran 60% off the lows.",
    "Genuine question, what is the bull case for TSLA at these multiples? Robotaxi feels priced in twice over.",
    "Took profits on COIN. Crypto beta is fun until it isn't and the last two weeks have been very not fun.",
]
SHORT = ["to the moon", "lol", "this", "buy the dip", "same", "yolo calls", "🚀🚀🚀"]
PROMO = ("Check out $ABCD, this company is about to explode. Huge contract "
         "coming next week, insiders are loading up, do your own research "
         "but this is the play of the year. Get in before the news!")


def _corpus(seed=0):
    rng = random.Random(seed)
    rows = []
    day = pd.Timestamp("2026-08-10")
    # humans: 60 varied posts across 30 accounts over 6 days
    for i in range(60):
        rows.append({"id": f"h{i}", "author": f"user_{i % 30}",
                     "date": day + pd.Timedelta(days=i % 6),
                     "title": rng.choice(HUMAN), "selftext": "",
                     "subreddit": "stocks", "source": "reddit"})
    # short exclamations from many people (must not be duplicates)
    for i in range(20):
        rows.append({"id": f"s{i}", "author": f"user_{i}",
                     "date": day, "title": rng.choice(SHORT),
                     "selftext": "", "subreddit": "wallstreetbets",
                     "source": "reddit"})
    # copy-paste promotion from four sock accounts, tiny variations
    for i in range(8):
        rows.append({"id": f"p{i}", "author": f"sock_{i % 4}",
                     "date": day + pd.Timedelta(days=i % 3),
                     "title": PROMO + (" Seriously." if i % 2 else ""),
                     "selftext": "", "subreddit": "pennystocks",
                     "source": "reddit"})
    # burst poster: 15 distinct posts in one day
    for i in range(15):
        rows.append({"id": f"b{i}", "author": "burst_guy",
                     "date": day, "title": f"Daily thread update number {i} "
                     f"with unique content about ticker number {i * 7} today",
                     "selftext": "", "subreddit": "stocks",
                     "source": "reddit"})
    # disclosed bot
    rows.append({"id": "bot1", "author": "AutoModerator", "date": day,
                 "title": "Welcome to the daily discussion thread.",
                 "selftext": "I am a bot, and this action was performed automatically.",
                 "subreddit": "stocks", "source": "reddit"})
    # templated account: 6 posts, same template, one number changes
    for i in range(6):
        rows.append({"id": f"t{i}", "author": "template_acct",
                     "date": day + pd.Timedelta(days=i),
                     "title": f"Signal alert: momentum score {40 + i} on the "
                     "watchlist name, breakout confirmed, volume rising, "
                     "target raised, stop moved up",
                     "selftext": "", "subreddit": "options",
                     "source": "reddit"})
    return pd.DataFrame(rows)


class TestSignals:
    def test_scores_are_bounded_and_columns_added(self):
        out = bs.screen_posts(_corpus(), jaccard=0.85, burst_per_day=12)
        assert {"bot_score", "bot_reasons"} <= set(out.columns)
        assert out["bot_score"].between(0, 1).all()

    def test_copy_paste_promotion_is_a_near_duplicate(self):
        out = bs.screen_posts(_corpus(), jaccard=0.85, burst_per_day=12)
        promo = out[out["id"].str.startswith("p")]
        assert (promo["bot_reasons"].str.contains("near_duplicate")).all()

    def test_burst_account_is_flagged(self):
        out = bs.screen_posts(_corpus(), jaccard=0.85, burst_per_day=12)
        burst = out[out["author"] == "burst_guy"]
        assert (burst["bot_reasons"].str.contains("burst")).all()

    def test_disclosed_bot_scores_one(self):
        out = bs.screen_posts(_corpus(), jaccard=0.85, burst_per_day=12)
        row = out[out["id"] == "bot1"].iloc[0]
        assert row["bot_score"] == 1.0 and "disclosed" in row["bot_reasons"]

    def test_templated_account_is_low_diversity(self):
        out = bs.screen_posts(_corpus(), jaccard=0.85, burst_per_day=12)
        tpl = out[out["author"] == "template_acct"]
        assert (tpl["bot_reasons"].str.contains("low_diversity")).all()

    def test_short_exclamations_are_never_duplicates(self):
        out = bs.screen_posts(_corpus(), jaccard=0.85, burst_per_day=12)
        short = out[out["id"].str.startswith("s")]
        assert not short["bot_reasons"].str.contains("near_duplicate").any()

    def test_ordinary_humans_are_not_excluded(self):
        """Humans repeat popular phrasings and share names; none may cross
        the default threshold."""
        out = bs.screen_posts(_corpus(), jaccard=0.85, burst_per_day=12)
        humans = out[out["id"].str.startswith("h")]
        assert (humans["bot_score"] < 0.6).all(), \
            humans[humans["bot_score"] >= 0.6][["author", "title", "bot_reasons"]]

    def test_one_person_cross_posting_once_is_not_flagged(self):
        df = pd.DataFrame([
            {"id": "a", "author": "u1", "date": "2026-08-10",
             "title": PROMO, "selftext": "", "subreddit": "stocks"},
            {"id": "b", "author": "u1", "date": "2026-08-10",
             "title": PROMO, "selftext": "", "subreddit": "investing"},
        ])
        out = bs.screen_posts(df, jaccard=0.85, burst_per_day=12)
        assert not out["bot_reasons"].str.contains("near_duplicate").any()

    def test_missing_optional_columns_disable_signals_gracefully(self):
        df = pd.DataFrame({"title": ["hello world " * 5, "other text here " * 5]})
        out = bs.screen_posts(df, jaccard=0.85, burst_per_day=12)
        assert (out["bot_score"] == 0.0).all()

    def test_empty_frame(self):
        out = bs.screen_posts(pd.DataFrame(columns=["title", "author"]),
                              jaccard=0.85, burst_per_day=12)
        assert out.empty and "bot_score" in out.columns


class TestApply:
    def test_apply_drops_flagged_and_reports(self):
        posts = _corpus()
        kept, rep = bs.apply_screen(posts, threshold=0.6, enabled=True)
        assert rep["rows_in"] == len(posts)
        assert rep["rows_excluded"] == len(posts) - len(kept)
        assert 0.0 < rep["exclusion_rate"] < 0.5
        assert list(kept.columns) == list(posts.columns)
        assert "bot_score" not in kept.columns
        assert rep["top_reasons"]

    def test_disabled_screen_changes_nothing(self):
        posts = _corpus()
        kept, rep = bs.apply_screen(posts, threshold=0.6, enabled=False)
        assert len(kept) == len(posts) and rep["rows_excluded"] == 0

    def test_report_round_trips_to_json(self, tmp_path):
        _, rep = bs.apply_screen(_corpus(), threshold=0.6, enabled=True)
        p = tmp_path / "ref" / "bot_screen_last.json"
        bs.write_report(rep, str(p))
        import json
        back = json.loads(p.read_text())
        assert back["rows_excluded"] == rep["rows_excluded"]
        assert "checked_utc" in back
        assert "excluded" in bs.format_report(rep)

    def test_settings_drive_the_defaults(self):
        from src import settings
        assert settings.get_bool("bot_screen_enabled") in (True, False)
        assert 0 < settings.get_float("bot_screen_duplicate_jaccard") <= 1
        assert settings.get_int("bot_screen_burst_posts_per_day") > 1


class TestWiring:
    def test_report_tool_runs_on_a_store(self, tmp_path, monkeypatch, capsys):
        from tools import bot_screen_report as rep
        store = tmp_path / "posts.parquet"
        df = _corpus()
        df["date"] = pd.to_datetime(df["date"])
        df.to_parquet(store, index=False)
        monkeypatch.setattr(rep, "POSTS_PATH", str(store))
        assert rep.main(["--start", "2026-08-01", "--examples", "3"]) == 0
        out = capsys.readouterr().out
        assert "bot screen: excluded" in out and "examples" in out

    def test_both_aggregation_entry_points_call_the_screen(self):
        for name in ("ingestion/build_aggregates.py",
                     "ingestion/append_live_abstracted.py"):
            src = (ROOT / name).read_text(encoding="utf-8")
            assert "apply_screen(" in src, name
            assert "bot_screen_last.json" in src, name
