# AI Pulse — how it works and what it is actually asked

Extracted verbatim from `analytics/ai_pulse.py` on 2026-08-12. READING COPY —
edit the prompts in `analytics/ai_pulse.py`, not here.

Calls 1, 2 and 4 see POSTS and nothing else. Call 3 is the single exception:
its `divergences` section compares the crowd's story against the measured
record, so it receives the evidence pack under its own system prompt.
`catalyst_watch` shares that call but is told to ignore the numbers.

## SYSTEM - calls 1, 2 and 4 (posts only)

```python
# POSTS ONLY, desk 2026-08-12: "can we make it so that it JUST reads
# from posts ... i want it to be a qualitative read of the posts and what
# people are saying and feeling".
#
# The evidence pack is STILL BUILT and still saved into ai_pulse.json -
# the dashboard orders the theme dropdown from it, and it remains the
# week's measured record - but it is NO LONGER SENT TO THE MODEL. The
# four prompts below carry posts and nothing else.
#
# What that trades away, stated plainly so it is a choice and not a
# regression: the model can no longer cite a share-vs-4-week ratio, name
# the emerging terms we measured, or contrast the story against our own
# numbers. In exchange every sentence it writes is now grounded in
# something a person actually posted, which is what the page is for. The
# old "story vs numbers" section had to be redefined rather than deleted
# - see _watch_prompt.
_PULSE_SYSTEM = (
    "You write the daily qualitative read of retail-investor chatter for "
    "a professional trading desk. You are reading POSTS - the actual "
    "words people wrote this week - and nothing else. Your job is to "
    "tell the desk what the crowd is saying, arguing about and FEELING.\n"
    "YOU HAVE NO STATISTICS. You are given no aggregates, no shares, no "
    "counts and no model output. Never invent a number, a percentage or "
    "a ranking. The only figures you may write are ones a post itself "
    "states, and those must be attributed as a claim the crowd is making "
    "('several posts put the number at ...'), never asserted as fact. "
    "Where you would reach for a statistic, describe the WEIGHT of the "
    "conversation instead - what dominates, what is a minority view, "
    "what only a handful of people are saying.\n"
    "PARAPHRASE - never quote verbatim, never name users; be concrete "
    "and falsifiable, never vague; mark genuine uncertainty plainly; no "
    "investment advice language, this is a description of the crowd, not "
    "a recommendation.\n"
    "THE NO-FILLER RULE, which overrides every length target below: a "
    "desk reads this page for the most interesting, most discussed and "
    "most recent things happening in the crowd. NEVER write that "
    "something is quiet, minimal, unremarkable, 'not much discussed' or "
    "'nothing notable' - if an item would say that, DELETE THE ITEM and "
    "return a shorter list. Returning three excellent entries beats six "
    "padded ones, and an empty list is a perfectly good answer. The one "
    "exception: a silence that is genuinely surprising is itself "
    "interesting, and you should say WHY it is surprising rather than "
    "merely noting it.\n"
    "Answer ONLY with the requested JSON object, no prose around it.")
```

## SYSTEM - call 3 (the one call with numbers)

```python
# THE ONE EXCEPTION, desk 2026-08-12: "i like the divergences part -
# that can be the only one which is using the numbers". Section 5 asks
# where the crowd's story and the measured record disagree, and that
# question has no second side without the numbers. So call 3 - and only
# call 3 - gets the evidence pack, under its own system prompt: the
# blanket "YOU HAVE NO STATISTICS" rule above would flatly contradict
# handing it a block of them. Calls 1, 2 and 4 stay posts-only.
_WATCH_SYSTEM = _PULSE_SYSTEM.replace(
    "YOU HAVE NO STATISTICS. You are given no aggregates, no shares, no "
    "counts and no model output. Never invent a number, a percentage or "
    "a ranking. The only figures you may write are ones a post itself "
    "states, and those must be attributed as a claim the crowd is making "
    "('several posts put the number at ...'), never asserted as fact. "
    "Where you would reach for a statistic, describe the WEIGHT of the "
    "conversation instead - what dominates, what is a minority view, "
    "what only a handful of people are saying.",
    "YOU HAVE EXACTLY ONE SOURCE OF NUMBERS: the EVIDENCE block supplied "
    "with this call. Cite figures ONLY from it - never invent one, never "
    "recall one from training, and never take one from a post as fact "
    "(a number inside a post is a CLAIM the crowd is making, and is "
    "interesting precisely when the evidence disagrees with it).")
assert _WATCH_SYSTEM != _PULSE_SYSTEM, (
    "the no-statistics paragraph moved - _WATCH_SYSTEM silently stopped "
    "granting the divergences call its numbers")
```

## CALL 1 - whole-market read (posts only)

```python
def _market_prompt(posts: list[dict]) -> str:
    """Call 1 - THE WHOLE MARKET, written from posts alone.  Section 1
    is 'general sentiment / feelings / vibes' as bullets plus one line
    that represents how the whole market feels; section 2 is the deep
    read of 'what all the forums are saying as a whole'.  Neither is a
    trending-topics list."""
    seg = {
        # Section 1 stays BULLETS - the format was right - at 30-45
        # words, which is room for the observation AND what it implies.
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
        # THE REGISTER, desk 2026-08-05, quoting the output they want:
        # "users on WSB are really talking a lot about this stock xx
        # because of this but many are worried about y ... sentiment
        # super bullish as everyone is posting that they are making
        # money."
        #
        # Three things in that sentence, and the spec below forces all
        # three: a NAMED forum and a NAMED instrument with the REASON
        # attached; the counter-worry alongside it; and - the one that
        # matters most - sentiment expressed as the BEHAVIOUR that
        # reveals it ("everyone is posting their gains") rather than as
        # an adjective ("sentiment is bullish"). An adjective is the
        # model's conclusion; the behaviour is the evidence, and a desk
        # can judge evidence. With the numbers gone this is now the
        # ONLY grounding the section has, so it matters more, not less.
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
                        "positive'. If a word or phrase keeps recurring "
                        "across the posts, say what it is and what the "
                        "crowd means by it. "
                        "This is what ALL the forums are saying, taken "
                        "as a whole. Paragraph 1 - the state of the "
                        "market conversation overall and what it is "
                        "preoccupied with. Paragraph 2 - "
                        "WHAT EACH BOARD IS ACTUALLY SAYING: for the "
                        "two or three "
                        "forums carrying the most conversation in the "
                        "posts below, give the SPECIFIC claims, trades "
                        "and arguments appearing there, paraphrased "
                        "from the posts. NEVER describe what a forum IS "
                        "or what it is generally about - the desk knows "
                        "that r/investing skews long-term and "
                        "r/wallstreetbets skews speculative, and a "
                        "sentence spent on it is a sentence wasted. "
                        "Write 'r/X is arguing that <claim>' and 'the "
                        "case being made on r/Y is <argument>', never "
                        "'r/X is a long-term-oriented community'. If "
                        "two boards disagree about the same name, say "
                        "what each one thinks and which is louder. "
                        "Paragraph 3 - the rotation, as the POSTS show "
                        "it: what the crowd has moved on TO and what it "
                        "has stopped talking about, including names "
                        "people mention only to say they have given up "
                        "on them. Paragraph 4 - positioning and "
                        "conviction: what the crowd is DOING vs merely "
                        "discussing - who reports being in a position, "
                        "who is watching, who is asking permission - "
                        "and where bulls and bears actually argue. "
                        "Paragraph 5 - what you would EXPECT this crowd "
                        "to be talking about and it simply is not, and "
                        "why that absence is interesting",
        "talk_of_the_town": "2 paragraphs (150-220 words): the specific "
                            "topics, threads, arguments and running "
                            "jokes the crowd keeps returning to - "
                            "concrete, never generic; name the recurring "
                            "arguments and who is winning them",
    }
    return (f"POSTS - this is your ONLY source. A sample spread across "
            f"the last days and across forums, ranked by engagement "
            f"inside each; `sub` is the forum, `day` the date, `themes` "
            f"the tags our pipeline put on it:\n"
            f"{json.dumps(posts, indent=0)}\n\n"
            f"Read them and write the whole-market read, at full depth. "
            f"Return ONE JSON object with exactly these keys:\n"
            f"{json.dumps(seg, indent=1)}")
```

## CALL 2 - theme briefs, batched 12 at a time (posts only)

```python
def _themes_prompt(by_theme: dict) -> str:
    """Call 2 - one brief per theme the desk can select in the dropdown,
    each written from THAT theme's own posts and nothing else."""
    # 220-300 WORDS, desk 2026-08-05: "i want LONGER thoughts about a
    # theme please". The extra words are spent on SUBSTANCE, so the
    # structure is prescriptive: four things to cover, in order.
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
           "whether they are being read correctly. Attribute these as "
           "the crowd's claims, not as established fact. "
           "(3) THE DISSENT - "
           "who is arguing the other side, what their strongest point "
           "is, and how seriously it is being taken; if there is no "
           "real dissent, say that the conversation is one-sided and "
           "what that implies. (4) WHAT CHANGED - what this "
           "conversation has become that it was not before, as visible "
           "in the posts themselves: a new worry, a new justification, "
           "a shift from arguing to celebrating, or the point where "
           "people stopped defending it. Say what a desk should watch "
           "next. Write about what THESE posts say, never about "
           "the theme in general or what people usually think about "
           "it. If a theme's posts genuinely contain nothing worth a "
           "desk's attention, OMIT that theme entirely rather than "
           "writing that it is quiet."}
    return (f"THEME POSTS - your ONLY source. The most-engaged recent "
            f"posts for each theme:\n{json.dumps(by_theme, indent=0)}\n\n"
            f"Return ONE JSON object with exactly this key:\n"
            f"{json.dumps(seg, indent=1)}")
```

## CALL 3 - catalysts (posts) + divergences (posts vs numbers)

```python
def _watch_prompt(ev: dict, posts: list[dict]) -> str:
    """Call 3 - catalysts (from the posts) and divergences (the ONE
    section that may use the measured numbers).

    Desk 2026-08-12, after the posts-only change: "i like the divergences
    part - that can be the only one which is using the numbers". Section
    5 exists to ask whether the story the crowd tells matches what we
    actually measure, and stripping the numbers left it with nothing to
    compare against. It gets them back here - under _WATCH_SYSTEM, not
    the posts-only system prompt - and the crowd-contradicts-itself lens
    written on 2026-08-12 is KEPT as a second admissible kind, because a
    thesis nobody will put a position behind is a real divergence even
    when every aggregate agrees with it.

    catalyst_watch stays posts-only inside this same call: it is a list
    of what people are positioning for, and no aggregate speaks to it."""
    seg = {
        "catalyst_watch": "list of 3-6 objects {event, themes[], "
                          "chatter: 30-50 words - how the crowd is "
                          "positioning for it, which side is louder, "
                          "and any date they cite}. Only events the "
                          "POSTS actually discuss - do not use the "
                          "EVIDENCE block for this section.",
        "divergences": "list of 3-5 objects {name, story: 50-70 words}. "
                       "This is the ONE section that uses the EVIDENCE "
                       "block. A divergence is a place where the story "
                       "does not match the record. Two kinds count, and "
                       "the first is the one to look for hardest: "
                       "(A) STORY vs NUMBERS - what the posts SAY "
                       "against what the measured numbers in EVIDENCE "
                       "show, and which of the two has been right "
                       "lately. Cite the specific figure you are "
                       "contrasting against: a theme the crowd has gone "
                       "quiet on whose mentions are still rising, a name "
                       "everyone is euphoric about that carries a GET "
                       "OUT flag, a theme running far above its 4-week "
                       "pace that nobody in the posts seems to have "
                       "noticed, or a phrase in emerging_terms_7d that "
                       "the posts have not caught up with. "
                       "(B) THE CROWD AGAINST ITSELF - a contradiction "
                       "inside the posts alone: confident views from "
                       "people who admit they hold nothing or have "
                       "already sold, a name everyone claims to be "
                       "bullish on that nobody reports owning, two "
                       "forums asserting opposite things about the same "
                       "instrument, or a thesis repeated long after the "
                       "reason given for it stopped being mentioned. "
                       "State both sides, and say which one you think "
                       "is the honest signal. Only include a row where "
                       "the two sides genuinely conflict.",
    }
    return (f"EVIDENCE - the measured record for this week, and the only "
            f"numbers you may cite. It is for the divergences section:\n"
            f"{json.dumps(ev, indent=1)}\n\n"
            f"POSTS - what people actually wrote:\n"
            f"{json.dumps(posts, indent=0)}\n\n"
            f"Return ONE JSON object with exactly these keys:\n"
            f"{json.dumps(seg, indent=1)}")
```

## CALL 4 - agentic digest (posts only)

```python
def _agentic_prompt(samples: list[dict]) -> str:
    """Call 4 - the agentic digest. Posts only: the 28-day category
    counts that used to open this prompt were the last aggregate
    reaching the model and went with the rest."""
    return (
        "These are posts where retail traders describe USING AI to "
        "trade (asking models for picks, running AI agents/bots, "
        "building AI strategies, or mocking those who do). They are "
        "your ONLY source - no counts are provided and none may be "
        f"invented.\n\nPOSTS:\n{json.dumps(samples, indent=0)}\n\n"
        "Return ONE JSON object: {digest: <=120 words - what retail is "
        "prompting AIs for and what the AIs appear to be telling/doing "
        "for them, as observed in these posts; asks: list of <=5 short "
        "strings - the typical prompts/questions; actions: list of <=5 "
        "short strings - what the AI reportedly said/did; risk_note: "
        "<=40 words - herding/correlation risk if many follow the same "
        "model outputs. PARAPHRASE only, no verbatim quotes, no "
        "usernames.}")
```

## WHICH THEMES GET A BRIEF (chosen from the posts)

```python
def _themes_from_posts(min_posts: int = 6,
                       cap: int = MAX_THEME_BRIEFS) -> list[str]:
    """Which themes earn a brief - decided by the POSTS themselves.

    This used to be driven by `theme_mention_share_7d` from the evidence
    pack (every theme above a share floor). With the pack no longer
    reaching the model, choosing the list from an aggregate the model
    cannot see would be incoherent: a theme could be selected for a
    brief and then have too few posts to write one from. Counting the
    harvested posts directly makes the two decisions the same decision -
    a theme is on the dropdown exactly when there is enough of its own
    conversation to read.
    """
    rows = _harvest()
    tally: dict[str, int] = {}
    for r in rows:
        for th in r.get("themes") or []:
            if th in THEME_ETFS:          # tradeable themes only
                tally[th] = tally.get(th, 0) + 1
    ranked = sorted((t for t, n in tally.items() if n >= min_posts),
                    key=lambda t: -tally[t])
    return ranked[:cap]
```
