"""
themes.py
=========
Two complementary theme signals, built to be compared side by side:

SIGNAL 1 - KEYWORD themes (direct mentions)
  Scan each post's raw text for curated words/phrases. A post about
  "HBM memory pricing" and "DRAM capacity" rolls up into the `memory`
  theme even if it never names a ticker.

  Entry point: build_daily_theme_counts(posts_df)
  Returns: DataFrame(date, theme, keyword_count, keyword_weighted)

  Matching is one tokenisation pass per post + hash lookups (fast: ~12k
  posts/sec vs ~600/sec for the old per-theme regex scan). Single-word
  keywords must match a whole token; multi-word phrases are matched as
  substrings of the lowercased text. Each post counts AT MOST ONCE per
  theme (same per-post dedupe rule as the ticker pipeline).

  NOTE (2026-07-06): the score**2 upvote weighting was REMOVED everywhere -
  archived scores are FINAL scores, so weighting day-t mentions by them
  leaks future information into any backtest (see build_mentions.py and
  design_decisions.xlsx #30). The *_weighted columns are still emitted for
  notebook compatibility but are ALWAYS 0 now.

  IMPORTANT: keyword lists contain WORDS AND PHRASES ONLY - no bare ticker
  symbols. Matching is case-insensitive, so a symbol like C (Citigroup) or
  O (Realty Income) would match every ordinary "c"/"o" in prose. Ticker
  exposure to a theme is Signal 2's job.

SIGNAL 2 - INFERRED themes (ticker -> theme)
  Map the already-extracted daily ticker counts (notebook 02's output,
  which had all the stop-list/screening precision applied) onto theme
  buckets: NVDA implies the AI trade, SHEL implies energy, and so on.
  A ticker may belong to several themes.

  Entry point: build_inferred_theme_counts(daily_ticker_counts_df)
  Returns: DataFrame(date, theme, inferred_count, inferred_weighted)

  This signal costs nothing to compute - it is a groupby over a file that
  already exists.

Combine both with combine_theme_signals() -> one row per (date, theme)
with all four count columns, zeros where a signal is silent.

TRADEABLE BY DESIGN: every theme is anchored to a liquid instrument in
THEME_ETFS (semiconductors -> SMH, gold_metals -> GLD, europe_defense ->
EUAD ...). If a theme's mentions spike, there is a concrete thing to
back-test it against and, eventually, trade. Vague non-tradeable themes
(options chatter, earnings chatter, IPO chatter) were deliberately removed;
short_squeeze / meme_stocks stay because their proxy (GME) is tradeable
and they are the project's home turf.

CLI (signal 2 only - signal 1 is called from notebook 04 directly):
  python -m src.themes --in daily_ticker_counts.parquet --out daily_theme_counts.parquet
"""

import argparse
import re

import pandas as pd


# ---------------------------------------------------------------------------
# SIGNAL 1 — keyword themes
# ---------------------------------------------------------------------------
# Words and phrases only — NO bare ticker symbols (see module docstring).
# Company names written as words (Nvidia, Micron, Exxon) are fine and useful:
# they catch posts that never use the symbol. Jargon acronyms that are not
# tradeable US symbols (HBM, DRAM, LLM, EUV, FDA, CPI...) are also fine.
# ---------------------------------------------------------------------------
THEME_KEYWORDS: dict[str, list[str]] = {
    "semiconductors": [
        "semiconductor", "semis", "chipmaker", "chips", "chip",
        "fab", "wafer", "foundry", "lithography", "EUV",
        "TSMC", "Intel", "Broadcom", "Qualcomm", "Texas Instruments",
        "Marvell", "ON Semiconductor", "Microchip", "ASML",
        "silicon", "process node", "3nm", "5nm", "7nm", "2nm",
        "advanced packaging", "CoWoS", "chiplet", "chiplets",
        "Tokyo Electron", "Advantest", "Lasertec", "Disco Corp",
        "SUMCO", "photoresist", "wafer fab equipment", "tape out",
    ],
    "memory": [
        "memory", "DRAM", "HBM", "HBM2", "HBM3", "HBM4",
        "NAND", "flash storage", "DDR4", "DDR5",
        "Micron", "Samsung", "SK Hynix", "Hynix",
        "bandwidth memory", "high bandwidth", "memory chip",
        "storage chip", "solid state", "SSD",
    ],
    "ai": [
        "AI", "artificial intelligence", "machine learning", "deep learning",
        "LLM", "large language model", "GPT", "ChatGPT", "generative AI",
        "neural network", "AI training", "AI chip",
        "data center AI", "Nvidia AI", "CUDA", "transformer",
        "foundation model", "AGI", "OpenAI", "Anthropic",
        "inference", "AI agent", "AI agents", "agentic", "copilot",
        "Gemini", "Claude", "DeepSeek", "Mistral", "xAI", "Grok",
        "AI capex", "AI bubble", "AI spending", "GPU", "GPUs",
    ],
    "datacenters": [
        "data center", "datacenter", "data centre", "datacenters",
        "colocation", "server farm", "hyperscale", "GPU cluster",
        "server rack", "compute capacity", "cloud infrastructure",
        "liquid cooling", "immersion cooling", "power density",
        "data center power", "gigawatt", "transformers shortage",
        "Stargate", "CoreWeave", "neocloud",
    ],
    "ai_megacap": [
        "Nvidia", "Microsoft", "Google", "Alphabet",
        "Facebook", "Meta Platforms", "Apple", "Amazon", "Tesla",
        "mag7", "magnificent seven", "big tech",
        "hyperscaler", "hyperscalers",
    ],
    "crypto": [
        "bitcoin", "BTC", "ethereum", "ETH", "crypto", "cryptocurrency",
        "defi", "blockchain", "altcoin", "NFT", "web3",
        "Coinbase", "MicroStrategy", "stablecoin",
        "halving", "mining rig", "hash rate",
        "XRP", "Ripple", "Solana", "dogecoin", "memecoin", "memecoins",
        "Tether", "USDC", "spot ETF", "onchain", "on-chain",
        "staking", "airdrop", "tokenized", "tokenization",
    ],
    "gold_metals": [
        "gold", "silver", "precious metal", "metals",
        "Newmont", "Barrick", "Agnico", "Freeport",
        "copper", "platinum", "palladium",
        "commodity", "commodities", "inflation hedge",
        "rare earth", "rare earths", "neodymium", "lithium miner",
        "gold miner", "gold miners", "bullion", "central bank buying",
        "cobalt", "nickel", "antimony", "gallium", "germanium",
    ],
    "energy": [
        "oil", "crude", "WTI", "Brent", "natural gas", "LNG",
        "energy stock", "oil stock", "Exxon", "Chevron",
        "shell oil", "ConocoPhillips", "refinery", "pipeline",
        "OPEC", "oilfield", "shale", "fracking",
    ],
    "ev_clean_energy": [
        "electric vehicle", "EV", "Tesla", "Rivian",
        "Lucid", "NIO", "Xpeng", "Li Auto",
        "battery", "lithium", "lithium ion", "charging station",
        "solar", "wind energy", "renewable", "clean energy",
        "Enphase", "First Solar",
        "BYD", "CATL", "solid state battery", "solid-state battery",
        "battery maker", "gigafactory", "cathode", "anode", "LFP",
        "Panasonic battery", "EV sales", "EV demand",
    ],
    "uranium_nuclear": [
        "uranium", "nuclear", "nuclear power", "nuclear energy",
        "reactor", "reactors", "SMR", "small modular reactor",
        "enrichment", "Cameco", "Kazatomprom", "fission",
        "nuclear renaissance", "yellowcake",
    ],
    "defense_aerospace": [
        "defense stock", "defence stock", "defense budget", "military spending",
        "Lockheed", "Raytheon", "Northrop", "General Dynamics",
        "Pentagon", "missile", "missiles", "artillery", "munitions",
        "air defense", "military contract", "defense contractor",
        "drone", "drones", "counter-drone", "drone warfare", "UAV",
        "Anduril", "Palantir defense", "hypersonic", "golden dome",
    ],
    "europe_defense": [
        "Rheinmetall", "BAE Systems", "Thales", "Saab",
        "European defense", "European defence", "EU defense",
        "rearmament", "rearm", "German defense",
        "NATO spending", "NATO target", "defense procurement",
        "Leonardo", "Hensoldt", "Kongsberg", "Renk", "Dassault",
        "Airbus defence", "MBDA", "Eurofighter", "Gripen",
    ],
    "short_squeeze": [
        "short squeeze", "gamma squeeze", "squeeze", "short interest",
        "days to cover", "float", "low float", "heavily shorted",
        "short seller", "short position", "naked short", "MOASS",
        "mother of all short squeezes", "cover shorts", "covering",
        "borrow rate",
    ],
    "meme_stocks": [
        "meme stock", "meme stocks", "GameStop", "BlackBerry", "Bed Bath",
        "reddit rally", "WSB", "wallstreetbets",
        "retail investor", "apes", "yolo", "diamond hands",
        "paper hands", "tendies", "to the moon",
    ],
    "biotech_pharma": [
        "biotech", "pharma", "pharmaceutical", "FDA", "FDA approval",
        "clinical trial", "phase 1", "phase 2", "phase 3",
        "drug approval", "cancer drug", "oncology",
        "Moderna", "Pfizer", "Merck",
        "AstraZeneca", "Eli Lilly",
        "weight loss drug", "GLP-1", "ozempic", "semaglutide",
        "gene therapy", "CRISPR", "antibody",
    ],
    "rates_bonds": [
        "interest rate", "federal reserve", "Fed", "FOMC",
        "rate hike", "rate cut", "inflation", "CPI", "PPI",
        "recession", "soft landing", "hard landing",
        "yield curve", "bond yield", "treasury", "10-year",
        "stagflation", "tightening", "pivot",
    ],
    "real_estate": [
        "real estate", "REIT", "housing market", "home price",
        "mortgage rate", "30-year mortgage", "refinancing",
        "commercial real estate", "office space", "multifamily",
        "Simon Property", "Realty Income",
        "landlord", "rent", "eviction",
    ],
    "cloud_saas": [
        "cloud", "SaaS", "software as a service", "AWS", "Azure",
        "Google Cloud", "GCP", "cloud computing", "subscription revenue",
        "ARR", "annual recurring revenue", "churn",
        "Salesforce", "Snowflake", "Palantir",
        "Datadog", "MongoDB", "Cloudflare",
    ],
    "china_geopolitics": [
        "China", "Chinese", "tariff", "trade war", "sanctions",
        "Taiwan", "geopolitical", "decoupling", "supply chain",
        "export control", "Alibaba", "JD.com",
        "Tencent", "Baidu", "Huawei",
    ],
    "financials": [
        "bank", "banking", "JPMorgan", "Goldman Sachs",
        "Morgan Stanley", "Bank of America",
        "Wells Fargo", "Citigroup",
        "credit card", "regional bank", "Silicon Valley Bank",
        "credit default swap",
    ],
    "consumer_retail": [
        "consumer", "retail", "spending", "Walmart",
        "Amazon", "Target", "Costco",
        "consumer sentiment", "discretionary", "e-commerce",
        "holiday sales", "Black Friday", "back to school",
    ],
    "cybersecurity": [
        "cybersecurity", "cyber attack", "cyberattack", "ransomware",
        "data breach", "hacked", "hackers", "phishing", "zero-day",
        "CrowdStrike", "Palo Alto", "Fortinet", "Zscaler",
    ],
    "fintech_payments": [
        "fintech", "payments", "digital wallet", "buy now pay later",
        "PayPal", "Visa", "Mastercard", "Stripe", "neobank",
        "payment processing", "interchange",
    ],
    "gaming_esports": [
        "gaming", "video game", "video games", "esports", "console",
        "playstation", "xbox", "nintendo", "game pass", "steam deck",
        "gamer", "gamers",
    ],
    "travel_airlines": [
        "airline", "airlines", "travel demand", "bookings", "cruise",
        "cruise line", "hotel occupancy", "revenge travel", "air travel",
        "airfare",
    ],
    "housing_builders": [
        "homebuilder", "homebuilders", "housing starts", "new homes",
        "home construction", "housing supply", "mortgage applications",
        "housing shortage",
    ],
    "robotics_automation": [
        "robotics", "robots", "humanoid", "humanoids", "automation",
        "industrial automation", "robotaxi", "self-driving", "autonomous vehicle",
        # the physical supply chain - where the humanoid trade actually trades
        "bearings", "ball bearing", "ball bearings", "actuator", "actuators",
        "servo", "servos", "harmonic drive", "strain wave", "gearbox",
        "reducer", "reducers", "planetary gear", "linear guide", "ball screw",
        "motion control", "end effector", "gripper", "grippers",
        "torque sensor", "force sensor", "lidar", "machine vision",
        # robot makers and integrators (names, not tickers)
        "Fanuc", "Yaskawa", "Keyence", "Kuka", "ABB robotics",
        "Universal Robots", "cobot", "cobots", "Optimus", "Figure AI",
        "Unitree", "Agility Robotics", "Boston Dynamics", "teleoperation",
        # bearings & motion names (Japan/Europe - keyword is how we see them)
        "THK", "Nabtesco", "Harmonic Drive Systems", "SKF", "Schaeffler",
        "NSK", "Timken", "RBC Bearings", "Regal Rexnord", "Rexnord",
    ],
    "space": [
        "space launch", "rocket launch", "satellite", "satellites",
        "SpaceX", "Starlink", "orbital", "space station", "moon landing",
        "space economy",
        "Starship", "Kuiper", "launch cadence", "reusable rocket",
        # NOT bare "constellation" - it would catch Constellation Energy/Brands
        "smallsat", "space force", "lunar lander", "satellite constellation",
    ],
    "quantum_computing": [
        "quantum computing", "quantum computer", "qubit", "qubits",
        "quantum chip", "quantum supremacy", "error correction",
        "Willow chip", "Majorana", "quantum advantage", "post-quantum",
        "quantum annealing", "trapped ion", "superconducting qubit",
    ],
    "weight_loss_glp1": [
        "GLP-1", "ozempic", "wegovy", "mounjaro", "zepbound",
        "semaglutide", "tirzepatide", "weight loss drug", "obesity drug",
    ],
    "cannabis": [
        "cannabis", "marijuana", "weed stock", "weed stocks",
        "dispensary", "rescheduling", "legalization",
    ],
    "solar": [
        "solar", "solar panel", "solar panels", "rooftop solar",
        "photovoltaic", "net metering", "solar farm",
    ],
    "agriculture_food": [
        "agriculture", "farmland", "fertilizer", "crop", "crops",
        "grain", "wheat", "corn prices", "food prices", "harvest",
    ],
    "shipping_logistics": [
        "shipping", "freight", "container rates", "supply chain",
        "trucking", "railroad", "railroads", "logistics", "port congestion",
        "red sea",
    ],
    "small_caps": [
        "small caps", "small cap", "russell 2000", "microcap", "microcaps",
        "small-cap rotation",
    ],
    "japan": [
        "nikkei", "yen", "bank of japan", "carry trade",
        "japanese stocks", "japan stocks",
        "topix", "BOJ", "JGB", "yen intervention", "Japan Inc",
        "Softbank", "Sony", "Toyota", "Mitsubishi Heavy",
        "trading houses", "sogo shosha", "Berkshire Japan",
    ],
    "utilities_power": [
        "utilities", "power grid", "electricity demand", "power plant",
        "grid buildout", "electricity prices", "power purchase agreement",
    ],
    "media_streaming": [
        "streaming", "subscribers", "box office", "netflix", "disney",
        "ad tier", "streaming wars", "cord cutting",
    ],
    "infrastructure": [
        "infrastructure", "construction spending", "roads and bridges",
        "grid upgrade", "data center construction", "megaproject",
        "infrastructure bill",
    ],
}

# ---------------------------------------------------------------------------
# TRADEABLE ANCHORS - the instrument each theme is judged against.
# One liquid primary per theme (mostly ETFs). short_squeeze / meme_stocks
# have no clean ETF; GME is the honest single-stock proxy.
# ---------------------------------------------------------------------------
# RESTRICTED TO THE FIRM-APPROVED TRADEABLE LIST (see the Tickers sheet):
# every anchor below is an instrument the desk can actually trade. Themes
# with no approved instrument (crypto, cannabis, small_caps, japan) are
# still TRACKED - keywords, counts, sentiment, conviction all work - but
# carry no anchor, so they are excluded from trade signals automatically.
THEME_ETFS: dict[str, str] = {
    "semiconductors": "SMH",
    "memory": "SMH",            # no pure memory ETF on the approved list
    "ai": "IYW",                # US tech; AIQ is not tradeable here
    "datacenters": "VPN",       # Global X data center REITs - on the list
    "ai_megacap": "QQQ",
    "gold_metals": "GLD",       # miners GDX / silver SLV,SIL / copper COPX in fallbacks
    "energy": "XLE",
    "ev_clean_energy": "LIT",   # lithium & battery - on the list
    "uranium_nuclear": "URA",
    "defense_aerospace": "ITA",
    "europe_defense": "ITA",    # no European defense line approved; US proxy
    "short_squeeze": "ARKK",    # speculative-growth basket; GME not tradeable
    "meme_stocks": "ARKK",      # same proxy - the retail-favourite basket
    "biotech_pharma": "XBI",
    "rates_bonds": "TLT",
    "real_estate": "XLRE",      # VNQ not approved; sector SPDR is
    "cloud_saas": "IGV",
    "china_geopolitics": "KWEB",
    "financials": "XLF",
    "consumer_retail": "XLY",
    "cybersecurity": "CIBR",
    "fintech_payments": "XLF",  # IPAY not approved; financials proxy
    "gaming_esports": "SOCL",   # ESPO not approved; social/interactive proxy
    "travel_airlines": "JETS",
    "housing_builders": "ITB",
    "robotics_automation": "XLI",  # BOTZ/ROBO not approved; industrials proxy
    "space": "ITA",             # ARKX not approved; aerospace & defense proxy
    "quantum_computing": "IYW",
    "weight_loss_glp1": "XLV",  # LLY single stock not tradeable; healthcare
    "solar": "LIT",             # TAN not approved; closest clean-energy line
    "agriculture_food": "XLP",  # MOO not approved; staples/food proxy
    "shipping_logistics": "IYT",
    "utilities_power": "XLU",
    "media_streaming": "XLC",
    "infrastructure": "XLI",    # PAVE not approved; industrials proxy
    "japan": "1622 JT",         # NF Topix-17 Auto & Transport Equip - the
                                # approved Japan line (narrow: autos/machinery,
                                # not the broad Nikkei - read charts accordingly)
    # NO approved instrument -> tracked but untradeable, no anchor:
    #   crypto, cannabis, small_caps
}

# Some primary anchors only started trading recently (IBIT Jan-2024,
# MAGS Apr-2023, EUAD 2024, DTCR renamed 2024...). For a window BEFORE the
# anchor existed, the overlay notebooks fall back down this list until a
# symbol with price data in the window is found - so a thematic chart is
# never empty just because the modern ETF is younger than the window.
# The price puller requests ALL of these too, so the fallback always has data.
# Fallback chains - RESTRICTED to the firm-approved list too. Order: primary
# anchor first, then approved alternates (nb 17 can also pick any directly).
THEME_ETF_FALLBACKS: dict[str, list[str]] = {
    "semiconductors": ["SMH", "SOXX"],
    "memory": ["SMH", "SOXX"],
    "ai": ["IYW", "QQQ", "XLK"],
    "datacenters": ["VPN", "IYW"],
    "ai_megacap": ["QQQ", "XLK"],
    "gold_metals": ["GLD", "GDX", "SLV", "SIL", "COPX"],
    "energy": ["XLE", "XOP", "OIH", "USO", "UNG"],
    "ev_clean_energy": ["LIT", "XLY"],
    "defense_aerospace": ["ITA", "XLI"],
    "europe_defense": ["ITA"],
    "biotech_pharma": ["XBI", "XLV"],
    "rates_bonds": ["TLT", "LQD", "HYG", "TIP"],
    "real_estate": ["XLRE", "ITB"],
    "cloud_saas": ["IGV", "IYW"],
    "china_geopolitics": ["KWEB", "FXI", "ASHR", "CQQQ"],
    "financials": ["XLF", "KBE"],
    "fintech_payments": ["XLF", "IGV"],
    "gaming_esports": ["SOCL", "IYW"],
    "housing_builders": ["ITB", "XLB"],
    "robotics_automation": ["XLI", "IYW", "ARKK"],
    "space": ["ITA"],
    "quantum_computing": ["IYW", "QQQ"],
    "weight_loss_glp1": ["XLV", "XBI"],
    "solar": ["LIT", "XLU"],
    "agriculture_food": ["XLP", "XLB"],
    "media_streaming": ["XLC", "SOCL"],
    "infrastructure": ["XLI", "XLB"],
}

# ---------------------------------------------------------------------------
# INTERNATIONAL COVERAGE (Europe / Japan) - retail posts refer to foreign
# companies by NAME ("Rheinmetall", "Fanuc"), almost never by local ticker
# ("RHM.DE", "6954.T"). So the counting side is handled by the company names
# in THEME_KEYWORDS above; this map provides the PRICING side - the US-listed
# ADR that proxies each name, so overlays and backtests have a price line.
# The Bloomberg puller requests every symbol here. Only liquid, verified ADR
# symbols - a wrong symbol silently pulls nothing.
# ---------------------------------------------------------------------------
INTERNATIONAL_ADRS: dict[str, str] = {
    # robotics / bearings / motion control (Japan + Europe)
    "Fanuc": "FANUY",
    "Yaskawa": "YASKY",
    "Keyence": "KYCCY",
    "THK": "THKLY",
    "Nabtesco": "NCTKY",
    "SKF": "SKFRY",
    # semiconductor equipment (Japan)
    "Tokyo Electron": "TOELY",
    "Advantest": "ATEYY",
    "Lasertec": "LSRCY",
    "Disco Corp": "DSCSY",
    "SUMCO": "SUOPY",
    # European defense
    "Rheinmetall": "RNMBY",
    "BAE Systems": "BAESY",
    "Thales": "THLLY",
    "Saab": "SAABY",
    "Leonardo": "FINMY",
    "Airbus": "EADSY",
    # EV / batteries / Japan majors
    "BYD": "BYDDY",
    "Softbank": "SFTBY",
}

# Tokens are runs of letters/digits in the lowercased text, so "0DTE" and
# "3nm" survive as single tokens. Anything with a space, hyphen or dot in
# the keyword is treated as a phrase and substring-matched instead.
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Built lazily from THEME_KEYWORDS on first use.
_WORD_TO_THEMES: dict[str, tuple[str, ...]] = {}
_PHRASE_TO_THEMES: dict[str, tuple[str, ...]] = {}


def _get_keyword_lookup():
    """word -> themes dict for single-word keywords, phrase -> themes dict
    for multi-word ones. Hash lookups replace the old 20-regexes-per-post
    scan - that is where the ~25x speedup comes from."""
    if _WORD_TO_THEMES or _PHRASE_TO_THEMES:
        return _WORD_TO_THEMES, _PHRASE_TO_THEMES
    word_acc: dict[str, list[str]] = {}
    phrase_acc: dict[str, list[str]] = {}
    for theme, keywords in THEME_KEYWORDS.items():
        for kw in keywords:
            k = kw.lower()
            acc = phrase_acc if (" " in k or "-" in k or "." in k) else word_acc
            acc.setdefault(k, [])
            if theme not in acc[k]:
                acc[k].append(theme)
    _WORD_TO_THEMES.update({w: tuple(t) for w, t in word_acc.items()})
    _PHRASE_TO_THEMES.update({p: tuple(t) for p, t in phrase_acc.items()})
    return _WORD_TO_THEMES, _PHRASE_TO_THEMES


def themes_in_text(text: str) -> set[str]:
    """All themes whose keywords appear in one post's text (lowercased once)."""
    word_map, phrase_map = _get_keyword_lookup()
    lowered = text.lower()
    found: set[str] = set()
    for token in set(_TOKEN_RE.findall(lowered)):
        hit = word_map.get(token)
        if hit:
            found.update(hit)
    for phrase, themes in phrase_map.items():
        if phrase in lowered:
            found.update(themes)
    return found


def build_daily_theme_counts(posts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Signal 1: scan each post's title + selftext for theme keywords.

    posts_df must have columns: date, title, selftext (score optional).

    Returns DataFrame(date, theme, keyword_count, keyword_weighted) where
      keyword_count    = number of posts that day mentioning the theme
                         (each post counts once per theme, however many
                         keywords it contains - same dedupe rule as the
                         ticker pipeline)
      keyword_weighted = ALWAYS 0 - the score**2 weighting was removed
                         because archived scores leak future information
                         (kept as a column only so older notebooks run)
    """
    titles = posts_df["title"].fillna("").astype(str)
    bodies = posts_df["selftext"].fillna("").astype(str)
    dates = posts_df["date"].astype(str)

    rows = []
    for date, title, body in zip(dates, titles, bodies):
        found = themes_in_text(title + " " + body)
        for theme in found:
            rows.append((date, theme))

    if not rows:
        return pd.DataFrame(columns=["date", "theme", "keyword_count", "keyword_weighted"])

    long_df = pd.DataFrame(rows, columns=["date", "theme"])
    daily = (
        long_df.groupby(["date", "theme"])
        .agg(keyword_count=("theme", "size"))
        .reset_index()
    )
    daily["keyword_weighted"] = 0   # deprecated, see docstring
    return daily


# ---------------------------------------------------------------------------
# SIGNAL 2 — inferred themes (ticker -> theme)
# ---------------------------------------------------------------------------
# Curated: liquid, well-known names per theme. A ticker may sit in several
# themes (NVDA is a semiconductor, an AI trade and a megacap). Extend freely -
# precision comes from notebook 02's extraction, not from this mapping.
# ---------------------------------------------------------------------------
THEME_TICKERS: dict[str, set[str]] = {
    "semiconductors": {
        "NVDA", "AMD", "INTC", "TSM", "ASML", "AVGO", "QCOM", "TXN",
        "MRVL", "MCHP", "MU", "LRCX", "AMAT", "KLAC", "SMCI",
        "SMH", "SOXX", "SOXL",
    },
    "memory": {"MU", "WDC", "STX"},
    "ai": {"NVDA", "AMD", "PLTR", "AI", "SMCI", "MSFT", "GOOGL", "BBAI", "SOUN", "AIQ"},
    "datacenters": {"EQIX", "DLR", "SMCI", "VRT", "IRM", "ANET", "DTCR"},
    "ai_megacap": {
        "NVDA", "MSFT", "GOOGL", "GOOG", "META", "AAPL", "AMZN", "TSLA", "MAGS",
    },
    "crypto": {
        "COIN", "MSTR", "MARA", "RIOT", "HUT", "BITF", "CLSK",
        "GBTC", "BITO", "ETHE", "SI", "IBIT",
    },
    "gold_metals": {
        "GLD", "SLV", "IAU", "GDX", "GDXJ", "NEM", "GOLD", "AEM",
        "FCX", "WPM", "FNV", "SCCO",
    },
    "energy": {
        "XOM", "CVX", "COP", "OXY", "SLB", "HAL", "BP", "SHEL",
        "PSX", "VLO", "MPC", "DVN", "FANG", "XLE", "USO",
    },
    "ev_clean_energy": {
        "TSLA", "RIVN", "LCID", "NIO", "XPEV", "LI", "PLUG", "FCEL",
        "ENPH", "FSLR", "RUN", "SEDG", "CHPT", "QS", "BLNK",
        "ALB", "LIT", "ICLN", "TAN",
    },
    "uranium_nuclear": {
        "CCJ", "URA", "URNM", "UEC", "DNN", "LEU", "SMR", "OKLO", "NNE",
    },
    "defense_aerospace": {
        "LMT", "RTX", "NOC", "GD", "LHX", "HII", "KTOS", "AVAV",
        "ITA", "PPA", "XAR",
    },
    "europe_defense": {"EUAD"},
    "short_squeeze": {
        "GME", "AMC", "BBBY", "KOSS", "EXPR", "NAKD", "WISH",
        "WKHS", "SPCE", "CLOV",
    },
    "meme_stocks": {
        "GME", "AMC", "BB", "BBBY", "CLOV", "SNDL", "KOSS", "NOK",
        "EXPR", "WISH", "HOOD", "TLRY",
    },
    "biotech_pharma": {
        "MRNA", "PFE", "MRK", "AZN", "LLY", "JNJ", "ABBV", "BMY",
        "GILD", "AMGN", "REGN", "VRTX", "NVAX", "BNTX", "CRSP",
        "OCGN", "XBI", "IBB",
    },
    "rates_bonds": {"TLT", "IEF", "SHY", "TBT", "HYG", "LQD"},
    "real_estate": {"SPG", "O", "VNQ", "AMT", "PLD", "EQR", "AVB"},
    "cloud_saas": {
        "CRM", "SNOW", "PLTR", "DDOG", "MDB", "NET", "ORCL", "ADBE",
        "NOW", "TEAM", "ZM", "WDAY", "OKTA", "ZS", "CRWD", "TWLO", "SHOP",
        "IGV",
    },
    "china_geopolitics": {
        "BABA", "JD", "PDD", "BIDU", "NIO", "XPEV", "LI",
        "FXI", "KWEB", "YINN", "DIDI", "TCEHY",
    },
    "financials": {
        "JPM", "GS", "MS", "BAC", "WFC", "C", "SCHW", "BLK",
        "V", "MA", "AXP", "XLF", "SOFI", "HOOD",
    },
    "consumer_retail": {
        "WMT", "AMZN", "TGT", "COST", "HD", "LOW", "NKE", "SBUX",
        "MCD", "LULU", "XLY",
    },
    "cybersecurity": {
        "CRWD", "PANW", "ZS", "OKTA", "FTNT", "CYBR", "TENB", "RPD",
        "CIBR", "HACK",
    },
    "fintech_payments": {
        "PYPL", "SQ", "XYZ", "V", "MA", "AXP", "AFRM", "SOFI", "HOOD",
        "TOST", "UPST", "IPAY", "FINX",
    },
    "gaming_esports": {
        "RBLX", "EA", "TTWO", "U", "SONY", "NTDOY", "MSFT", "DKNG",
        "ESPO",
    },
    "travel_airlines": {
        "DAL", "UAL", "AAL", "LUV", "ABNB", "BKNG", "EXPE", "CCL",
        "RCL", "NCLH", "MAR", "HLT", "JETS",
    },
    "housing_builders": {
        "DHI", "LEN", "PHM", "NVR", "TOL", "KBH", "BLDR", "ITB", "XHB",
    },
    "robotics_automation": {
        "ISRG", "TER", "ROK", "SYM", "PATH", "TSLA", "BOTZ", "ROBO",
        # bearings / motion-control supply chain (US-listed)
        "TKR", "RRX", "RBC", "SERV", "RR",
    },
    "space": {
        "RKLB", "LUNR", "ASTS", "SPCE", "BA", "RDW", "ARKX",
    },
    "quantum_computing": {
        "IONQ", "RGTI", "QBTS", "QUBT", "IBM", "QTUM",
    },
    "weight_loss_glp1": {
        "LLY", "NVO", "HIMS", "VKTX", "AMGN",
    },
    "cannabis": {
        "TLRY", "CGC", "ACB", "SNDL", "MSOS",
    },
    "solar": {
        "ENPH", "SEDG", "FSLR", "RUN", "NXT", "ARRY", "TAN",
    },
    "agriculture_food": {
        "ADM", "BG", "DE", "MOS", "NTR", "CF", "MOO", "DBA",
    },
    "shipping_logistics": {
        "FDX", "UPS", "ZIM", "MATX", "UNP", "CSX", "ODFL", "GXO", "IYT",
    },
    "small_caps": {
        "IWM",
    },
    "japan": {
        "EWJ", "DXJ",
    },
    "utilities_power": {
        "XLU", "NEE", "VST", "CEG", "D", "SO", "GEV",
    },
    "media_streaming": {
        "NFLX", "DIS", "WBD", "PARA", "ROKU", "SPOT", "XLC",
    },
    "infrastructure": {
        "CAT", "VMC", "MLM", "URI", "PWR", "PAVE",
    },
}


def build_ticker_to_themes(theme_tickers=THEME_TICKERS):
    lookup: dict[str, list[str]] = {}
    for theme, tickers in theme_tickers.items():
        for ticker in tickers:
            lookup.setdefault(ticker, []).append(theme)
    return lookup


def build_inferred_theme_counts(daily_ticker_counts: pd.DataFrame,
                                theme_tickers=THEME_TICKERS) -> pd.DataFrame:
    """
    Signal 2: roll the daily ticker counts (notebook 02's output) up into
    themes. NVDA mentions count toward semiconductors, ai AND ai_megacap.

    daily_ticker_counts must have columns: date, ticker, mention_count.

    Returns DataFrame(date, theme, inferred_count, inferred_weighted).
    inferred_weighted is ALWAYS 0 (score**2 weighting removed - archived
    scores leak future information; column kept for notebook compatibility).
    """
    df = daily_ticker_counts

    lookup = build_ticker_to_themes(theme_tickers)
    rows = []
    for date, ticker, count in zip(df["date"], df["ticker"], df["mention_count"]):
        for theme in lookup.get(ticker, ()):
            rows.append((date, theme, count))

    if not rows:
        return pd.DataFrame(columns=["date", "theme", "inferred_count", "inferred_weighted"])

    long_df = pd.DataFrame(rows, columns=["date", "theme", "inferred_count"])
    out = (
        long_df.groupby(["date", "theme"], as_index=False)[["inferred_count"]]
        .sum()
    )
    out["inferred_weighted"] = 0   # deprecated, see docstring
    return out


def combine_theme_signals(keyword_df: pd.DataFrame,
                          inferred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Outer-join the two signals on (date, theme); a theme silent in one signal
    gets 0 there. Result columns: date, theme, keyword_count,
    keyword_weighted, inferred_count, inferred_weighted.
    """
    merged = keyword_df.merge(inferred_df, on=["date", "theme"], how="outer")
    count_cols = ["keyword_count", "keyword_weighted", "inferred_count", "inferred_weighted"]
    for col in count_cols:
        if col not in merged.columns:
            merged[col] = 0
    merged[count_cols] = merged[count_cols].fillna(0).astype("int64")
    return merged.sort_values(["date", "theme"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI (signal 2 only — signal 1 is called from notebook 04 directly)
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Roll ticker mentions up into inferred themes")
    parser.add_argument("--in", dest="inp", required=True, help="daily ticker counts .parquet/.csv")
    parser.add_argument("--out", required=True, help="output daily inferred theme counts path")
    args = parser.parse_args(argv)

    df = pd.read_parquet(args.inp) if args.inp.endswith(".parquet") else pd.read_csv(args.inp)
    theme_df = build_inferred_theme_counts(df)

    if theme_df.empty:
        print("No tickers matched any theme - check THEME_TICKERS.")
        return 1

    if args.out.endswith(".parquet"):
        theme_df.to_parquet(args.out, index=False)
    else:
        theme_df.to_csv(args.out, index=False)

    totals = theme_df.groupby("theme")["inferred_count"].sum().sort_values(ascending=False)
    print("Saved", len(theme_df), "rows to", args.out)
    print("\nTotal inferred mentions per theme:")
    for theme, total in totals.items():
        print("  ", theme, ":", int(total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
