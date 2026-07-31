# ETF Research Note - retailAPOLLO theme mapping
Date: 2026-07-31. Holdings verified against current online sources (stockanalysis.com holdings pages, as-of dates 2026-07-15 to 2026-07-30, cross-checked with issuer/Cboe/CSI pages). All rows in config/etf_constituents.csv are verified=yes. Constituent tickers are normalized to US listings; foreign lines without a liquid US listing/ADR were skipped, so some ETFs carry fewer than 10 rows (better short and real than padded).

## ETFs researched (one line each)

Theme-pure equity ETFs (included in etf_constituents.csv):
- SMH - VanEck Semiconductor ETF (VanEck): 25 largest US-listed semis, NVDA ~21%; theme-pure (semiconductors, memory).
- SOXX - iShares Semiconductor ETF (BlackRock/iShares): ICE Semiconductor Index, more equal at the top than SMH; theme-pure.
- XLE - Energy Select Sector SPDR (State Street): S&P 500 energy, XOM+CVX ~36%; theme-pure (energy).
- XOP - SPDR S&P Oil & Gas Exploration & Production (State Street): equal-weight E&P + refiners; theme-pure.
- OIH - VanEck Oil Services ETF (VanEck): oilfield services, SLB/BKR/HAL; theme-pure.
- GDX - VanEck Gold Miners ETF (VanEck): global gold miners; theme-pure. Note Barrick now trades NYSE as B (renamed Barrick Mining, 2025).
- SIL - Global X Silver Miners ETF (Global X): silver miners/streamers, WPM ~22%; theme-pure.
- COPX - Global X Copper Miners ETF (Global X): global copper miners; theme-pure, but majority of holdings are Canada/UK/Asia-listed - only 5 top names map to US tickers (BHP, TECK, HBM, SCCO, FCX).
- URA - Global X Uranium ETF (Global X): miners + nuclear-fuel/SMR names (CCJ, OKLO, LEU); theme-pure.
- ITA - iShares U.S. Aerospace & Defense ETF (iShares): US A&D, GE+RTX ~39%; theme-pure for defense_aerospace (US).
- EUAD - Select STOXX Europe Aerospace & Defense ETF (Select Funds / Tuttle Capital Mgmt): European A&D via ADRs/swaps; theme-pure for europe_defense. See VERIFICATION below.
- XBI - SPDR S&P Biotech ETF (State Street): equal-weight US biotech, small/mid tilt; theme-pure. Top-weight order shuffles constantly (modified equal weight).
- BBH - VanEck Biotech ETF (VanEck): ~25 large biotechs, AMGN/GILD/VRTX heavy; theme-pure large-cap complement to XBI.
- XLRE - Real Estate Select Sector SPDR (State Street): S&P 500 REITs; theme-pure (real_estate).
- IGV - iShares Expanded Tech-Software Sector ETF (iShares): US software incl. security software; theme-pure (cloud_saas).
- KWEB - KraneShares CSI China Internet ETF (KraneShares): China internet platforms; theme-pure (china_geopolitics anchor). HK lines mapped to ADRs (TCEHY, MPNGY are OTC ADRs; rest NYSE/NASDAQ-listed).
- FXI - iShares China Large-Cap ETF (iShares): 50 HK-listed China large caps; theme-pure China proxy. China Life and PetroChina skipped (NYSE ADRs delisted 2022); banks/insurers mapped to OTC ADRs.
- CQQQ - Invesco China Technology ETF (Invesco): China tech incl. onshore A-share semis; only 6 top names have US ADRs, rest skipped.
- XLF - Financial Select Sector SPDR (State Street): S&P 500 financials incl. payments (V, MA) and BRK.B; theme-pure (financials).
- KBE - SPDR S&P Bank ETF (State Street): equal-weight banks + mortgage insurers; theme-pure.
- KRE - SPDR S&P Regional Banking ETF (State Street): equal-weight regional banks; theme-pure. Equal weight means top-10 is nearly arbitrary; membership matters more than rank here.
- XLY - Consumer Discretionary Select Sector SPDR (State Street): AMZN+TSLA ~39%; sector fund but retained as consumer_retail fallback.
- XRT - SPDR S&P Retail ETF (State Street): equal-weight retail incl. e-commerce; theme-pure (consumer_retail anchor). Good call replacing XLY as anchor.
- CIBR - First Trust NASDAQ Cybersecurity ETF (First Trust): cybersecurity; theme-pure.
- JETS - U.S. Global Jets ETF (U.S. Global): airlines + a tail of OTAs/aerospace; theme-pure (travel_airlines). Air Canada and Bombardier skipped (no US listing).
- ITB - iShares U.S. Home Construction ETF (iShares): homebuilders + building products/retail; theme-pure (housing_builders).
- IYT - iShares U.S. Transportation ETF (iShares): rails/parcel/truckers/airlines; theme-pure-ish for shipping_logistics (see mapping notes). New line FDXF = FedEx Freight Holding (2026 spin-off), verified in holdings.
- XLU - Utilities Select Sector SPDR (State Street): S&P 500 utilities incl. IPPs (CEG, VST); theme-pure (utilities_power).
- XLC - Communication Services Select Sector SPDR (State Street): META/GOOGL + media/telco/gaming; theme-pure (media_streaming).
- XLP - Consumer Staples Select Sector SPDR (State Street): staples; theme-pure (agriculture_food fallback).
- XLB - Materials Select Sector SPDR (State Street): chemicals/miners/aggregates; theme-pure (materials fallback).
- XLI - Industrial Select Sector SPDR (State Street): broad industrials; kept because it anchors robotics_automation and infrastructure (see mapping notes).
- LIT - Global X Lithium & Battery Tech ETF (Global X): lithium miners + battery chain + TSLA/BYD; RIO ~22% post its Arcadium acquisition. Majority Asia-listed; 6 US-mappable top names.
- MOO - VanEck Agribusiness ETF (VanEck): seeds/ag-chem/equipment/protein; theme-pure (agriculture_food anchor). Kubota, Mowi, Wilmar, Yara skipped (no liquid US line); Bayer mapped to BAYRY ADR.
- VPN - listed in config as VPN but THE TICKER IS STALE: Global X renamed the fund Global X Data Center & Digital Infrastructure ETF and changed the ticker to DTCR (late 2024). Holdings in the CSV (under etf=VPN to match the config key) are current DTCR holdings: data-center REITs + towers + AI-infra semis. ACTION: update theme_etfs.csv and approved_instruments.csv from VPN to DTCR.
- SOCL - Global X Social Media ETF (Global X): global social platforms, RDDT top weight; theme-pure for social media (NOT for gaming, see mapping notes). NAVER, Kakao, Kuaishou skipped.
- ARKK - ARK Innovation ETF (ARK Invest): concentrated speculative growth; included per spec as the short_squeeze/meme_stocks basket. Private holdings (SpaceX, OpenAI) skipped - no public tickers.

Researched but excluded from etf_constituents.csv (no single-name equity constituents):
- GLD - SPDR Gold Shares (State Street): physical gold bullion. No equities to map; gold_metals single names come via fallbacks GDX/SLV/SIL/COPX.
- SLV - iShares Silver Trust (iShares): physical silver bullion. No equities.
- TLT - iShares 20+ Year Treasury Bond ETF; LQD - iShares iBoxx $ Investment Grade Corporate Bond ETF; HYG - iShares iBoxx $ High Yield Corporate Bond ETF; TIP - iShares TIPS Bond ETF (all iShares): bond funds, holdings are CUSIP-level bonds, no equity tickers. rates_bonds theme cannot be mapped to single names via membership.
- USO / UNG (USCF): oil / natural gas futures funds, no equities (energy fallbacks only).
- ASHR - Xtrackers Harvest CSI 300 China A-Shares ETF (DWS/Xtrackers): onshore A-shares (Zhongji Innolight, CATL, Moutai...). Virtually no top holding has a plain US ticker; excluded rather than padded with pink-sheet lines.

Skipped as broad/style/multi-sector proxies per spec: QQQ, IYW, XLK, XLV, RSP, IVE, IVW, VTV, VUG, MTUM. Skipped non-US-listed: 159915 CS, 588000 CH, 1622 JT, CSIN0852.

## VERIFICATION - 13 newly added instruments (2026-07-31)

Bloomberg exchange-code conventions used: US=US composite, UN=NYSE, UP=NYSE Arca, UA=NYSE American, UW=NASDAQ GS, UQ=NASDAQ GM, UR=NASDAQ CM, UF=Cboe BZX, CH=China composite, CG=Shanghai, CS=Shenzhen, JT=Tokyo.

1. 159915 CS - E Fund ChiNext ETF. CONFIRMED: E Fund ChiNext (Price) Index ETF, listed Shenzhen Stock Exchange (159xxx = SZSE ETF range). CS = Bloomberg Shenzhen code - CORRECT. Bloomberg composite quote 159915:CH also exists; either works.
2. RSP US - Invesco S&P 500 Equal Weight ETF. CONFIRMED. Listed NYSE Arca; "US" composite code is fine (specific would be UP). Broad proxy - correctly excluded from constituents.
3. BBH UQ - VanEck Biotech ETF. CONFIRMED NASDAQ-listed (nasdaq.com/vaneck both show NASDAQ). UQ (NASDAQ GM) is plausible; note pre-existing rows use US composite - minor inconsistency, not an error.
4. IVE UP - iShares S&P 500 Value ETF. CONFIRMED, NYSE Arca-listed. UP correct.
5. IVW UP - iShares S&P 500 Growth ETF. CONFIRMED, NYSE Arca-listed. UP correct.
6. MOO UP - VanEck Agribusiness ETF. CONFIRMED, NYSE Arca-listed. UP correct.
7. VTV UP - Vanguard Value ETF. CONFIRMED, NYSE Arca-listed. UP correct.
8. VUG UP - Vanguard Growth ETF. CONFIRMED, NYSE Arca-listed. UP correct.
9. XRT UP - SPDR S&P Retail ETF. CONFIRMED, NYSE Arca-listed. UP correct.
10. 588000 CH - config names it "Huatai-PineBridge SSE STAR 50 ETF" - NAME IS WRONG. 588000 is the ChinaAMC (China Asset Management) SSE Science and Technology Innovation Board 50 (STAR 50) ETF, the largest STAR 50 tracker (Yahoo/Morningstar/Investing all show ChinaAMC). Huatai-PineBridge's STAR 50 ETF is 588090. The code 588000 + exchange are fine (588xxx = Shanghai; CH composite quotes on Bloomberg as 588000:CH; Shanghai-specific would be CG). ACTION: fix the name field to "ChinaAMC SSE STAR 50 ETF" (or switch code to 588090 if Huatai-PineBridge was intended).
11. CSIN0852 - IDENTIFIED WITH HIGH CONFIDENCE: this is the Bloomberg code for the CSI 1000 Index (Zhongzheng 1000, index code 000852), CSI's 1000-stock China A-share SMALL-CAP index (stocks outside the CSI 800; base date 2004-12-31, base 1000). Confirmed directly from the official CSI factsheet for index 000852, which lists Bloomberg code "CSIN0852 Index" and Reuters ".CSI1000I". It is an INDEX, not a tradable fund - fine as a signal/benchmark line, but you cannot execute it; the tradable expressions are CSI 1000 futures (IM) or onshore CSI 1000 ETFs (e.g. 512100 CG). Plausibly added for the small_caps (China) or china_geopolitics breadth leg.
12. KRE UP - SPDR S&P Regional Banking ETF. CONFIRMED, NYSE Arca-listed. UP correct.
13. MTUM TF - iShares MSCI USA Momentum Factor ETF. Name CONFIRMED; primary listing CONFIRMED as Cboe BZX (BlackRock annual report and Cboe listings page). FLAG: the standard Bloomberg exchange code for Cboe BZX-listed US equities is UF (legacy BATS), not TF - "MTUM TF Equity" looks like a typo and may not resolve on the terminal. RECOMMEND: use MTUM UF Equity or the composite MTUM US Equity. Verify on the terminal before go-live.

## EUAD confirmation (europe_defense remap)

CONFIRMED: Select STOXX Europe Aerospace & Defense ETF (ticker EUAD, ISIN US84858T7726) exists and is US-listed on Cboe BZX (it appears on Cboe's listed-products page; adviser Tuttle Capital Management, marketed by Select Funds; launched Oct 2024; SEC 497 on file). It tracks the STOXX Europe Total Market Aerospace & Defense (capped) index - Airbus, Rheinmetall, BAE, Thales, Leonardo, Saab, Safran, Rolls-Royce, MTU, Hensoldt etc. - so it IS the natural European-defense line, and the remap of europe_defense from ITA (US-only holdings, wrong region) to EUAD is financially correct. Two practical caveats: (a) the fund currently holds much of its exposure via a money-market + total-return-swap structure (~68% MM + swap overlay per current holdings), so look-through constituent weights come from the ADR/swap reference basket - membership mapping still works, weight precision is lower; (b) it is a young fund - check AUM/ADV against the desk's liquidity minimums. Constituents mapped to OTC ADRs in the CSV (EADSY, RNMBY, BAESY, ...).

## Theme -> ETF mapping issues found

WRONG (fix these):
1. datacenters -> VPN: STALE TICKER. VPN was renamed/re-tickered to DTCR (Global X Data Center & Digital Infrastructure ETF) in late 2024. The theme choice is right; the code will not resolve. Change VPN to DTCR in theme_etfs.csv and approved_instruments.csv.
2. japan -> 1622 JT: WRONG INSTRUMENT. 1622 JT is the Nomura NEXT FUNDS TOPIX-17 Automobiles & Transportation Equipment ETF - a Japan AUTOS sector fund, not broad Japan. For a "japan" theme use 1306 JT (NEXT FUNDS TOPIX) or 1321 JT (Nikkei 225), or US-listed EWJ if the desk prefers US hours. If autos exposure was actually intended, rename the theme japan_autos.
3. solar -> LIT: MISMATCH. LIT is lithium/battery-chain (RIO, Albemarle, CATL, BYD) with zero solar manufacturers. The solar ETF is TAN (Invesco Solar); fallback XLU only captures utility-scale demand, not solar names. Recommend adding TAN.
4. gaming_esports -> SOCL: MISMATCH. SOCL is social media (Reddit, Meta, Pinterest, Snap); its only gaming overlap is NetEase/Tencent. The gaming ETFs are ESPO (VanEck Video Gaming & eSports) or NERD. Recommend adding ESPO; until then the theme is mislabeled.
5. 588000 CH name wrong in approved_instruments.csv (ChinaAMC, not Huatai-PineBridge) - see VERIFICATION #10.
6. MTUM TF exchange code suspect (should be UF or US) - see VERIFICATION #13.

PROXY-ONLY (acceptable if intentional, but not theme-pure - flagging per spec):
7. fintech_payments -> XLF: XLF is broad financials (banks/insurers/BRK). V/MA are inside, but a purer line is IPAY or FINX.
8. weight_loss_glp1 -> XLV: broad healthcare; LLY is ~10% but the GLP-1 signal is diluted. Purer: OZEM (Roundhill GLP-1 & Weight Loss ETF) if approvable.
9. quantum_computing -> IYW: broad tech; purer is QTUM (Defiance Quantum).
10. robotics_automation -> XLI: broad industrials; purer is BOTZ or ROBO.
11. space -> ITA: US defense primes plus RKLB; purer is UFO or ARKX.
12. real_estate fallback ITB: ITB is homebuilders (companies), not REITs - as a fallback for real_estate it changes the economic exposure; XLRE alone is cleaner.
13. ev_clean_energy -> LIT: fine for the EV/battery chain, but fallback XLY reaches EVs only through TSLA/GM; ICLN or DRIV would be a more honest fallback.
14. shipping_logistics -> IYT: IYT is rails/parcel/airlines-heavy; container shipping proper is BOAT/BDRY. Acceptable for "logistics", weak for "shipping".
15. gold_metals fallback COPX: copper is an industrial metal - having COPX under gold_metals mixes precious and industrial exposure; consider a separate copper/industrial_metals theme.

Also note: rates_bonds (TLT/LQD/HYG/TIP) and gold/silver bullion (GLD/SLV) cannot map single names via membership at all (no equity constituents) - single-name attention hits on those themes need a different mechanism (e.g. rate-sensitive equity basket) or none.

## UPDATE 2026-07-31 (full holdings)

Per desk request, config/etf_constituents.csv was expanded from top-10/15 to the fullest retrievable constituent list per ETF. Sources: issuer full-holdings pages (VanEck, First Trust, KraneShares, arkfunds.io for ARK, Select Funds) where available; otherwise stockanalysis.com holdings pages, whose free tier exposes the TOP 25 rows only. All rows remain verified=yes (read from a current source). Rules unchanged: plain US tickers or liquid US ADRs only; foreign lines without one are skipped and re-ranked contiguously (so rank = order among US-mappable names, still descending by fund weight); cash/futures/money-market/private (SpaceX, OpenAI) lines excluded.

Coverage per ETF (rows in CSV / total holdings in fund; skips = non-US-mappable or non-equity lines dropped):
- FULL lists: SMH 25/25 (VanEck), OIH 25/25 (VanEck), XLE 21/21 equities (SPDR page shows all), XLC 23/23 equities, BBH 24/24 (VanEck), CIBR 40/42 (First Trust; skipped 2: Trend Micro JP, Atos FR), GDX 31/61 listed (VanEck full list; ~30 TSX/ASX/LSE/HK-only miners skipped), MOO 20/47 (VanEck full list; 27 non-US lines skipped), KWEB 19/33 (KraneShares full list; 14 HK-only lines skipped), ARKK 43/48 (arkfunds.io full list; skipped SpaceX, OpenAI, one money-market line, Brera Cl B + warrants), XLB 25/~26 equities (effectively full).
- TOP-25-ONLY (source free tier caps at 25 rows; fund total in parentheses): SOXX (34), XOP (54), ITA (53), XBI (157), XLRE (34), IGV (~120), XLF (80), KBE (105), KRE (~140), XLY (50), XRT (~78), ITB (48), IYT (47), XLU (34), XLP (38), XLI (84).
- TOP-25 SHOWN with foreign skips (rows kept / of 25 shown): SIL 17 (of 41 total; 8 foreign skipped incl. Penoles, Fresnillo, Korea Zinc, Aya), COPX 7 (of 44; 18 foreign skipped - COPX is majority non-US-listed), URA 10 (of 57; 15 skipped incl. Kazatomprom, Sprott U trust, Korean/Japanese E&C names), FXI 16 (of 60; 8 skipped incl. delisted-ADR China Life/PetroChina, Shenhua, CATL, Pop Mart), CQQQ 7 (of 188; overwhelmingly onshore A-share/HK tech without ADRs), LIT 8 (of 45; 17 skipped incl. CATL, Samsung SDI, LGES, EVE), SOCL 16 (of 50; 9 skipped incl. NAVER, Kakao, Kuaishou, NEXON; also skipped "PPLI/People Incorporated" - could not corroborate that ticker), JETS 18 (of 57; 7 skipped incl. Air Canada, Bombardier, Qantas, JAL, Turkish, IAG, Aena), VPN/DTCR 19 (of 28; 6 skipped incl. NEXTDC, Keppel DC REIT, SK hynix, China Tower).
- EUAD 12/23: issuer page itemizes only its top 10 (used for ranks 1-10); BCKIY and QNTQY appended at 11-12 (membership verified from the fund's holdings file); remaining ~11 small European lines not itemized by any free source.

Notable ticker facts encountered during expansion (all source-read): Barrick = B (NYSE, renamed 2025); Block = XYZ; Marsh McLennan appears as MMC (one aggregator showed a malformed "MRSH"); FedEx Freight spin-off trades as FDXF; Dril-Quip is now Innovex International (INVX); newly listed names present in funds: SARO (StandardAero, ITA), CRCL/BLSH/BMNR/CBRS/XE/FIG (ARKK), WYFI (DTCR), LIF (Life360 US listing, SOCL), AUGO (Aura Minerals, GDX).

Row-count changes (before -> after): SMH 15->25, SOXX 15->25, XLE 15->21, XOP 15->25, OIH 15->25, GDX 12->31, SIL 10->17, COPX 5->7, URA 8->10, ITA 15->25, EUAD 12->12 (ranks corrected to issuer weights), XBI 15->25, BBH 15->24, XLRE 15->25, IGV 15->25, KWEB 13->19, FXI 13->16, CQQQ 6->7, XLF 15->25, KBE 15->25, KRE 15->25, XLY 15->25, XRT 15->25, CIBR 15->40, JETS 13->18, ITB 15->25, IYT 15->25, XLU 15->25, XLC 15->23, XLP 15->25, XLB 15->25, XLI 15->25, LIT 6->8, MOO 11->20, VPN 12->19, SOCL 12->16, ARKK 13->43. Total 491 -> 826 rows.
