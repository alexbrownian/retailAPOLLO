"""
inflection.py
=============
Find the moment a ticker (or theme) "takes off" in Reddit mentions.

THE IDEA (this is the part to understand):

  - The daily mention count is the raw signal. It is bumpy day to day.
  - The "first derivative" just means: how much did it change since yesterday?
        first_derivative[today] = smoothed_count[today] - smoothed_count[yesterday]
    Think of the count as distance and the first derivative as speed.
    When speed jumps from ~0 to large positive, attention is accelerating -
    that is the inflection / take-off we want to catch early.
  - Because raw counts are noisy, we SMOOTH them first with an EWMA
    (exponentially weighted moving average - recent days weigh more).
    IMPORTANT: the velocity is ALWAYS the derivative of this EWMA line,
    NEVER of the raw mentions - otherwise every little wiggle looks like a
    derivative spike.
  - We then call a day an "inflection day" if its first derivative is unusually
    large compared to that ticker's normal day-to-day change. "Unusually large"
    = above mean + (k * standard deviation) of the derivative. That is just a
    simple, standard way of saying "much bigger than typical noise".

INPUT: a daily-counts table with columns: date, ticker, mention_count
       (this is exactly what extract_tickers.py --daily-out produces).

Run example:
  python3 inflection.py \\
      --in outputs/wsb_mentions/wsb_daily_ticker_counts_2021-01.parquet \\
      --ticker GME --smooth 3 --k 2.0
"""

import argparse
import datetime

import pandas as pd
from scipy.signal import find_peaks

# NOTE: don't call matplotlib.use("Agg") here - it forces a non-interactive
# backend for anyone who *imports* this module, which silently kills inline
# plotting in notebooks (plt.show() becomes a no-op). savefig()/close() in
# plot_inflection() below work fine on whatever backend is already active,
# so there's no need to force one at import time.
import matplotlib.pyplot as plt


def load_daily_counts(path):
    """Read the daily counts file (.parquet or .csv) into a DataFrame."""
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    # Make sure 'date' is a real date, not just text, so sorting/plotting works.
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_daily_series(df, ticker, value_col="mention_count"):
    """
    For one ticker, return a table with EVERY calendar day in the range,
    filling days with no mentions as 0.

    value_col : which column to use as the signal (normally "mention_count",
                the raw number of mentions; any other name falls back to
                mention_count if missing - score-based weighting was removed
                project-wide, see design_decisions.xlsx #30).

    Why fill the gaps? If a ticker is missing on a quiet day, the derivative
    would skip over it and look wrong. A continuous daily line is correct.
    """
    if value_col not in df.columns:
        value_col = "mention_count"

    one = df[df["ticker"] == ticker].copy()
    if one.empty:
        return None
    one = one.sort_values("date")

    # A complete list of days from the first to the last date in the data.
    full_range = pd.date_range(one["date"].min(), one["date"].max(), freq="D")

    # Reindex onto that full range; missing days become 0.
    series = (
        one.set_index("date")[value_col]
        .reindex(full_range)
        .fillna(0)
    )
    series.index.name = "date"
    return series


def compute_inflection(series, smooth_window, k, peak_k=3.0):
    """
    Take a daily count series and return a DataFrame with:
      count           - raw mentions
      smoothed        - EWMA of the raw mentions (less noisy)
      velocity        - first derivative OF THE EWMA (change in smoothed vs
                        yesterday - never computed on the raw mentions)
      is_inflection   - True on days where velocity is unusually high (alias
                        of is_rise, kept for backwards compatibility)
      is_rise         - "take-off": velocity spikes well above normal (concave
                        up - attention accelerating)
      is_fall         - "sell-off": velocity drops well below normal (concave
                        down and dropping fast - attention collapsing)
      is_peak         - a genuine local high of the smoothed line (it rose
                        into this day and fell back out of it)
      is_trough       - a genuine local low of the smoothed line (it fell
                        into this day and rose back out of it)

    smooth_window : EWMA span in days (e.g. 3 or 7; bigger = smoother)
    k             : how many standard deviations away from normal counts as
                    a spike (used symmetrically for both rise and fall)
    peak_k        : how many standard deviations of "prominence" a local
                    max/min needs to count as a real peak/trough, rather than
                    day-to-day noise. Naively marking every day the velocity
                    changes sign flags dozens of tiny wiggles per year; this
                    keeps only turns that actually stand out from their
                    surroundings (scipy's "prominence").
    """
    out = pd.DataFrame({"count": series})

    # 1) Smooth: EWMA with span `smooth_window` (recent days weigh more).
    out["smoothed"] = out["count"].ewm(span=smooth_window, min_periods=1).mean()

    # 2) First derivative OF THE EWMA: today's smoothed value minus
    #    yesterday's. Never diff the raw counts - differencing amplifies
    #    noise, so the derivative only makes sense on the smoothed line.
    out["velocity"] = out["smoothed"].diff().fillna(0)

    # 3) Thresholds = typical change +/- k * how spread out the changes are.
    average_change = out["velocity"].mean()
    spread = out["velocity"].std()
    rise_threshold = average_change + k * spread
    fall_threshold = average_change - k * spread

    # 4) Flag the rise/fall days (clearly above or below the threshold).
    out["is_rise"] = (out["velocity"] > rise_threshold) & (out["velocity"] > 0)
    out["is_fall"] = (out["velocity"] < fall_threshold) & (out["velocity"] < 0)
    out["is_inflection"] = out["is_rise"]  # backwards-compatible alias

    # 5) Peaks/troughs: prominent local turns of the smoothed line - i.e. the
    #    "rose here, fell here" moments, not every tiny up-down wiggle.
    smoothed_vals = out["smoothed"].to_numpy()
    prominence = peak_k * (out["smoothed"].std() or 1.0)
    peak_idx, _ = find_peaks(smoothed_vals, prominence=prominence)
    trough_idx, _ = find_peaks(-smoothed_vals, prominence=prominence)
    out["is_peak"] = False
    out["is_trough"] = False
    out.iloc[peak_idx, out.columns.get_loc("is_peak")] = True
    out.iloc[trough_idx, out.columns.get_loc("is_trough")] = True

    # Keep the thresholds around so we can draw them on the plot later.
    out.attrs["threshold"] = rise_threshold
    out.attrs["fall_threshold"] = fall_threshold
    return out


def plot_inflection(result, ticker, out_path):
    """Two stacked charts: mentions on top, velocity (first derivative) below.
    Rise/fall spikes and peak/trough turning points are annotated on both."""
    dates = result.index
    threshold = result.attrs["threshold"]
    fall_threshold = result.attrs["fall_threshold"]
    rise_days = result[result["is_rise"]]
    fall_days = result[result["is_fall"]]
    peak_days = result[result["is_peak"]]
    trough_days = result[result["is_trough"]]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    # --- Top: raw + smoothed mentions, with rise/fall/peak/trough marked ---
    ax_top.plot(dates, result["count"], color="lightgray", label="raw mentions")
    ax_top.plot(dates, result["smoothed"], color="steelblue", linewidth=2,
                label="EWMA of mentions")
    ax_top.scatter(rise_days.index, rise_days["smoothed"], color="green", marker="^",
                   s=60, zorder=5, label="rise (take-off)")
    ax_top.scatter(fall_days.index, fall_days["smoothed"], color="crimson", marker="v",
                   s=60, zorder=5, label="fall (sell-off)")
    for d in peak_days.index:
        ax_top.axvline(d, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    for d in trough_days.index:
        ax_top.axvline(d, color="gray", linestyle=":", linewidth=1, alpha=0.4)
    ax_top.set_title(ticker + " - mentions per day")
    ax_top.set_ylabel("mentions")
    ax_top.legend()
    ax_top.grid(True, alpha=0.3)

    # --- Bottom: the first derivative (velocity) and the thresholds ---
    ax_bottom.plot(dates, result["velocity"], color="darkorange",
                   label="first derivative of the EWMA (velocity)")
    ax_bottom.axhline(threshold, color="green", linestyle="--",
                      label="rise threshold")
    ax_bottom.axhline(fall_threshold, color="crimson", linestyle="--",
                      label="fall threshold")
    ax_bottom.axhline(0, color="black", linewidth=0.6)
    ax_bottom.scatter(rise_days.index, rise_days["velocity"], color="green", marker="^", zorder=5)
    ax_bottom.scatter(fall_days.index, fall_days["velocity"], color="crimson", marker="v", zorder=5)
    ax_bottom.set_title("First derivative of the EWMA (not raw mentions) - how fast attention is growing/shrinking")
    ax_bottom.set_ylabel("change vs prior day")
    ax_bottom.set_xlabel("date")
    ax_bottom.legend()
    ax_bottom.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print("Saved plot:", out_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Detect mention take-off (inflection) points")
    parser.add_argument("--in", dest="inp", required=True, help="daily counts .parquet or .csv")
    parser.add_argument("--ticker", required=True, help="ticker to analyse, e.g. GME")
    parser.add_argument("--smooth", type=int, default=3, help="rolling average window in days")
    parser.add_argument("--k", type=float, default=2.0, help="std-devs above normal = a spike")
    parser.add_argument("--plot-out", default=None, help="PNG path (default: <ticker>_inflection.png)")
    parser.add_argument("--csv-out", default=None, help="optional CSV of flagged inflection days")
    args = parser.parse_args(argv)

    df = load_daily_counts(args.inp)
    series = build_daily_series(df, args.ticker)
    if series is None:
        print("Ticker", args.ticker, "not found in the data.")
        return 1

    result = compute_inflection(series, args.smooth, args.k)

    plot_path = args.plot_out or (args.ticker + "_inflection.png")
    plot_inflection(result, args.ticker, plot_path)

    # Print the flagged days so you can see them in the terminal.
    flagged = result[result["is_inflection"]]
    print("\nInflection days for", args.ticker, "(", len(flagged), "found ):")
    for date, row in flagged.iterrows():
        print("  ", date.strftime("%Y-%m-%d"),
              "| mentions:", int(row["count"]),
              "| velocity:", round(row["velocity"], 1))

    if args.csv_out:
        flagged.to_csv(args.csv_out)
        print("Saved flagged days to", args.csv_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
