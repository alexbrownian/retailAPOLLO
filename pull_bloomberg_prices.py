#!/usr/bin/env python
"""Compatibility entry point. The price pull now lives in ``pull_prices.py``
and supports Bloomberg and yfinance; this name runs it with
``--provider bloomberg``."""
import runpy
import sys

if __name__ == "__main__":
    if not any(a.startswith("--provider") for a in sys.argv[1:]):
        sys.argv.extend(["--provider", "bloomberg"])
    runpy.run_path(__file__.replace("pull_bloomberg_prices.py",
                                    "pull_prices.py"), run_name="__main__")
