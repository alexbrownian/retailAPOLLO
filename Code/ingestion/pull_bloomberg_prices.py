#!/usr/bin/env python
"""Alias entry point: runs ``pull_prices.py`` (Bloomberg and Tiingo) with
``--provider bloomberg`` unless a provider is given."""
import runpy
import sys

if __name__ == "__main__":
    if not any(a.startswith("--provider") for a in sys.argv[1:]):
        sys.argv.extend(["--provider", "bloomberg"])
    runpy.run_path(__file__.replace("pull_bloomberg_prices.py",
                                    "pull_prices.py"), run_name="__main__")
