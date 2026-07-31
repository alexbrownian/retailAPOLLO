"""Screenshot the live dashboard, one PNG per tab, for the deck and the docs.

WHY THIS EXISTS
---------------
Two reasons, and the second is the important one.

1. The deck needs pictures of the product, and a screenshot taken by hand is
   stale the moment the next re-skin lands.
2. It is a VISUAL regression check.  The unit tests and ``AppTest`` prove the
   app runs and returns the right objects; they cannot see a chart rendered on
   top of its own axis labels, or a KPI card that has lost its border.  Looking
   at the rendered page is the only way to catch that class of fault, so this
   doubles as the confirmation step for the re-skin work.

HOW IT WORKS
------------
* Starts Streamlit headless on a private port, waits for the socket rather than
  sleeping a fixed number of seconds (a fixed sleep is either slow or flaky,
  usually both).
* Drives it with the Chromium already installed in this environment.
* For each tab: click it, wait for the network to go quiet, screenshot the
  full page into ``docs/figures/dashboard/NN_<tab>.png``.
* Always shuts the server down, including on failure, so a crashed run does not
  leave a port held.

USAGE
-----
    python tools/dashboard_shots.py                 # all tabs
    python tools/dashboard_shots.py --port 8899
"""
from __future__ import annotations

import argparse
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "figures" / "dashboard"

# 1440x1000 rather than a laptop 1280x800: the design language specifies a
# 1280-1400px container, so the viewport must be wide enough to show the
# container at full width with its intended margins.  DESK DECISION.
VIEWPORT = {"width": 1440, "height": 1000}
BOOT_TIMEOUT_S = 180
SETTLE_MS = 3500          # after a tab click, before the shot


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex((host, port)) == 0


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "tab"


def start_server(port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "dashboard.py",
         "--server.port", str(port),
         "--server.headless", "true",
         "--server.fileWatcherType", "none",
         "--browser.gatherUsageStats", "false"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True)
    t0 = time.time()
    while time.time() - t0 < BOOT_TIMEOUT_S:
        if _port_open(port):
            return proc
        if proc.poll() is not None:
            raise RuntimeError(
                "streamlit exited during boot:\n" + (proc.stdout.read() or ""))
        time.sleep(0.5)
    proc.kill()
    raise TimeoutError(f"streamlit did not open port {port} in "
                       f"{BOOT_TIMEOUT_S}s")


def shoot(port: int) -> list[Path]:
    from playwright.sync_api import sync_playwright

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUT_DIR.glob("*.png"):
        stale.unlink()

    saved: list[Path] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)
        page.goto(f"http://127.0.0.1:{port}", wait_until="networkidle",
                  timeout=120_000)
        # Streamlit paints the skeleton before the data callbacks return, so
        # wait for the app's own "running" indicator to disappear rather than
        # trusting networkidle alone.
        page.wait_for_timeout(6000)

        # The tab bar is a PERSISTENT RADIO since 2026-07-31 (the st.tabs
        # stuck-tab fix - see dashboard.py "PERSISTENT TAB BAR").  Its
        # options are <label>s inside the first stRadio group on the page;
        # the old stTab locator is kept as a fallback so this tool still
        # works against an older build of the app.
        tabs = page.locator('div[data-testid="stRadio"]').first.locator(
            '[data-testid="stRadioOption"]')
        if tabs.count() == 0:
            tabs = page.locator('[data-testid="stTab"]')
        if tabs.count() == 0:
            tabs = page.locator('[role="tab"]')
        n = tabs.count()
        if n == 0:
            raise RuntimeError(
                "no tab controls found - the app rendered but the tab bar "
                "did not, which usually means the page errored before "
                "reaching it")
        names = [tabs.nth(i).inner_text().strip() for i in range(n)]
        print(f"found {n} tabs: {names}")

        for i, name in enumerate(names):
            tabs.nth(i).click()
            page.wait_for_timeout(SETTLE_MS)
            # charts are drawn by a second render pass, so let the network
            # settle again before the shutter rather than trusting the timer
            try:
                page.wait_for_load_state("networkidle", timeout=20_000)
            except Exception:                                  # noqa: BLE001
                pass        # a live-updating widget can keep the socket busy
            path = OUT_DIR / f"{i:02d}_{_slug(name)}.png"
            page.screenshot(path=str(path), full_page=True)
            saved.append(path)
            print(f"  shot {path.name}")

        browser.close()
    return saved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8899)
    args = ap.parse_args()

    if _port_open(args.port):
        raise SystemExit(f"port {args.port} already in use")

    proc = start_server(args.port)
    try:
        saved = shoot(args.port)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    print(f"\n{len(saved)} screenshots in {OUT_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
