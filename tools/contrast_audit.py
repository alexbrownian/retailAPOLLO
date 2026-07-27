"""Measure text contrast on the live dashboard instead of eyeballing it.

WHY THIS EXISTS
---------------
"White text on a white background" is invisible, and invisible text cannot be
found by reading the stylesheet: the colour a browser actually paints is the
outcome of specificity, inheritance and Streamlit's own theme CSS all landing
on the same node.  Two rules that each look correct in isolation produce an
unreadable label when one supplies the colour and the other supplies the
background.  The only trustworthy way to find those pairs is to render the
page and ask the browser what it computed.

WHAT IT DOES
------------
Boots ``dashboard.py`` headlessly, opens it in Chromium, walks every element
that paints its own text, and for each one resolves

  * the computed text colour, and
  * the *effective* background: the first ancestor with a non-transparent
    background, composited down, because an element with
    ``background: rgba(0,0,0,0)`` is painted by whatever sits behind it,

then scores the pair with the WCAG 2.1 relative-luminance contrast ratio.
Anything at or below ``--fail-below`` is reported with its tag, testid, class
list and a sample of its text, which is enough to find the offending rule.

WHY WCAG AND WHY 3.0
--------------------
The ratio is a published, reproducible formula rather than a judgement, so the
same page scores the same on any machine - which is what makes this a test and
not an opinion.  The default gate is 3.0:1, WCAG's bar for large text.  It is
set there deliberately: this audit is hunting *unreadable* text, and a muted
grey label that lands at 4.2 against white is a design choice already signed
off in the brief, not a defect.  Raising the gate to 4.5 turns the report into
a list of things we chose on purpose, and a report full of expected entries
gets ignored - so the gate is set where every hit is a real bug.

USAGE
-----
    python tools/contrast_audit.py                 # every tab, gate at 3.0
    python tools/contrast_audit.py --fail-below 4.5 # stricter, incl. body text
    python tools/contrast_audit.py --tabs 0 3       # only these tab indices
    python tools/contrast_audit.py --shot audit.png # also save a screenshot

Exit status is 1 if any pair fails, so it can gate a commit.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The audit runs against a real browser, so the page needs a real server.  A
# free port is picked rather than hard-coded because a developer very often
# already has the dashboard open on 8501 while reading this report.
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# --------------------------------------------------------------------------
# The measurement itself, as a page script.
#
# Two details matter for correctness:
#
#   1. Only elements that paint their OWN text are scored.  A <div> wrapping a
#      paragraph has the paragraph's words in its textContent but paints none
#      of them, so scoring it double-counts and, worse, attributes the failure
#      to the wrong node.  The filter keeps elements with at least one direct
#      child text node.
#   2. Backgrounds are COMPOSITED, not merely inherited.  A semi-transparent
#      background over navy is not navy, so the walk accumulates alpha up the
#      ancestor chain until it reaches something opaque, defaulting to white
#      at the document root.
# --------------------------------------------------------------------------
PAGE_SCRIPT = r"""
() => {
  const parse = (s) => {
    const m = (s || "").match(/rgba?\(([^)]+)\)/);
    if (!m) return null;
    const p = m[1].split(",").map((x) => parseFloat(x.trim()));
    return { r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1 };
  };
  const over = (fg, bg) => ({          // composite fg (with alpha) onto bg
    r: fg.r * fg.a + bg.r * (1 - fg.a),
    g: fg.g * fg.a + bg.g * (1 - fg.a),
    b: fg.b * fg.a + bg.b * (1 - fg.a),
    a: 1,
  });
  const lum = (c) => {
    const f = (v) => {
      v /= 255;
      return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    };
    return 0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b);
  };
  const ratio = (a, b) => {
    const la = lum(a), lb = lum(b);
    return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
  };
  const effectiveBg = (el) => {
    let acc = null;
    // The element's own centre is the point whose backdrop matters.  An
    // ancestor is only BEHIND the text if it actually covers that point:
    // Streamlit's slider paints a navy thumb and hangs the value label off
    // it as a DOM child positioned ABOVE the thumb, so a naive walk up the
    // tree reported navy-on-navy for a label that is plainly readable on the
    // page.  Rejecting non-overlapping ancestors removes that whole class of
    // false positive - and false positives are what get an audit ignored.
    const r = el.getBoundingClientRect();
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    for (let n = el; n; n = n.parentElement) {
      if (n !== el) {
        const b = n.getBoundingClientRect();
        const covers = cx >= b.left && cx <= b.right
                    && cy >= b.top && cy <= b.bottom;
        if (!covers) continue;
      }
      const c = parse(getComputedStyle(n).backgroundColor);
      if (!c || c.a === 0) continue;
      acc = acc === null ? c : over(acc, c);
      if (acc.a >= 0.999) return acc;
    }
    const white = { r: 255, g: 255, b: 255, a: 1 };
    return acc === null ? white : over(acc, white);
  };
  const ownsText = (el) => {
    for (const n of el.childNodes)
      if (n.nodeType === 3 && n.textContent.trim().length) return true;
    return false;
  };

  // Where the FG comes from differs between HTML and SVG: HTML text is
  // painted with `color`, SVG text with `fill`.  Reading `color` on an SVG
  // <text> reports an inherited value the browser never paints, so a white
  // label on a white chart would score as a pass.  Plotly draws every chart
  // in this product as SVG, so this branch is the difference between
  // auditing the dashboard and auditing only its margins.
  const isSvg = (el) => el.namespaceURI === "http://www.w3.org/2000/svg";
  const textColour = (el, cs) => (isSvg(el) ? cs.fill : cs.color);

  // A short ancestor path makes a finding actionable: the failing node is
  // often an anonymous <p> or <text>, and the CSS rule that caused it is
  // attached to something three levels up.
  const pathOf = (el) => {
    const bits = [];
    for (let n = el, i = 0; n && i < 5; n = n.parentElement, i++) {
      const tid = n.getAttribute && n.getAttribute("data-testid");
      const cls = (n.className && n.className.toString)
        ? n.className.toString().trim().split(/\s+/)[0] : "";
      bits.unshift(n.tagName.toLowerCase()
        + (tid ? `[${tid}]` : (cls ? `.${cls}` : "")));
    }
    return bits.join(" > ");
  };

  const out = [];
  for (const el of document.querySelectorAll("body *, body svg *")) {
    if (!ownsText(el)) continue;
    const cs = getComputedStyle(el);
    if (cs.visibility === "hidden" || cs.display === "none") continue;
    if (parseFloat(cs.opacity) < 0.1) continue;
    const box = el.getBoundingClientRect();
    if (box.width < 2 || box.height < 2) continue;     // not actually painted
    const fgRaw = parse(textColour(el, cs));
    if (!fgRaw) continue;
    const bg = effectiveBg(el);
    const fg = fgRaw.a < 0.999 ? over(fgRaw, bg) : fgRaw;
    out.push({
      ratio: Math.round(ratio(fg, bg) * 100) / 100,
      color: textColour(el, cs),
      path: pathOf(el),
      bg: `rgb(${Math.round(bg.r)}, ${Math.round(bg.g)}, ${Math.round(bg.b)})`,
      tag: el.tagName.toLowerCase(),
      testid: el.getAttribute("data-testid") || "",
      cls: (el.className && el.className.toString ? el.className.toString() : "").slice(0, 70),
      size: cs.fontSize,
      weight: cs.fontWeight,
      text: el.textContent.trim().replace(/\s+/g, " ").slice(0, 70),
    });
  }
  return out;
}
"""


def audit(port: int, tab_indices: list[int] | None, gate: float,
          shot: Path | None) -> list[dict]:
    from playwright.sync_api import sync_playwright

    findings: dict[tuple, dict] = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        # A wide viewport matters: the layout is a 1400px container, and at
        # phone width Streamlit stacks columns, which changes which elements
        # are painted at all.  Audit the layout a PM will actually see.
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.goto(f"http://127.0.0.1:{port}", wait_until="load", timeout=120_000)
        # Streamlit renders in two stages: the shell, then the script's output
        # streamed in.  Waiting for the tab strip proves the second stage ran.
        #
        # SELECTOR: Streamlit renders each tab as a DIV carrying
        # data-testid="stTab", NOT as a BaseWeb-attributed button - querying
        # [data-baseweb="tab"] finds nothing and the audit times out having
        # measured a blank page.  Same lesson, and the same two-step fallback,
        # as tools/dashboard_shots.py.
        page.wait_for_selector('[data-testid="stTab"], [role="tab"]',
                               timeout=180_000)
        page.wait_for_timeout(6_000)

        def _tabs():
            t = page.query_selector_all('[data-testid="stTab"]')
            return t or page.query_selector_all('[role="tab"]')

        tabs = _tabs()
        order = list(tab_indices) if tab_indices is not None \
            else list(range(len(tabs)))
        print(f"tabs found: {len(tabs)}")

        for i in order:
            tabs = _tabs()
            if i >= len(tabs):
                print(f"  skip tab {i}: only {len(tabs)} present")
                continue
            label = (tabs[i].inner_text() or "").strip()[:40]
            print(f"  tab {i} [{label}] ...", flush=True)
            tabs[i].click(timeout=15_000)
            page.wait_for_timeout(3_500)
            # Expanders hide most of the explanation copy, and hidden text is
            # exactly where a colour bug survives review - so they are opened
            # before measuring.
            #
            # OPENED WITH .open, NOT .click().  Playwright's click waits for
            # the node to be stable, visible and unobscured, and retries until
            # its timeout; on a long page most expanders are below the fold
            # and every one of them burned the full default 30s, which turned
            # a two-minute audit into an apparent hang.  These are native
            # <details> elements, so setting the attribute opens them with no
            # scrolling, no hit-testing and no rerun.
            n_open = page.evaluate("""() => {
              let n = 0;
              for (const d of document.querySelectorAll('details')) {
                if (!d.open) { d.open = true; n++; }
              }
              return n;
            }""")
            page.wait_for_timeout(1_200)
            print(f"    opened {n_open} collapsed section(s)", flush=True)

            rows = page.evaluate(PAGE_SCRIPT)
            bad = [r for r in rows if r["ratio"] <= gate]
            print(f"    {len(rows)} text nodes, "
                  f"{len(bad)} at or below {gate}:1", flush=True)
            for r in bad:
                # De-duplicated on the STYLE, not the node: one bad rule shows
                # up on dozens of elements and the fix is the same rule, so a
                # per-node list buries the finding under its own repetition.
                key = (r["color"], r["bg"], r["tag"], r["testid"], r["cls"],
                       r["path"])
                r = dict(r, tab=label)
                findings.setdefault(key, r)

            if shot is not None and i == order[0]:
                page.screenshot(path=str(shot), full_page=True)
        browser.close()
    return sorted(findings.values(), key=lambda r: r["ratio"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fail-below", type=float, default=3.0,
                    help="report pairs at or below this WCAG ratio (default 3.0)")
    ap.add_argument("--tabs", type=int, nargs="*", default=None,
                    help="tab indices to audit (default: all)")
    ap.add_argument("--shot", type=Path, default=None,
                    help="save a full-page screenshot of the first tab here")
    ap.add_argument("--json", type=Path, default=None,
                    help="also write the findings as JSON")
    args = ap.parse_args()

    port = _free_port()
    env = dict(os.environ, MPLBACKEND="Agg")
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "dashboard.py",
         "--server.port", str(port), "--server.headless", "true",
         "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false"],
        cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        # Poll the port rather than sleeping a fixed amount: the app imports
        # pandas, sklearn and the whole analytics package, and that time is
        # very different on a laptop and in CI.
        deadline = time.time() + 180
        while time.time() < deadline:
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    break
            if proc.poll() is not None:
                print(proc.stdout.read() if proc.stdout else "")
                print("streamlit exited before serving")
                return 2
            time.sleep(1)
        else:
            print("streamlit did not start within 180s")
            return 2

        findings = audit(port, args.tabs, args.fail_below, args.shot)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()

    print(f"\n=== {len(findings)} distinct failing style(s) "
          f"at or below {args.fail_below}:1")
    for r in findings:
        print(f"\n  ratio {r['ratio']}:1   {r['color']}  on  {r['bg']}")
        print(f"    tab      {r['tab']}")
        print(f"    element  <{r['tag']}> testid={r['testid'] or '-'} "
              f"class={r['cls'] or '-'}")
        print(f"    path     {r.get('path', '-')}")
        print(f"    type     {r['size']} / weight {r['weight']}")
        print(f"    text     {r['text']!r}")

    if args.json is not None:
        args.json.write_text(json.dumps(findings, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")

    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
