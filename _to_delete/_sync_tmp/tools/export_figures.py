"""Export every notebook figure to PNG, for the slide deck and the report.

WHY THIS EXISTS
---------------
The deck must show the SAME charts the research record shows.  Re-drawing a
chart for a slide is how a presentation ends up making a claim the notebook
does not support: the axes get prettier, a filter creeps in, and the figure in
front of the PM is no longer the figure that was defended.

So the deck does not redraw anything.  This script executes a notebook exactly
as the headless verification does, with ``plt.show`` intercepted, and writes
every figure the notebook produced to ``docs/figures/<nb>/NN_<slug>.png``.
The figure files are therefore a pure by-product of a real notebook run - if
the notebook changes, re-running this changes the slides.

HOW IT WORKS
------------
* ``plt.show`` is replaced by a function that saves the CURRENT figure and
  closes it.  Nothing else about the notebook is altered - same code, same
  seeds, same data, same order.
* The slug comes from the axes title of the figure, so filenames are
  self-describing (``04_does-any-single-feature.png``) rather than positional
  only.  The leading counter keeps them in notebook order.
* ``MPLBACKEND=Agg`` is forced before pyplot is imported, so this runs with no
  display attached.
* Notebook stdout is left alone and streams to the console, because a run that
  quietly failed halfway should look like a failure, not like a short export.

USAGE
-----
    python tools/export_figures.py 01 02 05        # selected notebooks
    python tools/export_figures.py --all           # every notebook
    python tools/export_figures.py --list          # what is already exported
"""
from __future__ import annotations

import os
import re
import runpy
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib                                                # noqa: E402
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt                                  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
FIG_DIR = ROOT / "docs" / "figures"

# 160 dpi: sharp on a projector at slide width without producing 4 MB PNGs
# that make the .tex repo unpleasant to move around.  CONVENTION.
DPI = 160


def _jupyter_display(*objs) -> None:
    """Stand-in for the ``display`` that IPython injects into a notebook.

    Not a workaround.  ``display`` is a name Jupyter puts in the notebook's
    namespace for free, so a notebook that calls it is not broken - it is
    simply relying on its host.  This exporter IS the host for the duration of
    the run, so it supplies the name.  Rendering as text keeps the console log
    of an export readable and identical to the headless verification run.
    """
    for obj in objs:
        to_string = getattr(obj, "to_string", None)
        print(to_string() if callable(to_string) else obj)


def _slug(text: str, limit: int = 46) -> str:
    """A filesystem-safe, human-readable stub from a chart title."""
    text = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (text[:limit].rstrip("-") or "figure")


def _figure_title(fig) -> str:
    """Best human name for a figure: suptitle, else the first axes title."""
    sup = getattr(fig, "_suptitle", None)
    if sup is not None and sup.get_text().strip():
        return sup.get_text().strip().split("\n")[0]
    for ax in fig.get_axes():
        if ax.get_title().strip():
            return ax.get_title().strip().split("\n")[0]
    return "figure"


def export(nb_path: Path) -> list[Path]:
    """Run one notebook and save every figure it shows. Returns the paths."""
    out_dir = FIG_DIR / nb_path.stem.split("_")[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.png"):
        stale.unlink()          # a re-export must not leave last run's charts

    saved: list[Path] = []
    counter = {"n": 0}
    real_show = plt.show

    def capture_show(*_args, **_kwargs):
        fig = plt.gcf()
        if not fig.get_axes():          # a bare figure with nothing drawn
            plt.close(fig)
            return
        counter["n"] += 1
        name = f"{counter['n']:02d}_{_slug(_figure_title(fig))}.png"
        path = out_dir / name
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        saved.append(path)
        plt.close(fig)

    plt.show = capture_show
    cwd = Path.cwd()
    try:
        # run from notebooks/ so the notebook's own ROOT detection behaves
        # exactly as it does under Jupyter
        os.chdir(NB_DIR)
        sys.path.insert(0, str(ROOT))
        runpy.run_path(str(nb_path), run_name="__main__",
                       init_globals={"display": _jupyter_display})
    finally:
        plt.show = real_show
        os.chdir(cwd)
        plt.close("all")
    return saved


def main(argv: list[str]) -> int:
    all_nbs = sorted(NB_DIR.glob("[0-9][0-9]_*.py"))
    if "--list" in argv:
        for d in sorted(FIG_DIR.glob("*")):
            pngs = sorted(d.glob("*.png"))
            print(f"{d.name}: {len(pngs)} figures")
            for p in pngs:
                print(f"    {p.name}")
        return 0

    if "--all" in argv:
        targets = all_nbs
    else:
        wanted = [a for a in argv if a.isdigit()]
        if not wanted:
            print(__doc__)
            return 2
        targets = [p for p in all_nbs if p.stem.split("_")[0] in wanted]

    for nb in targets:
        t0 = time.time()
        print(f"\n=== exporting {nb.name}", flush=True)
        try:
            saved = export(nb)
        except Exception as exc:                       # noqa: BLE001
            # keep going: one broken notebook should not cost the whole export
            print(f"!!! {nb.name} FAILED after {time.time()-t0:.0f}s: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            continue
        print(f"--- {nb.name}: {len(saved)} figures in "
              f"{time.time()-t0:.0f}s", flush=True)
        for p in saved:
            print(f"      {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
