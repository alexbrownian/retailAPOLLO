#!/usr/bin/env python3
"""Does this repo reference anything that is not actually here?

    python tools/verify_deps.py            # from the project root
    python tools/verify_deps.py --quiet    # exit code only (CI / pre-commit)

WHY THIS EXISTS
---------------
On 2026-07-29 a routine `python update_data.py` died on

    ImportError: cannot import name 'pipeline_budget' from 'src'

and the investigation found that `src/pipeline_budget.py` was not on disk, not
in any git commit, and had no stale `.pyc` anywhere - i.e. it had NEVER been
imported successfully - while ARCHITECTURE.md and docs/RESEARCH_RECORD.md both
described it in detail, and four other files called into it.  Three further
gaps rode along invisibly: two missing `src/config.py` constants, a CLI flag
`fetch_all.py` did not accept, and a `default_lookback_days` import in
dashboard.py that a bare `except` had been swallowing for days.

DOCUMENTATION ASSERTING THAT CODE EXISTS IS NOT EVIDENCE THAT IT DOES, and
neither is a green dashboard: Python resolves imports lazily, so a broken
reference on a path nobody ran that day is invisible until the day somebody
runs it.  This script makes that class of breakage cheap to find, statically,
in under a second, without importing anything or touching the network.

WHAT IT CHECKS (five distinct ways a reference can dangle)
----------------------------------------------------------
1. UNRESOLVED LOCAL MODULES     - `import ingestion.foo` with no foo.py
2. IMPORTED NAMES              - `from src.config import BAR` with no BAR
3. ATTRIBUTES ON LOCAL MODULES - `config.BAZ` where src/config.py has no BAZ
4. REFERENCED REPO PATHS       - a repo path in a string, no such file there
5. CLI FLAGS                   - a script handed --flag its argparse rejects

WHAT IT DOES NOT CHECK: third-party packages (see requirements.txt), runtime
data files, and anything that only exists dynamically (getattr, globals()).
It is a fast structural check, not a type checker - a clean run means no
DANGLING reference, not that the code is correct.

Exit code 0 = clean, 1 = at least one finding.  Run it after any session that
edits across module boundaries.
"""

import argparse
import ast
import os
import re

SKIP_DIRS = ("_to_delete", ".git", "__pycache__", "node_modules",
             ".ipynb_checkpoints", "_salvaged_originals", "venv", ".venv",
             ".pytest_cache", "build", "dist")

# `config.py` inside a sentence is prose, not an attribute access; the same
# goes for a module named in a docstring.  Filtering these keeps the report
# free of noise that trains the reader to ignore it.
PROSE = {"py", "toml", "json", "md", "txt", "csv", "get"}


def _walk(root):
    py, allf = [], set()
    for r, d, f in os.walk(root):
        d[:] = [x for x in d if not x.startswith(SKIP_DIRS)]
        for n in f:
            p = os.path.relpath(os.path.join(r, n), root).replace("\\", "/")
            allf.add(p)
            if n.endswith(".py"):
                py.append(p)
    return py, allf


def _top_level_names(body, ns):
    """Names a module exports.  Walks into if/try/with/for bodies because
    config-style conditional definitions and `try: import x` fallbacks are
    real definitions, and flagging them would be a false alarm."""
    for n in body:
        if isinstance(n, ast.Assign):
            for tg in n.targets:
                elts = (tg.elts if isinstance(tg, (ast.Tuple, ast.List))
                        else [tg])
                for e in elts:
                    if isinstance(e, ast.Name):
                        ns.add(e.id)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            ns.add(n.target.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                            ast.ClassDef)):
            ns.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                ns.add(a.asname or a.name.split(".")[0])
        elif isinstance(n, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
            for attr in ("body", "orelse", "finalbody"):
                _top_level_names(getattr(n, attr, []) or [], ns)
            for h in getattr(n, "handlers", []) or []:
                _top_level_names(h.body, ns)


# Directory prefixes worth checking even when the directory is absent -
# absence is exactly the failure this catches. Extend the list rather than
# deriving it, so a deleted folder stays visible to the sweep.
_CITED_DIRS = ("src", "analytics", "ingestion", "tools", "helper", "docs",
               "notebooks", "config", "tests")


# A citation that already SAYS the file is absent is documentation, not a
# dangling reference. Without this the sweep punishes exactly the honest
# behaviour it is meant to encourage - notebook 07 says plainly that
# a since-removed module does not exist, and that sentence should not
# read as a defect.
# PRESENT ON THE DESK MACHINE, ABSENT FROM SOME CLONES.
#
# Learned the hard way on 2026-08-05: this checker was run inside an
# incomplete working copy, reported `helper/` and four docs as dangling,
# and five citations were "corrected" to say the directory did not exist.
# It does exist - it holds research_charts.py and find_emerging_terms.py.
#
# The tool cannot tell a DELETED file from an UN-CLONED one; both are
# simply not on disk. So paths known to live on the full repository are
# listed here rather than being reported every run and eventually
# ignored. Remove an entry only after confirming on the desk machine
# that the file has genuinely gone.
_DESK_ONLY = ("helper/", "docs/panel_review_latest.md",
              "docs/HANDOFF_PROMPT.md", "docs/LIVE_INGESTION.md",
              "docs/RESEARCH_RECORD.md", "docs/DATA_FLOW.tex")

_KNOWN_ABSENT = ("not in this repo", "does not exist", "no longer",
                 "is not on disk", "never present", "absent",
                 "was removed", "not present", "was deleted")


def _cited_paths(txt, pkgs=()):
    """Every repo-relative file path this text mentions, quoted or not.

    Two exclusions, both deliberate:
      * anything under `data/` - those are RUNTIME artefacts. Whether
        `data/processed/posts.parquet` exists depends on whether the
        pipeline has run on this machine, so its absence is never a
        broken reference and flagging it would train the reader to
        ignore this check.
      * any line that already declares the file missing (_KNOWN_ABSENT).
    """
    alt = "|".join(sorted((set(_CITED_DIRS) | set(pkgs)) - {"data"}))
    pat = (r"(?<![\w/.-])((?:%s)/[\w./-]*\.(?:py|json|txt|md|csv|parquet))"
           % alt)
    # The marker is looked for in a WINDOW around the citation, not on the
    # same line. Prose wraps: "removed the same day: the Index & factors
    # tab, the basket-breadth module, the sp500 theme" puts the verb
    # two lines above the path, and a same-line-only check would flag the
    # sentence that is doing exactly the right thing.
    lines = txt.splitlines()
    out = set()
    for i, line in enumerate(lines):
        window = " ".join(lines[max(0, i - 2):i + 3]).lower()
        if any(k in window for k in _KNOWN_ABSENT):
            continue
        out.update(m.rstrip(".,;:)") for m in re.findall(pat, line))
    return out


def sweep_docs(root):
    """Cited paths in MARKDOWN. Docs are where operator instructions live,
    so a dangling path here is a command somebody will run and watch fail -
    which is precisely how the deleted research-charts command survived in
    the RUNBOOK."""
    _py, allf = _walk(root)
    out = []
    for rel in sorted(allf):
        if not rel.endswith(".md") or rel.startswith("_to_delete"):
            continue
        try:
            txt = open(os.path.join(root, rel), encoding="utf-8").read()
        except OSError:
            continue
        for m in _cited_paths(txt):
            if m not in allf and not m.startswith(_DESK_ONLY):
                out.append((rel, m))
    return out


def sweep(root):
    py, allf = _walk(root)
    pkgs = {p.split("/")[0] for p in py if "/" in p}
    rootmods = {p[:-3] for p in py if "/" not in p}
    trees, names = {}, {}
    findings = {k: [] for k in ("module", "name", "attr", "path", "flag")}

    for p in py:
        try:
            trees[p] = ast.parse(open(os.path.join(root, p),
                                      encoding="utf-8").read())
        except (OSError, SyntaxError) as e:
            findings["module"].append((p, f"WILL NOT PARSE: {e}"))
            continue
        ns = set()
        _top_level_names(trees[p].body, ns)
        names[p[:-3]] = ns

    def resolve(mod):
        q = mod.replace(".", "/")
        for c in (q + ".py", q + "/__init__.py"):
            if c in allf:
                return c[:-3]
        return None

    def is_local(mod):
        return mod.split(".")[0] in pkgs or mod in rootmods

    # ---- argparse flags each script accepts (for check 5) ----
    accepts = {}
    for p, t in trees.items():
        fl = set()
        for n in ast.walk(t):
            if (isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "add_argument"):
                for a in n.args:
                    if isinstance(a, ast.Constant) and isinstance(a.value,
                                                                  str):
                        fl.add(a.value)
        accepts[p] = fl

    for p, t in trees.items():
        txt = open(os.path.join(root, p), encoding="utf-8").read()
        alias2mod = {}
        for n in ast.walk(t):
            # ---- 1 + 2 ----
            if isinstance(n, ast.Import):
                for a in n.names:
                    if not is_local(a.name):
                        continue
                    m = resolve(a.name)
                    if m is None:
                        findings["module"].append((p, a.name))
                    else:
                        alias2mod[a.asname or a.name.split(".")[0]] = m
            elif (isinstance(n, ast.ImportFrom) and n.level == 0
                  and n.module and is_local(n.module)):
                m = resolve(n.module)
                if m is None:
                    findings["module"].append((p, n.module))
                    continue
                for a in n.names:
                    if a.name == "*":
                        continue
                    sub = resolve(n.module + "." + a.name)
                    if sub:
                        alias2mod[a.asname or a.name] = sub
                    elif a.name not in names.get(m, set()):
                        findings["name"].append((p, f"{n.module}.{a.name}"))
        # ---- 3 ----
        for alias, mod in alias2mod.items():
            if mod == p[:-3]:
                continue
            known = names.get(mod, set())
            pat = r"(?<![\w.])%s\.([A-Za-z_][A-Za-z0-9_]*)" % re.escape(alias)
            for at in set(re.findall(pat, txt)):
                if at in known or at in PROSE:
                    continue
                if resolve(mod.replace("/", ".") + "." + at):
                    continue
                findings["attr"].append((p, f"{alias}.{at}"))
        # ---- 4 ----
        # WIDENED 2026-08-05, after an audit found FOURTEEN cited paths
        # that do not exist while this tool reported "Nothing dangles".
        #
        # The old pattern had two structural blind spots, and every one of
        # the fourteen sat in one of them:
        #   * it only matched inside QUOTED STRING LITERALS, so a path in
        #     a comment or a docstring was invisible - and that is where
        #     most cited paths live, because they are instructions to a
        #     human, not arguments to a function;
        #   * `pkgs` was derived from directories that EXIST, so a
        #     reference to a directory that had been deleted (`helper/`,
        #     cited five times including a RUNBOOK command) could never
        #     be a prefix it looked for. The one broken directory was the
        #     one it could not see.
        #
        # It now scans the raw text of the file rather than only its
        # literals, and it knows the directory names that have EVER been
        # cited rather than only those present. Markdown is swept
        # separately in `sweep_docs` for the same reason.
        for m in _cited_paths(txt, pkgs):
            if m not in allf and not m.startswith(_DESK_ONLY):
                findings["path"].append((p, m))
        # ---- 5 ----
        # only scripts that ACTUALLY parse argv: a script with no
        # add_argument at all (a Streamlit entry point, a plain module) is
        # not an argparse CLI, and flags near its name belong to whatever is
        # launching it.
        argv_cli = {k: v for k, v in accepts.items() if v}
        for target, flag in _command_flags(t, argv_cli):
            if flag not in argv_cli[target]:
                findings["flag"].append((p, f"{target} <- {flag}"))
    return findings


def _strings(node):
    return {n.value for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def _is_leaf_seq(node):
    """A list/tuple with no list/tuple inside it - i.e. ONE argv, not a list
    of argvs.  dashboard.py builds pipeline plans as

        [(["pull_bloomberg_prices.py"], env),
         (["update_data.py", "--start", s, "--skip-prices"], None)]

    and reading the outer sequence as one command would blame
    pull_bloomberg_prices.py for update_data.py's flags."""
    return not any(isinstance(c, (ast.List, ast.Tuple))
                   for c in ast.walk(node) if c is not node)


def _command_flags(tree, accepts):
    """Yield (target_script, --flag) pairs from ARGV LISTS specifically.

    Deliberately NOT a text window around the script name.  The first version
    scanned 400 characters after any script path and reported
    `append_live_abstracted.py <- --abstracted` in fetch_all.py, where the
    path and the flag are in two unrelated statements a few lines apart and no
    flag is passed at all.  A checker that cries wolf is worse than no checker,
    because the reader learns to skim its output.

    So a flag counts only when it travels in the SAME leaf argv list as the
    script -

        cmd = [py, "ingestion/fetch_all.py", "--no-merge"]

    - or when it is later appended to that same list variable, which is how
    argv is built conditionally:

        cmd.append("--skip-comments")
        cmd += ["--comment-pages", str(n)]

    `accepts` must contain ONLY scripts that actually parse argv.  A Streamlit
    entry point has no argparse at all, so `streamlit run dashboard.py
    --server.port 8501` passes flags to STREAMLIT, not to the script, and
    checking them against the script would be a category error."""
    targets = set(accepts)
    var2target = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple)) and _is_leaf_seq(node):
            here = _strings(node)
            hits = here & targets
            if len(hits) != 1:
                continue        # ambiguous: two scripts, one flag list
            target = next(iter(hits))
            for s in here:
                if s.startswith("--"):
                    yield target, s
        # remember the variable an argv list was assigned to
        if (isinstance(node, ast.Assign)
                and isinstance(node.value, (ast.List, ast.Tuple))
                and _is_leaf_seq(node.value)):
            hits = _strings(node.value) & targets
            if len(hits) == 1:
                for tg in node.targets:
                    if isinstance(tg, ast.Name):
                        var2target[tg.id] = next(iter(hits))
    # second pass: `cmd += [...]` / `cmd.append("--flag")` on a known argv var
    for node in ast.walk(tree):
        name = None
        if isinstance(node, ast.AugAssign) and isinstance(node.target,
                                                          ast.Name):
            name = node.target.id
        elif (isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute)
              and node.func.attr in ("append", "extend")
              and isinstance(node.func.value, ast.Name)):
            name = node.func.value.id
        if name and name in var2target:
            for s in _strings(node):
                if s.startswith("--"):
                    yield var2target[name], s


TITLES = {
    "module": "UNRESOLVED LOCAL MODULES",
    "name": "IMPORTED NAMES THAT DO NOT EXIST",
    "attr": "ATTRIBUTES ON LOCAL MODULES THAT DO NOT EXIST",
    "path": "REFERENCED REPO PATHS NOT ON DISK",
    "flag": "CLI FLAGS THE TARGET SCRIPT DOES NOT ACCEPT",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--quiet", action="store_true",
                    help="print nothing; exit 1 if anything dangles")
    a = ap.parse_args()
    f = sweep(a.root)
    f["path"] += sweep_docs(a.root)          # markdown counts too
    total = sum(len(set(v)) for v in f.values())
    if not a.quiet:
        for k in ("module", "name", "attr", "path", "flag"):
            rows = sorted(set(f[k]))
            print(f"\n=== {TITLES[k]} ({len(rows)}) ===")
            for src, what in rows:
                print(f"  {src}: {what}")
        print(f"\n{total} finding(s).",
              "Checks 1-3 and 5 are AST-based and should be exact; check 4 "
              "is a regex over the raw text of every .py AND .md file, so "
              "it sees comments, docstrings and operator instructions - "
              "the places cited paths actually live. It can flag a path "
              "that appears only as prose; confirm before acting."
              if total else "Nothing dangles.")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
