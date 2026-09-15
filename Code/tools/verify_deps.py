#!/usr/bin/env python3
"""Does this repo reference anything that is not actually here?

    python Code/tools/verify_deps.py            # from the project root
    python Code/tools/verify_deps.py --quiet    # exit code only (CI / pre-commit)

Documentation asserting that code exists is not evidence that it does,
and neither is a green dashboard: Python resolves imports lazily, so a
broken reference on a path nobody ran that day is invisible until the day
somebody runs it. This script makes that class of breakage cheap to find,
statically, in under a second, without importing anything or touching
the network.

It checks five distinct ways a reference can dangle:

1. unresolved local modules: ``import ingestion.foo`` with no ``foo.py``;
2. imported names: ``from src.config import BAR`` with no ``BAR``;
3. attributes on local modules: ``config.BAZ`` where ``src/config.py``
   has no ``BAZ``;
4. referenced repo paths: a repo path in any ``.py`` or ``.md`` text
   (comments and docstrings included) with no such file on disk;
5. CLI flags: a script handed ``--flag`` in an argv list that its
   argparse rejects.

It does not check third-party packages (see ``requirements.txt``),
runtime data files, or anything that only exists dynamically
(``getattr``, ``globals()``). It is a fast structural check, not a type
checker: a clean run means no dangling reference, not that the code is
correct.

Checks 1-3 and 5 are AST-based (``sweep``); check 4 is a regex over raw
text (``_cited_paths``, plus ``sweep_docs`` for Markdown). Exit code 0
means clean, 1 means at least one finding. Run it after any session that
edits across module boundaries.
"""

import argparse
import ast
import os
import re

SKIP_DIRS = ("_to_delete", ".git", "__pycache__", "node_modules",
             ".ipynb_checkpoints", "_salvaged_originals", "venv", ".venv",
             ".pytest_cache", "build", "dist")
# Exact directory names skipped at any depth. Reference Materials/archive/ (docs, presentations,
# research notebooks) is not part of the runtime and is not tracked. Matched by
# name, not prefix, so Data/research_record/ is still walked.
SKIP_NAMES = frozenset({"research", "presentations", "others", "archive",
                        "Presentations", "raw", "processed", "logs"})

# The project is <root>/Code (the Python tree), <root>/Data, <root>/Reports,
# <root>/Reference Materials. Citations inside the code are written
# relative to Code/ ("src/config.py") or to the project ("Data/abstracted"),
# so both spellings are accepted.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

# `config.py` inside a sentence is prose, not an attribute access; the same
# goes for a module named in a docstring.  Filtering these keeps the report
# free of noise that trains the reader to ignore it.
PROSE = {"py", "toml", "json", "md", "txt", "csv", "get"}


def _walk(root):
    """Return ``(python_files, all_files)`` as repo-relative paths under ``root``."""
    py, allf = [], set()
    for r, d, f in os.walk(root):
        d[:] = [x for x in d
               if not x.startswith(SKIP_DIRS) and x not in SKIP_NAMES]
        for n in f:
            p = os.path.relpath(os.path.join(r, n), root).replace("\\", "/")
            allf.add(p)
            if p.startswith("Code/"):
                allf.add(p[len("Code/"):])
            # A clone made from a case-insensitive checkout carries the
            # data folder as "data"; every citation in the code and the
            # docs spells it "Data". Register the cited spelling too, so
            # such a clone reports the paths it has rather than several
            # dozen dangling references to files that are present.
            # src/config.py accepts the same two spellings.
            if p.startswith("data/"):
                allf.add("Data/" + p[len("data/"):])
            # the Python tree is Code/; a top-level .py (the hosting
            # entry point that runs Code/dashboard.py) is not a module
            # anything imports
            if n.endswith(".py") and ("/" in p or not
                                      os.path.isdir(os.path.join(root, "Code"))):
                py.append(p)
    return py, allf


def _top_level_names(body, ns):
    """Add the names a module body defines to ``ns``.

    Walks into if/try/with/for bodies because config-style conditional
    definitions and ``try: import x`` fallbacks are real definitions, and
    flagging them would be a false alarm.
    """
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
_CITED_DIRS = ("src", "src/analytics", "ingestion", "tools",
               "config", "tests", "Code", "Data/research_record",
               "Data/abstracted", "Data/dashboard", "reference")


# Paths that may legitimately be absent. The checker cannot tell a
# deleted file from one that is simply not in this working copy. Paths
# under Reference Materials/archive/ are untracked by design (see .gitignore), so a citation
# into Reference Materials/archive/ from shipped code is documentation of provenance, not a
# dependency, and is never reported.
_OPTIONAL_PREFIXES = ("research/", "Reference Materials/archive/")

# A citation that already SAYS the file is absent is documentation, not
# a dangling reference. Without this the sweep would punish exactly the
# honest behaviour it is meant to encourage: a sentence stating that a
# since-removed module does not exist should not read as a defect.
_KNOWN_ABSENT = ("not in this repo", "does not exist", "no longer",
                 "is not on disk", "never present", "absent",
                 "was removed", "not present", "was deleted")


def _cited_paths(txt, pkgs=()):
    """Every repo-relative file path this text mentions, quoted or not.

    Two exclusions, both deliberate:

    * anything under ``Data/``: those are runtime artefacts. Whether
      ``Data/processed/posts.parquet`` exists depends on whether the
      pipeline has run on this copy, so its absence is never a broken
      reference and flagging it would train the reader to ignore this
      check;
    * any citation whose surrounding lines already declare the file
      missing (``_KNOWN_ABSENT``).

    Args:
        txt: File contents.
        pkgs: Extra top-level directory names to treat as citable.

    Returns:
        Set of repo-relative paths.
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
    """Find cited paths in Markdown files that do not exist.

    Docs are where operator instructions live, so a dangling path there
    is a command somebody will run and watch fail.

    Args:
        root: Repository root.

    Returns:
        List of ``(markdown_file, missing_path)`` pairs.
    """
    _py, allf = _walk(root)
    out = []
    for rel in sorted(allf):
        if not rel.endswith(".md") or rel.startswith("_to_delete"):
            continue
        try:
            txt = open(os.path.join(root, rel), encoding="utf-8").read()
        except (OSError, UnicodeDecodeError):
            continue                 # not UTF-8 text: nothing to sweep
        for m in _cited_paths(txt):
            if m not in allf and not m.startswith(_OPTIONAL_PREFIXES):
                out.append((rel, m))
    return out


def sweep(root):
    """Run checks 1-5 over every Python file under ``root``.

    Args:
        root: Repository root.

    Returns:
        Dict with keys ``module``, ``name``, ``attr``, ``path`` and
        ``flag``, each a list of ``(file, description)`` findings.
    """
    py, allf = _walk(root)
    pkgs = {p.split("/")[0] for p in py if "/" in p}
    rootmods = {p[:-3] for p in py if "/" not in p}
    trees, names = {}, {}
    findings = {k: [] for k in ("module", "name", "attr", "path", "flag")}

    for p in py:
        try:
            trees[p] = ast.parse(open(os.path.join(root, p),
                                      encoding="utf-8").read())
        except (OSError, SyntaxError, UnicodeDecodeError) as e:
            # A file saved in the console codepage rather than UTF-8 is a
            # finding of its own, not a reason for the sweep to stop.
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
        # Scans the raw text of the file rather than only its string
        # literals: most cited paths live in comments and docstrings,
        # because they are instructions to a human, not arguments to a
        # function. The directory prefixes come from _CITED_DIRS (names
        # that are cited, present or not) plus the packages that exist,
        # so a reference to a deleted directory is still visible.
        # Markdown is swept separately in `sweep_docs` for the same
        # reason.
        for m in _cited_paths(txt, pkgs):
            if m not in allf and not m.startswith(_OPTIONAL_PREFIXES):
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
    """Return every string constant inside an AST node."""
    return {n.value for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def _is_leaf_seq(node):
    """Return True for a list/tuple with no list/tuple inside it.

    A leaf sequence is one argv, not a list of argvs. ``dashboard.py``
    builds pipeline plans as::

        [(["ingestion/pull_prices.py"], env),
         (["update_data.py", "--start", s, "--skip-prices"], None)]

    and reading the outer sequence as one command would blame
    ``pull_prices.py`` for ``update_data.py``'s flags.
    """
    return not any(isinstance(c, (ast.List, ast.Tuple))
                   for c in ast.walk(node) if c is not node)


def _command_flags(tree, accepts):
    """Yield ``(target_script, flag)`` pairs found in argv lists.

    This is deliberately not a text window around the script name: a
    path and a flag can sit in two unrelated statements a few lines
    apart with no flag passed at all, and a checker that cries wolf
    teaches the reader to skim its output. A flag counts only when it
    travels in the same leaf argv list as the script::

        cmd = [py, "ingestion/fetch_all.py", "--no-merge"]

    or when it is later appended to that same list variable, which is
    how argv is built conditionally::

        cmd.append("--skip-comments")
        cmd += ["--comment-pages", str(n)]

    Args:
        tree: Parsed module.
        accepts: Mapping of script path to the flags it accepts. It must
            contain only scripts that actually parse argv: a Streamlit
            entry point has no argparse, so ``streamlit run dashboard.py
            --server.port 8501`` passes flags to Streamlit, not to the
            script.

    Yields:
        ``(target_script, flag)`` for every ``--flag`` string that
        travels with exactly one known script.
    """
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
    """Run the sweep and print the findings.

    Returns:
        ``1`` when anything dangles, otherwise ``0``.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", nargs="?", default=PROJECT_ROOT)
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
