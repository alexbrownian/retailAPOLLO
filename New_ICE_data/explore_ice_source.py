# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ICE — a new Reddit source, assessed before it is trusted
#
# **Desk request, 2026-08-20.** *"its reddit data from a different
# source and i want to see if it can be incorporated into our data
# pipeline moving forward"* — then, once it was unpacked: *"dont use
# the zip please just read straight from the unzipped folder now"*.
#
# ## What this notebook does, and what it refuses to do
#
# It reads the **unzipped delivery folder**, works out what is inside,
# normalises it into the project's own post schema, and answers the
# only question that matters — *would folding this in help, and is it
# comparable to what we already hold?*
#
# It **writes nothing outside `new_ice_data/`**. No live pipeline, no
# `data/processed`, no `ABSTRACTED_DATA`, no ledger, no config. The one
# artefact it produces is a normalised archive inside this folder that
# `tools/fold_historical.py` can read **if you decide to adopt it**.
# Adoption stays a separate, deliberate command.
#
# ## Reading the folder, not the zip
#
# The outer `Confide_*.zip` is ignored entirely. What the unpack left
# behind is `Reddit_2024Q1_split/Reddit_2024Q1.zip.part-0000..0002`,
# and those parts are **not three archives** - they are one archive cut
# into pieces, so no part opens on its own. Rather than write a 3 GB
# joined copy back to your disk, §1 presents the parts to `zipfile` as
# a single **seekable stream** (`SplitReader`). Nothing is duplicated,
# nothing new lands on disk, and the read is exactly as if the file
# had been joined.
#
# If you later extract that inner archive to loose files, this notebook
# needs no change: §1 picks up loose `.jsonl/.csv/.parquet/.zst` files
# in the tree and prefers them automatically.
#
# ## The question this notebook answers
#
# Desk clarification, 2026-08-21: *"this ICE is unrelated to the gap
# btw, i just need to check if this data source is ok for future
# ingestion"*. So this is a **vendor fitness test**, not a backfill.
# The 2024-08 → 2025-12 hole is a separate job (torrent dumps plus
# `tools/fold_historical.py --dumps`) and nothing here bears on it.
#
# That framing matters, because it changes what counts as enough data.
# Fitness is a question about the SHAPE of a source, and shape is
# visible in a sample — which is why §1c's partial salvage of a
# truncated delivery is not a compromise, it is sufficient. Coverage
# would have needed every byte; fitness does not.
#
# §5 runs the three tests that decide it:
#
# 1. **Does it carry the fields we ingest?** Every column in
#    `OUTPUT_COLUMNS` present and populated, ids stable, timestamps
#    real. A source missing `num_comments` or `score` cannot drive the
#    crowd features at all.
# 2. **Is it the same KIND of data?** Same subreddits, and comparable
#    mention density per post against our measured 0.11 benchmark. A
#    source that names our tickers ten times less often is dilution
#    dressed as volume — the single most likely way a vendor feed
#    quietly degrades the signal.
# 3. **Does it agree with what we already hold?** This delivery is
#    2024Q1, a quarter we restored ourselves, which makes it an unusually
#    good control: we can compare like for like rather than take the
#    vendor's word. Overlap is the point here, not a waste.

# %%
from __future__ import annotations

import bisect
import csv
import io
import json
import os
import re
import sys
import time
import zipfile
from collections import defaultdict

import pandas as pd

# Locate the project by a LANDMARK FILE, not by folder names.  The
# earlier version assumed this folder was called exactly "new_ice_data";
# the real one is "New_ICE_data", so it appended a second, non-existent
# path segment and PROJECT_ROOT came out one level too deep - hence
# "No module named 'src'".  Names drift, src/reddit_live_data.py does not.
_LANDMARK = os.path.join("src", "reddit_live_data.py")
_ICE_NAMES = {"new_ice_data", "ice_data", "new_ice"}


def _walk_up_to_root(start, levels=6):
    """Return the first ancestor of `start` that holds the landmark."""
    d = os.path.abspath(start)
    for _ in range(levels):
        if os.path.isfile(os.path.join(d, _LANDMARK)):
            return d
        parent = os.path.dirname(d)
        if parent == d:                      # hit the filesystem root
            break
        d = parent
    return None


_CWD = os.getcwd()
PROJECT_ROOT = _walk_up_to_root(_CWD)
if PROJECT_ROOT is None:
    raise RuntimeError(
        "Could not find the retailAPOLLO root above %s.\n"
        "Open the notebook from inside the project tree, or set\n"
        "PROJECT_ROOT by hand in this cell." % _CWD)

# This folder: the working directory when the notebook sits in it,
# otherwise the ICE folder under the root - matched CASE-INSENSITIVELY
# so New_ICE_data / new_ice_data / NEW_ICE_DATA all resolve.
if os.path.basename(_CWD).lower() in _ICE_NAMES:
    HERE = _CWD
else:
    HERE = next(
        (os.path.join(PROJECT_ROOT, n) for n in sorted(os.listdir(PROJECT_ROOT))
         if n.lower() in _ICE_NAMES
         and os.path.isdir(os.path.join(PROJECT_ROOT, n))),
        _CWD)

sys.path.insert(0, PROJECT_ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# The project's own post schema - imported, never re-typed, so this
# notebook cannot drift from the pipeline it is auditioning for.
from src.reddit_live_data import OUTPUT_COLUMNS          # noqa: E402

DATA_EXT = (".jsonl", ".ndjson", ".json", ".csv", ".tsv", ".parquet",
            ".zst", ".gz")
PART_RE = re.compile(r"^(?P<base>.+?)\.part-(?P<idx>\d+)$", re.IGNORECASE)

print(f"project root : {PROJECT_ROOT}")
print(f"this folder  : {HERE}")
print(f"target schema: {OUTPUT_COLUMNS}")

# %% [markdown]
# ## 1 — Find the data in the unzipped folder
#
# Three shapes are handled, in the order they are preferred:
#
# 1. **loose data files** already extracted in the tree — read directly;
# 2. **split archive parts** — stitched into one stream, no disk copy;
# 3. **a plain inner `.zip`** — opened normally.
#
# Any `.zip` sitting directly in `new_ice_data/` is treated as the
# original delivery wrapper and skipped: its contents are already on
# disk, and reading both would double-count everything.

# %%
class SplitReader(io.RawIOBase):
    """Present N `.part-NNNN` files as ONE seekable byte stream.

    `zipfile` needs seek/tell/read; it does not care whether the bytes
    come from one file or several. Implementing the stream is a few
    lines and saves writing a 3 GB duplicate of data that is already on
    the disk - which is the whole point of reading the folder in place.
    """

    def __init__(self, paths):
        super().__init__()
        self._paths = list(paths)
        self._sizes = [os.path.getsize(p) for p in self._paths]
        self._starts, run = [], 0
        for s in self._sizes:
            self._starts.append(run)
            run += s
        self._total = run
        self._pos = 0
        self._open_handles = {}

    def size(self):
        """Total bytes across all parts - what a salvage scan must not
        read past."""
        return self._total

    # --- the file-object protocol zipfile relies on
    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        else:
            self._pos = self._total + offset
        self._pos = max(0, min(self._pos, self._total))
        return self._pos

    MAX_OPEN = 8          # a read only ever needs one or two parts at once

    def _handle(self, i):
        """Open handles are CAPPED and least-recently-used is closed.

        The first version cached a handle per part forever. With three
        parts that is invisible; with a delivery split into thousands it
        exhausts the process file-descriptor limit and every later read
        dies with "OSError 24: Too many open files" - and, worse, so
        does everything else in the notebook, including reading our own
        parquet files. Found 2026-08-21 on a stress fixture.
        """
        h = self._open_handles.pop(i, None)
        if h is None:
            h = open(self._paths[i], "rb")
        self._open_handles[i] = h          # re-insert => most recent last
        while len(self._open_handles) > self.MAX_OPEN:
            _oldest = next(iter(self._open_handles))
            try:
                self._open_handles.pop(_oldest).close()
            except Exception:
                pass
        return h

    def close(self):
        for _h in self._open_handles.values():
            try:
                _h.close()
            except Exception:
                pass
        self._open_handles.clear()
        super().close()

    def readinto(self, buf):
        want, got = len(buf), 0
        while got < want and self._pos < self._total:
            i = bisect.bisect_right(self._starts, self._pos) - 1
            fh = self._handle(i)
            fh.seek(self._pos - self._starts[i])
            end_of_part = self._starts[i] + self._sizes[i]
            chunk = fh.read(min(want - got, end_of_part - self._pos))
            if not chunk:
                break
            buf[got:got + len(chunk)] = chunk
            got += len(chunk)
            self._pos += len(chunk)
        return got

    def close(self):
        for fh in self._open_handles.values():
            try:
                fh.close()
            except OSError:
                pass
        self._open_handles.clear()
        super().close()


def discover(root):
    """(loose data files, split sets, inner zips) under `root`."""
    loose, parts, zips = [], defaultdict(list), []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            full = os.path.join(dirpath, f)
            m = PART_RE.match(f)
            if m:
                parts[os.path.join(dirpath, m.group("base"))].append(
                    (int(m.group("idx")), full))
                continue
            low = f.lower()
            if low.endswith(".zip"):
                # a zip sitting at the top of new_ice_data/ is the
                # delivery wrapper - its contents are already unpacked
                if os.path.dirname(full) != root:
                    zips.append(full)
                continue
            if low.endswith(DATA_EXT):
                # Our OWN outputs live in this folder and are data files
                # too, so without this they get discovered as input on
                # the next run - and because loose files outrank split
                # sets, the notebook would quietly profile its own
                # results instead of the delivery. Found the hard way on
                # 2026-08-21: a second run reported 108 rows and three
                # tickers instead of 1,440 and forty.
                if (os.path.dirname(full) == root
                        and os.path.basename(low).startswith("ice_")):
                    continue
                loose.append(full)
    return loose, {k: sorted(v) for k, v in parts.items()}, zips


LOOSE, SPLITS, INNER_ZIPS = discover(HERE)

# If you joined a split set by hand (copy /b, or the vendor's PowerShell
# recipe), the result sits NEXT TO the parts under the name the parts
# were derived from. Prefer that file and retire the parts: otherwise
# the split branch below wins, re-reads the same pieces, and fails again
# in exactly the way the join was meant to cure.
_joined = {b for b in list(SPLITS) if os.path.isfile(b)}
for _b in _joined:
    print(f"  NOTE: {os.path.basename(_b)} exists as a joined file - "
          f"using it and ignoring its {len(SPLITS[_b])} parts.")
    SPLITS.pop(_b)
    if _b not in INNER_ZIPS and _b.lower().endswith(".zip"):
        INNER_ZIPS.insert(0, _b)

print("--- what is in the unzipped folder ---")
print(f"  loose data files : {len(LOOSE)}")
for p in LOOSE[:10]:
    print(f"      {os.path.relpath(p, HERE)} "
          f"({os.path.getsize(p) / 2**20:,.1f} MiB)")
print(f"  inner zips       : {len(INNER_ZIPS)}")
for p in INNER_ZIPS:
    print(f"      {os.path.relpath(p, HERE)}")
print(f"  split sets       : {len(SPLITS)}")
for base, ps in SPLITS.items():
    tot = sum(os.path.getsize(p) for _i, p in ps)
    sizes = {os.path.getsize(p) for _i, p in ps}
    print(f"      {os.path.relpath(base, HERE)} - {len(ps)} parts, "
          f"{tot / 2**30:.2f} GiB")
    for i, p in ps:
        print(f"          part-{i:04d}  {os.path.getsize(p):,} bytes")
    # A split set normally ends SHORT. Parts that are all exactly equal
    # are the signature of a transfer that stopped on a chunk boundary.
    if len(sizes) == 1 and len(ps) > 1:
        print(f"          WARNING: every part is exactly the same size, "
              f"so a part-{len(ps):04d} may be missing. The integrity "
              f"check below is what settles it.")

# %% [markdown]
# ### 1b — If the split set will not open, prove why
#
# `BadZipFile` says only "not a zip file", which is unhelpful because it
# is the same message for a corrupt head, a wrong file, and a perfectly
# good archive whose tail is missing. Those need different responses, so
# this reads the few bytes that tell them apart:
#
# * **offset 0 of part-0000** must be `PK\x03\x04` — the first local file
#   header. If it is, the head is a genuine zip and the split order is
#   right.
# * **later parts must NOT start with `PK\x03\x04`** — if they do, they
#   are separate archives, not pieces of one, and `SplitReader` is the
#   wrong tool.
# * **the last ~64 KiB must contain `PK\x05\x06`** (or `PK\x06\x06` for
#   zip64) — the End Of Central Directory record, which every zip closes
#   with. `zipfile` finds a zip by seeking to the end and scanning back
#   for it. No EOCD at the end means the end is not there.
#
# A head that is valid plus an EOCD that is absent is conclusive: the
# bytes are a real zip, cut short. Nothing local can recover it.

# %%
def diagnose_split(paths, tail_bytes=1 << 16):
    """Say WHICH failure this is, from the bytes rather than a guess."""
    LOCAL, EOCD, EOCD64 = b"PK\x03\x04", b"PK\x05\x06", b"PK\x06\x06"

    with open(paths[0], "rb") as fh:
        head = fh.read(4)
    head_ok = head == LOCAL
    # NB: no backslashes inside f-string expressions - that is a syntax
    # error before Python 3.12, and this notebook must run on the desk's
    # interpreter, not just a modern one.
    _verdict = "= PK 03 04, a real zip head" if head_ok else "= NOT a zip head"
    print("  head of part-0000        : " + head.hex(" ") + " " + _verdict)

    separate = False
    for n, p in enumerate(paths[1:], start=1):
        with open(p, "rb") as fh:
            b = fh.read(4)
        if b == LOCAL:
            separate = True
            print("  head of part-%04d        : also PK 03 04 - these are "
                  "SEPARATE archives, not one split file." % n)

    with open(paths[-1], "rb") as fh:
        fh.seek(max(0, os.path.getsize(paths[-1]) - tail_bytes))
        tail = fh.read()
    eocd = (EOCD in tail) or (EOCD64 in tail)
    _e = "FOUND" if eocd else ("ABSENT from the last %d KiB"
                               % (tail_bytes // 1024))
    print("  end-of-archive record    : " + _e)

    sizes = [os.path.getsize(p) for p in paths]
    equal = len(set(sizes)) == 1
    _s = ("all identical (%s bytes)" % format(sizes[0], ",") if equal
          else "last part is short, as a real split ends")
    print("  part sizes               : " + _s)

    print()
    # Order matters: "every part is its own archive" has to be ruled out
    # BEFORE the truncation branches, because such a set has a valid head
    # AND a valid EOCD and would otherwise fall through to "look
    # elsewhere" - the one verdict that helps nobody.
    if separate:
        print("  VERDICT: more than one part starts with a local file header,")
        print("  so these are independent archives that merely share a")
        print("  naming convention. Do not stream them as one file - open")
        print("  each on its own, or extract them side by side and let the")
        print("  loose-file path in section 1 pick the results up.")
    elif head_ok and not eocd and equal:
        print("  VERDICT: the head is a valid zip and every part is exactly")
        print("  the chunk size, but the archive's closing record is missing.")
        print("  A complete split ALWAYS ends short, because the last part is")
        print("  a remainder. This delivery stopped on a chunk boundary:")
        print(f"  part-{len(paths):04d} (and possibly more) was never sent.")
        print("  Joining the parts by hand cannot help: the bytes that are")
        print("  missing are missing. Ask the vendor for the rest - and see")
        print("  1c, which recovers the members IN FRONT of the cut, which")
        print("  is enough to judge whether the source is fit to ingest.")
    elif head_ok and not eocd:
        print("  VERDICT: valid head, no closing record -> truncated. Ask the")
        print("  vendor to re-send; the tail did not arrive intact.")
    elif not head_ok:
        print("  VERDICT: part-0000 is not the start of a zip. Either the")
        print("  parts are out of order, or the delivery is not a zip at all.")
    else:
        print("  VERDICT: head and closing record both present, so the")
        print("  failure is elsewhere - re-run and read the traceback.")


# %% [markdown]
# ### 1c — Salvage: read a truncated archive anyway
#
# The 2026-08-20 delivery arrived cut short (§1b). That is fatal to
# `zipfile`, which finds members by reading the **central directory** at
# the end of the file — but it is NOT fatal to the data. A zip is laid
# out as
#
# ```
# [local header][member 1 bytes][local header][member 2 bytes]...[central directory][EOCD]
# ```
#
# so every member whose bytes finished before the cut is sitting there
# intact and self-describing: the local header in front of it carries
# the name, the compression method, the compressed length and the CRC.
# Walking those headers from offset 0 recovers everything except the one
# member straddling the cut.
#
# **Why this is enough here.** The desk's question is not "does this fill
# the gap" — it is *"is this source OK for future ingestion"*. That is a
# question about schema, subreddit mix and mention density, all of which
# a representative sample answers. A vendor's fitness does not depend on
# our receiving the last 900 MB of one quarter.
#
# **The one case that limits it.** If the writer used *streaming* mode
# (general-purpose flag bit 3), the local header's size fields are zero
# and the real lengths trail the data in a descriptor. The scanner
# detects this, follows the `PK\x07\x08` descriptors, and refuses to
# guess if they do not check out — it will say so rather than hand back
# plausible rubbish.
#
# Salvaged output is marked `SALVAGED` everywhere it is used, and §7's
# verdict is required to state the recovered fraction, so no conclusion
# can quietly rest on a partial read.

# %%
class _SalvagedInfo:
    """Duck-type of zipfile.ZipInfo, only the fields this notebook uses."""

    def __init__(self, filename, file_size, compress_size, compress_type,
                 data_offset, CRC, partial=False):
        self.filename = filename
        self.file_size = file_size
        self.compress_size = compress_size
        self.compress_type = compress_type
        self.data_offset = data_offset
        self.CRC = CRC
        # True when this member runs past the end of the delivered bytes.
        # It is still readable up to the cut - a deflate stream decodes
        # from the front, so a prefix of the compressed bytes yields a
        # prefix of the content. Only the tail is lost.
        self.partial = partial

    def is_dir(self):
        return self.filename.endswith("/")


class _MemberStream(io.RawIOBase):
    """Streaming inflate of one salvaged member - never holds it whole."""

    def __init__(self, fh, info, chunk=1 << 20):
        super().__init__()
        self._fh, self._info, self._chunk = fh, info, chunk
        self._left = info.compress_size
        self._pos = info.data_offset
        self._buf = b""
        self._eof = False
        if info.compress_type == 0:                     # stored
            self._dec = None
        elif info.compress_type == 8:                   # deflate
            import zlib
            self._dec = zlib.decompressobj(-15)         # raw, no header
        else:
            raise NotImplementedError(
                "compression method %d is not supported by the salvage "
                "reader (only stored and deflate)" % info.compress_type)

    def readable(self):
        return True

    def _fill(self):
        while not self._buf and not self._eof:
            if self._left <= 0:
                self._eof = True
                if self._dec is not None:
                    # A PARTIAL member has no proper end, so flush() can
                    # raise on the unfinished stream. That is expected,
                    # not an error: everything decoded up to here is
                    # already correct and has been handed out.
                    try:
                        self._buf = self._dec.flush()
                    except Exception:
                        self._buf = b""
                break
            self._fh.seek(self._pos)
            raw = self._fh.read(min(self._chunk, self._left))
            if not raw:
                self._eof = True
                break
            self._pos += len(raw)
            self._left -= len(raw)
            if self._dec is None:
                self._buf = raw
            else:
                try:
                    self._buf = self._dec.decompress(raw)
                except Exception:
                    # Corruption or the ragged end of a truncated stream.
                    # Stop cleanly rather than throwing away the prefix.
                    self._eof = True
                    self._buf = b""

    def readinto(self, b):
        self._fill()
        if not self._buf:
            return 0
        n = min(len(b), len(self._buf))
        b[:n] = self._buf[:n]
        self._buf = self._buf[n:]
        return n


class SalvagedZip:
    """Open a zip that has no central directory, by walking local headers.

    Exposes only what this notebook asks of a ZipFile - `infolist()`,
    `open()`, `read()` - so every downstream cell works unchanged.
    """

    LFH, DD, CD = b"PK\x03\x04", b"PK\x07\x08", b"PK\x01\x02"

    def __init__(self, fileobj, total):
        self._fh, self._total = fileobj, total
        self.infos, self.truncated, self.streamed = [], None, False
        self.bytes_recovered = 0
        self.partial_members = []
        self._scan()

    # -- helpers
    @staticmethod
    def _u16(b, o):
        return int.from_bytes(b[o:o + 2], "little")

    @staticmethod
    def _u32(b, o):
        return int.from_bytes(b[o:o + 4], "little")

    def _zip64(self, extra, csize, usize):
        """Replace 0xFFFFFFFF placeholders from the zip64 extra field."""
        i = 0
        while i + 4 <= len(extra):
            hid, sz = self._u16(extra, i), self._u16(extra, i + 2)
            if hid == 0x0001:
                blk, j = extra[i + 4:i + 4 + sz], 0
                if usize == 0xFFFFFFFF and j + 8 <= len(blk):
                    usize = int.from_bytes(blk[j:j + 8], "little"); j += 8
                if csize == 0xFFFFFFFF and j + 8 <= len(blk):
                    csize = int.from_bytes(blk[j:j + 8], "little")
                break
            i += 4 + sz
        return csize, usize

    def _find_descriptor(self, start):
        """Streaming mode: locate the data descriptor that ends a member."""
        pos, CH = start, 1 << 20
        while pos < self._total:
            self._fh.seek(pos)
            blk = self._fh.read(min(CH, self._total - pos))
            if not blk:
                return None
            k = blk.find(self.DD)
            while k != -1:
                at = pos + k
                self._fh.seek(at)
                d = self._fh.read(16)
                if len(d) >= 16:
                    csize = self._u32(d, 8)
                    # the descriptor must agree with where it was found
                    if csize == at - start:
                        return at, csize, self._u32(d, 12)
                k = blk.find(self.DD, k + 1)
            pos += max(1, len(blk) - 4)        # overlap, signatures can split
        return None

    def _scan(self):
        pos = 0
        while pos + 30 <= self._total:
            self._fh.seek(pos)
            h = self._fh.read(30)
            if len(h) < 30 or h[:4] != self.LFH:
                break                          # central directory, or garbage
            flags = self._u16(h, 6)
            method = self._u16(h, 8)
            crc = self._u32(h, 14)
            csize, usize = self._u32(h, 18), self._u32(h, 22)
            nlen, elen = self._u16(h, 26), self._u16(h, 28)
            name = self._fh.read(nlen).decode("utf-8", "replace")
            extra = self._fh.read(elen)
            if 0xFFFFFFFF in (csize, usize):
                csize, usize = self._zip64(extra, csize, usize)
            data_off = pos + 30 + nlen + elen

            avail = self._total - data_off      # bytes we actually hold

            if flags & 0x08 and csize == 0:
                self.streamed = True
                found = self._find_descriptor(data_off)
                if found is None:
                    # The descriptor that would state this member's length
                    # is itself past the cut. We still hold the front of
                    # the member, which is what matters.
                    self._add_partial(name, usize, avail, method, data_off)
                    break
                dd_at, csize, usize = found
                crc = 0                        # trust the descriptor's sizes,
                nxt = dd_at + 16               # not a CRC we did not read
            else:
                nxt = data_off + csize

            if data_off + csize <= self._total:
                if not name.endswith("/"):
                    self.infos.append(_SalvagedInfo(
                        name, usize, csize, method, data_off, crc))
                    self.bytes_recovered += csize
                pos = nxt
            else:
                # This member straddles the cut. Do NOT throw it away: a
                # deflate stream decodes from the front, so the bytes we
                # hold yield a prefix of the content. When the archive is
                # ONE big member - which is the 2026-08-20 delivery - this
                # branch is the only thing standing between "2.84 GB of
                # real posts" and "nothing readable".
                self._add_partial(name, usize, avail, method, data_off)
                break

    def _add_partial(self, name, usize, avail, method, data_off):
        self.truncated = name
        if name.endswith("/") or avail <= 0:
            return
        self.infos.append(_SalvagedInfo(
            name, usize, avail, method, data_off, 0, partial=True))
        self.bytes_recovered += avail
        self.partial_members.append(name)

    # -- the ZipFile surface the notebook uses
    def infolist(self):
        return list(self.infos)

    def _info(self, name):
        for i in self.infos:
            if i.filename == name:
                return i
        raise KeyError(name)

    def open(self, name):
        return _MemberStream(self._fh, self._info(name))

    def read(self, name):
        return self.open(name).readall()

    def report(self):
        whole = [i for i in self.infos if not i.partial]
        part = [i for i in self.infos if i.partial]
        pct = 100.0 * self.bytes_recovered / self._total if self._total else 0
        print(f"  SALVAGED {len(whole)} complete + {len(part)} partial "
              f"member(s), {self.bytes_recovered / 2**30:.2f} GiB of "
              f"compressed data ({pct:.1f}% of the bytes present)")
        if self.streamed:
            print("  (streaming-mode archive: lengths taken from data "
                  "descriptors and cross-checked against their offsets)")
        for i in whole:
            print(f"      {i.filename}  "
                  f"{i.compress_size / 2**20:,.1f} MiB compressed  [complete]")
        for i in part:
            print(f"      {i.filename}  "
                  f"{i.compress_size / 2**20:,.1f} MiB compressed  [PARTIAL - "
                  f"decodes from the front, ends where the delivery stopped]")
        if part:
            print("  The partial member is READABLE. Deflate decodes")
            print("  sequentially, so every record before the cut comes out")
            print("  intact; the read simply stops early, and the final line")
            print("  may be half-written. Section 4 drops unparseable lines.")


# %% [markdown]
# ### 1d — Zips inside the zip
#
# The 2026-08-20 delivery turned out to be **nested**: the outer archive
# holds `REDDIT_EOD_KG_20_YYYYMM_partNNN.zip`, and each of those holds
# per-day `.parquet` files. Reading the outer members as text produced
# the `PK...PAR1...` gibberish in §2 — that was a zip being decoded as
# CSV, and it is also why §4 crawled: it was scanning gigabytes of
# binary looking for delimiters that were never there.
#
# Expanding a nested zip normally means extracting gigabytes to disk.
# It does not have to. A zip stored *inside* another zip is almost
# always `STORED` (already compressed, so deflating again is pointless),
# which means its bytes sit contiguously in the outer stream and a
# **seekable window** over that range behaves exactly like the file.
# `_SliceReader` is that window; `zipfile` and `pyarrow` both accept it,
# and nothing is copied. Only if an inner zip is deflated do we fall
# back to buffering it, and then only up to `MAX_BUFFER`.

# %%
MAX_BUFFER = 512 * 2 ** 20        # refuse to hold more than this in RAM


class _SliceReader(io.RawIOBase):
    """A seekable window [start, start+length) over a seekable parent."""

    def __init__(self, parent, start, length):
        super().__init__()
        self._p, self._start, self._len, self._pos = parent, start, length, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, off, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self._pos = off
        elif whence == io.SEEK_CUR:
            self._pos += off
        else:
            self._pos = self._len + off
        self._pos = max(0, min(self._pos, self._len))
        return self._pos

    def readinto(self, b):
        n = min(len(b), self._len - self._pos)
        if n <= 0:
            return 0
        self._p.seek(self._start + self._pos)
        chunk = self._p.read(n)
        b[:len(chunk)] = chunk
        self._pos += len(chunk)
        return len(chunk)

    def size(self):
        return self._len


def _stored_slice(zf, reader, info):
    """Seekable view of a STORED member, or None when that is impossible."""
    if reader is None or getattr(info, "compress_type", None) != 0:
        return None
    off = getattr(info, "data_offset", None)
    if off is None:                       # real ZipFile: ask it for the offset
        try:
            with zf.open(info.filename) as fh:
                off = fh._fileobj.tell() if hasattr(fh, "_fileobj") else None
        except Exception:
            off = None
        if off is None:
            hdr = getattr(info, "header_offset", None)
            if hdr is None:
                return None
            reader.seek(hdr + 26)
            nl = int.from_bytes(reader.read(2), "little")
            el = int.from_bytes(reader.read(2), "little")
            off = hdr + 30 + nl + el
    return _SliceReader(reader, off, info.compress_size)


def expand_members(zf, reader):
    """{display name -> opener()} with nested zips expanded in place.

    Display names use `outer.zip::inner.parquet` so §2/§4 can still key
    off a single string, and so the printout says where a file came from.
    """
    openers, sizes = {}, {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        nm = info.filename
        if not nm.lower().endswith(".zip"):
            openers[nm] = (lambda n=nm: zf.open(n))
            sizes[nm] = info.file_size
            continue

        # --- a zip inside the zip
        sub = _stored_slice(zf, reader, info)
        if sub is None:
            if info.compress_size > MAX_BUFFER:
                print(f"  SKIP {nm}: deflated inner zip of "
                      f"{info.compress_size / 2**20:,.0f} MiB exceeds the "
                      f"{MAX_BUFFER // 2**20} MiB buffer ceiling.")
                continue
            sub = io.BytesIO(zf.read(nm))

        try:
            inner = zipfile.ZipFile(sub)
            if not _ends_with_eocd(sub, sub.size() if hasattr(sub, "size")
                                   else len(sub.getbuffer())):
                raise zipfile.BadZipFile("interior directory, not the end")
            inner_infos = [i for i in inner.infolist() if not i.is_dir()]
        except zipfile.BadZipFile:
            # the inner zip is cut too - salvage it the same way
            total = sub.size() if hasattr(sub, "size") else len(sub.getbuffer())
            inner = SalvagedZip(sub, total)
            inner_infos = inner.infolist()
            if inner_infos:
                print(f"  {nm}: inner archive also truncated - salvaged "
                      f"{len(inner_infos)} member(s)")
            else:
                print(f"  {nm}: inner archive unreadable, skipped")
                continue

        for ii in inner_infos:
            disp = f"{nm}::{ii.filename}"
            openers[disp] = (lambda z=inner, n=ii.filename: z.open(n))
            sizes[disp] = ii.file_size
    return openers, sizes


# %%
def _ends_with_eocd(reader, total=None, tail=1 << 16):
    """Does the archive's closing record sit at the END of this stream?

    This guard exists because of a silent failure found on 2026-08-21.
    Our delivery is zips-inside-a-zip, STORED, so every inner archive
    carries its own End Of Central Directory record in the middle of the
    outer byte stream. `zipfile` locates an archive by scanning BACKWARDS
    from the end for that signature - so on a truncated outer stream it
    happily latched onto the last INNER zip's directory and opened
    successfully, reporting 15 members when the delivery actually held
    four inner archives. No exception, no warning, roughly three
    quarters of the data silently invisible.

    A well-formed archive ends with its EOCD. If the last 64 KiB does
    not contain one, whatever `zipfile` opened is not the whole stream,
    and the header walk in 1c is the honest reader.
    """
    if total is None:
        total = reader.size() if hasattr(reader, "size") else None
        if total is None:
            cur = reader.tell()
            total = reader.seek(0, io.SEEK_END)
            reader.seek(cur)
    n = min(tail, total)
    reader.seek(total - n)
    blk = reader.read(n)
    idx = blk.rfind(b"PK\x05\x06")
    if idx == -1:
        return False
    # "Present somewhere in the tail" is NOT enough - an inner archive's
    # directory can fall inside that window and still be interior. The
    # EOCD is 22 bytes plus an optional comment, and a real one finishes
    # exactly at the last byte of the file. Anything else is interior.
    start = total - n + idx
    reader.seek(start + 20)
    clen = int.from_bytes(reader.read(2), "little")
    return start + 22 + clen == total


# %%
# ---- resolve ONE source to read, preferring already-extracted files
ZF = None            # zipfile.ZipFile, or None when reading loose files
SALVAGED = False     # True when ZF is a SalvagedZip, not a real ZipFile
MEMBERS = []         # names inside ZF, or absolute paths when loose
SRC_LABEL = ""
_reader = None

if LOOSE:
    MEMBERS = sorted(LOOSE)
    SRC_LABEL = f"{len(MEMBERS)} loose file(s) on disk"
elif SPLITS or INNER_ZIPS:
    if SPLITS:
        base, ps = sorted(SPLITS.items())[0]
        _reader = SplitReader([p for _i, p in ps])
        SRC_LABEL = (f"{os.path.relpath(base, HERE)} "
                     f"({len(ps)} parts, streamed in place)")
        try:
            ZF = zipfile.ZipFile(_reader)
            if not _ends_with_eocd(_reader, _reader.size()):
                ZF = None
                raise zipfile.BadZipFile(
                    "opened, but the end-of-archive record is NOT at the end "
                    "of the stream - this is an interior directory from a "
                    "nested archive, not the whole delivery")
        except zipfile.BadZipFile as e:
            print(f"\nCANNOT OPEN the split set normally: {e}")
            diagnose_split([p for _i, p in ps])
            # The archive has no index, but the members in front of the
            # cut are intact and self-describing. Recover them.
            print("\nattempting salvage (see 1c) ...")
            _sz = SalvagedZip(_reader, _reader.size())
            if _sz.infolist():
                _sz.report()
                ZF, SALVAGED = _sz, True
                SRC_LABEL += "  [SALVAGED - PARTIAL]"
            else:
                print("  salvage recovered nothing readable; the head is "
                      "unusable, not merely incomplete.")
    else:
        path = INNER_ZIPS[0]
        SRC_LABEL = os.path.relpath(path, HERE)
        ZF = zipfile.ZipFile(path)
    if ZF is not None:
        OPENERS, SIZES = expand_members(ZF, _reader)
        MEMBERS = sorted(OPENERS)
else:
    print("NOTHING FOUND. Expected loose data files, a split set, or an "
          "inner zip under new_ice_data/.")

print(f"\nreading: {SRC_LABEL}")
print(f"members: {len(MEMBERS):,}")

# %%
# ---- inventory, whichever shape the source turned out to be
INV = pd.DataFrame()
if ZF is not None and MEMBERS:
    INV = pd.DataFrame([{
        "name": n,
        "ext": os.path.splitext(n)[1].lower(),
        "MB": round(SIZES.get(n, 0) / 1e6, 2),
    } for n in MEMBERS])
elif MEMBERS:
    INV = pd.DataFrame([{
        "name": p,
        "ext": os.path.splitext(p)[1].lower(),
        "MB": round(os.path.getsize(p) / 1e6, 2),
    } for p in MEMBERS])

if len(INV):
    print(f"{len(INV):,} file(s), {INV['MB'].sum():,.0f} MB\n")
    print(INV.groupby("ext").agg(files=("name", "size"), MB=("MB", "sum"))
          .round(1).sort_values("MB", ascending=False).to_string())
    print("\nlargest:")
    print(INV.nlargest(12, "MB").to_string(index=False))
    print("\nlayout (first 15):")
    for n in INV["name"].head(15):
        print("  ", os.path.relpath(n, HERE) if ZF is None else n)

# %% [markdown]
# ## 2 — Look at raw records before mapping anything
#
# One sample per distinct extension, decoded just far enough to reveal
# the field names. A second layer of compression (`.zst`/`.gz`) is
# opened transparently. **Read this before trusting §3** — the field
# map is a guess until a human has seen the real keys.

# %%
BINARY_EXT = (".parquet", ".zip", ".7z", ".gz.tar", ".bin", ".db",
              ".sqlite", ".arrow", ".feather")


def open_raw(name):
    """Binary stream for one member, nested or loose."""
    if ZF is not None:
        return OPENERS[name]()
    return open(name, "rb")


def open_member(name, limit=None):
    """TEXT stream for one member - from the zip or straight off disk,
    handling a nested .zst/.gz either way.

    Refuses binary formats outright. The first version did not, so a
    `.parquet` (or a nested `.zip`) was wrapped in a TextIOWrapper and
    then scanned line by line for delimiters - which is what produced
    the `PK...PAR1...` mojibake in section 2 and made section 4 take
    ages. Binary members belong to read_table_member(), not here.
    """
    low = name.lower()
    if low.endswith(BINARY_EXT):
        raise ValueError(
            "%s is a binary format - use read_table_member()" % name)
    raw = open_raw(name)
    if low.endswith(".zst"):
        import zstandard
        # Reddit dumps use long-distance matching; without the big
        # window this raises "Frame requires too much memory"
        dctx = zstandard.ZstdDecompressor(max_window_size=2 ** 31)
        return io.TextIOWrapper(dctx.stream_reader(raw),
                                encoding="utf-8", errors="replace")
    if low.endswith(".gz"):
        import gzip
        return io.TextIOWrapper(gzip.GzipFile(fileobj=raw),
                                encoding="utf-8", errors="replace")
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace")


def read_table_member(name, columns=None):
    """DataFrame from a parquet member, without copying it to disk.

    pyarrow needs a SEEKABLE stream (the parquet footer lives at the
    end). `_SliceReader` already is one; anything else gets buffered,
    with the ceiling enforced so a bad guess cannot exhaust memory.
    `columns=` is honoured, which is what makes scanning hundreds of
    daily files cheap - parquet is columnar, so asking for two columns
    reads two columns.
    """
    if ZF is None:
        return pd.read_parquet(name, columns=columns)
    fh = open_raw(name)
    if not (hasattr(fh, "seekable") and fh.seekable()):
        size = SIZES.get(name, 0)
        if size > MAX_BUFFER:
            raise MemoryError(
                "%s is %.0f MiB and not seekable; raise MAX_BUFFER or "
                "extract it first" % (name, size / 2 ** 20))
        fh = io.BytesIO(fh.read())
    return pd.read_parquet(fh, columns=columns)


# kept so older cells still resolve
read_parquet_member = read_table_member


def sniff(name, n_lines=3):
    print(f"\n--- {name} ---")
    if name.lower().endswith(".parquet"):
        # Schema first, from the footer alone - no row groups touched.
        try:
            import pyarrow.parquet as pq
            fh = open_raw(name)
            if not (hasattr(fh, "seekable") and fh.seekable()):
                fh = io.BytesIO(fh.read())
            pf = pq.ParquetFile(fh)
            print(f"parquet: {pf.metadata.num_rows:,} rows x "
                  f"{pf.metadata.num_columns} cols, "
                  f"{pf.metadata.num_row_groups} row group(s)")
            print("columns:")
            for f in pf.schema_arrow:
                print(f"    {f.name:<28} {f.type}")
        except Exception as e:                            # noqa: BLE001
            print(f"  (schema read failed: {e})")
        df = read_table_member(name).head(5)
        print(df.to_string())
        return df.head(1).to_dict("records")[0] if len(df) else None
    if name.lower().endswith(BINARY_EXT):
        print("  (binary member, not text - skipped)")
        return None
    try:
        fh = open_member(name)
        head = [next(fh) for _ in range(n_lines)]
    except StopIteration:
        head = []
    except Exception as e:                                # noqa: BLE001
        print(f"  could not read: {e}")
        return None
    if not head:
        print("  (empty)")
        return None
    first = head[0].strip()
    if first.startswith("{") or first.startswith("["):
        try:
            rec = json.loads(first if first.startswith("{")
                             else first.lstrip("[").rstrip(","))
            print("  JSON keys:", sorted(rec.keys()))
            print("  sample   :", json.dumps(rec, ensure_ascii=False)[:400])
            return rec
        except ValueError:
            pass
    print("  first lines:")
    for ln in head:
        print("   ", ln.rstrip()[:300])
    if "," in first or "\t" in first:
        sep = "\t" if first.count("\t") > first.count(",") else ","
        print("  -> delimited; columns:",
              [c.strip() for c in first.split(sep)][:25])
    return None


SAMPLES = {}
if len(INV):
    for ext in INV["ext"].unique():
        biggest = INV[INV["ext"] == ext].nlargest(1, "MB")["name"].iloc[0]
        SAMPLES[ext] = sniff(biggest)

# %% [markdown]
# ## 2b — One member, in full
#
# **This used to load every member into one frame and it blew up:**
# `MemoryError: unable to allocate 1.08 GiB for an array with shape
# (145272537,)`. 145 MILLION rows. The delivery is roughly 1.2 M edges
# *per day*, so concatenating ninety-odd daily files was never going to
# fit — and as `object` dtype, each string column alone wants gigabytes.
#
# The mistake was mine and it was a shape mistake: I sized this for an
# entity-per-day table, and it is an edge list, which is quadratic in
# the number of entities discussed. Nothing here needs the whole thing
# in memory at once. Section 4 streams it instead, reducing each file as
# it is read and keeping only the reductions.
#
# This section reads exactly ONE member — enough to show the schema and
# real rows, which is all it was ever for.

# %%
SAMPLE = pd.DataFrame()
_parqs = [m for m in MEMBERS if m.lower().endswith(".parquet")]
print(f"{len(_parqs):,} parquet member(s) available")
if _parqs:
    _first = _parqs[0]
    SAMPLE = read_table_member(_first)
    print(f"\nreading ONE member for the schema: {_first}")
    print(f"  {len(SAMPLE):,} rows x {len(SAMPLE.columns)} cols "
          f"in this single day")
    print(f"  -> {len(_parqs):,} members at this size would be roughly "
          f"{len(SAMPLE) * len(_parqs) / 1e6:,.0f} M rows. "
          f"Do NOT concatenate them.")
    print("\ncolumns, types, fill:")
    for _c in SAMPLE.columns:
        _nn = SAMPLE[_c].notna().mean() * 100
        _ex = SAMPLE[_c].dropna()
        _ex = repr(_ex.iloc[0])[:44] if len(_ex) else "(all null)"
        print(f"    {_c:<20} {str(SAMPLE[_c].dtype):<14} {_nn:5.1f}% "
              f"· {SAMPLE[_c].nunique():>8,} distinct · e.g. {_ex}")
    print("\nfirst 15 rows, unmodified:")
    with pd.option_context("display.max_columns", None, "display.width", 220,
                           "display.max_colwidth", 28):
        print(SAMPLE.head(15).to_string())
    print("\none row as a dict:")
    print(json.dumps({k: str(v) for k, v in SAMPLE.iloc[0].to_dict().items()},
                     indent=2))
else:
    print("no parquet members - check sections 1 and 2")

# %% [markdown]
# ## 3 — What ICE actually sends
#
# The columns are:
#
# ```
# Count  Date_of_comments
# Entity_1  Entity_1_ISIN  Entity1_Type  ICE_ID_1
# Entity_2  Entity_2_ISIN  Entity2_Type  ICE_ID_2
# ```
#
# This is not a post feed and it is not an entity-per-day table. It is a
# **CO-MENTION EDGE LIST**: one row = *"on this day, these two entities
# were mentioned together `Count` times"*. The unit is a PAIR — which is
# why there are 1.2 M rows a day, and why section 2b refuses to
# concatenate them.
#
# Three consequences, and the third is the interesting one.
#
# **1. There is no sentiment column.** The desk asked to "use theirs
# wherever we have a sentiment score". There isn't one — `ICE_ID_1` and
# `ICE_ID_2` are ICE's internal entity identifiers, the keys behind
# `Entity_1` / `Entity_2`, not scores. `Count` is a frequency. Sentiment
# stays ours, and nothing in this delivery can replace
# `daily_ticker_sentiment`. If ICE has a sentiment product it is a
# different file, and that is a question for them.
#
# **2. ISIN is a gift.** It answers *"is this row about a listed
# instrument?"* exactly — no normalising away INC / LTD / PLC and hoping.
# "Bok Choy" and "Miami Dolphins" drop out for free. We hold no ISIN
# column ourselves (checked: nothing in `config/` or `src/` carries one),
# so the ticker join still goes through names, but the ISIN is carried
# through every output as the stable key. That turns "map ICE to our
# universe" into one Bloomberg lookup later instead of a fuzzy-matching
# project.
#
# **3. The edges are the part we do not already have.** A daily mention
# count per ticker is something this project already computes from posts
# it can audit. But we have NOTHING that says what a name is being
# discussed *alongside*. That is the reason to keep reading.

# %%
ICE_SCHEMA = {
    # role            column name in the delivery
    "date":     "Date_of_comments",
    "count":    "Count",
    "e1":       "Entity_1",
    "e2":       "Entity_2",
    "isin1":    "Entity_1_ISIN",
    "isin2":    "Entity_2_ISIN",
    "type1":    "Entity1_Type",
    "type2":    "Entity2_Type",
    "id1":      "ICE_ID_1",
    "id2":      "ICE_ID_2",
}
SOURCE_TAG = "ice"          # keeps by-source aggregates separable


def _resolve_schema(df):
    """Declared names win; otherwise fall back to a case-insensitive
    match, so a delivery that renames `Count` to `count` still works
    without editing the dict."""
    out, lower = {}, {c.lower(): c for c in df.columns}
    for role, want in ICE_SCHEMA.items():
        out[role] = (want if want in df.columns
                     else lower.get(want.lower()))
    return out


COLS = _resolve_schema(SAMPLE) if len(SAMPLE) else {}
IS_EDGE_LIST = False
if COLS:
    print("schema resolution:")
    for k, v in COLS.items():
        print(f"    {k:<8} -> {v if v else 'MISSING'}")
    _missing = [k for k in ("date", "e1", "e2", "count") if not COLS.get(k)]
    if _missing:
        print(f"\n  CANNOT PROCEED: {_missing} not found. Edit ICE_SCHEMA "
              f"and re-run.")
    else:
        IS_EDGE_LIST = True
        print("\n  edge list confirmed: one row = a PAIR of entities "
              "co-mentioned on a day.")
        print(f"  reading only these columns per file, not all of them.")

# %% [markdown]
# ### 3b — The name → ticker lexicon
#
# Built before section 4, because it is what makes the streaming pass
# affordable: an edge with neither side in our universe can be dropped
# the moment it is read, and that is the overwhelming majority of them.

# %%
_SUFFIX = re.compile(
    r"\b(inc|corp|corporation|co|company|ltd|limited|plc|llc|lp|sa|nv|ag|se|"
    r"spa|holdings?|group|the|class\s+[abc]|adr|ads|technologies|technology|"
    r"international)\b", re.IGNORECASE)


def norm_name(x):
    x = re.sub(r"[^A-Za-z0-9 ]", " ", str(x)).lower()
    return " ".join(_SUFFIX.sub(" ", x).split())


def build_lexicon():
    """{normalised company name -> ticker} from config only.

    A name that maps to two different tickers is DROPPED, not guessed:
    a wrong join would put invented mentions on a real instrument, which
    is worse than a missing one.
    """
    lex, clash = {}, set()

    def add(name, tic):
        k = norm_name(name)
        if not k or len(k) < 3:
            return
        tic = str(tic).strip().upper()
        if k in lex and lex[k] != tic:
            clash.add(k)
            return
        lex[k] = tic

    f = os.path.join(PROJECT_ROOT, "config", "etf_constituents.csv")
    if os.path.exists(f):
        for _, r in pd.read_csv(f).iterrows():
            add(r.get("company"), r.get("ticker"))
    f = os.path.join(PROJECT_ROOT, "config", "approved_instruments.csv")
    if os.path.exists(f):
        for _, r in pd.read_csv(f).iterrows():
            if isinstance(r.get("name"), str) and r["name"].strip():
                add(r["name"], str(r.get("symbol", "")).split()[0])
    for k in clash:
        lex.pop(k, None)
    return lex, clash


LEX, CLASHES = build_lexicon()
print(f"lexicon: {len(LEX):,} company names -> tickers "
      f"({len(CLASHES)} ambiguous dropped)")
print("examples:", list(LEX.items())[:5])


def to_ticker(series, lex=None):
    """Map an entity-name column to tickers, working on the CATEGORIES.

    A day's file is ~1.6 M rows but only a few thousand DISTINCT
    entities. `series.map(norm_name).map(lex)` would run the regex 1.6 M
    times; doing it on the categories runs it a few thousand times and
    then does one integer take. Same answer, roughly two orders of
    magnitude less work - and it matters because this runs once per
    file, per side, per pass.
    """
    lex = LEX if lex is None else lex
    if isinstance(series.dtype, pd.CategoricalDtype):
        cats = series.cat.categories
        mapped = pd.Index([lex.get(norm_name(c)) for c in cats])
        return pd.Series(mapped[series.cat.codes].where(
            series.cat.codes >= 0), index=series.index)
    uniq = pd.Index(series.dropna().unique())
    lut = {u: lex.get(norm_name(u)) for u in uniq}
    return series.map(lut)


# Text columns repeat massively (thousands of distinct entities across
# millions of rows), so category dtype is the difference between ~1 GB
# and ~100 MB for a single day's file.
CATEGORICAL = ("entity_a", "entity_b", "isin_a", "isin_b",
               "type_a", "type_b", "ice_id_a", "ice_id_b")

# %% [markdown]
# ## 4 — The streaming pass
#
# One file at a time. Each file is reduced the moment it is read and the
# full frame is released before the next one is opened, so peak memory
# is one day, not ninety.
#
# Three reductions are kept, and each is small for a different reason:
#
# | kept | grain | why it stays small |
# |---|---|---|
# | `ENT_TOTALS` | entity | one row per entity, not per entity-day |
# | `ICE_EDGES` | date x pair | only pairs with at least ONE side in our universe |
# | `ICE_NODES` | date x our ticker | ~60 instruments x ~90 days |
#
# `ENT_TOTALS` exists so section 5a can still say what the WHOLE feed
# contains — how much of it is listed securities, what the big
# non-security entities are — without ever holding the whole feed.
#
# The filter on `ICE_EDGES` is the load-bearing one. We keep an edge if
# either endpoint maps to an instrument we follow. Edges between two
# entities we do not track cannot contribute to any output in sections
# 6 or 7, so dropping them at read time costs nothing and is the
# difference between this running and not.

# %%
MAX_MEMBERS = None        # None = all; set an int for a quick pass
PROGRESS_EVERY = 10

ICE_EDGES = pd.DataFrame()
ICE_NODES = pd.DataFrame()
ICE_SEC = pd.DataFrame()
ENT_TOTALS = pd.DataFrame()

if IS_EDGE_LIST and _parqs:
    c = COLS
    _want = [v for v in (c["date"], c["count"], c["e1"], c["e2"],
                         c["isin1"], c["isin2"], c["type1"], c["type2"],
                         c["id1"], c["id2"]) if v]
    _targets = _parqs if MAX_MEMBERS is None else _parqs[:MAX_MEMBERS]

    _ent_parts, _meta_parts, _edge_parts = [], [], []
    _rows_read = 0
    _t0 = time.time()

    for _i, _m in enumerate(_targets, 1):
        try:
            df = read_table_member(_m, columns=_want)
        except Exception as _e:                               # noqa: BLE001
            print(f"  [{_m}] {type(_e).__name__}: {_e}")
            continue
        _rows_read += len(df)

        df = df.rename(columns={
            c["date"]: "date", c["count"]: "count",
            c["e1"]: "entity_a", c["e2"]: "entity_b",
            c["isin1"]: "isin_a", c["isin2"]: "isin_b",
            c["type1"]: "type_a", c["type2"]: "type_b",
            c["id1"]: "ice_id_a", c["id2"]: "ice_id_b"})
        for _col in ("isin_a", "isin_b", "type_a", "type_b",
                     "ice_id_a", "ice_id_b"):
            if _col not in df:
                df[_col] = pd.NA
        df["date"] = pd.to_datetime(df["date"].astype(str), errors="coerce",
                                    format="mixed").dt.normalize()
        df["count"] = pd.to_numeric(df["count"], errors="coerce",
                                    downcast="integer").fillna(0)
        df = df[df["date"].notna()]
        for _col in CATEGORICAL:
            if _col in df and not isinstance(df[_col].dtype,
                                             pd.CategoricalDtype):
                df[_col] = df[_col].astype("category")

        # --- reduction 1: entity-level totals over the WHOLE file
        #
        # NEVER group on several CATEGORICAL keys at once. pandas builds
        # the full CROSS-PRODUCT of their levels and fills the empty
        # cells: grouping entity x isin x etype here asked for
        # 11,371,313,712 groups and a 42.4 GiB int32 array, on a file
        # holding a few tens of thousands of real combinations. That is
        # what `observed=` controls, and the default bit us.
        #
        # Two fixes, both applied: `observed=True` so only combinations
        # that OCCUR become groups, and - better - a single-key groupby,
        # because isin and etype are properties OF the entity, not
        # independent dimensions. They are carried separately in a tiny
        # lookup instead.
        _a = df[["entity_a", "count"]].rename(columns={"entity_a": "entity"})
        _b = df[["entity_b", "count"]].rename(columns={"entity_b": "entity"})
        _ent_parts.append(
            pd.concat([_a, _b], ignore_index=True)
            .groupby("entity", observed=True, dropna=False,
                     as_index=False)["count"].sum())

        # entity -> (isin, etype): one row per DISTINCT entity, not per
        # edge. drop_duplicates on the pair columns first so this never
        # touches the full row count either.
        _ma = df[["entity_a", "isin_a", "type_a"]].rename(
            columns={"entity_a": "entity", "isin_a": "isin",
                     "type_a": "etype"}).drop_duplicates("entity")
        _mb = df[["entity_b", "isin_b", "type_b"]].rename(
            columns={"entity_b": "entity", "isin_b": "isin",
                     "type_b": "etype"}).drop_duplicates("entity")
        _meta_parts.append(pd.concat([_ma, _mb], ignore_index=True)
                           .drop_duplicates("entity"))

        # --- reduction 2: keep only edges touching our universe
        _ka = to_ticker(df["entity_a"]).notna()
        _kb = to_ticker(df["entity_b"]).notna()
        _keep = df[_ka | _kb]
        if len(_keep):
            # categories carry the WHOLE day's vocabulary even when two
            # rows survive; drop the unused ones or the accumulator ends
            # up holding every entity string anyway
            _keep = _keep.copy()
            for _col in CATEGORICAL:
                if isinstance(_keep[_col].dtype, pd.CategoricalDtype):
                    _keep[_col] = _keep[_col].cat.remove_unused_categories()
            _edge_parts.append(_keep)

        del df, _a, _b, _keep
        if _i % PROGRESS_EVERY == 0 or _i == len(_targets):
            _kept = sum(len(x) for x in _edge_parts)
            print(f"  {_i}/{len(_targets)} files · {_rows_read:,} edges read "
                  f"· {_kept:,} kept ({100 * _kept / max(_rows_read, 1):.2f}%) "
                  f"· {time.time() - _t0:.0f}s", flush=True)

    if _ent_parts:
        ENT_TOTALS = (pd.concat(_ent_parts, ignore_index=True)
                      .groupby("entity", observed=True, dropna=False,
                               as_index=False)["count"].sum()
                      .rename(columns={"count": "comention_count"}))
        if _meta_parts:
            _meta = (pd.concat(_meta_parts, ignore_index=True)
                     .drop_duplicates("entity"))
            ENT_TOTALS = ENT_TOTALS.merge(_meta, on="entity", how="left")
        for _c in ("entity", "isin", "etype"):
            if _c in ENT_TOTALS and isinstance(ENT_TOTALS[_c].dtype,
                                               pd.CategoricalDtype):
                ENT_TOTALS[_c] = ENT_TOTALS[_c].astype(object)
    if _edge_parts:
        ICE_EDGES = pd.concat(_edge_parts, ignore_index=True)
        # Categories were the right call while STREAMING (they are what
        # kept a 1.6 M-row file at ~100 MB). They are the wrong call for
        # the kept set, because every later groupby uses these columns as
        # KEYS - and a categorical key makes pandas enumerate the full
        # product of levels. Casting back to object here reuses the same
        # interned strings, so it costs a pointer array and removes the
        # whole class of explosion downstream.
        for _c in ("entity_a", "entity_b", "type_a", "type_b"):
            if _c in ICE_EDGES and isinstance(ICE_EDGES[_c].dtype,
                                              pd.CategoricalDtype):
                ICE_EDGES[_c] = ICE_EDGES[_c].astype(object)
    del _ent_parts, _meta_parts, _edge_parts

    print(f"\nread    : {_rows_read:,} edges across {len(_targets):,} files")
    print(f"ENT_TOTALS: {len(ENT_TOTALS):,} distinct entities")
    print(f"ICE_EDGES : {len(ICE_EDGES):,} rows kept "
          f"(at least one side is an instrument we follow)")

# %%
# ---- fold the kept edges into per-name daily figures ------------------
# A name's daily figure is the sum of Count over every edge touching it:
# CO-MENTION volume, not mention volume. An entity discussed alone
# contributes nothing; one in a busy conversation is counted once per
# partner. `degree` - distinct partners that day - is reported beside
# it, because breadth and volume come apart in exactly the situations
# the desk cares about.
if len(ICE_EDGES):
    ICE_EDGES["ta"] = to_ticker(ICE_EDGES["entity_a"])
    ICE_EDGES["tb"] = to_ticker(ICE_EDGES["entity_b"])

    _sa = ICE_EDGES[ICE_EDGES["ta"].notna()].rename(columns={
        "ta": "name", "entity_b": "partner", "isin_a": "isin",
        "type_a": "etype", "ice_id_a": "ice_id"})
    _sb = ICE_EDGES[ICE_EDGES["tb"].notna()].rename(columns={
        "tb": "name", "entity_a": "partner", "isin_b": "isin",
        "type_b": "etype", "ice_id_b": "ice_id"})
    _cols = ["date", "name", "partner", "isin", "etype", "ice_id", "count"]
    NODE_LONG = pd.concat([_sa[_cols], _sb[_cols]], ignore_index=True)

    ICE_NODES = (NODE_LONG.groupby(["date", "name"], observed=True,
                                   as_index=False)
                 .agg(comention_count=("count", "sum"),
                      degree=("partner", "nunique"),
                      isin=("isin", "first"),
                      etype=("etype", "first"),
                      ice_id=("ice_id", "first")))
    ICE_NODES["comention_count"] = (ICE_NODES["comention_count"]
                                    .round().astype("int64"))
    ICE_NODES = ICE_NODES.rename(columns={"name": "entity"})
    ICE_NODES["ticker"] = ICE_NODES["entity"]

    print(f"ICE_NODES : {len(ICE_NODES):,} (ticker, date) rows, "
          f"{ICE_NODES['entity'].nunique():,} instruments, "
          f"{ICE_NODES['date'].nunique():,} days")
    print("\nbusiest ticker-days:")
    print(ICE_NODES.sort_values("comention_count", ascending=False)
          .head(10).to_string(index=False))

if len(ENT_TOTALS):
    _has = (ENT_TOTALS["isin"].notna()
            & ENT_TOTALS["isin"].astype(str).str.strip().ne("")
            & ENT_TOTALS["isin"].astype(str).str.lower().ne("nan"))
    ICE_SEC = ENT_TOTALS[_has].copy()

# %% [markdown]
# ## 5 — What is in it, measured

# %%
# ---- 5a. is it about listed securities, or about everything? ---------
if len(ENT_TOTALS):
    _tot, _sec = len(ENT_TOTALS), len(ICE_SEC)
    _v_all = ENT_TOTALS["comention_count"].sum()
    _v_sec = ICE_SEC["comention_count"].sum() if len(ICE_SEC) else 0
    print(f"entities         : {_sec:,} of {_tot:,} carry an ISIN "
          f"({100 * _sec / max(_tot, 1):.1f}%)")
    print(f"co-mention volume: {100 * _v_sec / max(_v_all, 1):.1f}% involves "
          f"an entity with an ISIN")
    print("\nentity types by volume:")
    print(ENT_TOTALS.groupby("etype", observed=True,
                             dropna=False)["comention_count"].sum()
          .sort_values(ascending=False).head(12).to_string())
    if len(ICE_SEC):
        print("\ntop 20 SECURITIES by co-mention volume:")
        print(ICE_SEC.nlargest(20, "comention_count")
              [["entity", "isin", "comention_count"]].to_string(index=False))
    print("\ntop 15 NON-security entities (the narrative side):")
    print(ENT_TOTALS[~ENT_TOTALS.index.isin(ICE_SEC.index)]
          .nlargest(15, "comention_count")[["entity", "etype",
                                            "comention_count"]]
          .to_string(index=False))

# %%
# ---- 5b. does it reach OUR universe? ---------------------------------
ICE_TICKERS = ICE_NODES.copy() if len(ICE_NODES) else pd.DataFrame()
if len(ENT_TOTALS):
    _m = to_ticker(ENT_TOTALS["entity"])
    print(f"entities matched to one of our tickers: {_m.notna().sum():,} "
          f"of {len(ENT_TOTALS):,}")
    print(f"distinct tickers reached              : {_m.dropna().nunique():,}")
    _vol_matched = ENT_TOTALS.loc[_m.notna(), "comention_count"].sum()
    print(f"share of ALL co-mention volume they carry: "
          f"{100 * _vol_matched / max(ENT_TOTALS['comention_count'].sum(), 1):.1f}%")
    if len(ICE_SEC):
        _unm = ICE_SEC[to_ticker(ICE_SEC["entity"]).isna()]
        print("\nBIGGEST UNMATCHED entities that DO carry an ISIN "
              "(real securities we have no name for):")
        print(_unm.nlargest(20, "comention_count")
              [["entity", "isin", "comention_count"]].to_string(index=False))
        print("\n  ^ each is a one-line fix once ISINs are mapped: "
              "pull_bloomberg_prices.py\n    already talks to blpapi and "
              "ISIN -> ticker is one reference request.\n    That is the "
              "cheapest way to widen this.")

# %%
# ---- 5c. is the daily series continuous? -----------------------------
if len(ICE_NODES):
    dd = ICE_NODES["date"]
    span = pd.date_range(dd.min(), dd.max(), freq="D")
    print(f"span     : {dd.min().date()} -> {dd.max().date()}")
    print(f"days     : {dd.nunique():,} of {len(span):,} "
          f"({100 * dd.nunique() / len(span):.0f}% dense)")
    per = dd.value_counts().sort_index()
    print(f"rows/day : median {per.median():,.0f}, min {per.min():,.0f}, "
          f"max {per.max():,.0f}")
    _gaps = sorted(set(span.date) - set(dd.dt.date))
    if _gaps:
        print(f"missing  : {len(_gaps)} day(s), e.g. "
              f"{[str(g) for g in _gaps[:8]]}")

# %%
# ---- 5d. does it AGREE with our own counts? --------------------------
_OURS = os.path.join(PROJECT_ROOT, "ABSTRACTED_DATA",
                     "daily_ticker_counts.parquet")
if len(ICE_NODES) and os.path.exists(_OURS):
    ice_t = (ICE_NODES.groupby(["date", "ticker"], observed=True,
                               as_index=False)
             ["comention_count"].sum())
    ours = pd.read_parquet(_OURS)
    ours["date"] = pd.to_datetime(ours["date"]).dt.normalize()
    lo, hi = ice_t["date"].min(), ice_t["date"].max()
    ours = ours[(ours["date"] >= lo) & (ours["date"] <= hi)]
    j = ice_t.merge(ours, on=["date", "ticker"], how="outer", indicator=True)
    print(f"overlap window {lo.date()} -> {hi.date()}")
    print(j["_merge"].value_counts()
          .rename({"left_only": "ICE only", "right_only": "ours only",
                   "both": "both"}).to_string())
    b = j[j["_merge"] == "both"]
    if len(b) > 5:
        r = b["comention_count"].corr(b["mention_count"], method="spearman")
        print(f"\nSpearman(ICE co-mention volume, our mention count) "
              f"= {r:.3f}  (n={len(b):,})")
        print("  Near 1: ICE is re-measuring what we already have.")
        print("  Near 0: different quantities - EXPECTED here, and the case")
        print("          in which the edges are worth having.")

# %% [markdown]
# ## 6 — Making it useful, without pretending it is posts
#
# Three concrete uses, cheapest first. None of them writes to a live
# store; every output lands in `new_ice_data/` and is gitignored.
#
# **A. A second opinion on attention** — `ice_daily_ticker_counts`, in
# the exact shape of `ABSTRACTED_DATA/daily_ticker_counts.parquet`
# (`date, ticker, mention_count`), tagged `source="ice"`. Useful as a
# CROSS-CHECK, not a replacement: if ICE and our own counts disagree
# sharply on a name-day, one of the two is wrong and it is worth
# knowing which before a call fires on it.
#
# **B. The co-mention graph** — `ice_daily_cooccurrence`
# (`date, ticker_a, ticker_b, count`) restricted to pairs where BOTH
# sides are instruments we follow. This is the layer we have nothing
# like. Concretely it answers: is a name being discussed with its
# sector, or on its own? Retail manias tend to broaden — one name, then
# its peers, then the theme — and a widening neighbourhood is that
# process made visible a step before it shows up in per-name volume.
#
# **C. Narrative attachment** — `ice_daily_narrative`
# (`date, ticker, narrative_entity, count`): our instruments paired with
# NON-security entities. What is attached to a ticker — a person, a
# product, a country, a policy — and when that attachment changes.
#
# The honest limits, stated with the outputs rather than buried:
# co-mention volume is not mention volume; there is no sentiment here;
# the ticker join goes through names until ISINs are mapped; and this
# delivery is a truncated 2024Q1, so every number is a sample.

# %%
_OUT = []


def _emit(name, df):
    if not len(df):
        return
    path = os.path.join(HERE, name + ".parquet")
    df.to_parquet(path, index=False)
    _OUT.append((name, len(df)))


# --- A. our schema, their numbers
if len(ICE_TICKERS):
    A = (ICE_TICKERS.groupby(["date", "ticker"], observed=True,
                             as_index=False)
         ["comention_count"].sum()
         .rename(columns={"comention_count": "mention_count"}))
    A["date"] = pd.to_datetime(A["date"]).astype("datetime64[ns]")
    A["ticker"] = A["ticker"].astype(str)
    A["mention_count"] = A["mention_count"].round().astype("int64")
    _emit("ice_daily_ticker_counts", A)
    B = A.copy()
    B["source"] = SOURCE_TAG
    _emit("ice_daily_ticker_counts_by_source",
          B[["date", "ticker", "source", "mention_count"]])
    print("A) ice_daily_ticker_counts - our columns exactly:")
    print(A.head(6).to_string(index=False))

# --- B. ticker x ticker co-mention
if len(ICE_EDGES):
    ed = ICE_EDGES.copy()
    ed["ta"] = to_ticker(ed["entity_a"])
    ed["tb"] = to_ticker(ed["entity_b"])
    pair = ed[ed["ta"].notna() & ed["tb"].notna() & (ed["ta"] != ed["tb"])]
    if len(pair):
        # undirected: order the pair so A-B and B-A are one row
        lo = pair[["ta", "tb"]].min(axis=1)
        hi = pair[["ta", "tb"]].max(axis=1)
        C = (pair.assign(ticker_a=lo, ticker_b=hi)
             .groupby(["date", "ticker_a", "ticker_b"], observed=True,
                      as_index=False)
             ["count"].sum().rename(columns={"count": "comention_count"}))
        C["comention_count"] = C["comention_count"].round().astype("int64")
        _emit("ice_daily_cooccurrence", C)
        print(f"\nB) ice_daily_cooccurrence - {len(C):,} ticker-pair-days")
        print(C.sort_values("comention_count", ascending=False)
              .head(10).to_string(index=False))
        print("\n   strongest PAIRS over the whole sample:")
        print(C.groupby(["ticker_a", "ticker_b"],
                        observed=True)["comention_count"].sum()
              .sort_values(ascending=False).head(12).to_string())
    else:
        print("\nB) no edge had BOTH sides matched to our universe on this "
              "sample - widen the lexicon (5b) or map ISINs.")

# --- C. ticker x narrative entity
if len(ICE_EDGES):
    ed = ICE_EDGES.copy()
    ed["ta"] = to_ticker(ed["entity_a"])
    ed["tb"] = to_ticker(ed["entity_b"])

    def _blank(x):
        return (x.isna() | x.astype(str).str.strip().eq("")
                | x.astype(str).str.lower().eq("nan"))

    left = ed[ed["ta"].notna() & _blank(ed["isin_b"])][
        ["date", "ta", "entity_b", "type_b", "count"]].rename(
        columns={"ta": "ticker", "entity_b": "narrative_entity",
                 "type_b": "narrative_type"})
    right = ed[ed["tb"].notna() & _blank(ed["isin_a"])][
        ["date", "tb", "entity_a", "type_a", "count"]].rename(
        columns={"tb": "ticker", "entity_a": "narrative_entity",
                 "type_a": "narrative_type"})
    D = pd.concat([left, right], ignore_index=True)
    if len(D):
        D = (D.groupby(["date", "ticker", "narrative_entity",
                        "narrative_type"], observed=True,
                       as_index=False)["count"].sum()
             .rename(columns={"count": "comention_count"}))
        D["comention_count"] = D["comention_count"].round().astype("int64")
        _emit("ice_daily_narrative", D)
        print(f"\nC) ice_daily_narrative - {len(D):,} ticker-narrative-days")
        print("   what our instruments are being discussed ALONGSIDE:")
        print(D.groupby(["ticker", "narrative_entity"],
                        observed=True)["comention_count"]
              .sum().sort_values(ascending=False).head(15).to_string())

for _n, _k in _OUT:
    print(f"\nwrote {_n}.parquet ({_k:,} rows)")
if _OUT:
    print("\nAll of these are in new_ice_data/ and gitignored. Nothing was "
          "written to ABSTRACTED_DATA or data/processed.")


# %% [markdown]
# ## 7 — How this would actually be used
#
# The live pipeline has a fixed shape, and any new source has to enter
# through one of its doors. There are exactly three, and they are not
# equally good.
#
# ```
#   posts.parquet ──► aggregates ──► build_day_frame ──► ML_BANK ──► GET IN / GET OUT
#   (raw text)        daily_ticker_    one row per        9 crowd      frozen cuts
#                     counts/sentiment (name, date)      features
#                                       + PRICE_FEATURES (2)
# ```
#
# **Door 1 — as posts.** Closed. `tools/fold_historical.py` and
# `ingestion/` consume `OUTPUT_COLUMNS` records and derive the
# aggregates themselves. ICE has no post text, no author, no per-post
# engagement. Nothing to fold.
#
# **Door 2 — as the aggregates.** Technically open: section 6A emits
# `date, ticker, mention_count` in exactly the right shape. **Do not use
# it.** `daily_ticker_counts` is the input to every downstream number —
# the `EUPHORIA_MIN_COVERAGE` gate, `e1`/`e3`, `hype_ratio`, the lot. If
# it came from ICE, the desk's headline numbers would rest on extraction
# rules that live in a vendor's code, that we cannot inspect, reproduce
# or fix, and that can change between deliveries without telling us. It
# would also be irreversible in practice: once history is rebuilt on
# vendor counts, the thresholds are fitted to them. Keep 6A as a
# **cross-check**, which is genuinely useful and costs nothing.
#
# **Door 3 — as FEATURES.** This is the one. `build_day_frame` produces
# one row per `(name, date)` and `ML_BANK` names the columns the model
# reads. A new feature is a new column on that frame and a new entry in
# the bank. That door is:
#
# * **reversible** — remove the column, the old model returns exactly;
# * **measurable** — the existing tournament already scores banks
#   against each other, walk-forward, on the episode ledger;
# * **contained** — a vendor feature that degrades is one of eleven
#   inputs, not the substrate everything else stands on.
#
# ### What ICE can say that we cannot
#
# Our nine crowd features all describe ONE name's own attention: its
# level, its rate of change, its acceleration, its mood. Every one is a
# univariate time series per instrument. ICE's edges describe the name's
# **neighbourhood** — who it is being discussed with. That is
# structurally new information, not a re-measurement, which is the only
# honest reason to add a feature.
#
# The specific claim worth testing: **retail manias broaden before they
# peak.** One name, then its peers, then the theme. If that is true, a
# widening co-mention neighbourhood should lead a name's own attention —
# and leading our existing features is exactly what a GET IN needs.
#
# ### The six candidate features
#
# All trailing-only (day *t* uses edges dated <= *t*), all per
# `(name, date)`, all in the same normalised idiom the bank already
# uses (trailing z against the name's own 84-day baseline — `BASELINE`
# in `src/config.py` — so no cross-sectional lookahead).
#
# | feature | what it measures | why it might lead |
# |---|---|---|
# | `ice_degree_z` | distinct co-mention partners today vs own normal | the conversation is widening |
# | `ice_peer_share` | share of volume spent with OTHER instruments we follow | sector talk vs standalone talk |
# | `ice_new_partner_rate` | partners today unseen in the trailing 28d | genuinely new names being pulled in |
# | `ice_narrative_hhi` | concentration over non-security partners | one story, or many |
# | `ice_peer_heat_z` | how hot the name's PEERS are | the neighbourhood lighting up around it |
# | `ice_vol_z` | co-mention volume vs own normal | the control — should overlap our `e1`/`e3` |
#
# `ice_vol_z` is in there deliberately as a **control**. If it is the
# only ICE feature the model uses, ICE is re-measuring attention we
# already have and the graph added nothing — that is a real possible
# outcome and the test should be able to return it.
#
# ### What would have to be true to adopt
#
# Pre-stated here, before any number is computed:
#
# 1. **Enough overlapping history to fit.** The tournament is
#    walk-forward: fit on years < Y, score Y blind. That needs ICE
#    covering at least two full years that also contain graded episodes.
#    **This delivery is a partial 2024Q1 — nowhere near enough.** Every
#    number below is a shape check, not evidence.
# 2. **Coverage alignment.** Candidate days are days that pass
#    `EUPHORIA_MIN_COVERAGE` (100 tagged posts per name per trailing
#    28d), computed from OUR posts. An ICE feature that is null on most
#    of those days cannot help, whatever its correlation elsewhere.
#    Section 7c measures that overlap directly.
# 3. **A real lift, on both heads.** The existing criterion, unchanged:
#    combined test-year AP lift over the two heads must beat the
#    incumbent bank. One head improving while the other degrades is a
#    reject, not a trade-off to argue about.
# 4. **Stability across deliveries.** `ICE_ID_*` must mean the same
#    entity next quarter. Worth asking ICE in writing before any of
#    this is built.
#
# ### The staged plan
#
# * **Stage 0 (now, free).** Ship nothing. Use 6A as a cross-check and
#   watch for name-days where ICE and our counts disagree sharply.
# * **Stage 1 (this section).** Build the features and prove they are
#   causal and joinable. Done below.
# * **Stage 2 (needs 2+ years from ICE).** Add `ICE_BANK` to the
#   tournament as a challenger bank alongside `ML_BANK` and
#   `DESK_ML_BANK`, and let the existing walk-forward machinery judge
#   it. No new evaluation code is needed — that is the point of
#   entering through door 3.
# * **Stage 3.** Adopt only on criterion 3. If it fails, the answer is
#   "measured, rejected, recorded" — which is a result, not a waste.

# %%
# ---- 7a. build the candidate features (trailing only) ----------------
BASELINE_D = 84       # matches src/config.py BASELINE
MIN_DAYS_D = 28       # matches src/config.py MIN_DAYS
NEW_PARTNER_WINDOW = 28


def _tz(s, win=BASELINE_D, minp=MIN_DAYS_D):
    """Trailing z-score: day t uses days <= t only. Shifted by one so a
    day is never normalised against itself - the same discipline the
    live features use."""
    prev = s.shift(1)
    mu = prev.rolling(win, min_periods=minp).mean()
    sd = prev.rolling(win, min_periods=minp).std(ddof=0)
    return (s - mu) / sd.replace(0, pd.NA)


def build_ice_features(edges, lex):
    """(date, name) -> the six candidate features. Pure function of the
    edges handed in, which is what makes the causality check in 7b
    possible: call it again on a truncated history and compare."""
    if not len(edges) or not lex:
        return pd.DataFrame(), []

    ed = edges.copy()
    ed["ta"] = to_ticker(ed["entity_a"], lex)
    ed["tb"] = to_ticker(ed["entity_b"], lex)

    # long form: one row per (date, our ticker, partner); a partner is
    # either a PEER (an instrument we follow too) or NARRATIVE (not one)
    la = ed[ed["ta"].notna()].rename(
        columns={"ta": "name", "entity_b": "partner", "tb": "partner_tkr"})
    lb = ed[ed["tb"].notna()].rename(
        columns={"tb": "name", "entity_a": "partner", "ta": "partner_tkr"})
    L = pd.concat([la[["date", "name", "partner", "partner_tkr", "count"]],
                   lb[["date", "name", "partner", "partner_tkr", "count"]]],
                  ignore_index=True)
    L = L[L["partner_tkr"].isna() | (L["partner_tkr"] != L["name"])]
    if not len(L):
        return pd.DataFrame(), []

    daily = (L.groupby(["date", "name"], observed=True, as_index=False)
             .agg(vol=("count", "sum"), degree=("partner", "nunique")))
    peer = (L[L["partner_tkr"].notna()]
            .groupby(["date", "name"], observed=True,
                     as_index=False)["count"].sum()
            .rename(columns={"count": "peer_vol"}))
    daily = daily.merge(peer, on=["date", "name"], how="left")
    daily["peer_vol"] = daily["peer_vol"].fillna(0.0)
    daily["ice_peer_share"] = (daily["peer_vol"]
                               / daily["vol"].replace(0, pd.NA))

    # narrative concentration: HHI over NON-security partners
    nar = L[L["partner_tkr"].isna()].copy()
    if len(nar):
        tot = nar.groupby(["date", "name"], observed=True)["count"].transform("sum")
        nar["sh2"] = (nar["count"] / tot.replace(0, pd.NA)) ** 2
        daily = daily.merge(
            nar.groupby(["date", "name"], observed=True,
                        as_index=False)["sh2"].sum()
            .rename(columns={"sh2": "ice_narrative_hhi"}),
            on=["date", "name"], how="left")
    else:
        daily["ice_narrative_hhi"] = pd.NA

    # NEW PARTNERS: partners today unseen in the trailing 28 days. A
    # rolling SET is the one thing pandas will not roll for you, so it
    # is done explicitly - and strictly over days BEFORE today, because
    # getting this wrong would be a lookahead dressed as a feature.
    parts = (L.groupby(["name", "date"], observed=True)["partner"]
             .apply(lambda x: set(x)).sort_index())
    rows = []
    for nm, grp in parts.groupby(level=0):
        dates = list(grp.index.get_level_values(1))
        sets = list(grp.values)
        for i, d in enumerate(dates):
            lo = d - pd.Timedelta(days=NEW_PARTNER_WINDOW)
            hist = set()
            for j in range(i):                      # strictly earlier
                if dates[j] > lo:
                    hist |= sets[j]
            today = sets[i]
            rows.append((d, nm,
                         (len(today - hist) / len(today)) if today
                         else float("nan")))
    daily = daily.merge(
        pd.DataFrame(rows, columns=["date", "name",
                                    "ice_new_partner_rate"]),
        on=["date", "name"], how="left")

    daily = daily.sort_values(["name", "date"]).reset_index(drop=True)
    g = daily.groupby("name", group_keys=False)
    daily["ice_vol_z"] = g["vol"].apply(_tz)
    daily["ice_degree_z"] = g["degree"].apply(_tz)

    # PEER HEAT: how hot this name's peers are today, in THEIR own terms.
    # Built from ice_vol_z (already own-history normalised) so it cannot
    # smuggle raw cross-sectional level in through the back door.
    vz = daily[["date", "name", "ice_vol_z"]].rename(
        columns={"name": "partner_tkr", "ice_vol_z": "_pz"})
    daily = daily.merge(
        L[L["partner_tkr"].notna()]
        .merge(vz, on=["date", "partner_tkr"], how="left")
        .groupby(["date", "name"], observed=True,
                 as_index=False)["_pz"].mean()
        .rename(columns={"_pz": "ice_peer_heat_z"}),
        on=["date", "name"], how="left")

    bank = ["ice_degree_z", "ice_peer_share", "ice_new_partner_rate",
            "ice_narrative_hhi", "ice_peer_heat_z", "ice_vol_z"]
    return daily[["date", "name"] + bank].copy(), bank


ICE_FEATURES, ICE_BANK = build_ice_features(ICE_EDGES, LEX)

if len(ICE_FEATURES):
    print(f"ICE_FEATURES: {len(ICE_FEATURES):,} (name, date) rows, "
          f"{ICE_FEATURES['name'].nunique():,} instruments")
    print("\nfill rate per feature (the blanks are the 28-day warm-up, "
          "exactly as the live features have):")
    for c in ICE_BANK:
        print(f"    {c:<24} {ICE_FEATURES[c].notna().mean() * 100:5.1f}%")
    print("\nsample:")
    print(ICE_FEATURES.dropna().head(8).to_string(index=False))
    _emit("ice_features", ICE_FEATURES)
else:
    print("no edges or no lexicon - sections 4 and 5b must run first")

# %%
# ---- 7b. prove the features are CAUSAL -------------------------------
# A feature that peeks at the future looks brilliant in research and
# fails live. The check: recompute every feature on a truncated history
# and require the overlapping values to be IDENTICAL. If day t's value
# changes when later days are removed, it was using them.
if len(ICE_FEATURES):
    _cut = ICE_FEATURES["date"].quantile(0.7)
    _full = (ICE_FEATURES[ICE_FEATURES["date"] <= _cut]
             .set_index(["name", "date"]).sort_index())
    _e = ICE_EDGES[ICE_EDGES["date"] <= _cut]
    print(f"recomputing on history truncated at {_cut.date()} "
          f"({len(_e):,} of {len(ICE_EDGES):,} edges) ...")
    _t, _ = build_ice_features(_e, LEX)
    _trunc = _t.set_index(["name", "date"]).sort_index()
    _common = _full.index.intersection(_trunc.index)
    print(f"comparing {len(_common):,} overlapping (name, date) rows")
    _bad = []
    for c in ICE_BANK:
        a = pd.to_numeric(_full.loc[_common, c], errors="coerce")
        b = pd.to_numeric(_trunc.loc[_common, c], errors="coerce")
        same = float(((a.isna() & b.isna())
                      | ((a - b).abs() < 1e-9)).mean())
        print(f"    {c:<24} {same * 100:6.2f}% identical")
        if same < 0.999:
            _bad.append(c)
    print("\n  CAUSAL: every feature reproduces exactly from truncated "
          "history." if not _bad else
          f"\n  !! LOOKAHEAD in {_bad} - do NOT take these to stage 2.")

# %%
# ---- 7c. would they even be available on candidate days? -------------
# Criterion 2. Candidate days are days that pass EUPHORIA_MIN_COVERAGE,
# computed from OUR posts. An ICE feature that is null on most of them
# cannot help however well it behaves elsewhere.
_OURS = os.path.join(PROJECT_ROOT, "ABSTRACTED_DATA",
                     "daily_ticker_counts.parquet")
if len(ICE_FEATURES) and os.path.exists(_OURS):
    ours = pd.read_parquet(_OURS)
    ours["date"] = pd.to_datetime(ours["date"]).dt.normalize()
    lo, hi = ICE_FEATURES["date"].min(), ICE_FEATURES["date"].max()
    ours = ours[(ours["date"] >= lo) & (ours["date"] <= hi)]
    ours_days = set(zip(ours["date"], ours["ticker"]))
    ice_days = set(zip(ICE_FEATURES["date"], ICE_FEATURES["name"]))
    both = ours_days & ice_days
    print(f"window {lo.date()} -> {hi.date()}")
    print(f"  our (ticker, day) rows with any mentions : {len(ours_days):,}")
    print(f"  ICE (ticker, day) rows                   : {len(ice_days):,}")
    print(f"  present in BOTH                          : {len(both):,} "
          f"({100 * len(both) / max(len(ours_days), 1):.1f}% of ours)")
    if len(both) < 0.3 * max(len(ours_days), 1):
        print("\n  On this sample ICE reaches a minority of the name-days we")
        print("  actually score. Before stage 2, that number has to come up -")
        print("  by mapping ISINs (5b) so more of ICE's securities resolve to")
        print("  our tickers, or by accepting the features only on the subset")
        print("  where ICE is present and letting the model see NaN elsewhere")
        print("  (HistGradientBoosting handles missing natively; the logit")
        print("  half of the ens pair does NOT, so this needs a decision, not")
        print("  an assumption).")

# %%
# ---- 7d. the exact patch stage 2 would apply -------------------------
print("""
Stage 2 is a three-line change plus a merge - written out here so the
size of the commitment is visible before anyone starts.

  1. analytics/ml_detector.py
         ICE_BANK = ["ice_degree_z", "ice_peer_share",
                     "ice_new_partner_rate", "ice_narrative_hhi",
                     "ice_peer_heat_z", "ice_vol_z"]
         DESK_PLUS_ICE = DESK_ML_BANK + ICE_BANK

  2. analytics/euphoria_phases.py, inside build_day_frame, after the
     price features are attached:
         df = df.merge(ice_features, on=["name", "date"], how="left")
     (LEFT join: ICE must never remove a candidate day. A name-day we
     can score stays scoreable whether or not ICE has an opinion.)

  3. run_ml_tournament gains one more variant per head, so the existing
     walk-forward judge scores DESK_PLUS_ICE against DESK_ML_BANK on the
     same episodes, same folds, same criterion. Note this re-opens the
     tournament: set DESK_MODEL_FAMILY = None in src/config.py for that
     run, because the pinned family was chosen under the OLD bank.

  Nothing else moves. No new store, no new gate, no change to the
  aggregates, no change to any frozen threshold unless the bank wins.

  WHAT IS BLOCKING IT: history. The judge needs at least two full years
  of ICE overlapping graded episodes. This delivery is a partial 2024Q1.
  Everything above is a shape check that the plumbing works - it is not
  evidence that the features help, and it must not be quoted as any.
""")

# %% [markdown]
# ## 8 — Stage nothing, decide deliberately
#
# Nothing here is adopted. The parquet files section 6 wrote sit in
# `new_ice_data/`, which is gitignored, and no live store was touched.
# Adoption would be a separate, deliberate piece of work — and for this
# source it is NOT a fold: `tools/fold_historical.py` consumes posts,
# and there are no posts here.

# %%
print("files this notebook produced (all inside new_ice_data/, all "
      "gitignored):")
_any = False
for _f in sorted(os.listdir(HERE)):
    if _f.startswith("ice_") and _f.endswith(".parquet"):
        _any = True
        _p = os.path.join(HERE, _f)
        print(f"    {_f:<42} {os.path.getsize(_p) / 1024:>8,.0f} KB")
if not _any:
    print("    (none - run sections 4-6)")
print("\nNOTHING was written to ABSTRACTED_DATA/ or data/processed/.")

# %% [markdown]
# ## 9 — Verdict
#
# Fill in from sections 5, 6 and 7.
#
# | question | answer |
# |---|---|
# | Read complete, or SALVAGED partial? (§1c) | |
# | — if salvaged, share of bytes recovered | |
# | Shape: posts / entity-days / **co-mention edges** | |
# | Sentiment available? | |
# | % of entity-days carrying an ISIN (§5a) | |
# | % matched to our tickers by name (§5b) | |
# | Daily continuity (§5c) | |
# | Spearman vs our own counts (§5d) | |
# | **Worth building A / B / C?** | |
#
# ### How to read the answer
#
# The mention counts (use **A**) are the least interesting part. We
# already compute daily attention from posts we can audit, under
# extraction rules that live in this repo; importing a vendor's version
# means our headline numbers would come from rules we cannot inspect or
# reproduce, which is exactly what the abstracted-data boundary exists
# to prevent. Keep A as a **cross-check** — a name-day where ICE and we
# disagree sharply is worth a look before a call fires on it — not as an
# input.
#
# The **edges (B and C) are the case for this source**. Nothing in the
# project currently says what a name is being discussed *alongside*.
# If §5d shows a low correlation, that is not a failure — it confirms
# co-mention volume is a different quantity from mention volume, which
# is the whole reason it could add something.
#
# ### What to ask ICE next
#
# 1. **Sentiment.** This delivery has none. If they have a sentiment
#    product, that is the file that would matter for the desk's original
#    ask.
# 2. **The missing parts** (§1b): `part-0003` onward, or a single
#    archive. This one was read at a partial salvage.
# 3. **2024-08 → 2025-12** rather than 2024Q1 — though note that is a
#    separate job from this evaluation, and the torrent dumps plus
#    `tools/fold_historical.py --dumps` remain the route for the gap.
# 4. **An ISIN → ticker mapping**, or confirmation that `ICE_ID_*` is
#    stable across deliveries. §5b shows real securities we simply have
#    no name for; ISINs turn that from a fuzzy-matching problem into one
#    Bloomberg reference request.
#
# ### What this notebook does NOT claim
#
# Not that the source is good — it measures, it does not decide. Not
# that the numbers are final: the archive arrived truncated, so every
# figure is a sample. And nothing here has touched the live pipeline.
