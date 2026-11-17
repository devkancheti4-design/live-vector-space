#!/usr/bin/env python3
"""
code_map.py - a cheap map of a codebase, plus exact numbered pieces to look up.

An LLM reads the map first: each file's (or, for big codebases, each folder's) one-line
intent and the code's shape (cores of similar statements inside the cube). When the
map is not enough, it zooms in and looks up exact code by number:

    python3 code_map.py build live_vector_space.py      # one file
    python3 code_map.py build src/ tools/extra.py       # many files and folders
    python3 code_map.py build lib/ --exclude test tests # skip folders by name
    python3 code_map.py map                             # the map (what the LLM reads first)
    python3 code_map.py files asyncio                   # zoom into a folder: its files ("." = top level)
    python3 code_map.py find fluid svg                  # numbered pieces that mention these words
    python3 code_map.py find timeout --in asyncio       # ... only inside one folder or file
    python3 code_map.py get 41 42                       # the exact code of those pieces
    python3 code_map.py rebuild out/                    # every file back, byte-exact

The map and each `files` page stay within MAP_BUDGET bytes however big the codebase is:
with many entries, files are named without their intents first (`files` still has them).
`find` shows, for each piece, the line that matched.

--store PREFIX keeps several maps apart (default "code": code_map.json, code_store.bin,
code_index.bin). Pieces are compressed with a shared zstd dictionary, so any one of
them can be read without unpacking the rest.
"""

import argparse
import ast
import hashlib
import json
import math
import os
import random
import re
import struct
import textwrap
from collections import Counter, OrderedDict, defaultdict

from compression import zstd

HERE = os.path.dirname(os.path.abspath(__file__))
MAGIC = b"CMAP2"
STRONG = re.compile(r"(?:def |class |async def |@|if __name__|PAGE = |# ---|function |async function "
                    r"|// ---|<style>|<script>|<body>)")
MAX_LINES = 60
COLLAPSE_FILES = 60          # above this many files, the map lists folders, not every file
MAP_BUDGET = 4000            # bytes for a listing, so the map stays this size however big the code
SAMPLE = 12000               # statements used to find the shape of a big codebase
WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+")


def store_paths(prefix):
    base = prefix if os.path.isabs(prefix) else os.path.join(HERE, prefix)
    return base + "_map.json", base + "_store.bin", base + "_index.bin"


# ---- numbered pieces -----------------------------------------------------------

def collect(paths, exclude=()):
    """Expand folders into their .py files; return (root, [absolute file paths])."""
    skip = set(exclude) | {"__pycache__"}
    files, parents = [], []
    for p in paths:
        p = os.path.abspath(p)
        parents.append(os.path.dirname(p))
        if os.path.isdir(p):
            for dirpath, dirnames, names in os.walk(p):
                dirnames[:] = sorted(d for d in dirnames if d not in skip and not d.startswith("."))
                files += [os.path.join(dirpath, n) for n in sorted(names) if n.endswith(".py")]
        else:
            files.append(p)
    return os.path.commonpath(parents), files


def split_pieces(text):
    """Cut a file into pieces at natural starts (def, class, function, sections,
    blank-line breaks). The pieces cover every byte, in order."""
    lines = text.splitlines(keepends=True)
    cuts = [0]
    for i in range(1, len(lines)):
        line, prev, size = lines[i], lines[i - 1], i - cuts[-1]
        col0 = line[:1].strip() and line[:1] not in "})]" and not line.startswith(('"""', "'''", "</"))
        strong = STRONG.match(line) and not prev.startswith("@")
        soft = not prev.strip() and size >= 4
        if (col0 and (strong or soft)) or (size >= MAX_LINES and not prev.strip()) or size >= MAX_LINES + 20:
            cuts.append(i)
    cuts.append(len(lines))
    spans = list(zip(cuts, cuts[1:]))
    return ["".join(lines[a:b]) for a, b in spans], [(a + 1, b) for a, b in spans]


def intent_of(source):
    try:
        doc = ast.get_docstring(ast.parse(source))
    except (SyntaxError, ValueError):
        doc = None
    line = doc.strip().splitlines()[0] if doc and doc.strip() else ""
    if not line:
        line = next((ln.strip("# ").strip() for ln in source.splitlines() if ln.startswith("#") and not ln.startswith("#!")), "")
    return line if len(line) <= 100 else line[:99] + "…"


def label_of(piece):
    first = next((ln.strip() for ln in piece.splitlines() if ln.strip()), "")
    return first if len(first) <= 72 else first[:71] + "…"


def words_of(text, stop):
    found = set()
    for token in WORD.findall(text):
        for part in token.split("_"):
            for w in CAMEL.findall(part):
                w = w.lower()
                if len(w) >= 3 and w not in stop:
                    found.add(w)
        joined = token.lower().replace("_", "")
        if len(joined) >= 3 and joined.isalpha() and joined not in stop:
            found.add(joined)
    return found


def ranges(numbers, limit=None):
    out, run = [], []
    for n in sorted(numbers):
        if run and n == run[-1] + 1:
            run.append(n)
            continue
        if run:
            out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
        run = [n]
    if run:
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    if limit and len(out) > limit:
        return ", ".join(out[:limit]) + f", … (+{len(out) - limit} more ranges)"
    return ", ".join(out)


# ---- the shape, pointing at pieces -------------------------------------------------

def shape_cores(sources, piece_of, total_pieces):
    """Cores of similar statements: position in the cube, telling words and, for
    small codebases, the numbered pieces that hold each core's statements. Big
    codebases use a fixed random sample of statements."""
    import numpy as np
    import live_vector_space as lvs
    import skeleton as sk
    rng, stmts, seen = random.Random(0), [], 0
    for fi, src in enumerate(sources):
        for first, _, col, toks in lvs.lex(src)[0]:
            if not toks:
                continue
            seen += 1
            if len(stmts) < SAMPLE:
                stmts.append((fi, first, col, toks))
            else:
                j = rng.randrange(seen)
                if j < SAMPLE:
                    stmts[j] = (fi, first, col, toks)
    sampled = seen > len(stmts)
    if sampled:
        stmts.sort(key=lambda s: (s[0], s[1]))
    rows = [lvs.features(toks, col) for _, _, col, toks in stmts]
    X = lvs.vectorize(rows)
    labels = sk.kmeans(X, min(8 if seen < 2000 else 12, len(rows)))
    coords, _ = lvs.project(X, [str(i) for i in range(len(rows))], {})
    pos = np.array([[sk.cube(c) for c in row] for row in coords])
    order = [j for j, _ in Counter(labels.tolist()).most_common()]
    names = {j: chr(ord("A") + i) for i, j in enumerate(order)}
    n, df = len(rows), Counter(f for r in rows for f in r)
    total, per = Counter(), {j: Counter() for j in order}
    for r, lab in zip(rows, labels):
        for f, w in r.items():
            kind, _, word = f.partition(":")
            if kind in ("sub", "word") and len(word) >= 3 and word.isalpha() and word not in sk.STOP:
                v = math.log1p(w) * (math.log((1 + n) / (1 + df[f])) + 1)
                total[word] += v
                per[lab][word] += v
    main = pos[labels == order[0]].mean(axis=0)
    scale = seen / n
    cores = []
    for j in order:
        members = labels == j
        size = int(members.sum())
        lift = {w: per[j][w] / size - total[w] / n for w in per[j]}
        center = pos[members].mean(axis=0)
        core = {"name": names[j], "n": int(round(size * scale)), "at": [round(float(v), 2) for v in center],
                "d": round(float(np.linalg.norm(center - main)), 2),
                "words": [w for w, _ in sorted(lift.items(), key=lambda kv: -kv[1])[:4]]}
        if total_pieces <= 2000:
            core["pieces"] = ranges({piece_of[(fi, first)] for (fi, first, _, _), lab in zip(stmts, labels) if lab == j}, limit=30)
        cores.append(core)
    links = Counter()
    if not sampled:
        for (fa, *_), (fb, *_), a, b in zip(stmts, stmts[1:], labels, labels[1:]):
            if fa == fb and a != b:
                links[tuple(sorted((names[a], names[b])))] += 1
    note = f"shape from a random sample of {n:,} of {seen:,} statements" if sampled else ""
    return cores, ", ".join(f"{a}-{b} {c}" for (a, b), c in links.most_common(6)), note


def top_groups(files):
    """Group files by the first folder below their common root."""
    parts = [f["path"].split("/") for f in files]
    depth = 0
    while all(len(p) > depth + 1 for p in parts) and len({p[depth] for p in parts}) == 1:
        depth += 1
    groups = OrderedDict()
    for f, p in zip(files, parts):
        key = "/".join(p[:depth + 1]) + ("/" if len(p) > depth + 1 else "")
        groups.setdefault(key, []).append(f)
    return groups


def display_root(paths):
    """The folder every path shares; listings name it once instead of on every line."""
    common = []
    for level in zip(*(p.split("/")[:-1] for p in paths)):
        if len(set(level)) > 1:
            break
        common.append(level[0])
    return "".join(part + "/" for part in common)


def cut(text, n):
    return text if len(text) <= n else text[:n - 1] + "…"


def rows_of(files, root):
    """What a listing shows: every file or, above COLLAPSE_FILES files, each top-level
    folder (collapsed) and each loose file."""
    groups = OrderedDict((f["path"], [f]) for f in files) if len(files) <= COLLAPSE_FILES else top_groups(files)
    rows = []
    for key, members in groups.items():
        row = {"name": key[len(root):], "folder": key.endswith("/"), "first": members[0]["first"],
               "last": members[-1]["last"], "intent": members[0]["intent"]}
        if row["folder"]:
            init = next((m for m in members if m["path"] == key + "__init__.py"), members[0])
            row.update(intent=init["intent"], n=len(members), kb=sum(m["bytes"] for m in members) / 1000)
        rows.append(row)
    return rows


def row_line(r, intent_len, numbers):
    about = f": {cut(r['intent'] or '(no docstring)', intent_len)}" if intent_len else ""
    nums = f" -> pieces {r['first']}-{r['last']}" if numbers else ""
    size = f" ({r['n']} files, {r['kb']:,.0f} KB)" if r["folder"] else ""
    return f"  {r['name']}{size}{about}{nums}"


# intent length for folders, intent length for files, piece numbers? (most detail first)
LEVELS = ((80, 80, True), (60, 60, False), (60, 0, False), (40, 0, False))


def wrap_names(names):
    return textwrap.wrap(", ".join(names), 100, initial_indent="  ", subsequent_indent="  ",
                         break_long_words=False, break_on_hyphens=False)


def render(rows, budget=MAP_BUDGET):
    """List rows in the most detail that fits the budget. Files lose their intents before
    folders do (a file listed by name alone still has its intent in `files`); if the names
    still do not fit, the biggest files are named and the rest counted."""
    for folder_len, file_len, numbers in LEVELS:
        lines, bare = [], []
        for r in rows:
            if r["folder"] or file_len:
                lines.append(row_line(r, folder_len if r["folder"] else file_len, numbers))
            else:
                bare.append(r["name"])
        text = "\n".join(lines + wrap_names(bare))
        if len(text.encode()) <= budget:
            return text, bool(bare)
    lines = [row_line(r, 40, False) for r in rows if r["folder"]]
    loose = sorted((r for r in rows if not r["folder"]), key=lambda r: -(r["last"] - r["first"]))
    room = budget - len("\n".join(lines).encode()) - 90
    named = []
    for r in loose:
        room -= len(r["name"]) + 2
        if room < 0:
            break
        named.append(r["name"])
    lines += wrap_names(sorted(named))
    if len(named) < len(loose):
        lines.append(f"  … and {len(loose) - len(named)} smaller top-level files: `files .`")
    kept, size = [], 0
    for line in lines:
        size += len(line.encode()) + 1
        if size > budget:
            kept.append(f"  … {len(lines) - len(kept)} more lines left out: zoom in with `files <folder>`")
            break
        kept.append(line)
    return "\n".join(kept), True


def map_text(meta):
    files = meta["files"]
    root = display_root([f["path"] for f in files])
    out = [f"{meta['name']}: {len(files)} file(s), {meta['bytes']:,} bytes in {meta['count']:,} "
           f"numbered exact pieces (rebuilds byte-exact)." + (f" Paths below are inside {root}" if root else "")]
    listing, bare = render(rows_of(files, root))
    if len(files) <= COLLAPSE_FILES:
        out.append("Files:")
    else:
        out.append("Top level, folders collapsed (zoom in with `files <folder>`"
                   + ("; `files .` gives each top-level file's intent" if bare else "") + "):")
    out.append(listing)
    note = f" ({meta['shape_note']})" if meta.get("shape_note") else ""
    out.append(f"Shape: {len(meta['cores'])} cores of similar statements in a cube frame (x y z in [-1, 1]); "
               f"A is the main core; d = distance from A{note}.")
    for c in meta["cores"]:
        x, y, z = c["at"]
        line = f"{c['name']} n={c['n']:,} at ({x:+.2f}, {y:+.2f}, {z:+.2f}) d={c['d']:.2f}  {', '.join(c['words'])}"
        out.append(line + (f"  -> pieces {c['pieces']}" if c.get("pieces") else ""))
    if meta["links"]:
        out.append("lines (file order): " + meta["links"])
    out.append("Look up: `files <folder>` lists a folder's files; `find <words> [--in <folder>]` lists the numbered "
               "pieces that mention them, with the line that matches; `get <numbers>` prints those pieces exactly.")
    return "\n".join(out) + "\n"


# ---- the exact store and the word index ------------------------------------------------

def write_store(path, blobs):
    big = sum(map(len, blobs)) > 2_000_000
    level, sizes = (19, (65536,)) if big else (22, (0, 4096, 16384, 65536))
    best = None
    for size in sizes:
        try:
            d = zstd.train_dict(blobs, size) if size else None
        except zstd.ZstdError:
            continue
        comp = [zstd.compress(b, level=level, zstd_dict=d) for b in blobs]
        raw = d.dict_content if d else b""
        total = len(raw) + sum(map(len, comp))
        if best is None or total < best[0]:
            best = (total, raw, comp)
    if best is None:
        best = (0, b"", [zstd.compress(b, level=level) for b in blobs])
    _, raw, comp = best
    with open(path, "wb") as fh:
        fh.write(MAGIC + struct.pack("<II", len(raw), len(comp)) + raw)
        fh.write(struct.pack(f"<{len(comp)}I", *map(len, comp)))
        for c in comp:
            fh.write(c)
    return len(raw), level


def read_pieces(path, numbers=None):
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:5] != MAGIC:
        raise SystemExit(f"{path} is not a code map store (rebuild it with `build`)")
    dict_len, n = struct.unpack_from("<II", data, 5)
    pos = 13
    d = zstd.ZstdDict(data[pos:pos + dict_len]) if dict_len else None
    pos += dict_len
    lengths = struct.unpack_from(f"<{n}I", data, pos)
    pos += 4 * n
    offsets = []
    for size in lengths:
        offsets.append(pos)
        pos += size
    wanted = range(1, n + 1) if numbers is None else [k for k in numbers if 1 <= k <= n]
    return {k: zstd.decompress(data[offsets[k - 1]:offsets[k - 1] + lengths[k - 1]], zstd_dict=d) for k in wanted}


def load_index(store):
    with open(store_paths(store)[2], "rb") as fh:
        return json.loads(zstd.decompress(fh.read()))


# ---- commands ----------------------------------------------------------------------------

def load_meta(store):
    with open(store_paths(store)[0], encoding="utf-8") as fh:
        return json.load(fh)


def where(meta, k, root=""):
    fi, a, b = meta["spans"][k - 1]
    return f"{meta['files'][fi]['path'][len(root):]} lines {a}-{b}"


def pick_folder(meta, text, root):
    """A folder or file as the model writes it ('.' = top level), and the indices of the files
    inside it. It is read inside the shared root first, then with the shared root taken off,
    because a folder in the root may share the root's name (reef/reef/)."""
    w = text.strip().strip("/")
    readings = ["" if w == "." else w]
    if root and (w + "/").startswith(root):
        readings.append(w[len(root):].strip("/"))
    for want in readings:
        keep = [fi for fi, f in enumerate(meta["files"]) if inside(f["path"][len(root):], want)]
        if keep:
            return want, keep
    return readings[0], []


def inside(rel, want):
    if not want:
        return "/" not in rel
    return rel == want or rel.startswith(want + "/") or ("/" + want + "/") in ("/" + rel) or rel.endswith("/" + want)


def matching_line(text, forms):
    """The first line of a piece that holds a search form, the way the index sees words
    (line offset, stripped text); the piece's first line if none does."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        words = words_of(line, ())
        if any(w.startswith(f) for f in forms for w in words):
            return i, line.strip()
    return 0, next((ln.strip() for ln in lines if ln.strip()), "")


def log(path, **entry):
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")


def cmd_build(args):
    from skeleton import STOP
    root, paths = collect(args.paths, args.exclude)
    files, blobs, spans, labels, sources, piece_of = [], [], [], [], [], {}
    index, total = defaultdict(list), 0
    for fi, path in enumerate(paths):
        raw = open(path, "rb").read()
        source = raw.decode("utf-8")
        pieces, file_spans = split_pieces(source)
        assert "".join(pieces) == source
        first = len(blobs) + 1
        for (a, b), piece in zip(file_spans, pieces):
            k = len(blobs) + 1
            blobs.append(piece.encode("utf-8"))
            spans.append((fi, a, b))
            labels.append(label_of(piece))
            for word in words_of(piece, STOP):
                index[word].append(k)
            for line in range(a, b + 1):
                piece_of[(fi, line)] = k
        files.append({"path": os.path.relpath(path, root), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                      "intent": intent_of(source), "first": first, "last": len(blobs)})
        sources.append(source)
        total += len(raw)
    cores, links, note = shape_cores(sources, piece_of, len(blobs))
    name = args.name or (files[0]["path"] if len(files) == 1 else os.path.basename(os.path.abspath(args.paths[0])))
    meta = {"name": name, "bytes": total, "count": len(blobs), "files": files, "spans": spans,
            "labels": labels, "cores": cores, "links": links, "shape_note": note}
    map_file, store_file, index_file = store_paths(args.store)
    dict_len, level = write_store(store_file, blobs)
    with open(index_file, "wb") as fh:
        fh.write(zstd.compress(json.dumps(index, separators=(",", ":")).encode(), level=19))
    with open(map_file, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    the_map = len(map_text(meta).encode())
    whole = len(zstd.compress(b"".join(blobs), level=level))
    print(f"{name}: {len(files)} file(s), {total:,} bytes -> {len(blobs):,} numbered pieces")
    print(f"  map (what the LLM reads first)       {the_map:>11,} bytes")
    print(f"  exact pieces, zstd + shared dict     {os.path.getsize(store_file):>11,} bytes  (dictionary {dict_len:,})")
    print(f"  word index                           {os.path.getsize(index_file):>11,} bytes")
    print(f"  all files as one zstd -{level} stream     {whole:>11,} bytes")


def cmd_map(args):
    print(map_text(load_meta(args.store)), end="")


def cmd_files(args):
    """A folder's files (sub-folders collapsed when there are many), with intents and
    piece numbers, in pages that each fit MAP_BUDGET."""
    meta = load_meta(args.store)
    root = display_root([f["path"] for f in meta["files"]])
    want, idx = pick_folder(meta, args.folder, root)
    rows = [meta["files"][fi] for fi in idx]
    if not rows:
        out = f"no files under {args.folder}"
    else:
        pages, size = [[]], 0
        for line in (row_line(r, 80, True) for r in rows_of(rows, root)):
            if pages[-1] and size + len(line.encode()) + 1 > MAP_BUDGET:
                pages.append([])
                size = 0
            pages[-1].append(line)
            size += len(line.encode()) + 1
        n = max(1, min(args.page, len(pages)))
        out = f"{want or 'top level'}: {len(rows)} file(s)" + (f", page {n} of {len(pages)}" if len(pages) > 1 else "")
        out += "\n" + "\n".join(pages[n - 1])
        if n < len(pages):
            out += f"\n  (more: `files {args.folder} --page {n + 1}`)"
    print(out)
    log(args.log, cmd="files", args=[args.folder, args.page], bytes=len(out.encode()))


def query_forms(q):
    """A search word as written ('MAX_LINES' -> 'maxlines', 'k-means' -> 'kmeans') plus its
    parts ('max', 'lines'), which are used only when the whole word matches nothing."""
    parts = re.findall(r"[a-z0-9]+", q.lower())
    joined = "".join(parts)
    return joined, {p for p in parts if len(p) >= 3 and p != joined}


def cmd_find(args):
    """Each search word gets its own short list of hits, so no word is crowded out.
    Exact matches come first, ranked by how many of the search words a piece holds."""
    meta, index = load_meta(args.store), load_index(args.store)
    root = display_root([f["path"] for f in meta["files"]])
    keep = None
    if args.within is not None:
        keep = set(pick_folder(meta, args.within, root)[1])
    keys = list(index)

    def pieces_for(form):
        out = set()
        for key in keys:
            if key.startswith(form):
                out.update(index[key])
        return out if keep is None else {k for k in out if meta["spans"][k - 1][0] in keep}
    exact, partial, forms = {}, {}, {}
    for q in args.words:
        whole, parts = query_forms(q)
        exact[q] = pieces_for(whole) if whole else set()
        partial[q] = set().union(*(pieces_for(p) for p in parts)) - exact[q] if parts else set()
        forms[q] = [whole] if exact[q] else sorted(parts)
    per_word = 15 if len(args.words) == 1 else 4
    picked = {q: sorted(exact[q] or partial[q], key=lambda k: (-sum(k in exact[o] for o in args.words), k))[:per_word]
              for q in args.words}
    texts = read_pieces(store_paths(args.store)[1], sorted({k for ks in picked.values() for k in ks}))
    lines = []
    for q in args.words:
        found = exact[q] or partial[q]
        if not found:
            lines.append(f"{q}: no piece mentions it" + (f" in {args.within}" if keep is not None else ""))
            continue
        kind = "" if exact[q] else " (only parts of the word match)"
        lines.append(f"{q}: {len(found)} piece(s){kind}" + (f", first {per_word}" if len(found) > per_word else ""))
        for k in picked[q]:
            fi, a, _ = meta["spans"][k - 1]
            at, text = matching_line(texts[k].decode("utf-8"), forms[q])
            owner = re.match(r"(?:async\s+)?(?:def|class)\s+(\w+)", meta["labels"][k - 1]) if at else None
            lines.append(f"  {k:>6}  {meta['files'][fi]['path'][len(root):]}:{a + at}  {cut(text, 110)}"
                         + (f"  (in {owner.group(1)})" if owner else ""))
    out = "\n".join(lines)
    print(out)
    log(args.log, cmd="find", args=args.words + ([f"--in {args.within}"] if keep is not None else []),
        bytes=len(out.encode()))


def cmd_get(args):
    meta = load_meta(args.store)
    root = display_root([f["path"] for f in meta["files"]])
    out = [f"--- piece {k} ({where(meta, k, root)}) ---\n" + text.decode("utf-8")
           for k, text in read_pieces(store_paths(args.store)[1], args.numbers).items()]
    text = "".join(out)
    print(text, end="" if text.endswith("\n") else "\n")
    log(args.log, cmd="get", args=args.numbers, bytes=len(text.encode()))


def cmd_rebuild(args):
    meta = load_meta(args.store)
    pieces = read_pieces(store_paths(args.store)[1])
    exact = 0
    for f in meta["files"]:
        data = b"".join(pieces[k] for k in range(f["first"], f["last"] + 1))
        target = os.path.join(args.out, f["path"])
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "wb") as fh:
            fh.write(data)
        exact += hashlib.sha256(data).hexdigest() == f["sha256"]
    print(f"rebuilt {len(meta['files'])} file(s) into {args.out}: {exact} of {len(meta['files'])} byte-exact")


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--store", default="code", help="map/store prefix (default: code)")
    ap = argparse.ArgumentParser(description="A cheap map of a codebase, plus exact numbered pieces to look up.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", parents=[common], help="split, index, compress")
    b.add_argument("paths", nargs="*", default=[os.path.join(HERE, "live_vector_space.py")])
    b.add_argument("--name", help="name shown at the top of the map")
    b.add_argument("--exclude", nargs="*", default=[], help="folder names to skip, e.g. test tests")
    sub.add_parser("map", parents=[common], help="print the map")
    fl = sub.add_parser("files", parents=[common], help="list the files under a folder")
    fl.add_argument("folder")
    fl.add_argument("--page", type=int, default=1)
    fl.add_argument("--log")
    f = sub.add_parser("find", parents=[common], help="numbered pieces that mention these words")
    f.add_argument("words", nargs="+")
    f.add_argument("--in", dest="within", help="only pieces in this folder or file")
    f.add_argument("--log")
    g = sub.add_parser("get", parents=[common], help="exact code of these pieces")
    g.add_argument("numbers", nargs="+", type=int)
    g.add_argument("--log")
    r = sub.add_parser("rebuild", parents=[common], help="write every file back")
    r.add_argument("out")
    args = ap.parse_args()
    {"build": cmd_build, "map": cmd_map, "files": cmd_files, "find": cmd_find, "get": cmd_get,
     "rebuild": cmd_rebuild}[args.cmd](args)


if __name__ == "__main__":
    main()
