#!/usr/bin/env python3
"""
code_map.py - a cheap map of a Python file, plus exact numbered pieces to look up.

An LLM reads the map first: the file's one-line intent and its shape (a few cores of
similar statements inside the cube, each pointing at the numbered pieces that hold
it). When the map is not enough, it looks up the exact code by number:

    python3 code_map.py build [file]       # write code_map.json + code_store.bin
    python3 code_map.py map                # print the map (what the LLM reads first)
    python3 code_map.py find fluid svg     # which numbered pieces mention these words
    python3 code_map.py get 41 42          # the exact code of those pieces
    python3 code_map.py rebuild out.py     # every piece in order: a byte-exact copy

The pieces are compressed with a shared zstd dictionary, so any one of them can be
read without unpacking the rest.
"""

import argparse
import ast
import hashlib
import json
import os
import re
import struct
from collections import Counter

from compression import zstd

HERE = os.path.dirname(os.path.abspath(__file__))
MAP_FILE = os.path.join(HERE, "code_map.json")
STORE_FILE = os.path.join(HERE, "code_store.bin")
MAGIC = b"CMAP1"
STRONG = re.compile(r"(?:def |class |async def |@|if __name__|PAGE = |# ---|function |async function "
                    r"|// ---|<style>|<script>|<body>)")
MAX_LINES = 60
WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+")


# ---- numbered pieces -----------------------------------------------------------

def split_pieces(text):
    """Cut the file into pieces at natural starts (def, class, function, sections,
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


def ranges(numbers):
    out, run = [], []
    for n in sorted(numbers):
        if run and n == run[-1] + 1:
            run.append(n)
        else:
            if run:
                out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
            run = [n]
    if run:
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    return ", ".join(out)


# ---- the shape, pointing at pieces -------------------------------------------------

def shape_cores(source, piece_of_line):
    import live_vector_space as lvs
    import skeleton as sk
    anchors, links = sk.skeleton(source)
    statements = [s for s in lvs.lex(source)[0] if s[3]]
    X, _ = sk.exact_tfidf([lvs.features(toks, col) for _, _, col, toks in statements])
    labels = sk.kmeans(X, min(8, len(X)))                    # same seed, same cores as the skeleton
    order = [j for j, _ in Counter(labels.tolist()).most_common()]
    cores = []
    for anchor, j in zip(anchors, order):
        pieces = {piece_of_line[first] for (first, _, _, _), lab in zip(statements, labels) if lab == j}
        cores.append({"name": anchor["name"], "n": anchor["n"], "at": [round(float(v), 2) for v in anchor["at"]],
                      "d": round(anchor["d"], 2), "words": anchor["words"], "pieces": ranges(pieces)})
    return cores, ", ".join(f"{a}-{b} {n}" for (a, b), n in links)


def map_text(meta):
    out = [meta["intent"] or meta["file"],
           f"{meta['bytes']:,} bytes in {meta['count']} numbered exact pieces (rebuilds byte-exact).",
           "Shape: 8 cores of similar statements in a cube frame (x y z in [-1, 1]); A is the main core; "
           "d = distance from A; each core lists the pieces that hold its statements."]
    for c in meta["cores"]:
        x, y, z = c["at"]
        out.append(f"{c['name']} n={c['n']} at ({x:+.2f}, {y:+.2f}, {z:+.2f}) d={c['d']:.2f}  "
                   f"{', '.join(c['words'])}  -> pieces {c['pieces']}")
    out.append("lines (file order): " + meta["links"])
    out.append("Look up: `find <words>` lists the numbered pieces that mention them; "
               "`get <numbers>` prints those pieces exactly.")
    return "\n".join(out) + "\n"


# ---- the exact store -----------------------------------------------------------------

def write_store(blobs):
    best = None
    for size in (0, 2048, 4096, 8192, 16384):
        try:
            d = zstd.train_dict(blobs, size) if size else None
        except zstd.ZstdError:
            continue
        comp = [zstd.compress(b, level=22, zstd_dict=d) for b in blobs]
        raw = d.dict_content if d else b""
        total = len(raw) + sum(map(len, comp))
        if best is None or total < best[0]:
            best = (total, raw, comp)
    _, raw, comp = best
    with open(STORE_FILE, "wb") as fh:
        fh.write(MAGIC + struct.pack("<II", len(raw), len(comp)) + raw)
        fh.write(struct.pack(f"<{len(comp)}I", *map(len, comp)))
        for c in comp:
            fh.write(c)
    return len(raw)


def read_pieces(numbers=None):
    with open(STORE_FILE, "rb") as fh:
        data = fh.read()
    if data[:5] != MAGIC:
        raise SystemExit("code_store.bin is not a code map store")
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


# ---- commands ----------------------------------------------------------------------------

def load_meta():
    with open(MAP_FILE, encoding="utf-8") as fh:
        return json.load(fh)


def log(path, **entry):
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")


def cmd_build(args):
    path = os.path.abspath(args.file)
    raw = open(path, "rb").read()
    source = raw.decode("utf-8")
    pieces, spans = split_pieces(source)
    assert "".join(pieces) == source
    piece_of_line = {}
    for k, (a, b) in enumerate(spans, 1):
        for line in range(a, b + 1):
            piece_of_line[line] = k
    try:
        intent = (ast.get_docstring(ast.parse(source)) or "").strip().splitlines()[0]
    except (SyntaxError, IndexError):
        intent = ""
    cores, links = shape_cores(source, piece_of_line)
    meta = {"file": os.path.basename(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "intent": intent, "count": len(pieces), "spans": spans, "labels": [label_of(p) for p in pieces],
            "cores": cores, "links": links}
    dict_len = write_store([p.encode("utf-8") for p in pieces])
    with open(MAP_FILE, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    store, the_map = os.path.getsize(STORE_FILE), len(map_text(meta).encode())
    whole = len(zstd.compress(raw, level=22))
    print(f"{meta['file']}: {len(raw):,} bytes -> {len(pieces)} numbered pieces")
    print(f"  map (what the LLM reads first)       {the_map:>7,} bytes")
    print(f"  exact pieces, zstd + shared dict     {store:>7,} bytes  (dictionary {dict_len:,})")
    print(f"  whole file, zstd -22, no lookups     {whole:>7,} bytes")


def cmd_map(args):
    print(map_text(load_meta()), end="")


def cmd_find(args):
    from skeleton import STOP
    meta = load_meta()
    queries = [w.lower() for w in args.words]
    hits = []
    for k, text in read_pieces().items():
        words = words_of(text.decode("utf-8"), STOP)
        matched = [q for q in queries if any(w.startswith(q) for w in words)]
        if matched:
            hits.append((-len(matched), k, matched))
    hits.sort()
    lines = []
    for _, k, matched in hits[:15]:
        a, b = meta["spans"][k - 1]
        lines.append(f"{k:>3}  lines {a}-{b}  {meta['labels'][k - 1]}  ({', '.join(matched)})")
    out = "\n".join(lines) if lines else "no piece mentions those words"
    print(out)
    log(args.log, cmd="find", args=args.words, bytes=len(out.encode()))


def cmd_get(args):
    meta = load_meta()
    pieces = read_pieces(args.numbers)
    out = []
    for k, text in pieces.items():
        a, b = meta["spans"][k - 1]
        out.append(f"--- piece {k} (lines {a}-{b}) ---\n" + text.decode("utf-8"))
    text = "".join(out)
    print(text, end="" if text.endswith("\n") else "\n")
    log(args.log, cmd="get", args=args.numbers, bytes=len(text.encode()))


def cmd_rebuild(args):
    meta = load_meta()
    data = b"".join(v for _, v in sorted(read_pieces().items()))
    with open(args.out, "wb") as fh:
        fh.write(data)
    exact = hashlib.sha256(data).hexdigest() == meta["sha256"]
    print(f"rebuilt {len(data):,} bytes into {args.out}; byte-exact: {'yes' if exact else 'NO'}")


def main():
    ap = argparse.ArgumentParser(description="A cheap map of a Python file, plus exact numbered pieces to look up.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="split, index, compress")
    b.add_argument("file", nargs="?", default=os.path.join(HERE, "live_vector_space.py"))
    sub.add_parser("map", help="print the map")
    f = sub.add_parser("find", help="numbered pieces that mention these words")
    f.add_argument("words", nargs="+")
    f.add_argument("--log")
    g = sub.add_parser("get", help="exact code of these pieces")
    g.add_argument("numbers", nargs="+", type=int)
    g.add_argument("--log")
    r = sub.add_parser("rebuild", help="write every piece in order")
    r.add_argument("out")
    args = ap.parse_args()
    {"build": cmd_build, "map": cmd_map, "find": cmd_find, "get": cmd_get, "rebuild": cmd_rebuild}[args.cmd](args)


if __name__ == "__main__":
    main()
