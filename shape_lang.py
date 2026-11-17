#!/usr/bin/env python3
"""
shape_lang.py - Shape Python: a program as points in the 3-D cube, the shapes those points form,
and exact lookups by number.

The pipeline is the one this project grew step by step:
  1. live_vector_space.py turns every statement into a vector and places it in the cube.
  2. shape_codec.py snaps the points onto the cube's 4096-step grid, exactly.
  3. optimal_shapes.py lets the points choose their own shapes (blob, line, curve, sheet).
  4. Here every shape is named, given its telling words, and pointed at its numbered statements.
     The shape view is what the LLM reads; `get` prints any statement exactly, `find` lists the
     statements that mention a word, and the numbered statements rebuild the file byte-exact.

    python3 shape_lang.py build live_vector_space.py --store lvs   # lvs.shp (shape view) + lvs.json
    python3 shape_lang.py shapes --store lvs                        # the shape view
    python3 shape_lang.py shape S3 --store lvs                      # zoom into a shape: its statements
    python3 shape_lang.py find fluid particles --store lvs          # statements that mention the words
    python3 shape_lang.py get 41 42 --store lvs                     # those statements, exactly
    python3 shape_lang.py rebuild out.py --store lvs                # the file back, byte-exact
"""

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np

import live_vector_space as lvs
import optimal_shapes as osh
import shape_codec as sc
import skeleton as sk

WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
AXES = "xyz"


def read(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read()


def file_lines(source):
    parts = source.split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def words_of(text):
    """Every word of a statement, whole and in parts: 'MAX_LINES' -> maxlines, max, lines."""
    out = set()
    for w in WORD.findall(text):
        out.add(w.lower().replace("_", ""))
        out.update(p.lower() for p in CAMEL.findall(w) if len(p) >= 2)
    return out


def spans(numbers):
    numbers, out = sorted(numbers), []
    for n in numbers:
        if out and n == out[-1][1] + 1:
            out[-1][1] = n
        else:
            out.append([n, n])
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in out)


def telling_words(source, members_of, k=4):
    """For each shape, the words its statements have more of than the file does (as skeleton.py)."""
    lexed = [s for s in lvs.lex(source)[0] if s[3]]
    X, vocab = sk.exact_tfidf([lvs.features(t, c) for _, _, c, t in lexed])
    cols = [(i, f.partition(":")[2]) for i, f in enumerate(vocab)
            if f.partition(":")[0] in ("sub", "word") and len(f.partition(":")[2]) >= 3
            and f.partition(":")[2].isalpha() and f.partition(":")[2] not in sk.STOP]
    overall = X.mean(axis=0)
    out = []
    for members in members_of:
        lift = X[members].mean(axis=0) - overall
        best = {}
        for i, w in cols:
            best[w] = max(best.get(w, -1e9), lift[i])
        out.append([w for w, _ in sorted(best.items(), key=lambda kv: -kv[1])[:k]])
    return out


def where(g):
    """A shape's place and form in the cube, in words and numbers."""
    p = lambda v: "(" + ", ".join(f"{c:+.2f}" for c in v) + ")"
    if g["type"] == "blob":
        return f"clump around {p(g['center'])}, radius {g['radius']:.2f}"
    if g["type"] in ("line", "curve"):
        return f"arm from {p(g['path'][0])} to {p(g['path'][-1])}"
    poly = np.array(g["poly"])
    flat = AXES[int(np.argmin(poly.std(axis=0)))] if len(poly) > 2 else "?"
    return f"flat spread around {p(poly.mean(axis=0))}, thin along {flat}"


def build(path, store):
    source = read(path)
    data, _, q = sc.grid_points(source)
    units = data["units"]
    blob, shapes, lab = osh.find_shapes(q)
    back, _, back_lab = osh.decode(blob)
    exact_points = np.array_equal(back, q) and np.array_equal(back_lab, lab)
    geo, _ = osh.geometry(q, shapes, lab)
    order = sorted(range(len(shapes)), key=lambda j: -int((lab == j).sum()))
    members_of = [np.flatnonzero(lab == j) for j in order]
    words = telling_words(source, members_of)

    lines = file_lines(source)
    chunks, prev = [], 0
    for u in units:                                # statement i owns the lines after statement i-1
        chunks.append("".join(lines[prev:u["end"]]))
        prev = u["end"]
    tail = "".join(lines[prev:])
    assert "".join(chunks) + tail == source, "statements do not rebuild the file"

    shape_of = {}
    for rank, members in enumerate(members_of, 1):
        for i in members:
            shape_of[int(i)] = rank
    name = os.path.basename(path)
    head = (f"#shape-python 2 · {name} · {len(source.encode()):,} bytes · {len(units)} statements, each a point on "
            f"the cube's {sc.G}-step grid · {len(shapes)} shapes the points chose cover them all"
            f"{' (exact)' if exact_points else ''}\n"
            "cube: x y z in [-1, 1]; similar statements sit close together. A clump is a blob of points, an arm a "
            "line or curve of them, a flat spread a sheet. Statements are numbered in file order.\n")
    rows = []
    for rank, (j, members, ws) in enumerate(zip(order, members_of, words), 1):
        rows.append(f"S{rank} {geo[j]['type']} n={len(members)}: {where(geo[j])}; words: {', '.join(ws)}; "
                    f"statements {spans(int(i) + 1 for i in members)}\n")
    foot = ("look up: `shape <S#>` lists a shape's statements (number, line, first line of code); `get <numbers>` "
            "prints statements exactly; `find <words>` lists the statements that mention words.\n")
    view = head + "".join(rows) + foot
    with open(store + ".shp", "w", encoding="utf-8") as fh:
        fh.write(view)
    with open(store + ".json", "w", encoding="utf-8") as fh:
        json.dump({"name": name, "chunks": chunks, "tail": tail, "shape": [shape_of[i] for i in range(len(units))],
                   "lines": [[u["line"], u["end"]] for u in units]}, fh)
    print(f"{name}: {len(units)} statements -> {len(shapes)} shapes; shape view {len(view.encode()):,} bytes; "
          f"points exact: {exact_points}; statements rebuild byte-exact: True")


def load(store):
    with open(store + ".json", encoding="utf-8") as fh:
        return json.load(fh)


def own_text(st, i):
    """Statement i's own lines, without the blank lines before it."""
    a, b = st["lines"][i]
    chunk = st["chunks"][i].split("\n")
    keep = b - a + 1
    body = chunk[-keep - 1:] if chunk[-1] == "" else chunk[-keep:]
    return "\n".join(body).rstrip("\n")


def log(path, **entry):
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")


def cmd_shapes(args):
    with open(args.store + ".shp", encoding="utf-8") as fh:
        print(fh.read(), end="")


def cmd_get(args):
    st = load(args.store)
    out = []
    for n in args.numbers:
        if 1 <= n <= len(st["chunks"]):
            a, b = st["lines"][n - 1]
            out.append(f"== statement {n} (lines {a}-{b}, shape S{st['shape'][n - 1]}) ==\n{own_text(st, n - 1)}")
        else:
            out.append(f"== statement {n}: no such statement ==")
    text = "\n".join(out)
    print(text)
    log(args.log, cmd="get", args=args.numbers, bytes=len(text.encode()))


def cmd_shape(args):
    """Zoom into shapes: each of their statements with its number, line and first line of code."""
    st = load(args.store)
    out = []
    for name in args.names:
        rank = int(name.upper().lstrip("S")) if name.upper().lstrip("S").isdigit() else 0
        members = [i for i, s in enumerate(st["shape"]) if s == rank]
        out.append(f"S{rank}: {len(members)} statement(s)" if members else f"{name}: no such shape")
        for i in members:
            first = own_text(st, i).split("\n")[0].strip()
            first = first if len(first) <= 80 else first[:79] + "…"
            more = " …" if st["lines"][i][1] > st["lines"][i][0] else ""
            out.append(f"  {i + 1:>4}  line {st['lines'][i][0]:<5} {first}{more}")
    text = "\n".join(out)
    print(text)
    log(args.log, cmd="shape", args=args.names, bytes=len(text.encode()))


def cmd_find(args):
    st = load(args.store)
    texts = [own_text(st, i) for i in range(len(st["chunks"]))]
    sets = [words_of(t) for t in texts]
    queries = [re.sub(r"[^a-z0-9]", "", w.lower()) for w in args.words]
    hits = {}
    for q in queries:
        exact = [i for i, s in enumerate(sets) if q in s]
        hits[q] = exact or [i for i, s in enumerate(sets) if any(w.startswith(q) for w in s)]
    per = 15 if len(queries) == 1 else 6
    out = []
    for q in queries:
        ranked = sorted(hits[q], key=lambda i: (-sum(i in hits[o] for o in queries), i))
        out.append(f"{q}: {len(ranked)} statement(s)" + (f", first {per}" if len(ranked) > per else ""))
        for i in ranked[:per]:
            line_no, lines = st["lines"][i][0], texts[i].split("\n")
            at = next((k for k, l in enumerate(lines) if q in words_of(l) or any(w.startswith(q) for w in words_of(l))), 0)
            show = lines[at].strip()
            show = show if len(show) <= 110 else show[:109] + "…"
            out.append(f"  {i + 1:>4}  S{st['shape'][i]:<3} line {line_no + at:<5} {show}")
    text = "\n".join(out)
    print(text)
    log(args.log, cmd="find", args=args.words, bytes=len(text.encode()))


def cmd_rebuild(args):
    st = load(args.store)
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        fh.write("".join(st["chunks"]) + st["tail"])
    print(f"rebuilt {st['name']} -> {args.out}")


def main():
    ap = argparse.ArgumentParser(description="Shape Python: a program as shapes in the cube, with exact lookups.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("file")
    for name in ("build", "shapes", "shape", "get", "find", "rebuild"):
        p = b if name == "build" else sub.add_parser(name)
        p.add_argument("--store", default="shapes")
        if name in ("shape", "get", "find"):
            p.add_argument("--log")
        if name == "shape":
            p.add_argument("names", nargs="+")
        if name == "get":
            p.add_argument("numbers", nargs="+", type=int)
        if name == "find":
            p.add_argument("words", nargs="+")
        if name == "rebuild":
            p.add_argument("out")
    args = ap.parse_args()
    {"build": lambda a: build(a.file, a.store), "shapes": cmd_shapes, "shape": cmd_shape, "get": cmd_get,
     "find": cmd_find, "rebuild": cmd_rebuild}[args.cmd](args)


if __name__ == "__main__":
    main()
