#!/usr/bin/env python3
"""
skeleton.py - squeeze a Python file into a few labeled anchors inside the cube.

You don't need every statement's coordinates. Group the statements into a few
"cores" (k-means on the same token vectors live_vector_space.py builds), pin each
core inside the cube frame, label it with its most telling words, and keep only a
few lines: each core's distance from the main core, plus the strongest links
between cores when the file is read top to bottom.

    python3 skeleton.py                         # skeleton of live_vector_space.py
    python3 skeleton.py path/to/code.py --cores 8
    python3 skeleton.py --mode numbers          # positions and lines only, no words
    python3 skeleton.py --mode words            # words only, no numbers
    python3 skeleton.py --stats                 # also report the sizes
"""

import argparse
import gzip
import math
import os
from collections import Counter

import numpy as np

import live_vector_space as lvs

STOP = set("""
the and for with that this from into are was its not but all any can has have one two out per via
each when then than else only also just more most very over under after before while where what
which who how why our your their them they there here these those been being will would should
could may might must does did about again once both few other some such own same too let use used
self none true false return def class import const var function new null undefined args kwargs
value values data item items result list dict str int float len range print get set key keys
""".split())


def exact_tfidf(rows):
    """TF-IDF over the exact vocabulary (no hashing), rows scaled to unit length."""
    vocab = sorted({f for row in rows for f in row})
    index = {f: i for i, f in enumerate(vocab)}
    n = len(rows)
    df = Counter(f for row in rows for f in row)
    X = np.zeros((n, len(vocab)))
    for i, row in enumerate(rows):
        for f, w in row.items():
            X[i, index[f]] = math.log1p(w) * (math.log((1 + n) / (1 + df[f])) + 1)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X, vocab


def kmeans(X, k, seed=0, iters=100):
    """Spherical (cosine) k-means with seeded k-means++, so results repeat."""
    rng = np.random.default_rng(seed)
    n = len(X)
    centers = [X[rng.integers(n)]]
    while len(centers) < k:
        best = np.max(np.stack([X @ c for c in centers]), axis=0)
        weight = np.clip(1 - best, 0, None) ** 2
        centers.append(X[rng.choice(n, p=weight / weight.sum() if weight.sum() > 0 else None)])
    C = np.stack(centers)
    for _ in range(iters):
        labels = np.argmax(X @ C.T, axis=1)
        new = np.stack([X[labels == j].mean(axis=0) if np.any(labels == j) else C[j] for j in range(k)])
        new /= np.linalg.norm(new, axis=1, keepdims=True) + 1e-12
        if np.allclose(new, C):
            break
        C = new
    return np.argmax(X @ C.T, axis=1)


def cube(c, box=1.2):
    """The viewer's soft squeeze into the box, expressed in cube units (-1..1)."""
    limit, knee = box * 0.94, box * 0.94 * 0.7
    m = abs(c)
    if m > knee:
        m = knee + (limit - knee) * math.tanh((m - knee) / (limit - knee))
    return math.copysign(m, c) / box


def skeleton(source, cores=8, words_per_core=4):
    """Return (anchors, links): a few labeled cores in the cube and the lines between them."""
    statements = [s for s in lvs.lex(source)[0] if s[3]]
    rows = [lvs.features(toks, col) for _, _, col, toks in statements]
    X, vocab = exact_tfidf(rows)
    coords, _ = lvs.project(X, [str(i) for i in range(len(X))], {})
    pos = np.array([[cube(c) for c in row] for row in coords])
    labels = kmeans(X, min(cores, len(X)))
    order = [j for j, _ in Counter(labels.tolist()).most_common()]      # biggest group first
    names = {j: chr(ord("A") + i) for i, j in enumerate(order)}
    word_cols = []
    for i, feat in enumerate(vocab):
        kind, _, word = feat.partition(":")
        if kind in ("sub", "word") and len(word) >= 3 and word.isalpha() and word not in STOP:
            word_cols.append((i, word))
    overall = X.mean(axis=0)
    main = pos[labels == order[0]].mean(axis=0)
    anchors = []
    for j in order:
        members = labels == j
        lift = X[members].mean(axis=0) - overall      # what this core has more of than the file does
        best = {}
        for i, word in word_cols:
            best[word] = max(best.get(word, -1e9), lift[i])
        top = [w for w, _ in sorted(best.items(), key=lambda kv: -kv[1])[:words_per_core]]
        center = pos[members].mean(axis=0)
        anchors.append({"name": names[j], "n": int(members.sum()), "at": center,
                        "d": float(np.linalg.norm(center - main)), "words": top})
    links = Counter()
    for a, b in zip(labels, labels[1:]):
        if a != b:
            links[tuple(sorted((names[a], names[b])))] += 1
    return anchors, links.most_common(6)


def render(anchors, links, mode="full"):
    out = [] if mode == "words" else ["cube frame: x y z in [-1, 1]; A is the main core"]
    for a in anchors:
        if mode == "words":
            out.append(f"{a['name']}: {', '.join(a['words'])}")
            continue
        x, y, z = a["at"]
        line = f"{a['name']} n={a['n']} at ({x:+.2f}, {y:+.2f}, {z:+.2f}) d={a['d']:.2f}"
        out.append(line + ("  " + ", ".join(a["words"]) if mode == "full" else ""))
    if mode != "words":
        out.append("lines (file order): " + ", ".join(f"{a}-{b} {n}" for (a, b), n in links))
    return "\n".join(out) + "\n"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Squeeze a Python file into a few labeled anchors in the cube.")
    parser.add_argument("file", nargs="?", default=os.path.join(here, "live_vector_space.py"))
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--mode", choices=("full", "numbers", "words"), default="full")
    parser.add_argument("--stats", action="store_true", help="also report sizes against the source file")
    args = parser.parse_args()
    with open(args.file, encoding="utf-8-sig", errors="replace") as fh:
        source = fh.read()
    text = render(*skeleton(source, args.cores), mode=args.mode)
    print(text, end="")
    if args.stats:
        def sizes(b):
            return f"{len(b):>7,} bytes   gzip {len(gzip.compress(b, 9)):>6,}   ≈{len(b) // 4:>6,} tokens"
        print(f"\nsource    {sizes(source.encode())}")
        print(f"skeleton  {sizes(text.encode())}")


if __name__ == "__main__":
    main()
