#!/usr/bin/env python3
"""
shape_codec.py - store the code's 3-D shape as anchor shapes, and get it back perfectly.

1. Snap every statement's position to a grid inside the cube (12 bits per axis:
   4096 steps from wall to wall). The grid is the "double check" frame.
2. Tile the points with anchor shapes: a few cores, each a small shape (segment,
   triangle or tetrahedron) whose corners are anchor points. Every statement is
   tied to one corner, so the shapes cover 100% of the points, and moving a shape
   moves exactly the points tied to it.
3. Store the shapes once; for each statement store only which corner it is tied
   to and its small offset from that corner, written with a bit-level Rice code.
4. Decode, then check every point lands on exactly the same grid cell, and check
   a CRC-32 of the whole grid as well.

    python3 shape_codec.py                     # test on live_vector_space.py
    python3 shape_codec.py other.py --svg shape_map.svg
"""

import argparse
import json
import math
import os
import zlib
from collections import Counter

import numpy as np

import live_vector_space as lvs
from skeleton import cube

BITS = 12
G = 1 << BITS                 # grid steps per axis across the cube


def grid_points(source):
    data, _ = lvs.build(source, {})
    pts = np.array([[cube(c) for c in u["p"]] for u in data["units"]])
    q = np.rint((pts + 1) / 2 * (G - 1)).astype(np.int64)
    return data, pts, q


def crc(q):
    return zlib.crc32(q.astype("<u2").tobytes())


# ---- byte plumbing ----------------------------------------------------------

def put(out, n):
    """Unsigned varint."""
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | 0x80 if n else b)
        if not n:
            return


def zig(v):
    return v * 2 if v >= 0 else -v * 2 - 1


def unzig(n):
    return n // 2 if n % 2 == 0 else -(n + 1) // 2


class Reader:
    def __init__(self, data):
        self.data, self.i = data, 0

    def get(self):
        n = shift = 0
        while True:
            b = self.data[self.i]
            self.i += 1
            n |= (b & 0x7F) << shift
            shift += 7
            if not b & 0x80:
                return n


class BitWriter:
    def __init__(self):
        self.out, self.acc, self.n = bytearray(), 0, 0

    def bits(self, value, count):
        for i in range(count - 1, -1, -1):
            self.acc = (self.acc << 1) | ((value >> i) & 1)
            self.n += 1
            if self.n == 8:
                self.out.append(self.acc)
                self.acc = self.n = 0

    def rice(self, value, k):
        """Rice code: value >> k in unary (ones, then a zero), then the low k bits."""
        q = value >> k
        self.bits((1 << (q + 1)) - 2, q + 1)
        self.bits(value & ((1 << k) - 1), k)

    def done(self):
        return bytes(self.out) + (bytes([self.acc << (8 - self.n)]) if self.n else b"")


class BitReader:
    def __init__(self, data):
        self.data, self.pos = data, 0

    def bits(self, count):
        v = 0
        for _ in range(count):
            v = (v << 1) | ((self.data[self.pos >> 3] >> (7 - (self.pos & 7))) & 1)
            self.pos += 1
        return v

    def rice(self, k):
        q = 0
        while self.bits(1):
            q += 1
        return (q << k) | self.bits(k)


def best_k(values):
    """The Rice parameter that writes these values in the fewest bits."""
    return min(range(20), key=lambda k: sum((v >> k) + 1 + k for v in values))


def entropy_bits(values):
    counts, n = Counter(values), len(values)
    return -sum(k * math.log2(k / n) for k in counts.values())


# ---- plain grid -------------------------------------------------------------

def pack_grid(q):
    out, bits, nbits = bytearray(), 0, 0
    put(out, len(q))
    for v in q.ravel():
        bits, nbits = (bits << BITS) | int(v), nbits + BITS
        while nbits >= 8:
            nbits -= 8
            out.append((bits >> nbits) & 0xFF)
        bits &= (1 << nbits) - 1
    if nbits:
        out.append((bits << (8 - nbits)) & 0xFF)
    return bytes(out)


def unpack_grid(blob):
    r = Reader(blob)
    n = r.get()
    vals, bits, nbits = [], 0, 0
    for byte in blob[r.i:]:
        bits, nbits = (bits << 8) | byte, nbits + 8
        while nbits >= BITS and len(vals) < n * 3:
            nbits -= BITS
            vals.append((bits >> nbits) & (G - 1))
        bits &= (1 << nbits) - 1
    return np.array(vals, dtype=np.int64).reshape(n, 3)


# ---- the path: each point as a step from the previous statement ---------------

def encode_path(q):
    w = BitWriter()
    w.bits(len(q), 16)
    for axis in range(3):
        steps = [zig(int(q[0, axis]))] + [zig(int(q[i, axis] - q[i - 1, axis])) for i in range(1, len(q))]
        k = best_k(steps)
        w.bits(k, 5)
        for s in steps:
            w.rice(s, k)
    return w.done()


def decode_path(blob):
    r = BitReader(blob)
    n = r.bits(16)
    q = np.zeros((n, 3), dtype=np.int64)
    for axis in range(3):
        k, prev = r.bits(5), 0
        for i in range(n):
            prev += unzig(r.rice(k))
            q[i, axis] = prev
    return q


# ---- anchor shapes ------------------------------------------------------------

def kmeans_xyz(P, k, seed=0, iters=60):
    rng = np.random.default_rng(seed)
    k = min(k, len(P))
    C = P[rng.choice(len(P), k, replace=False)].astype(float)
    for _ in range(iters):
        lab = np.argmin(((P[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        new = np.array([P[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
        if np.allclose(new, C):
            break
        C = new
    return np.argmin(((P[:, None, :] - C[None]) ** 2).sum(-1), axis=1), C


def make_shapes(q, cores, corners):
    """Split the points into `cores` groups and give each a shape with up to `corners`
    corners (1 = point, 2 = segment, 3 = triangle, 4 = tetrahedron). Every point is
    then tied to its nearest corner, so the shapes cover 100% of the points."""
    lab, _ = kmeans_xyz(q.astype(float), cores)
    anchors, shape_of = [], []
    for j in range(lab.max() + 1):
        members = q[lab == j]
        if not len(members):
            continue
        _, C = kmeans_xyz(members.astype(float), corners, seed=j + 1)
        for corner in np.rint(C).astype(np.int64):
            anchors.append(corner)
            shape_of.append(j)
    anchors = np.array(anchors)
    tie = np.argmin(((q[:, None, :] - anchors[None]) ** 2).sum(-1), axis=1)
    return anchors, np.array(shape_of), tie


def tie_bits(m):
    return math.ceil(math.log2(m)) if m > 1 else 0


def encode_shapes(q, anchors, tie):
    w = BitWriter()
    w.bits(len(q), 16)
    w.bits(len(anchors), 16)
    for corner in anchors:                     # the shapes' corners, once
        for v in corner:
            w.bits(int(v), BITS)
    for t in tie:                              # which corner each point is tied to
        w.bits(int(t), tie_bits(len(anchors)))
    for axis in range(3):                      # each point's small offset from its corner
        offsets = [zig(int(q[i, axis] - anchors[tie[i], axis])) for i in range(len(q))]
        k = best_k(offsets)
        w.bits(k, 5)
        for o in offsets:
            w.rice(o, k)
    return w.done()


def decode_shapes(blob):
    r = BitReader(blob)
    n, m = r.bits(16), r.bits(16)
    anchors = np.array([[r.bits(BITS) for _ in range(3)] for _ in range(m)], dtype=np.int64)
    tie = np.array([r.bits(tie_bits(m)) for _ in range(n)], dtype=np.int64)
    q = anchors[tie].copy()
    for axis in range(3):
        k = r.bits(5)
        for i in range(n):
            q[i, axis] += unzig(r.rice(k))
    return q


# ---- picture --------------------------------------------------------------------

def shape_map_svg(q, anchors, shape_of, tie, size=1024):
    fx = lvs._fit_axis([float(v) for v in q[:, 0]])
    fy = lvs._fit_axis([float(v) for v in q[:, 1]])

    def at(x, y):
        return fx(float(x)) * size, (1 - fy(float(y))) * size

    pts = [at(x, y) for x, y, _ in q]
    cor = [at(x, y) for x, y, _ in anchors]
    share = Counter(shape_of[tie].tolist())
    font = 'font-family="system-ui, -apple-system, Segoe UI, sans-serif"'
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size + 60}" '
           f'width="{size}" height="{size + 60}">',
           f'<rect width="{size}" height="{size + 60}" fill="#fcfcfb"/>',
           '<g stroke="#898781" stroke-width="1" opacity="0.55">']
    out += [f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{cor[t][0]:.1f}" y2="{cor[t][1]:.1f}"/>'
            for (x, y), t in zip(pts, tie)]
    out.append("</g>")
    for j in sorted(share):
        corners = [cor[i] for i, s in enumerate(shape_of) if s == j]
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in corners)
        out.append(f'<polygon points="{poly}" fill="#52514e" fill-opacity="0.10" stroke="#0b0b0b" '
                   f'stroke-width="2.5" stroke-linejoin="round"/>')
    out.append('<g fill="#0b0b0b">' + "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3"/>' for x, y in pts) + "</g>")
    out.append('<g fill="#fcfcfb" stroke="#0b0b0b" stroke-width="2.5">'
               + "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7"/>' for x, y in cor) + "</g>")
    for j, count in share.items():
        corners = [cor[i] for i, s in enumerate(shape_of) if s == j]
        cx = sum(x for x, _ in corners) / len(corners)
        cy = sum(y for _, y in corners) / len(corners)
        out.append(f'<text x="{cx:.1f}" y="{cy:.1f}" {font} font-size="26" font-weight="600" fill="#0b0b0b" '
                   f'stroke="#fcfcfb" stroke-width="5" paint-order="stroke" text-anchor="middle">'
                   f'{count / len(q):.0%}</text>')
    out.append(f'<text x="24" y="{size + 38}" {font} font-size="22" fill="#52514e">{len(q)} code points · '
               f'{len(share)} shapes · every point tied to a corner (100% covered)</text></svg>\n')
    return "".join(out)


# ---- the test ----------------------------------------------------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Store the code's 3-D shape as anchor shapes and check it decodes perfectly.")
    ap.add_argument("file", nargs="?", default=os.path.join(here, "live_vector_space.py"))
    ap.add_argument("--svg", help="also draw the three-triangle map to this SVG file")
    args = ap.parse_args()
    with open(args.file, encoding="utf-8-sig", errors="replace") as fh:
        source = fh.read()
    data, pts, q = grid_points(source)
    n, check = len(q), crc(q)
    rows = []

    def row(label, size, decoded=None):
        exact = "-" if decoded is None else ("yes" if np.array_equal(decoded, q) and crc(decoded) == check else "NO")
        rows.append((label, size, exact))

    row("JSON text (what the viewer sends)", len(json.dumps([u["p"] for u in data["units"]], separators=(",", ":"))))
    row("float32 binary", len(pts.astype("<f4").tobytes()))
    blob = pack_grid(q)
    row("cube grid, packed 12+12+12 bits", len(blob), unpack_grid(blob))
    blob = encode_path(q)
    row("cube grid + steps along file order (Rice)", len(blob), decode_path(blob))

    tri = make_shapes(q, 3, 3)                  # three triangles, about a third of the points each
    blob = encode_shapes(q, tri[0], tri[2])
    row("3 triangles + offsets (Rice)", len(blob), decode_shapes(blob))

    best = None
    for cores in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64):
        for corners in (1, 2, 3, 4):
            a, s, t = make_shapes(q, cores, corners)
            blob = encode_shapes(q, a, t)
            if best is None or len(blob) < len(best[0]):
                best = (blob, a, s, t, corners)
    blob, a, s, t, corners = best
    kind = {1: "points", 2: "segments", 3: "triangles", 4: "tetrahedra"}[corners]
    row(f"best: {len(set(s.tolist()))} {kind} + offsets (Rice)", len(blob), decode_shapes(blob))
    floor = 36 * len(a) + entropy_bits(t.tolist()) + sum(entropy_bits((q[:, k] - a[t, k]).tolist()) for k in range(3))
    row("entropy floor of that shape model", math.ceil(floor / 8))

    print(f"{n} code points, snapped to a {G}-step grid in the cube ({BITS} bits per axis)\n")
    print(f"{'how the shape is stored':<46}{'bytes':>8}{'per point':>12}   exact copy?")
    for label, size, exact in rows:
        print(f"{label:<46}{size:>8,}{size / n:>10.2f} B   {exact}")

    anchors, shape_of, tie = tri
    share = Counter(shape_of[tie].tolist())
    print("\nthree triangles: " + ", ".join(f"triangle {j + 1} holds {share[j] / n:.0%}" for j in sorted(share))
          + f"  (total {sum(share.values()) / n:.0%})")
    moved_anchors = anchors.copy()
    moved_anchors[shape_of == 0, 0] += round(0.10 / 2 * (G - 1))   # slide triangle 1 by 0.10 along x
    q_moved = moved_anchors[tie] + (q - anchors[tie])
    moved = np.any(q_moved != q, axis=1)
    print(f"slide triangle 1 by 0.10 along x: {moved.mean():.0%} of the points move with it, "
          f"the other {1 - moved.mean():.0%} stay exactly where they were")
    if args.svg:
        with open(args.svg, "w", encoding="utf-8") as fh:
            fh.write(shape_map_svg(q, anchors, shape_of, tie))
        print(f"\nwrote {os.path.abspath(args.svg)}")


if __name__ == "__main__":
    main()
