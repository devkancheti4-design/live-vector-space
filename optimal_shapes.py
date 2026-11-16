#!/usr/bin/env python3
"""
optimal_shapes.py - let the code's points choose their own shapes.

No shape is forced. Each region of the 3-D point cloud is fitted with whichever
shape describes it in the fewest bits:

    blob   points clumped around a centre        stores 3 offsets per point
    line   points stretched along an arm         stores 1 position + 2 tiny offsets
    curve  points along a bending arm            stores 1 position + 2 tiny offsets
    sheet  points spread over a flat surface     stores 2 positions + 1 tiny offset

How many shapes there are, which kind each one is, and which points belong to
which are all chosen by the total bit count (minimum description length).
Positions live on the cube's 12-bit grid, so decoding gives back every point
exactly, and a CRC double-checks the whole grid.

    python3 optimal_shapes.py                          # test on live_vector_space.py
    python3 optimal_shapes.py --html optimal_shapes.html
"""

import argparse
import json
import math
import os
from collections import Counter

import numpy as np

import shape_codec as sc

SLOPE = 12            # fixed-point fraction bits for linear coefficients
CURVE = 24            # fixed-point fraction bits for the curve's squared term
COEF_BITS = 20        # stored width of a linear coefficient (signed)
CURVE_BITS = 24       # stored width of a squared-term coefficient (signed)
TYPES = ("blob", "line", "curve", "sheet")


def coef_widths(kind):
    return {"blob": [], "line": [COEF_BITS] * 2, "sheet": [COEF_BITS] * 2,
            "curve": [COEF_BITS, CURVE_BITS] * 2}[kind]


def param_bits(kind):
    return 2 + 3 * sc.BITS + (0 if kind == "blob" else 2) + sum(coef_widths(kind)) + 3 * 5


def others(m):
    return [k for k in range(3) if k != m]


# ---- prediction: where a shape says a point should be (identical when decoding) ----

def predict_axis(shape, idx, d):
    c = shape["coefs"]
    if shape["type"] == "line":
        val = c[idx] * d / (1 << SLOPE)
    else:
        val = c[2 * idx] * d / (1 << SLOPE) + c[2 * idx + 1] * d * d / (1 << CURVE)
    return np.floor(val + 0.5).astype(np.int64)


def sheet_pred(shape, d1, d2):
    b1, b2 = shape["coefs"]
    return np.floor((b1 * d1 + b2 * d2) / (1 << SLOPE) + 0.5).astype(np.int64)


def coded(shape, P):
    """The three numbers stored per point for this shape: positions on the shape,
    then the tiny offsets from it."""
    D = P - shape["a"]
    kind, m = shape["type"], shape["m"]
    if kind == "blob":
        return D.copy()
    o, out = others(m), np.empty_like(D)
    if kind in ("line", "curve"):
        out[:, 0] = D[:, m]
        d = D[:, m].astype(float)
        for idx, k in enumerate(o):
            out[:, idx + 1] = D[:, k] - predict_axis(shape, idx, d)
        return out
    out[:, 0], out[:, 1] = D[:, o[0]], D[:, o[1]]
    out[:, 2] = D[:, m] - sheet_pred(shape, D[:, o[0]].astype(float), D[:, o[1]].astype(float))
    return out


def rebuild(shape, V):
    kind, m = shape["type"], shape["m"]
    D = V.copy()
    if kind in ("line", "curve"):
        o = others(m)
        D[:, m] = V[:, 0]
        d = D[:, m].astype(float)
        for idx, k in enumerate(o):
            D[:, k] = V[:, idx + 1] + predict_axis(shape, idx, d)
    elif kind == "sheet":
        o = others(m)
        D[:, o[0]], D[:, o[1]] = V[:, 0], V[:, 1]
        D[:, m] = V[:, 2] + sheet_pred(shape, D[:, o[0]].astype(float), D[:, o[1]].astype(float))
    return shape["a"] + D


# ---- fitting and costing -------------------------------------------------------

def q_signed(x, frac, width):
    lim = (1 << (width - 1)) - 1
    return max(-lim, min(lim, int(round(float(x) * (1 << frac)))))


def fit(P, kind, m):
    a = np.rint(P.mean(axis=0)).astype(np.int64)
    D = (P - a).astype(float)
    shape = {"type": kind, "a": a, "m": m, "coefs": []}
    if kind in ("line", "curve"):
        d = D[:, m]
        for k in others(m):
            y = D[:, k]
            if kind == "line":
                den = float(d @ d)
                shape["coefs"].append(q_signed(float(d @ y) / den if den else 0.0, SLOPE, COEF_BITS))
            else:
                c, *_ = np.linalg.lstsq(np.stack([d, d * d], axis=1), y, rcond=None)
                shape["coefs"] += [q_signed(c[0], SLOPE, COEF_BITS), q_signed(c[1], CURVE, CURVE_BITS)]
    elif kind == "sheet":
        c, *_ = np.linalg.lstsq(D[:, others(m)], D[:, m], rcond=None)
        shape["coefs"] = [q_signed(v, SLOPE, COEF_BITS) for v in c]
    return shape


def zz(V):
    return np.where(V >= 0, V * 2, -V * 2 - 1)


def stream_cost(values):
    z = zz(values)
    return min(((k, int(((z >> k) + 1 + k).sum())) for k in range(16)), key=lambda kc: kc[1])


def shape_cost(shape, P):
    V = coded(shape, P)
    ks, bits = [], 0
    for j in range(3):
        k, c = stream_cost(V[:, j])
        ks.append(k)
        bits += c
    shape["ks"] = ks
    return param_bits(shape["type"]) + bits


def best_shape(P):
    candidates = [fit(P, "blob", 0)] + [fit(P, kind, m) for kind in ("line", "curve", "sheet") for m in range(3)]
    costs = [shape_cost(s, P) for s in candidates]
    return candidates[int(np.argmin(costs))]


def point_costs(shape, P):
    ks = np.array(shape["ks"])
    return ((zz(coded(shape, P)) >> ks) + 1 + ks).sum(axis=1)


def compact(lab):
    _, inverse = np.unique(lab, return_inverse=True)
    return inverse.astype(np.int64)


def id_bits(count):
    return math.ceil(math.log2(count)) if count > 1 else 0


# ---- the bit stream -------------------------------------------------------------

def encode(q, shapes, lab):
    w = sc.BitWriter()
    w.bits(len(q), 16)
    w.bits(len(shapes), 8)
    for s in shapes:
        w.bits(TYPES.index(s["type"]), 2)
        for v in s["a"]:
            w.bits(int(v), sc.BITS)
        if s["type"] != "blob":
            w.bits(s["m"], 2)
        for c, width in zip(s["coefs"], coef_widths(s["type"])):
            w.bits(c + (1 << (width - 1)), width)
        for k in s["ks"]:
            w.bits(k, 5)
    V = np.zeros_like(q)
    for j, s in enumerate(shapes):
        V[lab == j] = coded(s, q[lab == j])
    bits = id_bits(len(shapes))
    for i in range(len(q)):
        w.bits(int(lab[i]), bits)
        for j in range(3):
            w.rice(sc.zig(int(V[i, j])), shapes[lab[i]]["ks"][j])
    return w.done()


def decode(blob):
    r = sc.BitReader(blob)
    n, count = r.bits(16), r.bits(8)
    shapes = []
    for _ in range(count):
        kind = TYPES[r.bits(2)]
        a = np.array([r.bits(sc.BITS) for _ in range(3)], dtype=np.int64)
        m = r.bits(2) if kind != "blob" else 0
        coefs = [r.bits(w) - (1 << (w - 1)) for w in coef_widths(kind)]
        shapes.append({"type": kind, "a": a, "m": m, "coefs": coefs, "ks": [r.bits(5) for _ in range(3)]})
    bits = id_bits(count)
    lab, V = np.zeros(n, dtype=np.int64), np.zeros((n, 3), dtype=np.int64)
    for i in range(n):
        lab[i] = r.bits(bits)
        for j in range(3):
            V[i, j] = sc.unzig(r.rice(shapes[lab[i]]["ks"][j]))
    q = np.zeros((n, 3), dtype=np.int64)
    for j, s in enumerate(shapes):
        q[lab == j] = rebuild(s, V[lab == j])
    return q, shapes, lab


# ---- the search: shapes, kinds and membership, all chosen by bits ------------------

def find_shapes(q, sizes=(1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32), seeds=(0, 1, 2), rounds=6):
    best = None
    for K in sizes:
        for seed in seeds:
            lab = compact(sc.kmeans_xyz(q.astype(float), K, seed=seed)[0])
            for _ in range(rounds):
                shapes = [best_shape(q[lab == j]) for j in range(lab.max() + 1)]
                new = compact(np.argmin(np.stack([point_costs(s, q) for s in shapes]), axis=0))
                if np.array_equal(new, lab):
                    break
                lab = new
            shapes = [best_shape(q[lab == j]) for j in range(lab.max() + 1)]
            blob = encode(q, shapes, lab)
            if best is None or len(blob) < len(best[0]):
                best = (blob, shapes, lab)
    return best


# ---- a 3-D view of what was found --------------------------------------------------

def to_cube(v):
    return (np.asarray(v, dtype=float) / (sc.G - 1) * 2 - 1).round(4).tolist()


def hull2d(pts):
    pts = sorted(set(pts))
    if len(pts) < 3:
        return pts
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def geometry(q, shapes, lab):
    out, on = [], np.zeros((len(q), 3))
    for j, s in enumerate(shapes):
        P = q[lab == j]
        D = (P - s["a"]).astype(float)
        a = s["a"].astype(float)
        g = {"type": s["type"], "n": int(len(P))}
        if s["type"] == "blob":
            g["center"] = to_cube(a)
            g["radius"] = round(float(np.sqrt((D ** 2).sum(1).mean())) / (sc.G - 1) * 2, 4)
            on[lab == j] = a
        elif s["type"] in ("line", "curve"):
            m, o = s["m"], others(s["m"])
            ds = np.linspace(D[:, m].min(), D[:, m].max(), 32)

            def along(d):
                pt = np.zeros((len(d), 3))
                pt[:, m] = d
                for idx, k in enumerate(o):
                    c = s["coefs"]
                    pt[:, k] = (c[idx] * d / (1 << SLOPE) if s["type"] == "line"
                                else c[2 * idx] * d / (1 << SLOPE) + c[2 * idx + 1] * d * d / (1 << CURVE))
                return pt + a
            g["path"] = [to_cube(p) for p in along(ds)]
            on[lab == j] = along(D[:, m])
        else:
            m, (o0, o1) = s["m"], others(s["m"])
            b1, b2 = s["coefs"]

            def lift(d1, d2):
                pt = np.zeros(3)
                pt[o0], pt[o1], pt[m] = d1, d2, (b1 * d1 + b2 * d2) / (1 << SLOPE)
                return pt + a
            g["poly"] = [to_cube(lift(x, y)) for x, y in hull2d([(float(x), float(y)) for x, y in D[:, [o0, o1]]])]
            on[lab == j] = [lift(x, y) for x, y in D[:, [o0, o1]]]
        out.append(g)
    return out, [to_cube(p) for p in on]


def html_view(q, shapes, lab, title):
    geo, on = geometry(q, shapes, lab)
    data = {"title": title, "points": [to_cube(p) for p in q], "on": on, "of": lab.tolist(), "shapes": geo}
    return PAGE.replace("/*DATA*/", json.dumps(data, separators=(",", ":")))


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Found Shapes</title>
<style>
  :root { color-scheme: light; --surface: #fcfcfb; --panel: rgba(252,252,251,.9); --text-1: #0b0b0b; --text-2: #52514e;
    --muted: #898781; --axis: #c3c2b7; --border: rgba(11,11,11,.1);
    --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #b9b8b1; }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --surface: #1a1a19;
    --panel: rgba(26,26,25,.9); --text-1: #f0efec; --text-2: #c3c2b7; --axis: #383835; --border: rgba(255,255,255,.1);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #6b6a66; } }
  :root[data-theme="dark"] { color-scheme: dark; --surface: #1a1a19; --panel: rgba(26,26,25,.9); --text-1: #f0efec;
    --text-2: #c3c2b7; --axis: #383835; --border: rgba(255,255,255,.1); --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #6b6a66; }
  html, body { margin: 0; height: 100%; overflow: hidden; background: var(--surface); color: var(--text-1);
    font: 13px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
  canvas { display: block; width: 100vw; height: 100vh; cursor: grab; touch-action: none; }
  #hud { position: fixed; top: 12px; left: 12px; max-width: min(340px, calc(100vw - 24px)); padding: 10px 12px;
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px; }
  h1 { margin: 0 0 4px; font-size: 14px; } #sub { color: var(--text-2); }
  #legend { display: grid; grid-template-columns: 10px auto 1fr; gap: 3px 8px; align-items: center; margin-top: 10px; }
  #legend .n { color: var(--text-2); font-variant-numeric: tabular-nums; }
  .dot { width: 9px; height: 9px; border-radius: 50%; }
  #help { position: fixed; left: 12px; bottom: 12px; color: var(--text-2); font-size: 12px; }
</style></head><body>
<canvas id="c"></canvas>
<div id="hud"><h1 id="title"></h1><div id="sub">Every code point is tied to one shape the points themselves formed.
Thin grey lines join each point to where its shape puts it; that gap is all that is stored besides the shapes.</div>
<div id="legend"></div></div>
<div id="help">Drag to rotate · scroll to zoom · space pauses the spin</div>
<script>
"use strict";
const D = /*DATA*/;
const cv = document.getElementById("c"), ctx = cv.getContext("2d");
const KIND = { line: "--s1", curve: "--s2", sheet: "--s3", blob: "--s4" };
const NAMES = { line: "Lines (arms)", curve: "Curves (bending arms)", sheet: "Sheets (flat spreads)", blob: "Blobs (clumps)" };
let T = {}, W = 0, H = 0, yaw = 0.7, pitch = 0.35, zoom = 1, spin = true, drag = null;
function theme() {
  const css = getComputedStyle(document.documentElement), v = (n) => css.getPropertyValue(n).trim();
  T = { surface: v("--surface"), muted: v("--muted"), axis: v("--axis"), kind: {} };
  for (const k in KIND) T.kind[k] = v(KIND[k]);
}
theme();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", theme);
function size() { const r = Math.min(devicePixelRatio || 1, 2); W = innerWidth; H = innerHeight;
  cv.width = W * r; cv.height = H * r; ctx.setTransform(r, 0, 0, r, 0, 0); }
addEventListener("resize", size); size();
const CAM = 6;
function P(p) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = p[0] * cy + p[2] * sy, z1 = p[2] * cy - p[0] * sy, y2 = p[1] * cp - z1 * sp, z2 = p[1] * sp + z1 * cp;
  const f = Math.min(W, H) * 0.27 * zoom * CAM / (CAM - z2);
  return [W / 2 + x1 * f, H / 2 - y2 * f, f];
}
function seg(a, b) { const p = P(a), q = P(b); ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke(); }
function poly(list, close) { ctx.beginPath(); list.forEach((v, i) => { const p = P(v); if (i) ctx.lineTo(p[0], p[1]); else ctx.moveTo(p[0], p[1]); }); if (close) ctx.closePath(); }
function draw() {
  ctx.fillStyle = T.surface; ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = T.axis; ctx.lineWidth = 1;
  for (const a of [-1, 1]) for (const b of [-1, 1]) { seg([-1, a, b], [1, a, b]); seg([a, -1, b], [a, 1, b]); seg([a, b, -1], [a, b, 1]); }
  ctx.strokeStyle = T.muted; ctx.globalAlpha = 0.45;
  D.points.forEach((p, i) => seg(p, D.on[i]));
  ctx.globalAlpha = 1;
  for (const s of D.shapes) {
    const c = T.kind[s.type];
    ctx.strokeStyle = c; ctx.fillStyle = c;
    if (s.type === "blob") {
      const p = P(s.center);
      ctx.globalAlpha = 0.14; ctx.beginPath(); ctx.arc(p[0], p[1], Math.max(4, s.radius * p[2]), 0, 7); ctx.fill();
      ctx.globalAlpha = 1; ctx.lineWidth = 1.5; ctx.stroke();
    } else if (s.type === "sheet") {
      ctx.globalAlpha = 0.2; poly(s.poly, true); ctx.fill(); ctx.globalAlpha = 1; ctx.lineWidth = 1.5; ctx.stroke();
    } else {
      ctx.lineWidth = 3; ctx.lineCap = "round"; poly(s.path, false); ctx.stroke();
    }
  }
  D.points.forEach((p, i) => { const s = P(p); ctx.fillStyle = T.kind[D.shapes[D.of[i]].type];
    ctx.beginPath(); ctx.arc(s[0], s[1], 3, 0, 7); ctx.fill(); });
}
document.getElementById("title").textContent = D.title;
const tally = {};
D.shapes.forEach((s) => { tally[s.type] = tally[s.type] || { shapes: 0, points: 0 }; tally[s.type].shapes++; tally[s.type].points += s.n; });
document.getElementById("legend").innerHTML = Object.keys(KIND).filter((k) => tally[k]).map((k) =>
  `<span class="dot" style="background:var(${KIND[k]})"></span><span>${NAMES[k]}</span>` +
  `<span class="n">${tally[k].shapes} · ${Math.round(100 * tally[k].points / D.points.length)}% of points</span>`).join("");
cv.addEventListener("pointerdown", (e) => { drag = [e.clientX, e.clientY]; cv.setPointerCapture(e.pointerId); });
cv.addEventListener("pointermove", (e) => { if (!drag) return; yaw += (e.clientX - drag[0]) * 0.008;
  pitch = Math.max(-1.4, Math.min(1.4, pitch + (e.clientY - drag[1]) * 0.008)); drag = [e.clientX, e.clientY]; });
cv.addEventListener("pointerup", () => { drag = null; });
cv.addEventListener("wheel", (e) => { e.preventDefault(); zoom = Math.max(0.4, Math.min(6, zoom * Math.exp(-e.deltaY * 0.0015))); }, { passive: false });
addEventListener("keydown", (e) => { if (e.key === " ") { spin = !spin; e.preventDefault(); } });
let last = performance.now();
(function frame(now) { if (spin && !drag) yaw += (now - last) / 1000 * 0.2; last = now; draw(); requestAnimationFrame(frame); })(last);
</script></body></html>
"""


# ---- the test ------------------------------------------------------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Let the code's points choose their own shapes, then check the copy is perfect.")
    ap.add_argument("file", nargs="?", default=os.path.join(here, "live_vector_space.py"))
    ap.add_argument("--html", help="also write a 3-D view of the found shapes to this file")
    args = ap.parse_args()
    with open(args.file, encoding="utf-8-sig", errors="replace") as fh:
        source = fh.read()
    _, _, q = sc.grid_points(source)
    n, check = len(q), sc.crc(q)

    grid = sc.pack_grid(q)
    forced = min((sc.encode_shapes(q, *sc.make_shapes(q, c, k)[::2]) for c in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
                  for k in (1, 2, 3, 4)), key=len)
    blob, shapes, lab = find_shapes(q)
    back, _, back_lab = decode(blob)
    exact = np.array_equal(back, q) and sc.crc(back) == check and np.array_equal(back_lab, lab)

    print(f"{n} code points on the cube's {sc.G}-step grid\n")
    print(f"{'how the shape is stored':<44}{'bytes':>7}{'per point':>12}   exact copy?")
    for label, size, ok in (("cube grid, packed 12+12+12 bits", len(grid), "yes"),
                            ("forced shapes (k-means corners), earlier best", len(forced), "yes"),
                            ("shapes the points chose themselves", len(blob), "yes" if exact else "NO")):
        print(f"{label:<44}{size:>7,}{size / n:>10.2f} B   {ok}")

    kinds = Counter(s["type"] for s in shapes)
    members = Counter(shapes[j]["type"] for j in lab.tolist())
    print(f"\nfound {len(shapes)} shapes covering {len(lab) / n:.0%} of the points:")
    for kind in TYPES:
        if kinds[kind]:
            print(f"  {kind:<6} {kinds[kind]:>3} shapes   {members[kind] / n:>5.0%} of the points")
    if args.html:
        title = f"{n} code points · {len(shapes)} found shapes · {len(blob):,} bytes, perfect copy"
        with open(args.html, "w", encoding="utf-8") as fh:
            fh.write(html_view(q, shapes, lab, title))
        print(f"\nwrote {os.path.abspath(args.html)}")


if __name__ == "__main__":
    main()
