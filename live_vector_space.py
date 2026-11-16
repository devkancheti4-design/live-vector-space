#!/usr/bin/env python3
"""
live_vector_space.py - watch a Python file, live, as vectors in 3-D space.

Every statement in the target file becomes a 1024-dimensional vector built
from its tokens: identifiers and their sub-words, keywords, operators, words
in strings and comments, and the statement's syntactic "shape". TF-IDF
weighting keeps rare, telling tokens strong. PCA then squeezes the vectors
down to the three directions of greatest variance, and a small built-in web
viewer draws each statement as an arrow from the origin. Save the file and
the points glide to their new places; similar statements sit close together.

The whole space sits in a box filled with 14,000 particles of random,
maximum-entropy "fluid". They swirl through a divergence-free flow field,
and as the heat drops each particle flows along its statement's vector until
the fluid condenses into the shape of the code. The entropy meter tracks the
disorder falling; press H (Shake) to scramble it and watch the shape re-form.

    python3 live_vector_space.py                  # visualise this very file
    python3 live_vector_space.py path/to/code.py  # visualise any Python file
    python3 live_vector_space.py --port 9000 --no-browser
    python3 live_vector_space.py --square square.svg  # save the square, then exit

Needs numpy. Stop with Ctrl+C.
"""

import argparse
import io
import json
import keyword
import math
import os
import re
import threading
import time
import tokenize
import webbrowser
import zlib
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

DIMS = 1024           # width of the hashed feature space
NEIGHBORS = 5         # nearest neighbours reported for each statement
POLL_SECONDS = 0.2    # how often the watcher looks at the file

GROUPS = ("Definitions", "Control flow", "Assignments", "Other")
FLOW = {"if", "elif", "else", "for", "while", "try", "except", "finally",
        "with", "return", "yield", "raise", "break", "continue", "pass", "assert"}
AUGMENTED = {"+=", "-=", "*=", "/=", "//=", "%=", "**=", "@=",
             "&=", "|=", "^=", ">>=", "<<="}
CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
WORDS = re.compile(r"[A-Za-z]{3,}")
ROUGH_TOKEN = re.compile(
    r'#.*|[A-Za-z_]\w*|\d[\w.]*|"[^"]*"?|'
    r"'[^']*'?|\*\*=|//=|>>=|<<=|->|\*\*|//|>>|<<|[-+*/%@&|^=!<>:]=|\S"
)
STRING_TOKENS = {tokenize.STRING} | {
    getattr(tokenize, name)
    for name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END",
                 "TSTRING_START", "TSTRING_MIDDLE", "TSTRING_END")
    if hasattr(tokenize, name)
}
TOKEN_KIND = {tokenize.NAME: "NAME", tokenize.NUMBER: "NUMBER",
              tokenize.OP: "OP", tokenize.COMMENT: "COMMENT",
              **{t: "STRING" for t in STRING_TOKENS}}


# ---------------------------------------------------------------- lexing ----

def lex(source):
    """Split source into statements: (first_line, last_line, column, tokens).

    tokens is a list of (kind, text) pairs, kind in NAME/NUMBER/OP/STRING/COMMENT.
    Python's own tokenizer is used when the file tokenizes; while it is mid-edit
    (unclosed bracket, unterminated string...) a forgiving line lexer takes over.
    """
    try:
        return list(_python_statements(source)), "python tokenizer"
    except Exception:
        return list(_rough_statements(source)), "line by line (the file doesn't tokenize right now)"


def _python_statements(source):
    tokens, first, last, column = [], 0, 0, 0
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.NEWLINE:
            if tokens:
                yield first, last, column, tokens
            tokens = []
            continue
        kind = TOKEN_KIND.get(tok.type)
        if kind is None:
            continue
        if kind == "COMMENT" and not tokens:       # a comment on its own line
            yield tok.start[0], tok.start[0], tok.start[1], [(kind, tok.string)]
            continue
        if not tokens:
            first, column = tok.start
        last = tok.end[0]
        tokens.append((kind, tok.string))
    if tokens:
        yield first, last, column, tokens


def _rough_statements(source):
    for number, line in enumerate(source.split("\n"), 1):
        text = line.strip()
        if not text:
            continue
        column = len(line) - len(line.lstrip())
        tokens = []
        for match in ROUGH_TOKEN.finditer(text):
            tok = match.group()
            if tok[0] == "#":
                kind = "COMMENT"
            elif tok[0] in "\"'":
                kind = "STRING"
            elif tok[0].isdigit():
                kind = "NUMBER"
            elif tok[0].isalpha() or tok[0] == "_":
                kind = "NAME"
            else:
                kind = "OP"
            tokens.append((kind, tok))
        yield number, number, column, tokens


# ------------------------------------------------------ statement → vector --

def classify(tokens):
    """Return (kind, group index, label) for one statement."""
    code = [(k, t) for k, t in tokens if k != "COMMENT"]
    if not code:
        return "comment", 3, ""
    words = [t for _, t in code]
    head = words[1] if words[0] == "async" and len(words) > 1 else words[0]
    if words[0] == "@":
        return "decorator", 0, ""
    if head in ("def", "class"):
        rest = words[words.index(head) + 1:]
        return head, 0, f"{head} {rest[0]}" if rest else head
    if head in ("import", "from"):
        return "import", 0, ""
    if head in FLOW or (head in ("match", "case") and words[-1] == ":"):
        return head, 1, ""
    depth = 0
    for kind, text in code:
        if kind != "OP":
            continue
        if text in ("(", "[", "{"):
            depth += 1
        elif text in (")", "]", "}"):
            depth -= 1
        elif depth == 0 and (text == "=" or text in AUGMENTED):
            return "assignment", 2, ""
    if all(kind == "STRING" for kind, _ in code):
        return "docstring", 3, ""
    return ("call" if "(" in words else "expression"), 3, ""


def subwords(name):
    return [w.lower() for part in name.split("_") for w in CAMEL.findall(part) if len(w) > 1]


def features(tokens, column):
    """Bag of weighted features describing one statement."""
    bag = Counter()
    shape = []
    for kind, text in tokens:
        if kind == "NAME" and keyword.iskeyword(text):
            bag["kw:" + text] += 1.5
            shape.append(text)
        elif kind == "NAME":
            bag["id:" + text] += 1.0
            for word in subwords(text):
                bag["sub:" + word] += 0.6
            shape.append("ID")
        elif kind == "NUMBER":
            bag["number"] += 0.5
            shape.append("NUM")
        elif kind == "STRING":
            bag["string"] += 0.5
            for word in WORDS.findall(text)[:200]:
                bag["word:" + word.lower()] += 0.3
            shape.append("STR")
        elif kind == "COMMENT":
            bag["comment"] += 1.0
            for word in WORDS.findall(text)[:60]:
                bag["word:" + word.lower()] += 0.5
        else:
            bag["op:" + text] += 0.4
            shape.append(text)
    if shape:
        bag["head:" + shape[0]] += 1.5
    for a, b in zip(shape, shape[1:]):
        bag[f"pair:{a} {b}"] += 0.5
    bag[f"indent:{min(column // 4, 8)}"] += 0.5
    return bag


def vectorize(rows):
    """Hash feature bags into a TF-IDF matrix with unit-length rows."""
    n = len(rows)
    df = Counter()
    for row in rows:
        df.update(row.keys())
    X = np.zeros((n, DIMS))
    for i, row in enumerate(rows):
        for feat, weight in row.items():
            h = zlib.crc32(feat.encode())
            idf = math.log((1 + n) / (1 + df[feat])) + 1.0
            X[i, h % DIMS] += (1.0 if h & 0x80000000 else -1.0) * math.log1p(weight) * idf
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def project(X, ids, previous):
    """PCA down to 3-D, then turn the result to line up with the last frame."""
    n = len(X)
    coords = np.zeros((n, 3))
    explained = [0.0, 0.0, 0.0]
    if n >= 2:
        centered = X - X.mean(axis=0)
        _, sing, vt = np.linalg.svd(centered, full_matrices=False)
        k = min(3, len(sing))
        coords[:, :k] = centered @ vt[:k].T
        total = float((sing ** 2).sum()) or 1.0
        explained[:k] = [float(s * s) / total for s in sing[:k]]
        radius = float(np.percentile(np.linalg.norm(coords, axis=1), 95))
        if radius > 0:
            coords /= radius
    # Orthogonal Procrustes on statements that survived the edit keeps the view steady.
    shared = [i for i, uid in enumerate(ids) if uid in previous]
    if len(shared) >= 4:
        u, _, vt = np.linalg.svd(coords[shared].T @ np.array([previous[ids[i]] for i in shared]))
        coords = coords @ (u @ vt)
    return coords, explained


def nearest(X, k=NEIGHBORS):
    """Top-k cosine neighbours of every row, measured in the full space."""
    n = len(X)
    k = min(k, n - 1)
    if k <= 0:
        return [[] for _ in range(n)]
    out = []
    for start in range(0, n, 512):
        sims = X[start:start + 512] @ X.T
        for r, row in enumerate(sims):
            row[start + r] = -np.inf
            idx = np.argpartition(-row, k - 1)[:k]
            idx = idx[np.argsort(-row[idx])]
            out.append([[int(j), round(float(row[j]), 2)] for j in idx])
    return out


def shorten(text, limit):
    return text if len(text) <= limit else text[:limit - 1] + "…"


def snippet(lines, first, last, column, limit=8):
    body = []
    for line in lines[first - 1:min(last, first - 1 + limit)]:
        body.append(shorten((line[column:] if not line[:column].strip() else line.strip()).rstrip(), 100))
    if last - first + 1 > limit:
        body.append(f"… {last - first + 1 - limit} more lines")
    return "\n".join(body)


def build(source, previous):
    """Turn source text into the viewer's JSON data plus positions for next time."""
    statements, parser = lex(source)
    lines = source.split("\n")
    units, rows, ids, seen = [], [], [], Counter()
    for first, last, column, tokens in statements:
        if not tokens:
            continue
        kind, group, label = classify(tokens)
        key = zlib.crc32(" ".join(" ".join(lines[first - 1:last]).split()).encode())
        seen[key] += 1
        ids.append(f"{key:08x}-{seen[key]}")
        rows.append(features(tokens, column))
        units.append({
            "line": first, "end": last, "kind": kind, "group": group, "label": label,
            "text": shorten(lines[first - 1].strip(), 90) + (" …" if last > first else ""),
            "snippet": snippet(lines, first, last, column),
        })
    X = vectorize(rows) if rows else np.zeros((0, DIMS))
    coords, explained = project(X, ids, previous)
    for unit, uid, xyz, nn in zip(units, ids, coords, nearest(X)):
        unit.update(id=uid, p=[round(float(c), 4) for c in xyz], nn=nn)
    data = {
        "parser": parser,
        "dims": DIMS,
        "count": len(units),
        "explained": [round(e, 4) for e in explained],
        "updated": time.strftime("%H:%M:%S"),
        "units": units,
    }
    return data, dict(zip(ids, coords))


# ------------------------------------------------- the square minus the path ---

def _fit_axis(values):
    """Map values onto 0..1 so the square is used: the middle 90% spreads linearly
    over 12%..88%, and the outliers are eased into the margins."""
    lo, hi = np.percentile(values, [5, 95]) if values else (0.0, 1.0)
    span = float(hi - lo) or 1.0

    def place(v):
        d = 0.12 + 0.76 * (v - lo) / span - 0.5      # offset from the centre
        m = abs(d)
        if m > 0.38:
            m = 0.38 + 0.09 * math.tanh((m - 0.38) / 0.09)
        return 0.5 + math.copysign(m, d)
    return place


def square_svg(units, size=1024, ink="#0b0b0b"):
    """The square minus the code path.

    Every statement is placed on the square by its first two principal components
    (PC1 across, PC2 up, each stretched so the middle 90% of statements fills the
    square). The points are joined in file order, and that path is cut out of a
    solid square: what is left is the square minus the path.
    """
    fx = _fit_axis([u["p"][0] for u in units])
    fy = _fit_axis([u["p"][1] for u in units])
    pts = [(fx(u["p"][0]) * size, (1 - fy(u["p"][1])) * size) for u in units]
    path = " ".join(f"{'L' if i else 'M'}{x:.1f} {y:.1f}" for i, (x, y) in enumerate(pts))
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{size * 0.005:.1f}"/>' for x, y in pts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size}" height="{size}">'
        f'<mask id="cut" maskUnits="userSpaceOnUse" x="0" y="0" width="{size}" height="{size}">'
        f'<rect width="{size}" height="{size}" fill="#fff"/>'
        f'<path d="{path}" fill="none" stroke="#000" stroke-width="{size * 0.003:.1f}" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f'<g fill="#000">{dots}</g></mask>'
        f'<rect width="{size}" height="{size}" fill="{ink}" mask="url(#cut)"/></svg>\n'
    )


# ------------------------------------------------------- live web viewer ---

class State:
    """The latest picture of the file, shared by the watcher and the server."""

    def __init__(self, path):
        self.path = path
        self.version = 0
        self.payload = b'{"version":0,"units":[]}'
        self.units = []
        self.previous = {}
        self.changed = threading.Condition()

    def refresh(self):
        try:
            with open(self.path, encoding="utf-8-sig", errors="replace") as fh:
                source = fh.read()
            data, self.previous = build(source, self.previous)
        except Exception as exc:  # keep showing the last good picture
            print(f"  (skipped an update: {exc!r})", flush=True)
            return
        data["file"] = os.path.basename(self.path)
        data["path"] = self.path
        with self.changed:
            self.version += 1
            data["version"] = self.version
            self.payload = json.dumps(data, separators=(",", ":")).encode()
            self.units = data["units"]
            self.changed.notify_all()
        print(f"  v{self.version}  {data['count']} statements  ({data['updated']})", flush=True)


def watch(state):
    seen = None
    while True:
        try:
            info = os.stat(state.path)
            stamp = (info.st_mtime_ns, info.st_size)
        except OSError:          # mid-save: the file can vanish for a moment
            stamp = seen
        if stamp != seen:
            time.sleep(0.05)     # let the editor finish writing
            seen = stamp
            state.refresh()
        time.sleep(POLL_SECONDS)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        state = self.server.state
        if url.path in ("/", "/index.html"):
            self._send(PAGE.encode(), "text/html; charset=utf-8")
        elif url.path == "/data":
            try:
                since = int(parse_qs(url.query).get("since", ["0"])[0])
            except ValueError:
                since = 0
            with state.changed:  # long-poll: answer the moment there is something new
                state.changed.wait_for(lambda: state.version != since, timeout=20)
                body = state.payload if state.version != since else b'{"same":true}'
            self._send(body, "application/json")
        elif url.path == "/square.svg":
            ink = "#f0efec" if parse_qs(url.query).get("ink", [""])[0] == "dark" else "#0b0b0b"
            with state.changed:
                units = state.units
            self._send(square_svg(units, ink=ink).encode(), "image/svg+xml")
        else:
            self.send_error(404)

    def _send(self, body, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="Watch a Python file live as vectors in 3-D space.")
    parser.add_argument("file", nargs="?", default=__file__,
                        help="Python file to visualise (default: this script)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    parser.add_argument("--square", metavar="OUT.svg",
                        help="save the square-minus-the-path picture of the file and exit")
    args = parser.parse_args()

    path = os.path.abspath(args.file)
    if not os.path.isfile(path):
        parser.error(f"no such file: {path}")
    if args.square:
        with open(path, encoding="utf-8-sig", errors="replace") as fh:
            data, _ = build(fh.read(), {})
        with open(args.square, "w", encoding="utf-8") as fh:
            fh.write(square_svg(data["units"]))
        print(f"Wrote {os.path.abspath(args.square)}  ({data['count']} statements)")
        return
    state = State(path)
    threading.Thread(target=watch, args=(state,), daemon=True).start()

    server = None
    for port in range(args.port, args.port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit("no free port found")
    server.state = state
    url = f"http://localhost:{port}/"
    print(f"Watching {path}\nViewer:  {url}   (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(0.6, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Live Vector Space</title>
<style>
  :root {
    color-scheme: light;
    --surface: #fcfcfb;
    --panel: rgba(252, 252, 251, 0.9);
    --text-1: #0b0b0b;
    --text-2: #52514e;
    --muted: #898781;
    --grid: #e1e0d9;
    --axis: #c3c2b7;
    --border: rgba(11, 11, 11, 0.1);
    --wash: rgba(11, 11, 11, 0.05);
    --good: #0ca30c;
    --s1: #2a78d6;
    --s2: #eb6834;
    --s3: #1baf7a;
    --s4: #b9b8b1;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --surface: #1a1a19;
      --panel: rgba(26, 26, 25, 0.9);
      --text-1: #f0efec;
      --text-2: #c3c2b7;
      --grid: #2c2c2a;
      --axis: #383835;
      --border: rgba(255, 255, 255, 0.1);
      --wash: rgba(255, 255, 255, 0.06);
      --s1: #3987e5;
      --s2: #d95926;
      --s3: #199e70;
      --s4: #6b6a66;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface: #1a1a19;
    --panel: rgba(26, 26, 25, 0.9);
    --text-1: #f0efec;
    --text-2: #c3c2b7;
    --grid: #2c2c2a;
    --axis: #383835;
    --border: rgba(255, 255, 255, 0.1);
    --wash: rgba(255, 255, 255, 0.06);
    --s1: #3987e5;
    --s2: #d95926;
    --s3: #199e70;
    --s4: #6b6a66;
  }
  html, body {
    margin: 0;
    height: 100%;
    overflow: hidden;
    background: var(--surface);
    color: var(--text-1);
    font: 13px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  #c { display: block; width: 100vw; height: 100vh; touch-action: none; cursor: grab; }
  .card {
    position: fixed;
    box-sizing: border-box;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 12px;
    -webkit-backdrop-filter: blur(10px);
    backdrop-filter: blur(10px);
  }
  #hud { top: 12px; left: 12px; width: min(320px, calc(100vw - 24px)); }
  #hud h1 { margin: 0; font-size: 14px; font-weight: 600; overflow-wrap: anywhere; }
  #sub { margin-top: 2px; color: var(--text-2); }
  #legend { display: grid; grid-template-columns: 10px 1fr auto; gap: 3px 8px; align-items: center; margin-top: 10px; }
  #legend .n { color: var(--muted); font-variant-numeric: tabular-nums; }
  .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--muted); }
  #fluid { display: grid; grid-template-columns: auto minmax(0, 1fr) auto; gap: 6px 10px; align-items: center; margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border); }
  #fluid .label { color: var(--text-2); }
  #entropy { min-width: 2.6em; text-align: right; font-variant-numeric: tabular-nums; }
  #spark { display: block; width: 100%; height: 24px; }
  #heat { width: 100%; margin: 0; accent-color: var(--text-2); }
  button { font: inherit; font-size: 12px; color: var(--text-1); background: var(--wash); border: 1px solid var(--border); border-radius: 6px; padding: 1px 8px; cursor: pointer; }
  button:hover { background: var(--border); }
  #status { display: flex; align-items: center; gap: 7px; margin-top: 10px; color: var(--text-2); font-variant-numeric: tabular-nums; }
  #status.live .dot { background: var(--good); animation: pulse 1.6s ease-in-out infinite; }
  @keyframes pulse { 50% { opacity: 0.3; } }
  #help { left: 12px; bottom: 12px; max-width: calc(100vw - 24px); color: var(--text-2); font-size: 12px; }
  kbd { font: 11px ui-monospace, SFMono-Regular, Menlo, monospace; padding: 0 4px; border: 1px solid var(--border); border-radius: 4px; }
  #tip { display: none; z-index: 3; pointer-events: none; width: max-content; max-width: min(460px, calc(100vw - 24px)); }
  #tip .meta { display: flex; align-items: center; gap: 7px; color: var(--text-2); }
  #tip .meta b { color: var(--text-1); }
  pre, code { font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; }
  #tip pre { margin: 6px 0 0; white-space: pre-wrap; overflow-wrap: anywhere; }
  #tip h2 { margin: 10px 0 4px; font-size: 11px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
  #tip .nn { display: grid; grid-template-columns: auto auto minmax(0, 1fr); gap: 2px 10px; color: var(--text-2); font-variant-numeric: tabular-nums; }
  #tip .nn code { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--text-1); }
  #toast { top: 12px; left: 50%; transform: translateX(-50%); opacity: 0; transition: opacity 0.3s; pointer-events: none; color: var(--text-2); white-space: nowrap; }
  #table { display: none; top: 12px; right: 12px; bottom: 12px; width: min(440px, calc(100vw - 24px)); padding: 0; overflow: auto; z-index: 2; }
  #table .head { position: sticky; top: 0; padding: 8px 12px; background: var(--panel); border-bottom: 1px solid var(--border); color: var(--text-2); font-weight: 600; }
  #table .row { display: grid; grid-template-columns: 3em 9px minmax(0, 1fr); gap: 8px; align-items: center; padding: 2px 12px; cursor: pointer; }
  #table .row:hover { background: var(--wash); }
  #table .ln { color: var(--muted); text-align: right; font-variant-numeric: tabular-nums; font-size: 12px; }
  #table code { white-space: pre; overflow: hidden; text-overflow: ellipsis; }
  #square { top: 12px; right: 12px; width: min(220px, 34vw); padding: 8px; }
  #square img { display: block; width: 100%; aspect-ratio: 1; }
  #square .cap { display: flex; justify-content: space-between; gap: 8px; margin-top: 6px; color: var(--text-2); font-size: 12px; }
  #square a { color: var(--text-1); white-space: nowrap; }
  @media (max-width: 640px) { #help, #square { display: none; } }
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="hud" class="card">
  <h1 id="title">Connecting…</h1>
  <div id="sub">Waiting for the first snapshot of the file.</div>
  <div id="legend"></div>
  <div id="fluid">
    <span class="label" title="Shannon entropy of the particles over a 10×10×10 grid of cells. 1.00 = spread perfectly evenly (maximum disorder).">Entropy</span>
    <canvas id="spark" aria-hidden="true"></canvas>
    <span id="entropy">–</span>
    <label class="label" for="heat">Heat</label>
    <input id="heat" type="range" min="0" max="100" value="100">
    <button id="shake" type="button" title="Scramble the fluid back to maximum entropy (H)">Shake</button>
  </div>
  <div id="status"><span class="dot"></span><span id="statusText">Connecting…</span></div>
</div>
<div id="square" class="card">
  <img id="squareImg" src="/square.svg" alt="The square minus the code path: every statement placed by PC1 and PC2, joined in file order, and cut out of a solid square">
  <div class="cap"><span>Square − path · PC1 × PC2</span><a id="saveSquare" href="/square.svg" download="code-square.svg">Save SVG</a></div>
</div>
<div id="help" class="card">Drag to rotate · scroll to zoom · hover or click a point ·
  <kbd>H</kbd> shake · <kbd>F</kbd> fluid · <kbd>V</kbd> vectors · <kbd>P</kbd> path · <kbd>L</kbd> labels · <kbd>T</kbd> table · <kbd>S</kbd> square · <kbd>Space</kbd> spin</div>
<div id="tip" class="card"></div>
<div id="toast" class="card"></div>
<div id="table" class="card"></div>
<script>
"use strict";
const $ = (id) => document.getElementById(id);
const canvas = $("c");
const ctx = canvas.getContext("2d");
const GROUPS = ["Definitions", "Control flow", "Assignments", "Other"];
const BOX = 1.2;                           // the box spans -BOX..BOX on every axis
const LIMIT = BOX * 0.94, KNEE = LIMIT * 0.7;
const CAM = 7;                             // camera distance from the origin
const FIT = 0.46 / (BOX * Math.sqrt(3));   // the box's bounding sphere fills ~92% of the short side
const N = 14000;                            // fluid particles
const FONT = "11px system-ui, -apple-system, sans-serif";
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

let theme = {};
function readTheme() {
  const css = getComputedStyle(document.documentElement);
  const v = (name) => css.getPropertyValue(name).trim();
  theme = {
    surface: v("--surface"), text1: v("--text-1"), text2: v("--text-2"), muted: v("--muted"),
    grid: v("--grid"), axis: v("--axis"), series: [v("--s1"), v("--s2"), v("--s3"), v("--s4")],
  };
  const hex = theme.surface.replace("#", "");
  theme.dark = parseInt(hex.slice(0, 2), 16) + parseInt(hex.slice(2, 4), 16) + parseInt(hex.slice(4, 6), 16) < 384;
}
readTheme();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { readTheme(); updateSquare(); });

const view = { yaw: 0.7, pitch: 0.38, zoom: 1, spin: true, vectors: false, path: false, labels: true, fluid: true };
let units = [], byId = new Map(), dying = [], meta = null, version = 0, live = false;
let hover = null, rowHover = null, pinned = null, pointer = null, hits = [];
let dragging = false, dragMoved = false, lastX = 0, lastY = 0;
let W = 0, H = 0, CX = 0, tableOpen = false, tipFor = null, tipVersion = -1, fps = 0;
let cosY = 1, sinY = 0, cosP = 1, sinP = 0;

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  W = window.innerWidth;
  H = window.innerHeight;
  CX = W / 2;
  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener("resize", resize);
resize();

// ---- 3-D -> screen ---------------------------------------------------------
function project(p) {
  const x1 = p[0] * cosY + p[2] * sinY;
  const z1 = p[2] * cosY - p[0] * sinY;
  const y2 = p[1] * cosP - z1 * sinP;
  const z2 = p[1] * sinP + z1 * cosP;
  const depth = Math.max(CAM - z2, 0.2);
  const k = CAM / depth;                        // perspective scale, 1 at the origin
  const f = Math.min(W, H) * FIT * view.zoom * k;
  return { x: CX + x1 * f, y: H / 2 - y2 * f, depth, k };
}
const NEAR = CAM - BOX * 1.6, SPAN = BOX * 3.2;
const fog = (depth) => 1 - 0.5 * clamp((depth - NEAR) / SPAN, 0, 1);

function segment(a, b) {
  const p = project(a), q = project(b);
  ctx.beginPath();
  ctx.moveTo(p.x, p.y);
  ctx.lineTo(q.x, q.y);
  ctx.stroke();
}

// keep every statement inside the box: identity in the middle, a soft squeeze near the walls
function squash(c) {
  const m = Math.abs(c);
  return m <= KNEE ? c : Math.sign(c) * (KNEE + (LIMIT - KNEE) * Math.tanh((m - KNEE) / (LIMIT - KNEE)));
}

// ---- the fluid: N particles of maximum-entropy noise -----------------------
const P = new Float32Array(N * 3);     // positions
const V = new Float32Array(N * 3);     // velocities
const J = new Float32Array(N * 3);     // each particle's offset from its spot on the shape
const T = new Float32Array(N);         // where along its statement's vector it settles (0..1)
const SX = new Float32Array(N), SY = new Float32Array(N), SK = new Float32Array(N);
const BUCKET = new Uint8Array(N);
const owner = new Array(N).fill(null); // the statement each particle belongs to
const gauss = () => Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
for (let i = 0; i < N; i++) {
  const tip = Math.random() < 0.45;     // 45% gather at the tip, the rest spread along the vector
  T[i] = tip ? 1 : Math.sqrt(Math.random());
  const spread = tip ? 0.07 : 0.03;
  for (let c = 0; c < 3; c++) J[i * 3 + c] = gauss() * spread;
}
function scramble() {
  for (let i = 0; i < N * 3; i++) {
    P[i] = (Math.random() * 2 - 1) * BOX;
    V[i] = (Math.random() * 2 - 1) * 0.3;
  }
}
scramble();

function assignOwners(added, replaced) {
  if (!units.length) { owner.fill(null); return; }
  const alive = new Set(units);
  const any = () => units[(Math.random() * units.length) | 0];
  for (let i = 0; i < N; i++) {
    let u = owner[i];
    if (u && replaced.has(u)) u = replaced.get(u);
    owner[i] = u && alive.has(u) ? u : any();
  }
  if (added.length && added.length < units.length) {
    // new statements pull a fair share of fluid out of the rest
    const quota = Math.max(1, Math.floor(N / units.length));
    for (const u of added) for (let q = 0; q < quota; q++) owner[(Math.random() * N) | 0] = u;
  }
}

const heatInput = $("heat");
let heat = 1, cooling = null;
function coolDown(delay) {
  cooling = { start: performance.now() + delay, from: heat, to: 0.12, dur: 5500 };
}
coolDown(1500);
function updateHeat(now) {
  if (!cooling) return;
  const t = clamp((now - cooling.start) / cooling.dur, 0, 1);
  heat = cooling.from + (cooling.to - cooling.from) * t * t * (3 - 2 * t);
  heatInput.value = String(Math.round(heat * 100));
  if (t >= 1) cooling = null;
}
heatInput.addEventListener("input", () => { cooling = null; heat = Number(heatInput.value) / 100; });
function shake() {
  scramble();
  heat = 1;
  heatInput.value = "100";
  coolDown(1200);
}
$("shake").addEventListener("click", (e) => { shake(); e.currentTarget.blur(); });

function fluidStep(dt, time) {
  const pull = (1 - heat) * 2.4;            // how hard particles are drawn to the shape
  const turb = 0.07 + 0.6 * heat;           // how hard the flow stirs them
  const blend = 1 - Math.exp(-6 * dt);
  for (let i = 0; i < N; i++) {
    const i3 = i * 3;
    const x = P[i3], y = P[i3 + 1], z = P[i3 + 2];
    // curl noise: the curl of a drifting sine potential, so the flow is divergence-free
    // and swirls like an incompressible fluid (uniform noise stays uniform under it)
    let fx = 0, fy = 0, fz = 0;
    for (let o = 0; o < 2; o++) {
      const f = o ? 4.4 : 2.2, a = o ? 0.5 : 1, ph = o ? 1.9 : 0, ts = time * (o ? 0.9 : 0.45);
      const sY = Math.sin(f * y + ts + ph), cY = Math.cos(f * y + ts + ph);
      const sZ = Math.sin(f * z + 1.3 + ph), cZ = Math.cos(f * z + 1.3 + ph);
      const sZ2 = Math.sin(f * z + ts * 0.8 + 2.1 + ph), cZ2 = Math.cos(f * z + ts * 0.8 + 2.1 + ph);
      const sX = Math.sin(f * x + 0.4 + ph), cX = Math.cos(f * x + 0.4 + ph);
      const sX2 = Math.sin(f * x + ts * 1.1 + 4.2 + ph), cX2 = Math.cos(f * x + ts * 1.1 + 4.2 + ph);
      const sY2 = Math.sin(f * y + 2.7 + ph), cY2 = Math.cos(f * y + 2.7 + ph);
      fx -= a * (sX2 * sY2 + cZ2 * cX);
      fy -= a * (sY * sZ + cX2 * cY2);
      fz -= a * (sZ2 * sX + cY * cZ);
    }
    // the walls: no flow through them, so the fluid slides along the inside of the box
    if (fx * x > 0) fx *= 1 - (Math.abs(x) / BOX) ** 6;
    if (fy * y > 0) fy *= 1 - (Math.abs(y) / BOX) ** 6;
    if (fz * z > 0) fz *= 1 - (Math.abs(z) / BOX) ** 6;
    let tx = fx * turb, ty = fy * turb, tz = fz * turb;
    const u = owner[i];
    if (u && u.b) {
      const t = T[i];
      tx += (u.b[0] * t + J[i3] - x) * pull;
      ty += (u.b[1] * t + J[i3 + 1] - y) * pull;
      tz += (u.b[2] * t + J[i3 + 2] - z) * pull;
    }
    V[i3] += (tx - V[i3]) * blend;
    V[i3 + 1] += (ty - V[i3 + 1]) * blend;
    V[i3 + 2] += (tz - V[i3 + 2]) * blend;
    for (let c = 0; c < 3; c++) {
      let p = P[i3 + c] + V[i3 + c] * dt;
      if (p > BOX) { p = BOX; V[i3 + c] *= -0.5; }
      else if (p < -BOX) { p = -BOX; V[i3 + c] *= -0.5; }
      P[i3 + c] = p;
    }
  }
}

// Shannon entropy of the particles over a GRID^3 lattice, 1 = perfectly even
const GRID = 10, cells = new Uint16Array(GRID * GRID * GRID), history = [];
function measureEntropy() {
  cells.fill(0);
  const s = GRID / (2 * BOX);
  for (let i = 0; i < N; i++) {
    const i3 = i * 3;
    const gx = Math.min(GRID - 1, ((P[i3] + BOX) * s) | 0);
    const gy = Math.min(GRID - 1, ((P[i3 + 1] + BOX) * s) | 0);
    const gz = Math.min(GRID - 1, ((P[i3 + 2] + BOX) * s) | 0);
    cells[gx + GRID * (gy + GRID * gz)]++;
  }
  let h = 0;
  for (let c = 0; c < cells.length; c++) {
    if (cells[c]) { const p = cells[c] / N; h -= p * Math.log(p); }
  }
  return h / Math.log(cells.length);
}

const spark = $("spark"), sctx = spark.getContext("2d");
function drawSpark() {
  const w = spark.clientWidth, h = spark.clientHeight;
  if (!w || !h) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (spark.width !== Math.round(w * dpr) || spark.height !== Math.round(h * dpr)) {
    spark.width = Math.round(w * dpr);
    spark.height = Math.round(h * dpr);
  }
  sctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  sctx.clearRect(0, 0, w, h);
  sctx.strokeStyle = theme.grid;              // baseline = entropy 0
  sctx.lineWidth = 1;
  sctx.beginPath();
  sctx.moveTo(0, h - 0.5);
  sctx.lineTo(w, h - 0.5);
  sctx.stroke();
  if (history.length < 2) return;
  const step = (w - 6) / 119;
  const y = (v) => 3 + (1 - v) * (h - 6);
  sctx.strokeStyle = theme.text2;
  sctx.lineWidth = 2;
  sctx.lineJoin = "round";
  sctx.beginPath();
  history.forEach((v, i) => {
    const x = w - 3 - (history.length - 1 - i) * step;
    if (i) sctx.lineTo(x, y(v)); else sctx.moveTo(x, y(v));
  });
  sctx.stroke();
  sctx.fillStyle = theme.text1;
  sctx.beginPath();
  sctx.arc(w - 3, y(history[history.length - 1]), 2.5, 0, Math.PI * 2);
  sctx.fill();
}

// ---- data from the Python side ---------------------------------------------
function applyData(d) {
  const now = performance.now();
  const incoming = new Set(d.units.map((u) => u.id));
  const gone = [...byId.values()].filter((u) => !incoming.has(u.id));
  const goneByLine = new Map(gone.map((u) => [u.line, u]));
  const next = new Map(), replaced = new Map(), fresh = [];
  let edited = 0;
  const list = d.units.map((raw) => {
    let u = byId.get(raw.id);
    if (!u) {
      const old = goneByLine.get(raw.line);     // same line, new text: morph it in place
      if (old) {
        goneByLine.delete(raw.line);
        old.morphed = true;
        u = { cur: old.cur.slice(), born: now - 1000 };
        replaced.set(old, u);
        edited++;
      } else {
        u = { cur: [0, 0, 0], born: now };       // brand new: shoot out of the origin
        fresh.push(u);
      }
    }
    Object.assign(u, raw);
    next.set(raw.id, u);
    return u;
  });
  let removed = 0;
  for (const u of gone) {
    if (!u.morphed) { u.dyingSince = now; dying.push(u); removed++; }
  }
  for (const u of list) u.neighbors = u.nn.map(([j, sim]) => ({ u: list[j], sim }));
  const first = meta === null;
  units = list;
  byId = next;
  meta = d;
  version = d.version;
  assignOwners(fresh, replaced);
  if (pinned && !next.has(pinned.id)) pinned = null;
  if (rowHover && !next.has(rowHover.id)) rowHover = null;
  renderHud();
  updateSquare();
  if (tableOpen) renderTable();
  if (!first) {
    const parts = [];
    if (fresh.length) parts.push(`${fresh.length} new`);
    if (edited) parts.push(`${edited} edited`);
    if (removed) parts.push(`${removed} removed`);
    toast(`File changed · ${parts.join(" · ") || "same statements"}`);
  }
}

function step(dt, now) {
  const k = 1 - Math.exp(-dt * 6);
  for (const u of units) {
    for (let i = 0; i < 3; i++) u.cur[i] += (u.p[i] - u.cur[i]) * k;
    u.alpha = clamp((now - u.born) / 500, 0, 1);
    u.b = [squash(u.cur[0]), squash(u.cur[1]), squash(u.cur[2])];
  }
  dying = dying.filter((u) => now - u.dyingSince < 600);
}

function pick() {
  if (!pointer || (dragging && dragMoved)) return null;
  let best = null;
  for (const h of hits) {
    const close = Math.hypot(h.x - pointer.x, h.y - pointer.y) <= Math.max(h.r + 5, 9);
    if (close && (!best || h.depth < best.depth)) best = h;
  }
  return best ? best.u : null;
}

// ---- drawing -------------------------------------------------------------
function drawBox() {
  ctx.strokeStyle = theme.muted;
  ctx.globalAlpha = 0.55;
  ctx.lineWidth = 1.25;
  for (const a of [-BOX, BOX]) {
    for (const b of [-BOX, BOX]) {
      segment([-BOX, a, b], [BOX, a, b]);
      segment([a, -BOX, b], [a, BOX, b]);
      segment([a, b, -BOX], [a, b, BOX]);
    }
  }
  ctx.globalAlpha = 1;
  ctx.lineWidth = 1;
}

function drawFluid(related) {
  const scale = Math.min(W, H) * FIT * view.zoom;
  const cy = H / 2;
  for (let i = 0; i < N; i++) {
    const i3 = i * 3;
    const x = P[i3], y = P[i3 + 1], z = P[i3 + 2];
    const x1 = x * cosY + z * sinY;
    const z1 = z * cosY - x * sinY;
    const y2 = y * cosP - z1 * sinP;
    const z2 = y * sinP + z1 * cosP;
    const k = CAM / Math.max(CAM - z2, 0.2);
    SX[i] = CX + x1 * scale * k;
    SY[i] = cy - y2 * scale * k;
    SK[i] = k;
    const level = clamp(Math.floor(((CAM - z2 - NEAR) / SPAN) * 3), 0, 2);
    const u = owner[i];
    const dim = related !== null && !(u && related.has(u)) ? 1 : 0;
    BUCKET[i] = ((u ? u.group : 3) * 3 + level) * 2 + dim;
  }
  // one batched path per colour x depth band x dimmed, far bands fainter
  const alpha = theme.dark ? [0.5, 0.34, 0.2] : [0.55, 0.38, 0.22];
  ctx.globalCompositeOperation = theme.dark ? "lighter" : "source-over";
  for (let b = 0; b < 24; b++) {
    const dim = b & 1, level = (b >> 1) % 3, group = Math.floor((b >> 1) / 3);
    ctx.fillStyle = theme.series[group];
    ctx.globalAlpha = alpha[level] * (dim ? 0.22 : 1);
    ctx.beginPath();
    let any = false;
    for (let i = 0; i < N; i++) {
      if (BUCKET[i] !== b) continue;
      const s = 2 * SK[i];
      ctx.rect(SX[i] - s / 2, SY[i] - s / 2, s, s);
      any = true;
    }
    if (any) ctx.fill();
  }
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "source-over";
}

function draw(focus, now) {
  ctx.fillStyle = theme.surface;
  ctx.fillRect(0, 0, W, H);
  ctx.lineWidth = 1;
  const related = focus ? new Set([focus, ...focus.neighbors.map((n) => n.u)]) : null;
  const faded = (u) => related !== null && !related.has(u);
  const shape = view.fluid ? clamp((0.75 - heat) / 0.45, 0, 1) : 1;   // statements surface as the fluid cools

  // floor grid on the bottom of the box, and the principal axes through the middle
  ctx.strokeStyle = theme.grid;
  for (let i = 1; i < 6; i++) {
    const t = -BOX + (i * BOX) / 3;
    segment([t, -BOX, -BOX], [t, -BOX, BOX]);
    segment([-BOX, -BOX, t], [BOX, -BOX, t]);
  }
  ctx.strokeStyle = theme.axis;
  ctx.fillStyle = theme.muted;
  ctx.font = FONT;
  for (let i = 0; i < 3; i++) {
    const end = [0, 0, 0];
    end[i] = BOX;
    segment(end.map((c) => -c), end);
    const p = project(end);
    ctx.fillText(`PC${i + 1}`, p.x + 5, p.y - 5);
  }

  if (view.fluid) drawFluid(related);
  drawBox();

  const origin = project([0, 0, 0]);

  // every statement as a vector from the origin
  if (view.vectors) {
    for (const u of units) {
      ctx.globalAlpha = u.alpha * shape * (faded(u) ? 0.05 : 0.2);
      ctx.strokeStyle = theme.series[u.group];
      ctx.beginPath();
      ctx.moveTo(origin.x, origin.y);
      ctx.lineTo(u.s.x, u.s.y);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  // statements in file order
  if (view.path && units.length > 1) {
    ctx.strokeStyle = theme.muted;
    ctx.globalAlpha = (related ? 0.2 : 0.55) * shape;
    ctx.beginPath();
    units.forEach((u, i) => (i ? ctx.lineTo(u.s.x, u.s.y) : ctx.moveTo(u.s.x, u.s.y)));
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  if (focus) {
    // links to the nearest neighbours (measured in the full high-dimensional space)
    ctx.strokeStyle = theme.text2;
    ctx.lineWidth = 1.25;
    ctx.setLineDash([4, 4]);
    for (const n of focus.neighbors) {
      ctx.beginPath();
      ctx.moveTo(focus.s.x, focus.s.y);
      ctx.lineTo(n.u.s.x, n.u.s.y);
      ctx.stroke();
    }
    ctx.setLineDash([]);
    // the focused statement's own vector, with an arrowhead
    const color = theme.series[focus.group];
    const dx = focus.s.x - origin.x, dy = focus.s.y - origin.y, len = Math.hypot(dx, dy);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(origin.x, origin.y);
    ctx.lineTo(focus.s.x, focus.s.y);
    ctx.stroke();
    if (len > 24) {
      const ux = dx / len, uy = dy / len;
      const tx = focus.s.x - ux * 9, ty = focus.s.y - uy * 9;
      ctx.beginPath();
      ctx.moveTo(tx, ty);
      ctx.lineTo(tx - ux * 9 - uy * 4.5, ty - uy * 9 + ux * 4.5);
      ctx.lineTo(tx - ux * 9 + uy * 4.5, ty - uy * 9 - ux * 4.5);
      ctx.closePath();
      ctx.fill();
    }
    ctx.lineWidth = 1;
  }

  // points, far to near, each with a surface-colored ring so overlaps stay readable
  const all = units.concat(dying);
  all.sort((a, b) => b.s.depth - a.s.depth);
  hits = [];
  for (const u of all) {
    const fade = u.dyingSince ? clamp(1 - (now - u.dyingSince) / 600, 0, 1) : u.alpha;
    const isRelated = related !== null && related.has(u);
    // with the fluid on, statements shrink to small beads so the fluid's shape leads
    const base = view.fluid && !isRelated ? 2.6 : (u.group < 3 ? 4.5 : 4);
    const r = base * clamp(u.s.k, 0.9, 1.5) + (u === focus ? 2 : 0);
    if (!u.dyingSince && shape > 0.4) hits.push({ u, x: u.s.x, y: u.s.y, r, depth: u.s.depth });
    const a = fade * fog(u.s.depth) * (faded(u) ? 0.18 : 1) * (isRelated ? 1 : shape);
    if (a <= 0.01) continue;
    ctx.globalAlpha = a;
    ctx.fillStyle = theme.surface;
    ctx.beginPath();
    ctx.arc(u.s.x, u.s.y, r + (view.fluid ? 1.25 : 2), 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = theme.series[u.group];
    ctx.beginPath();
    ctx.arc(u.s.x, u.s.y, r, 0, Math.PI * 2);
    ctx.fill();
    if (isRelated) {
      ctx.globalAlpha = 1;
      ctx.strokeStyle = theme.text1;
      ctx.lineWidth = u === focus ? 2 : 1.25;
      ctx.beginPath();
      ctx.arc(u.s.x, u.s.y, r + 3.5, 0, Math.PI * 2);
      ctx.stroke();
      ctx.lineWidth = 1;
    }
  }
  ctx.globalAlpha = 1;

  // origin marker
  ctx.fillStyle = theme.text2;
  ctx.beginPath();
  ctx.arc(origin.x, origin.y, 2.5, 0, Math.PI * 2);
  ctx.fill();

  drawLabels(focus, related, shape);
}

function drawLabels(focus, related, shape) {
  ctx.font = FONT;
  ctx.lineJoin = "round";
  const boxes = ["hud", "help", "table", "square"].map((id) => {   // keep labels out from under the cards
    const r = $(id).getBoundingClientRect();
    return { x: r.left, y: r.top, w: r.width, h: r.height };
  });
  const free = (x, y, w, h) => boxes.every((b) => x >= b.x + b.w || x + w <= b.x || y >= b.y + b.h || y + h <= b.y);
  const queue = [];
  if (focus) {
    queue.push([focus, true]);
    for (const n of focus.neighbors) queue.push([n.u, true]);
  }
  if (view.labels && shape > 0.05) {
    units.filter((u) => u.label).sort((a, b) => a.s.depth - b.s.depth).forEach((u) => queue.push([u, false]));
  }
  const done = new Set();
  for (const [u, strong] of queue) {
    if (done.has(u)) continue;
    done.add(u);
    const text = u.label || `L${u.line}`;
    const w = ctx.measureText(text).width;
    const x = u.s.x + 9, y = u.s.y - 7;
    if (!free(x - 2, y - 11, w + 4, 15)) continue;
    boxes.push({ x: x - 2, y: y - 11, w: w + 4, h: 15 });
    ctx.globalAlpha = strong ? 1 : u.alpha * shape * fog(u.s.depth) * (related && !related.has(u) ? 0.25 : 1);
    ctx.strokeStyle = theme.surface;
    ctx.lineWidth = 3;
    ctx.strokeText(text, x, y);
    ctx.fillStyle = strong ? theme.text1 : theme.text2;
    ctx.fillText(text, x, y);
  }
  ctx.globalAlpha = 1;
  ctx.lineWidth = 1;
}

// ---- HUD, tooltip, table -----------------------------------------------------
function renderHud() {
  $("title").textContent = meta.file;
  $("title").title = meta.path;
  const kept = Math.round(meta.explained.reduce((a, b) => a + b, 0) * 100);
  let sub = `${meta.count} statements → ${meta.dims}-d token vectors → PCA → 3-D (${kept}% of variance kept). ` +
    `${N.toLocaleString()} particles of random fluid fill the box and settle onto the statements as it cools.`;
  if (!meta.parser.startsWith("python")) sub += ` Parsing ${meta.parser}.`;
  $("sub").textContent = sub;
  const counts = [0, 0, 0, 0];
  for (const u of units) counts[u.group]++;
  $("legend").innerHTML = GROUPS.map((g, i) =>
    `<span class="dot" style="background:var(--s${i + 1})"></span><span>${g}</span><span class="n">${counts[i]}</span>`).join("");
  updateStatus();
}

function updateSquare() {
  if (!meta) return;
  const src = `/square.svg?v=${version}&ink=${theme.dark ? "dark" : "light"}`;
  $("squareImg").src = src;
  $("saveSquare").href = src;
}

function updateStatus() {
  $("status").classList.toggle("live", live);
  $("statusText").textContent = live
    ? `Live · v${version}${meta ? ` · updated ${meta.updated}` : ""} · ${fps} fps`
    : (meta ? "Server stopped · retrying…" : "Connecting…");
}

let toastTimer = 0;
function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.style.opacity = "1";
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.style.opacity = "0"; }, 2200);
}

function tipHtml(u) {
  const where = u.end > u.line ? `L${u.line}–${u.end}` : `L${u.line}`;
  let html = `<div class="meta"><span class="dot" style="background:var(--s${u.group + 1})"></span>` +
    `<b>${where}</b><span>${esc(u.kind)} · ${GROUPS[u.group]}</span></div><pre>${esc(u.snippet)}</pre>`;
  if (u.neighbors.length) {
    html += `<h2>Most similar · cosine in ${meta.dims}-d</h2><div class="nn">` +
      u.neighbors.map((n) => `<span>${n.sim.toFixed(2)}</span><span>L${n.u.line}</span><code>${esc(n.u.text)}</code>`).join("") +
      "</div>";
  }
  return html;
}

function placeTip(focus) {
  const el = $("tip");
  if (!focus) { el.style.display = "none"; tipFor = null; return; }
  if (focus !== tipFor || version !== tipVersion) {
    el.innerHTML = tipHtml(focus);
    tipFor = focus;
    tipVersion = version;
  }
  el.style.display = "block";
  const anchor = hover && pointer ? pointer : focus.s;
  const w = el.offsetWidth, h = el.offsetHeight;
  let x = anchor.x + 18, y = anchor.y + 18;
  if (x + w > W - 12) x = anchor.x - w - 18;
  if (y + h > H - 12) y = H - h - 12;
  el.style.left = `${Math.max(12, x)}px`;
  el.style.top = `${Math.max(12, y)}px`;
}

function toggleTable() {
  tableOpen = !tableOpen;
  $("table").style.display = tableOpen ? "block" : "none";
  if (tableOpen) renderTable();
}
function renderTable() {
  $("table").innerHTML = `<div class="head">${units.length} statements · press T to close</div>` +
    units.map((u, i) => `<div class="row" data-i="${i}"><span class="ln">${u.line}</span>` +
      `<span class="dot" style="background:var(--s${u.group + 1})"></span><code>${esc(u.text)}</code></div>`).join("");
}
$("table").addEventListener("mouseover", (e) => {
  const row = e.target.closest(".row");
  rowHover = row ? units[Number(row.dataset.i)] : null;
});
$("table").addEventListener("mouseleave", () => { rowHover = null; });
$("table").addEventListener("click", (e) => {
  const row = e.target.closest(".row");
  if (!row) return;
  const u = units[Number(row.dataset.i)];
  pinned = pinned === u ? null : u;
});

// ---- input ----------------------------------------------------------------
canvas.addEventListener("pointerdown", (e) => {
  dragging = true;
  dragMoved = false;
  lastX = e.clientX;
  lastY = e.clientY;
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", (e) => {
  pointer = { x: e.clientX, y: e.clientY };
  if (!dragging) return;
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  if (Math.abs(dx) + Math.abs(dy) > 3) dragMoved = true;
  if (!dragMoved) return;
  view.yaw += dx * 0.008;
  view.pitch = clamp(view.pitch + dy * 0.008, -1.45, 1.45);
  lastX = e.clientX;
  lastY = e.clientY;
});
canvas.addEventListener("pointerup", () => {
  if (dragging && !dragMoved) pinned = hover && hover !== pinned ? hover : null;
  dragging = false;
  dragMoved = false;
});
canvas.addEventListener("pointercancel", () => { dragging = false; dragMoved = false; });
canvas.addEventListener("pointerleave", () => { if (!dragging) pointer = null; });
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  view.zoom = clamp(view.zoom * Math.exp(-e.deltaY * 0.0015), 0.35, 8);
}, { passive: false });
canvas.addEventListener("dblclick", () => {
  Object.assign(view, { yaw: 0.7, pitch: 0.38, zoom: 1 });
  pinned = null;
});
window.addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const key = e.key.toLowerCase();
  if (key === " ") { view.spin = !view.spin; e.preventDefault(); }
  else if (key === "h") shake();
  else if (key === "f") view.fluid = !view.fluid;
  else if (key === "v") view.vectors = !view.vectors;
  else if (key === "p") view.path = !view.path;
  else if (key === "l") view.labels = !view.labels;
  else if (key === "t") toggleTable();
  else if (key === "s") $("square").style.display = $("square").style.display === "none" ? "" : "none";
  else if (key === "escape") pinned = null;
});

// ---- live loop ------------------------------------------------------------
async function poll() {
  for (;;) {
    try {
      const res = await fetch(`/data?since=${version}`, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const d = await res.json();
      if (!live) { live = true; updateStatus(); }
      if (!d.same && d.version !== version) applyData(d);
    } catch (err) {
      if (live || meta === null) { live = false; updateStatus(); }
      await new Promise((r) => setTimeout(r, 1000));
    }
  }
}
poll();

let lastT = performance.now(), frames = 0, fpsSince = lastT, sampledAt = 0;
function frame(now) {
  const dt = Math.min(0.05, Math.max(0, (now - lastT) / 1000));
  lastT = now;
  hover = pick();
  const focus = hover || rowHover || pinned;
  if (view.spin && !dragging && !focus) view.yaw += dt * 0.18;
  cosY = Math.cos(view.yaw); sinY = Math.sin(view.yaw);
  cosP = Math.cos(view.pitch); sinP = Math.sin(view.pitch);
  updateHeat(now);
  step(dt, now);
  if (view.fluid) fluidStep(dt, now / 1000);
  for (const u of units) u.s = project(u.b);
  for (const u of dying) u.s = project(u.b || u.cur);
  draw(focus, now);
  placeTip(focus);
  canvas.style.cursor = dragging ? "grabbing" : hover ? "pointer" : "grab";
  if (now - sampledAt >= 100) {
    sampledAt = now;
    const h = measureEntropy();
    history.push(h);
    if (history.length > 120) history.shift();
    $("entropy").textContent = h.toFixed(2);
    drawSpark();
  }
  frames++;
  if (now - fpsSince >= 500) {
    fps = Math.round((frames * 1000) / (now - fpsSince));
    frames = 0;
    fpsSince = now;
    updateStatus();
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
