"""Four views of one codebase, all cut to the same byte budget:
  map   - what a map agent sees: `map` plus the given `files` zooms (intents, sizes, piece numbers, shape)
  docs  - plain docstring first lines (folder: intent, file: intent), the map view's folders first
  tree  - folders with their file counts and file names, the map view's folders first
  code  - the raw source of the package's entry files, in order
Run with python3 -I; the repo is only read."""
import json
import os
import subprocess
import sys
from collections import defaultdict

TOOLS = os.path.dirname(os.path.abspath(__file__))
store, repo, out, zooms, entries = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(","), sys.argv[5].split(",")
cm = [sys.executable, "-m", "lvs"]            # the research runs used code_map.py, whose output is the same
view_map = subprocess.run(cm + ["map", "--store", store], capture_output=True, text=True).stdout
for z in zooms:
    view_map += "\n" + subprocess.run(cm + ["files", z, "--store", store], capture_output=True, text=True).stdout
budget = len(view_map.encode())

meta = json.load(open(store + "_map.json"))
root = os.path.commonpath([f["path"] for f in meta["files"]]) + "/"
files = [dict(f, rel=f["path"][len(root):]) for f in meta["files"]]
intent = {f["rel"]: f.get("intent") or "" for f in files}
folders = defaultdict(list)
for f in files:
    parts = f["rel"].split("/")
    for d in range(1, len(parts)):
        folders["/".join(parts[:d]) + "/"].append(f["rel"])


def depth(p):
    return p.rstrip("/").count("/")


shown = {""} | {z[len(root):].strip("/") + "/" if z.startswith(root) else z.strip("/") + "/" for z in zooms}


def parent(p):
    head = p.rstrip("/").rsplit("/", 1)
    return head[0] + "/" if len(head) == 2 else ""


def order(p):
    return (parent(p) not in shown, depth(p), p)


def fit(lines):
    text, size = [], 0
    for line in lines:
        if size + len(line.encode()) + 1 > budget:
            break
        text.append(line)
        size += len(line.encode()) + 1
    return "\n".join(text) + "\n"


docs = []
for p in sorted(list(folders) + [f["rel"] for f in files], key=order):
    about = intent.get(p + "__init__.py") if p.endswith("/") else intent.get(p)
    if about:
        docs.append(f"{p}: {about}")
tree = []
for d in sorted(folders, key=lambda p: (p not in shown, depth(p), p)):
    here = sorted(os.path.basename(r) for r in folders[d] if r.rsplit("/", 1)[0] + "/" == d)
    subs = sorted({r[len(d):].split("/")[0] + "/" for r in folders[d] if "/" in r[len(d):]})
    tree.append(f"{d} ({len(folders[d])} files): " + ", ".join(subs + here))
code = []
for e in entries:
    with open(os.path.join(repo, e), encoding="utf-8", errors="replace") as fh:
        code.append(f"# ===== {e} =====\n{fh.read()}")
code_text = "\n".join(code).encode()[:budget].decode("utf-8", "ignore")

views = {"a": view_map, "b": fit(docs), "c": fit(tree), "d": code_text}
for name, text in views.items():
    with open(f"{out}_{name}.txt", "w") as fh:
        fh.write(text)
    print(f"{name}: {len(text.encode()):,} bytes")
