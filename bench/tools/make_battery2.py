"""Five question sets for one repo: the fixed checked questions plus random ones drawn from the code
(documented constants and docstring-only functions), never repeated across sets. Keys come from the code.
Run with python3 -I; the repo is only read and parsed, never run.
Usage: make_battery2.py ROOT OUTDIR TAG N_RANDOM_PER_SET EXCLUDE_NAMES(comma) FIXED_JSON"""
import ast
import json
import os
import random
import re
import sys

ROOT, OUT, TAG, NR = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
EXCLUDE, FIXED = set(sys.argv[5].split(",")), json.load(open(sys.argv[6]))
OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
       ast.Div: lambda a, b: a / b, ast.Pow: lambda a, b: a ** b, ast.FloorDiv: lambda a, b: a // b}


def number(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](number(node.left), number(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -number(node.operand)
    raise ValueError


consts, funcs, seen_doc = [], [], {}
for d, _, names in os.walk(ROOT):
    for f in sorted(names):
        if not f.endswith(".py"):
            continue
        rel = os.path.relpath(os.path.join(d, f), ROOT)
        src = open(os.path.join(d, f), encoding="utf-8", errors="replace").read()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        lines = src.split("\n")
        body = tree.body
        for i, node in enumerate(body):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                continue
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if not isinstance(target, ast.Name) or not re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", target.id) or target.id in EXCLUDE:
                continue
            try:
                value = number(node.value)
            except (ValueError, ZeroDivisionError, TypeError, OverflowError):
                continue
            k, above = node.lineno - 2, []
            while k >= 0 and lines[k].strip().startswith("#"):
                above.insert(0, lines[k].strip().lstrip("#:").strip())
                k -= 1
            desc = " ".join(above)
            if not desc and i + 1 < len(body) and isinstance(body[i + 1], ast.Expr) \
                    and isinstance(getattr(body[i + 1], "value", None), ast.Constant) and isinstance(body[i + 1].value.value, str):
                desc = " ".join(body[i + 1].value.value.split())
            if not desc:
                tail = lines[node.end_lineno - 1].split("#", 1)
                desc = tail[1].strip() if len(tail) == 2 and node.lineno == node.end_lineno else ""
            if len(desc) >= 25 and not re.search(r"\d", desc):
                consts.append({"kind": "const", "file": rel, "line": node.lineno, "name": target.id, "value": value,
                               "expr": ast.get_source_segment(src, node.value), "desc": desc})
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node)
                first = doc.strip().split("\n")[0].strip() if doc else ""
                if len(first) >= 40:
                    funcs.append({"kind": "func", "file": rel, "line": node.lineno, "name": node.name, "doc": first})
                    seen_doc[first] = seen_doc.get(first, 0) + 1

descs = {}
for c in consts:
    descs[c["desc"]] = descs.get(c["desc"], 0) + 1
consts = [c for c in consts if descs[c["desc"]] == 1]
funcs = [f for f in funcs if seen_doc[f["doc"]] == 1]
rng = random.Random(2026)
rng.shuffle(consts)
rng.shuffle(funcs)
nc = min(NR // 2, len(consts) // 5)
nf = NR - nc
print(f"{TAG}: {len(consts)} documented constants, {len(funcs)} docstring-only functions; per set {nc} constants + {nf} functions")
os.makedirs(OUT, exist_ok=True)
for s in range(1, 6):
    picks = consts[(s - 1) * nc:s * nc] + funcs[(s - 1) * nf:s * nf]
    qs = [q["q"] for q in FIXED]
    keys = [dict(kind="fixed", **{k: v for k, v in q.items() if k != "q"}) for q in FIXED]
    for p in picks:
        if p["kind"] == "const":
            qs.append(f'A constant is documented in the code as: "{p["desc"]}" What is its value?')
        else:
            qs.append(f'Which function has this docstring: "{p["doc"]}"? Give its name and file.')
        keys.append(p)
    with open(f"{OUT}/{TAG}{s}_questions.txt", "w") as fh:
        fh.write(f"{len(qs)} questions about one codebase. Answer every one.\n\n")
        fh.writelines(f"{i}. {q}\n" for i, q in enumerate(qs, 1))
    json.dump(keys, open(f"{OUT}/{TAG}{s}_key.json", "w"), indent=1)
print(f"wrote {TAG}1-{TAG}5: {len(FIXED) + NR} questions each")
