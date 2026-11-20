"""Grade one answers file against its key. Returns per-question correctness by kind.
Answers are lines 'N. answer — file:line'; only the part before ' — ' is graded."""
import json
import re
import sys

NUM = re.compile(r"-?\d[\d,_]*\.?\d*(?:e[+-]?\d+)?", re.I)


def left(text):
    for sep in (" — ", " -- ", " – "):
        if sep in text:
            return text.split(sep, 1)[0]
    return text


def numbers(text):
    out = []
    for m in NUM.findall(text):
        try:
            out.append(float(m.replace(",", "").replace("_", "")))
        except ValueError:
            pass
    return out


def const_ok(ans, key):
    v = float(key["value"])
    expr = re.sub(r"\s+", "", str(key.get("expr", "")))
    if expr and expr in re.sub(r"\s+", "", ans):
        return True
    for x in numbers(ans):
        for scale in (1, 1024, 1024 ** 2, 1024 ** 3, 1000, 1e6, 1e9, 60, 3600, 86400, 604800, 0.01, 0.001):
            if abs(x * scale - v) <= 1e-9 * max(1.0, abs(v)):
                return True
    return False


def fixed_ok(ans, key):
    a = ans.lower()
    if "all" in key:
        return all(re.search(p, a, re.I) for p in key["all"])
    if "order" in key:
        pos = [re.search(p, a, re.I) for p in key["order"]]
        return all(pos) and all(pos[i].start() < pos[i + 1].start() for i in range(len(pos) - 1))
    if "pct" in key:
        lo, hi = key["pct"]
        return "half" in a or any(lo <= x <= hi for x in numbers(a))
    return False


def grade(answers_path, key_path):
    keys = json.load(open(key_path))
    got = {}
    for line in open(answers_path, encoding="utf-8", errors="replace"):
        m = re.match(r"^\s*(\d+)\s*[.)]\s*(.*)$", line)
        if m and int(m.group(1)) not in got:
            got[int(m.group(1))] = m.group(2).strip()
    rows = []
    for i, key in enumerate(keys, 1):
        ans = left(got.get(i, ""))
        if ans.strip().lower().startswith("unsure"):
            ok = False                            # a hedge is not an answer, whatever guess follows it
        elif key["kind"] == "const":
            ok = const_ok(ans, key)
        elif key["kind"] == "func":
            ok = re.search(r"\b" + re.escape(key["name"]) + r"\b", ans) is not None
        else:
            ok = fixed_ok(ans, key)
        rows.append({"n": i, "kind": key["kind"], "ok": ok, "answer": got.get(i, "")[:140]})
    return rows


if __name__ == "__main__":
    rows = grade(sys.argv[1], sys.argv[2])
    print(f"{sum(r['ok'] for r in rows)}/{len(rows)} right")
    for r in rows:
        if not r["ok"]:
            print(f"  wrong {r['n']} ({r['kind']}): {r['answer']}")
