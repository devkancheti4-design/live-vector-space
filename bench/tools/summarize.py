"""Score every bench run (answers graded against keys built from the code), read every run's token total
from its transcript, and print per-condition accuracy with 95% Wilson intervals plus paired map-vs-grep
differences with 95% t intervals."""
import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grade import grade  # noqa: E402

B = os.environ.get("BENCH") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.exists(os.path.join(B, "runs.json")):     # token totals saved from the run transcripts
    TOK = {tag: run["tokens"] for tag, run in json.load(open(os.path.join(B, "runs.json"))).items()}
else:                                                  # or read them from the transcripts again
    from tokens import runs as token_runs  # noqa: E402
    TOK = {tag: run["tokens"] for tag, run in token_runs().items()}
T975 = {4: 2.776, 9: 2.262}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (100 * (c - h), 100 * (c + h))


def score(tag, key):
    path = f"{B}/answers/{tag}.txt"
    if not os.path.exists(path):
        return None
    rows = grade(path, f"{B}/sets/{key}_key.json")
    by = {}
    for r in rows:
        by.setdefault(r["kind"], [0, 0])
        by[r["kind"]][0] += r["ok"]
        by[r["kind"]][1] += 1
    return {"k": sum(r["ok"] for r in rows), "n": len(rows), "by": by, "tok": TOK.get(tag)}


out = {"battery": {}, "views": {}, "controls": {}}
print("== 48-question sets: map + lookups vs grep (5 sets per repo; same questions for both methods)")
for repo, name in (("R", "reef"), ("P", "PRAXIST")):
    mem = score(f"{repo}1_mem", f"{repo}1")
    base = mem["tok"] if mem else None
    out["controls"][f"{name}_memory"] = mem
    if mem:
        print(f"{name} memory-only control: {mem['k']}/{mem['n']} right, {mem['tok']:,} tokens (baseline for extra tokens)")
    for method in ("map", "grep"):
        res = [score(f"{repo}{s}_{method}", f"{repo}{s}") for s in range(1, 6)]
        res = [r for r in res if r]
        if not res:
            continue
        k, n = sum(r["k"] for r in res), sum(r["n"] for r in res)
        accs = [100 * r["k"] / r["n"] for r in res]
        toks = [r["tok"] - (base or 0) for r in res if r["tok"]]
        kinds = {}
        for r in res:
            for kd, (a, b) in r["by"].items():
                kinds.setdefault(kd, [0, 0])
                kinds[kd][0] += a
                kinds[kd][1] += b
        lo, hi = wilson(k, n)
        kd_txt = ", ".join(f"{kd} {a}/{b}" for kd, (a, b) in sorted(kinds.items()))
        print(f"  {name} {method:<4} runs={len(res)}  right {k}/{n} = {100 * k / n:.1f}% (95% CI {lo:.1f}-{hi:.1f}); "
              f"per run {min(accs):.0f}-{max(accs):.0f}%; extra tokens mean {st.mean(toks):,.0f} sd {st.pstdev(toks):,.0f}; [{kd_txt}]")
        out["battery"][f"{name}_{method}"] = {"k": k, "n": n, "accs": accs, "extra_tokens": toks, "kinds": kinds}
pairs_acc, pairs_tok = [], []
for repo in ("R", "P"):
    for s in range(1, 6):
        a, b = score(f"{repo}{s}_map", f"{repo}{s}"), score(f"{repo}{s}_grep", f"{repo}{s}")
        if a and b and a["tok"] and b["tok"]:
            pairs_acc.append(100 * (a["k"] - b["k"]) / a["n"])
            pairs_tok.append(a["tok"] - b["tok"])
if len(pairs_acc) >= 2:
    n = len(pairs_acc)
    t = T975.get(n - 1, 2.0)
    for label, xs, unit in (("accuracy, map minus grep", pairs_acc, " points"), ("tokens, map minus grep", pairs_tok, " tokens")):
        m, sd = st.mean(xs), st.stdev(xs)
        print(f"  paired over {n} sets, {label}: mean {m:+,.1f}{unit}, 95% CI {m - t * sd / math.sqrt(n):+,.1f} to "
              f"{m + t * sd / math.sqrt(n):+,.1f}")
    out["paired"] = {"acc": pairs_acc, "tok": pairs_tok}

print("\n== same-size views, 10 architecture questions (5 runs per view per repo)")
for repo, name in (("R", "reef"), ("P", "PRAXIST")):
    for v, vname in (("a", "map"), ("b", "docstrings"), ("c", "file tree"), ("d", "raw code"), ("n", "nothing")):
        res = [score(f"{repo}V{v}{i}", f"{repo}U") for i in range(1, 6)]
        res = [r for r in res if r]
        if not res:
            continue
        k, n = sum(r["k"] for r in res), sum(r["n"] for r in res)
        lo, hi = wilson(k, n)
        per = [r["k"] for r in res]
        print(f"  {name} {vname:<10} runs={len(res)}  {k}/{n} = {100 * k / n:.0f}% (95% CI {lo:.0f}-{hi:.0f}); per run {per}")
        out["views"][f"{name}_{vname}"] = {"k": k, "n": n, "per": per}
for vname in ("map", "docstrings", "file tree", "raw code", "nothing"):
    both = [out["views"].get(f"{name}_{vname}") for name in ("reef", "PRAXIST")]
    both = [x for x in both if x]
    if len(both) == 2:
        k, n = sum(x["k"] for x in both), sum(x["n"] for x in both)
        lo, hi = wilson(k, n)
        print(f"  both repos {vname:<10} {k}/{n} = {100 * k / n:.0f}% (95% CI {lo:.0f}-{hi:.0f})")
json.dump(out, open(f"{B}/summary.json", "w"), indent=1)
