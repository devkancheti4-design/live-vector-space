"""Find every bench run's transcript by the answers file named in its prompt, and read its model and token
total (the last assistant message's input + cache + output tokens, which equals the total the harness
reports). Prints runs.json. TRANSCRIPTS can point at another set of Claude Code subagent transcripts."""
import glob
import json
import os
import re

SUB = os.environ.get("TRANSCRIPTS", os.path.expanduser("~/.claude/projects/*/subagents/agent-*.jsonl"))
TAG = re.compile(r"answers/([A-Za-z0-9_]+)\.txt")


def runs():
    out = {}
    for path in glob.glob(SUB):
        tag, last, model = None, None, None
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                msg = rec.get("message") or {}
                if tag is None and msg.get("role") == "user":
                    content = msg.get("content")
                    text = content if isinstance(content, str) else " ".join(
                        p.get("text", "") for p in content if isinstance(p, dict)) if isinstance(content, list) else ""
                    m = TAG.search(text)
                    if m:
                        tag = m.group(1)
                if msg.get("role") == "assistant" and msg.get("usage"):
                    last, model = msg["usage"], msg.get("model", model)
        if tag and last:
            total = sum(last.get(k, 0) or 0 for k in ("input_tokens", "cache_creation_input_tokens",
                                                       "cache_read_input_tokens", "output_tokens"))
            out[tag] = {"model": model, "tokens": total}
    return out


if __name__ == "__main__":
    print(json.dumps(dict(sorted(runs().items())), indent=1))
