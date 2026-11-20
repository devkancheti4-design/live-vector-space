# The 66-run benchmark

Everything needed to check or rerun the scaled test in
[MILESTONE.md](../MILESTONE.md#method-of-the-66-runs), except the two codebases. They have no license
file, so their code isn't copied here: rebuild them from the pinned commits below.

| File | What it is |
|---|---|
| `tools/make_battery2.py` | makes a repo's five 48-question sets: its hand-written questions plus random documented constants and docstring-only functions (seed 2026), with keys from the parsed code |
| `tools/fixed_reef.json`, `tools/fixed_praxist.json` | the hand-written questions (architecture and details) and their regex keys |
| `tools/build_views.py` | makes the four same-size views: map, docstrings, file tree and raw code |
| `tools/grade.py` | grades one answers file against its key, by script |
| `tools/summarize.py` | grades every run, then prints accuracy with 95% intervals and the paired map − grep differences |
| `tools/tokens.py` | reads each run's model and token total from its transcript (it made `runs.json`) |
| `prompts.json` | the exact prompt each run got, with local paths shown as `{BENCH}` and `{CODE_MAP}` |
| `sets/` | the question sets and keys: `R` = reef, `P` = PRAXIST, `RU`/`PU` = the views test's 10 questions |
| `answers/` | what each run answered |
| `logs/` | every lookup the map runs made: the command, its arguments and the bytes it returned |
| `runs.json` | each run's model and total tokens |
| `summary.json` | the graded results, as `summarize.py` writes them |

## Regrade

```bash
python3 bench/tools/summarize.py
```

## Rerun from scratch

You need `lvs` installed (`pip install live-vector-space`). Work in an empty folder, which plays the part
of `{BENCH}` in the prompts; `LVS` is this repository. First the codebases, at the tested commits,
without hidden folders:

```bash
LVS=path/to/live-vector-space
git clone https://github.com/Human-Agent-Society/reef dl/reef && git -C dl/reef checkout 6668ea1add0e35b411e86b48845b5060ac4e0a84
git clone https://github.com/sapientinc/PRAXIST dl/praxist && git -C dl/praxist checkout 5b50658b219d292fe471cfae99a9ec0797429ed7
rsync -a --exclude '.git' --exclude '.*' dl/reef/ big/reef/
rsync -a --exclude '.git' --exclude '.*' dl/praxist/ big/praxist/
mkdir -p answers logs
```

Then the stores, the question sets and the views. The names left out of the sets are constants the
hand-written questions already ask about:

```bash
(cd big && lvs build reef --store ../stores/reefL && lvs build praxist --store ../stores/praxP)
python3 -I $LVS/bench/tools/make_battery2.py big/reef sets R 28 FIRST_RETRY_DELAY_SECONDS,MAX_RETRY_DELAY_SECONDS,SWEEP_INTERVAL_SECONDS,RECORD_PAGE_SIZE,MAX_RECORDED_BYTES,POSTGRES_SCHEMA_VERSION,JOB_DIRECTORY_TTL_SECONDS,DEFAULT_GENERATION_ATTEMPTS,DEFAULT_GENERATION_DEADLINE_SECONDS $LVS/bench/tools/fixed_reef.json
python3 -I $LVS/bench/tools/make_battery2.py big/praxist sets P 30 DEFAULT_COHORT_SIZE,DEFAULT_PROMOTE_TOP_K,DEFAULT_FULL_AUTO_MAX_RUNTIME_SECONDS,API_BILLING_RETRY_INTERVAL,DEFAULT_MAX_PER_GPU,CGROUP_MEMORY_WARNING_RATIO,CGROUP_MEMORY_SHUTDOWN_RATIO,STALE_AFTER_DAYS,DEFAULT_CHUNK_CHARS $LVS/bench/tools/fixed_praxist.json
python3 $LVS/bench/tools/build_views.py stores/reefL big/reef views/r reef reef/__init__.py,reef/__main__.py,reef/cli.py,reef/pipeline_context.py
python3 $LVS/bench/tools/build_views.py stores/praxP big/praxist views/p praxist/praxist,praxist/praxist/plugins praxist/__init__.py,praxist/__main__.py,praxist/run.py,praxist/config.py
```

The views test's sets (`RU`, `PU`) are the first 10 hand-written questions of each repo; copy them from
`$LVS/bench/sets/`.

Give each prompt in `prompts.json` to a fresh agent, with `{BENCH}` replaced by this folder's full path
and `python3 '{CODE_MAP}'` by `lvs`. The tested runs were Claude Sonnet 5.5 subagents in Claude Code, at
most 20 at a time. Then grade (token totals are read from Claude Code's transcripts when there is no
`runs.json`):

```bash
BENCH=$PWD python3 $LVS/bench/tools/summarize.py
```
