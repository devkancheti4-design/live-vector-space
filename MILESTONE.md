# Milestone 1: a cheap map plus exact lookups (16 Nov 2026)

**The idea:** an LLM doesn't need to read all of a code file. Give it a cheap map (one line of intent
plus the code's shape), and let it look up exact, numbered pieces of the code only where the map isn't
enough. The pieces stay exact, so the whole file can always be rebuilt byte for byte.

**The result**, measured on `live_vector_space.py` (52,552 bytes) with fresh models that had never seen it:

| What the model had | Extra tokens | Purpose (out of 14) | Details (out of 4) |
|---|---|---|---|
| The map only (one line + shape) | +572 | 12 | 0 |
| **The map + numbered lookups** | **+9,221** | **14** | **4** |
| The whole file | +30,416 | 14 | 4 |

With lookups it read 8 of the 80 pieces (17% of the file) and matched the full read exactly. The 80 pieces
rebuild the file byte-exact (same SHA-256).

## How we got here

1. **Code as vectors in 3-D** (`live_vector_space.py`). Every statement becomes a 1024-number vector from
   its words and syntax; PCA brings them down to 3-D; a browser view redraws live whenever the file is
   saved. Similar statements cluster.
2. **The cube and the fluid.** The space went into a cube filled with 14,000 random particles, the
   high-entropy "fluid", which settles into the code's shape as it cools. Entropy falls from about 0.95
   to about 0.6.
3. **The square minus the path** (`code_square.svg`). The shape was flattened into a square: the
   statements joined in file order and cut out of a solid square, a 19 KB fingerprint.
4. **Can an LLM read meaning from the shape? No.** Blind models could count the statements but not say
   what the program does. Swapping all 964 names and words for gibberish left the fingerprint identical
   (exactly, with unhashed counts), so the shape cannot hold meaning.
5. **The skeleton** (`skeleton.py`): 8 cores of similar statements, each with a position and a few
   words, in 676 bytes. Quiz: words + numbers 10/14, words only 10/14, numbers only 7/14 (the same as
   answering "no" every time). The meaning came from the words.
6. **The cube grid as an exact frame** (`shape_codec.py`). Points snapped to a 12-bit grid inside the
   cube decode as a perfect copy. The grid gave the big saving (8,690 → 1,609 bytes); forced triangles
   added little.
7. **Shapes the points choose themselves** (`optimal_shapes.py`). Choosing by bit count, the code's
   points picked 8 blobs (1,471 bytes): the cloud is a nearly round ball. On control data that really
   forms lines and a sheet, the same method found them and saved 32%, so the method works.
8. **The compression floor.** An exact copy of the file needs about 16 KB (brotli). A 1 KB map can
   never be byte-exact; only the exact pieces can.
9. **One line of intent + the shape.** 12/14 on purpose and 0/4 on details at about 2% of the tokens: a
   great map, but it can't answer specifics.
10. **The milestone** (`code_map.py`). The map, a word index (`find fluid` → piece numbers) and exact
    numbered pieces (`get 50`): 14/14 and 4/4 for about a third of the tokens, with a byte-exact rebuild.

## What we keep

- The **shape** is the overview; the **words** carry the meaning; the **numbered exact pieces** are the
  truth; the **cube grid** is the exact frame.
- Every claim gets measured: blind quizzes on fresh models, byte-exact checks, and control data.
- Each quiz so far was a single run on one file. Next: a larger multi-file codebase, and repeated runs
  to see how much the scores vary.

# Milestone 2: the whole codebase from a small map (17 Nov 2026)

**The idea, at scale:** the same map plus exact lookups, now on big multi-file codebases no model has
seen. The question that matters for an LLM: does the map give it the whole codebase?

**The result**, on two Python repositories that went public after the models were trained. Models that
saw nothing scored 0, so both are truly unseen. Each fresh model saw just one view of the same size
(about 4.6 KB, 0.03% of the code), then answered 10 questions about how the codebase is built:

| What the model saw (same bytes) | reef (14.5 MB) | PRAXIST (11.9 MB) | Total (out of 20) |
|---|---|---|---|
| **The map (shapes, words, sizes)** | **10** | **10** | **20** |
| Plain docstrings | 9 | 8 | 17 |
| File tree | 8 | 8 | 16 |
| Raw Python | 3 | 1 | 4 |

With lookups added, every exact detail tested came back right. The numbered pieces keep every byte:

| How reef's 1,231 files are kept | Size of the code | Read one piece alone? | Byte-exact? |
|---|---|---|---|
| **Numbered pieces (zstd, shared dictionary)** | **30.7%** | **yes, any of 17,660** | **yes** |
| zip -9 | 30.3% | one whole file at a time | yes |
| One zstd -19 stream | 17.0% | no | yes |
| FluidCompressor, one stream | 11.5% | no | yes |

**6 million tokens in a 1M window.** Reef is about 6.2 million tokens of code (estimated), so no
model's 1M window can hold it raw. One fresh model session, using only the map and lookups, answered
48 questions drawn from all over reef:

| One session on reef | Result |
|---|---|
| Right answers | **46 of 48** |
| Tokens the whole session used | **111,501**, 11% of a 1M window |
| Pieces the tool unpacked | 862 of 17,660 (4.9%), most only to show one matching line |
| Code that stayed compressed | **88.7%**; the model itself never decompressed anything |

## How we got here

1. **Bigger codebases.** Map + lookups held up on numpy (3.7 MB) and the Python standard library
   (11.8 MB): full marks while reading 0.02% of the code. But ripgrep matched it, for fewer tokens,
   whenever the questions named the code.
2. **What was off.** The map grew with the repo (19 KB on the standard library). It is now capped at
   about 5 KB at any size, and `find` shows the matching line.
3. **Code nobody has seen.** First the project's own files, then reef and PRAXIST, downloaded Python
   only. On the way, zooming into a folder named like the repo (`reef/reef/`) was fixed.
4. **Finding single facts is grep's ground.** Grep matched map + lookups on accuracy and was 14-42%
   cheaper at every size. Given a free choice, models used grep for 41 of 42 commands.
5. **The whole codebase is the map's ground.** 4.6 KB of map: 20/20 on architecture questions; the same
   bytes of Python: 4/20. With tools, the map was 26% cheaper than grep on reef and even on PRAXIST.
6. **Shape Python** (`shape_lang.py`). The pipeline as planned: program → points in the cube → the shapes
   the points choose (exact on the grid) → numbered statements with `shape`, `find` and `get`. It rebuilds
   byte-exact and scored 18/18 on the first test for about a third of a full read.
7. **Shapes as the only way in.** With word search banned it still scored 18/18, but cost about a full
   read: 8 clumps with 4 words each say what is similar, not what it does.
8. **Compression.** The exact store is as small as zip (31% against 30%) and can still hand over any
   single piece instantly. FluidCompressor took reef's raw code to 11.5%, byte-exact, but as one
   stream that has to be unpacked whole.
9. **6 million tokens in a 1M window.** One session, map and lookups only, answered 46 of 48 random
   questions about reef in 111,501 tokens, while 88.7% of the code stayed compressed.

## What we keep

- A map of a few KB gives an LLM the whole codebase; the numbered exact pieces give every detail and
  every byte back.
- An LLM can work with a codebase several times bigger than its context window: the store holds it
  all, compressed and exact, and the LLM pays only for the map and the pieces it asks for.
- The **shape is the heart**: it is what the program becomes in the cube. It still needs words to guide
  an LLM to the right place.
- For one fact, grep is cheaper; the map's strength is the whole picture.
- Next: test another compressor against the store, and give each shape a one-line intent, then rerun
  the shapes-only test.

## Scaled up: 66 blind runs (20 Nov 2026)

The claims were then retested at scale on reef and PRAXIST (method, prompts, grading and the grep
baseline are in [Method of the 66 runs](#method-of-the-66-runs)):
- 5 runs per condition, each on fresh question sets that never repeat;
- every answer graded by script against keys taken from the code;
- every token count read from the run's own transcript.

A model with no access to the code scored 1 of 48 on each repo, so both are truly unseen.

**Exact facts through the map and lookups** (48 questions per session):

| Repo | Right | 95% range | Tokens per session | Of reading the whole code |
|---|---|---|---|---|
| reef (14.5 MB, about 6.2M tokens) | 231/240 (96.2%) | 93.0–98.0% | 108k–233k | 1.7–3.8% |
| PRAXIST (11.9 MB, about 5.1M tokens) | 237/240 (98.8%) | 96.4–99.6% | 89k–117k | 1.7–2.3% |
| **Both** | **468/480 (97.5%)** | **95.7–98.6%** | | |

Architecture and detail questions were 190 of 190. All 12 misses came from finding a function by its docstring
or a constant by its comment.

**The whole codebase from one small view** (same bytes, about 4.6 KB; 10 architecture questions; 5 runs on
each repo):

| View | Right | 95% range |
|---|---|---|
| **The map** | **100/100** | **96–100%** |
| Plain docstrings | 85/100 | 77–91% |
| File tree | 80/100 | 71–87% |
| Raw code | 34/100 | 25–44% |
| Nothing (guessing) | 10/40 | 14–40% |

The map was perfect in all 10 runs. Through it, models answered 97.5% of exact questions about codebases 25 to
60 times bigger than their context, for about 2–4% of the tokens a full read would take.

## Method of the 66 runs

How the scaled test above was run, so it can be checked and rerun. Everything except the two codebases is
in [`bench/`](bench/): the scripts, every prompt, the question sets with their keys, every answer, the
lookup logs and each run's token count. `python3 bench/tools/summarize.py` regrades every answer and
prints the numbers on this page.

**Codebases.** Two public Python repositories, at the commits tested; hidden folders were dropped:
- reef: [Human-Agent-Society/reef](https://github.com/Human-Agent-Society/reef) at `6668ea1` (17 Nov 2026), 1,231 Python files, 14.5 MB;
- PRAXIST: [sapientinc/PRAXIST](https://github.com/sapientinc/PRAXIST) at `5b50658` (23 Sep 2026), 546 Python files, 11.9 MB.

Given only the questions, a model scored 1/48 on each, so it didn't know either codebase.

**Model.** Every run was a fresh Claude Sonnet 5.5 agent (`claude-sonnet-5-5`), started as a subagent in
Claude Code, with no memory of other runs, no web and no other agents. Each got one prompt
(`bench/prompts.json`), read its questions from a file and wrote its answers to a file. At most 20 ran at
once.

**What each run could use.**

| Condition | Allowed | Runs |
|---|---|---|
| Map + lookups | only `map`, `files`, `find` and `get` on the store (the research `code_map.py`, whose output is the same as `lvs`'s) | 5 per repo |
| Grep (baseline) | read-only `rg`, `grep`, `find`, `ls`, `head`, `tail`, `sed -n`, `cat`, `wc` and `du`, and Read, on the raw files | 5 per repo |
| Memory only (control) | the questions, nothing else | 1 per repo |
| One same-size view | one file, read once: the map view (`map` plus a `files` zoom into the main package), plain docstrings, the file tree or raw code, each about 4.6 KB | 5 per view per repo |
| Nothing (control) | the 10 architecture questions, nothing else | 2 per repo |

That is 10 + 10 + 2 + 40 + 4 = 66 runs.

**Questions.** Each 48-question set holds two kinds of question:
- The repo's hand-written questions on architecture and details (20 for reef, 18 for PRAXIST), each keyed
  by regular expressions (`bench/tools/fixed_*.json`).
- Random questions drawn from the parsed code by `bench/tools/make_battery2.py` (seed 2026). Constants:
  "A constant is documented in the code as: '…' What is its value?" Functions: "Which function has this
  docstring: '…'? Give its name and file."
  - A constant qualifies if its comment is unique, has no digits and is at least 25 characters long.
  - A function qualifies if the first line of its docstring is unique and at least 40 characters long.
  - Constants the hand-written questions already ask about are left out, and no item repeats across the
    five sets.
  - The keys come straight from the code.

The views test asked each repo's first 10 hand-written questions (sets `RU` and `PU`).

**Grading.** By script, with no human or model judging (`bench/tools/grade.py`). Only the answer before
" — file:line" counts:
- A constant is right if a number in the answer equals the key, allowing for units (×1024ⁿ, ×1000ⁿ,
  minutes, hours, days, weeks, percent), or if the key's expression appears as written.
- A function is right if its name appears as a word.
- A hand-written question is right if all its patterns match (some must appear in order, some are a
  number range).
- An answer that starts with "unsure" is wrong, whatever follows.

**Tokens.** Each run's total comes from its own transcript: the last assistant message's input,
cache-creation, cache-read and output tokens, which add up to the total Claude Code reports
(`bench/runs.json`). "Of reading the whole code" compares that total with the code's size in tokens,
estimated at 2.33 bytes per token: 6.2M for reef, 5.1M for PRAXIST.

**Statistics.** Accuracy has 95% Wilson intervals. Map minus grep uses paired 95% t intervals over the 10
question sets (`bench/tools/summarize.py`).

**The baseline: grep on the same questions.**

| 48-question sets (10 per method) | Right | 95% range | Hand-written | Constants | Functions | Tokens per session |
|---|---|---|---|---|---|---|
| Map + lookups | 468/480 (97.5%) | 95.7–98.6% | 190/190 | 73/75 | 205/215 | 89k–232k |
| Grep on the raw files | 477/480 (99.4%) | 98.2–99.8% | 188/190 | 75/75 | 214/215 | 70k–80k |

Paired over the 10 sets, map minus grep:
- accuracy: −1.9 points (95% CI −4.7 to +1.0), no clear difference;
- tokens: +64k per session (95% CI +31k to +97k).

Grep reads the raw files on disk. Map + lookups reads only the compressed store, unpacking just the pieces
it asks for. What the map adds is the whole codebase at once: from 4.6 KB of map view, 100/100 on
architecture questions, against 34/100 for the same bytes of raw code (table above).

**Limits.** Two codebases, one model, Python only, five sets per condition.
