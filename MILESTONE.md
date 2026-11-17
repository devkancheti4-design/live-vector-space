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
