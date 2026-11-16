# Live Vector Space

Watch a Python file live, as vectors in 3-D space, plus a set of experiments on
squeezing code into shapes and maps that an LLM can read cheaply.

![The square minus the code path](code_square.svg)

## What's here

| File | What it does |
|---|---|
| `live_vector_space.py` | Turns every statement of a Python file into a 1024-d token vector, projects them to 3-D with PCA, and serves a live browser view: a cube filled with 14,000 particles of random "fluid" that settle into the code's shape. It updates every time the file is saved. |
| `skeleton.py` | Squeezes a file into 8 labeled cores of similar statements (a position in the cube plus a few words each). |
| `shape_codec.py` | Stores the 3-D points on a 12-bit grid in the cube, tied to anchor shapes, and decodes a perfect copy. |
| `optimal_shapes.py` | Lets the points choose their own shapes (blob, line, curve, sheet) by bit count (minimum description length). |
| `code_map.py` | A cheap map of a file plus exact, numbered pieces an LLM can look up; rebuilds the file byte-exact. |

Generated outputs: `code_square.svg`, `square_vs_scrambled.svg`, `shape_map.svg`,
`optimal_shapes.html`, `optimal_shapes_control.html` (open locally in a browser),
`code_map.json` and `code_store.bin`.

## Run it

Needs Python 3 with numpy; `code_map.py` needs Python 3.14+ for the built-in `compression.zstd`.

```bash
python3 live_vector_space.py                      # live 3-D view of itself at http://localhost:8765
python3 live_vector_space.py path/to/other.py     # any other Python file
python3 live_vector_space.py --square square.svg  # save the square-minus-the-path picture
python3 skeleton.py --stats
python3 shape_codec.py --svg shape_map.svg
python3 optimal_shapes.py --html optimal_shapes.html
python3 code_map.py build
python3 code_map.py find fluid svg
python3 code_map.py get 50
```

Viewer controls: drag to rotate, scroll to zoom, hover or click a point. Keys: `H` shake the fluid,
`F` fluid on/off, `V` vectors, `P` code path, `L` labels, `T` table, `S` square, `Space` spin.

## What the experiments found

Measured on `live_vector_space.py` itself (52,552 bytes, 357 statements).

- **The shape alone doesn't carry meaning.** Swapping all 964 names and words for gibberish leaves the
  fingerprint identical (exactly, with unhashed word counts). Fresh models given only the shape's numbers
  scored 7/14 on a quiz about the program, the same as answering "no" to everything.
- **Words carry the gist cheaply.** The file's one-line intent plus the skeleton (about 750 bytes) scored
  12/14 on what the program does, but 0/4 on details.
- **A map plus numbered lookups is the sweet spot.** With the map and exact lookups by number
  (`code_map.py`), a fresh model scored 14/14 and 4/4, the same as reading the whole file, using about
  9,200 extra tokens instead of about 30,400, and reading 8 of the 80 pieces.
- **Compression.** The cube grid stores the 357 points in 1,609 bytes; shapes the points chose themselves
  (8 blobs) take 1,471. The cloud is a nearly round ball, so lines and sheets don't help here (on control
  data that really forms lines and a sheet they save 32%). An exact copy of the code needs about 16 KB
  (brotli); numbered pieces with a shared zstd dictionary take 24 KB, the price of reading any piece on
  its own.

Each LLM test was a single run on one file, so small score differences may be chance.
