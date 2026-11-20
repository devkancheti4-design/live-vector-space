"""Piece splitting and byte-exact rebuilds on many kinds of Python file: line endings, encodings, odd
characters, empty and huge files, and HTML, CSS and JavaScript inside strings. Every piece must cover
its exact lines, and every file must come back byte for byte."""

import contextlib
import io
import os
import random
import re
import shutil
import tempfile
import unittest

from lvs import codemap

SAMPLES = {
    "empty.py": "",
    "one_line.py": "x = 1",
    "bom.py": "\ufeffprint('a byte order mark comes first')\n",
    "crlf.py": "def alpha():\r\n    return 1\r\n\r\n\r\ndef quokka():\r\n    return 2\r\n",
    "old_mac.py": "x = 1\ry = 2\rdef f():\r    return x\r",
    "mixed_endings.py": "a = 1\nb = 2\r\nc = 3\rd = 4\n",
    "formfeed.py": "import os\n\x0c\ndef wombat():\n    pass\n",
    "unicode_breaks.py": 's = "a\u2028b\u2029c\x85d\x0be\x1cf"\ndef narwhal():\n    return s\n',
    "tabs.py": "class T:\n\tdef m(self):\n\t\treturn 1\n",
    "unicode_names.py": "café = 1\n名前 = '🙂'\nπ = 3.14159\n",
    "comments_only.py": "# a comment\n# another\n",
    "docstring_only.py": '"""Only a docstring."""\n',
    "decorators.py": "@first\n@second(1)\ndef f():\n    pass\n\n\n@third\nclass K:\n    pass\n",
    "nested_async.py": "class Outer:\n    class Inner:\n        async def run(self):\n            await thing()\n",
    "main_block.py": "def main():\n    pass\n\n\nif __name__ == '__main__':\n    main()\n",
    "embedded_web.py": ('PAGE = r"""<!doctype html>\n<style>\nbody { margin: 0 }\n</style>\n<script>\n'
                        "function draw() { return 1 }\nasync function load() {}\n// --- events\n"
                        'window.onload = load;\n</script>\n<body></body>\n"""\n'),
    "long_function.py": "def big():\n" + "".join(f"    x{i} = {i}\n" for i in range(200)),
    "long_line.py": "s = '" + "y" * 100_000 + "'\n",
    "no_final_newline.py": "def f():\n    return 1",
    "blank_tail.py": "x = 1   \n\n\n   \n",
    "null_byte.py": "x = '\x00'\n",
}
LATIN1 = "# -*- coding: latin-1 -*-\nname = 'caf\xe9'\n".encode("latin-1")
BITS = ["def f():\n", "class C:\n", "    pass\n", "\n", "\r\n", "\r", "\x0c", "\u2028", "\x85", "x = 1", "@deco\n",
        "# --- part\n", "<script>\n", "function g() {}\n", "async def h():\n", '"""doc"""\n', "}\n", "</div>\n",
        "if __name__ == '__main__':\n", "\t", "é", "🙂", "    return 1\n", "'''", " ", "\n\n\n"]


def run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        codemap.main(list(argv))
    return out.getvalue()


class Splitting(unittest.TestCase):
    def check(self, text):
        pieces, spans = codemap.split_pieces(text)
        self.assertEqual("".join(pieces), text)
        if not text:
            return
        lines = codemap.lines_of(text)
        self.assertEqual(spans[0][0], 1)
        self.assertEqual(spans[-1][1], len(lines))
        for (_, b), (c, _) in zip(spans, spans[1:]):
            self.assertEqual(c, b + 1)                                 # no gaps, no overlaps
        for (a, b), piece in zip(spans, pieces):
            self.assertEqual(piece, "".join(lines[a - 1:b]))           # each piece is exactly its lines

    def test_every_kind_of_file_splits_into_pieces_that_cover_every_byte(self):
        for name, text in SAMPLES.items():
            with self.subTest(name):
                self.check(text)

    def test_random_files_split_into_pieces_that_cover_every_byte(self):
        rng = random.Random(784)
        for _ in range(500):
            self.check("".join(rng.choice(BITS) for _ in range(rng.randrange(0, 150))))

    def test_only_real_line_breaks_count_as_lines(self):
        """Form feeds and Unicode separators inside a line don't shift the line numbers."""
        for name, word, line in (("formfeed.py", "def wombat", 3), ("unicode_breaks.py", "def narwhal", 2),
                                 ("crlf.py", "def quokka", 5), ("old_mac.py", "def f", 3)):
            with self.subTest(name):
                pieces, spans = codemap.split_pieces(SAMPLES[name])
                at = [a for (a, _), p in zip(spans, pieces) if p.startswith(word)]
                self.assertEqual(at, [line])

    def test_long_code_is_cut_into_pieces_of_bounded_size(self):
        pieces, spans = codemap.split_pieces(SAMPLES["long_function.py"])
        self.assertGreater(len(pieces), 2)
        self.assertTrue(all(b - a + 1 <= codemap.MAX_LINES + 20 for a, b in spans))

    def test_decorators_stay_with_what_they_decorate(self):
        pieces, _ = codemap.split_pieces(SAMPLES["decorators.py"])
        self.assertTrue(any(p.startswith("@first\n@second(1)\ndef f():") for p in pieces))
        self.assertTrue(any(p.startswith("@third\nclass K:") for p in pieces))

    def test_web_code_inside_strings_is_cut_at_its_own_joints(self):
        starts = [p.split("\n", 1)[0] for p in codemap.split_pieces(SAMPLES["embedded_web.py"])[0]]
        for joint in ("<style>", "<script>", "function draw() { return 1 }", "async function load() {}", "// --- events"):
            self.assertIn(joint, starts)


class Rebuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="lvs-pieces-")
        cls.proj = os.path.join(cls.tmp, "samples")
        os.makedirs(cls.proj)
        for name, text in SAMPLES.items():
            with open(os.path.join(cls.proj, name), "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
        with open(os.path.join(cls.proj, "latin1.py"), "wb") as fh:
            fh.write(LATIN1)
        cls.store = os.path.join(cls.tmp, "store", "code")
        cls.built = run("build", cls.proj, "--store", cls.store)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_every_file_comes_back_byte_exact(self):
        out = os.path.join(self.tmp, "out")
        self.assertIn(f"{len(SAMPLES)} of {len(SAMPLES)} byte-exact", run("rebuild", out, "--store", self.store))
        for name, text in SAMPLES.items():
            with open(os.path.join(out, "samples", name), "rb") as fh:
                self.assertEqual(fh.read(), text.encode("utf-8"), name)

    def test_any_file_can_be_read_back_from_its_pieces_alone(self):
        meta = codemap.load_meta(self.store)
        store_file = codemap.store_paths(self.store)[1]
        for f in meta["files"]:
            with self.subTest(f["path"]):
                pieces = codemap.read_pieces(store_file, range(f["first"], f["last"] + 1))
                self.assertEqual(b"".join(pieces.values()), SAMPLES[os.path.basename(f["path"])].encode("utf-8"))

    def test_a_file_that_is_not_utf8_is_left_out_and_named(self):
        self.assertIn("left out 1 file(s) that are not UTF-8: samples/latin1.py", self.built)
        self.assertNotIn("samples/latin1.py", [f["path"] for f in codemap.load_meta(self.store)["files"]])

    def test_find_reports_the_real_line_numbers(self):
        for word, where in (("quokka", "crlf.py:5"), ("wombat", "formfeed.py:3"), ("narwhal", "unicode_breaks.py:2")):
            with self.subTest(word):
                self.assertRegex(run("find", word, "--store", self.store), re.escape(where) + r"\s+def " + word)


if __name__ == "__main__":
    unittest.main()
