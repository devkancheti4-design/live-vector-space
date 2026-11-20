"""End-to-end tests for lvs: build a small project, read its map, look things up, rebuild it
byte-exact, and talk to the MCP server over stdio. Run: python3 -m unittest discover -s tests"""

import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

import lvs
from lvs import codemap

FILES = {
    "proj/__init__.py": '"""Proj: a tiny sample package for testing lvs."""\n',
    "proj/core.py": textwrap.dedent('''\
        """Core helpers: retries with backoff and a small cache."""

        import time

        RETRY_LIMIT = 7          # how many times fetch() tries before giving up


        def backoff(attempt):
            """Seconds to wait before the next try."""
            return min(30, 2 ** attempt)


        def fetch(url, get):
            """Fetch a URL, retrying with backoff."""
            for attempt in range(RETRY_LIMIT):
                try:
                    return get(url)
                except OSError:
                    time.sleep(backoff(attempt))
            raise RuntimeError(f"gave up on {url}")


        class Cache:
            """A tiny least-recently-used cache."""

            def __init__(self, size=128):
                self.size, self.items = size, {}

            def put(self, key, value):
                self.items.pop(key, None)
                self.items[key] = value
                while len(self.items) > self.size:
                    self.items.pop(next(iter(self.items)))
        '''),
    "tools/run.py": textwrap.dedent('''\
        """Command line entry point: fetch every URL given."""

        import sys

        from proj.core import fetch


        def main():
            for url in sys.argv[1:]:
                print(fetch(url, get=lambda u: u.upper()))


        if __name__ == "__main__":
            main()
        '''),
    "venv/lib/junk.py": "JUNK = 1\n",          # a virtualenv: never part of the map
    ".hidden/junk.py": "JUNK = 2\n",           # hidden folders are skipped too
    "notes.txt": "not python\n",
}
CODE = ["proj/__init__.py", "proj/core.py", "tools/run.py"]


def make_project(where):
    for rel, text in FILES.items():
        path = os.path.join(where, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)


def run(*argv):
    """Run an lvs command in-process; return what it printed."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        codemap.main(list(argv))
    return out.getvalue()


def setUpModule():
    global TMP, PROJ, STORE
    TMP = tempfile.mkdtemp(prefix="lvs-test-")
    PROJ = os.path.join(TMP, "proj")           # the project folder shares its name with its package
    make_project(PROJ)
    STORE = os.path.join(PROJ, ".lvs", "code")
    run("build", PROJ, "--store", STORE)


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


class Store(unittest.TestCase):
    def test_only_the_projects_python_is_kept(self):
        meta = codemap.load_meta(STORE)
        self.assertEqual(sorted(f["path"] for f in meta["files"]), ["proj/" + p for p in CODE])

    def test_rebuild_is_byte_exact(self):
        out = os.path.join(TMP, "rebuilt")
        self.assertIn("3 of 3 byte-exact", run("rebuild", out, "--store", STORE))
        for rel in CODE:
            with open(os.path.join(PROJ, rel), "rb") as a, open(os.path.join(out, "proj", rel), "rb") as b:
                self.assertEqual(a.read(), b.read(), rel)

    def test_map_is_small_and_shows_the_whole_project(self):
        text = run("map", "--store", STORE)
        self.assertLess(len(text.encode()), codemap.MAP_BUDGET + 2000)
        for name in ("core.py", "run.py", "Core helpers", "Shape:", "Look up:"):
            self.assertIn(name, text)

    def test_find_shows_the_matching_line(self):
        text = run("find", "RETRY_LIMIT", "--store", STORE)
        self.assertRegex(text, r"\d+\s+proj/core\.py:5\s+RETRY_LIMIT = 7")
        self.assertIn("no piece mentions it", run("find", "zebra", "--store", STORE))

    def test_get_gives_back_the_exact_code(self):
        meta = codemap.load_meta(STORE)
        f = next(f for f in meta["files"] if f["path"].endswith("core.py"))
        text = run("get", *map(str, range(f["first"], f["last"] + 1)), "--store", STORE)
        pieces = re.split(r"(?m)^--- piece \d+ \([^)]*\) ---\n", text)[1:]
        self.assertEqual("".join(pieces), FILES["proj/core.py"])
        self.assertIn("no piece 9999", run("get", "9999", "--store", STORE))

    def test_files_zooms_into_a_folder_named_like_the_project(self):
        text = run("files", "proj", "--store", STORE)
        self.assertIn("core.py", text)
        self.assertNotIn("run.py", text)

    def test_lookups_find_the_store_from_a_subfolder(self):
        here = os.getcwd()
        os.chdir(os.path.join(PROJ, "tools"))
        try:
            self.assertIn("RETRY_LIMIT = 7", run("find", "retry_limit"))
        finally:
            os.chdir(here)

    def test_no_store_says_how_to_make_one(self):
        with self.assertRaises(SystemExit) as caught:
            run("map", "--store", os.path.join(TMP, "nothing", "code"))
        self.assertIn("lvs build", str(caught.exception.code))


class Command(unittest.TestCase):
    def lvs(self, *argv, **kw):
        env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(lvs.__file__)))
        return subprocess.run([sys.executable, "-m", "lvs", *argv], env=env, capture_output=True,
                              text=True, timeout=120, **kw)

    def test_version(self):
        self.assertEqual(self.lvs("--version").stdout.strip(), f"lvs {lvs.__version__}")

    def test_live_draws_the_square(self):
        svg = os.path.join(TMP, "square.svg")
        done = self.lvs("live", os.path.join(PROJ, "proj", "core.py"), "--square", svg)
        self.assertEqual(done.returncode, 0, done.stderr)
        with open(svg, encoding="utf-8") as fh:
            self.assertTrue(fh.read().lstrip().startswith("<svg"))


class Mcp(unittest.TestCase):
    def session(self, root, messages):
        """Send messages to `lvs mcp --root root` over stdio; return its replies by id."""
        env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(lvs.__file__)))
        lines = [m if isinstance(m, str) else json.dumps(m) for m in messages]
        done = subprocess.run([sys.executable, "-m", "lvs", "mcp", "--root", root], env=env,
                              input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=300)
        self.assertEqual(done.returncode, 0, done.stderr)
        replies = [json.loads(line) for line in done.stdout.splitlines()]   # every stdout line is JSON-RPC
        return {r.get("id"): r for r in replies}

    @staticmethod
    def call(mid, name, **arguments):
        return {"jsonrpc": "2.0", "id": mid, "method": "tools/call",
                "params": {"name": name, "arguments": arguments}}

    def test_a_session(self):
        meta = codemap.load_meta(STORE)
        core = next(f for f in meta["files"] if f["path"].endswith("core.py"))
        r = self.session(PROJ, [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                        "clientInfo": {"name": "test", "version": "0"}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            self.call(3, "map"),
            self.call(4, "find", words=["RETRY_LIMIT"]),
            self.call(5, "get", numbers=list(range(core["first"], core["last"] + 1))),
            self.call(6, "files", folder="proj"),
            self.call(7, "get", numbers=[0]),
            {"jsonrpc": "2.0", "id": 8, "method": "resources/list"},
            {"jsonrpc": "2.0", "id": 9, "method": "ping"},
            self.call(10, "delete_everything"),
            "{not json",
        ])
        init = r[1]["result"]
        self.assertEqual(init["protocolVersion"], "2025-06-18")
        self.assertEqual(init["serverInfo"]["name"], "lvs")
        self.assertIn("tools", init["capabilities"])
        self.assertEqual({t["name"] for t in r[2]["result"]["tools"]}, {"map", "files", "find", "get", "build"})
        text = lambda i: r[i]["result"]["content"][0]["text"]
        self.assertIn("core.py", text(3))
        self.assertIn("RETRY_LIMIT = 7", text(4))
        pieces = re.split(r"(?m)^--- piece \d+ \([^)]*\) ---\n", text(5) + "\n")[1:]
        self.assertEqual("".join(pieces), FILES["proj/core.py"])
        self.assertNotIn("run.py", text(6))
        self.assertTrue(r[7]["result"]["isError"])
        self.assertEqual(r[8]["error"]["code"], -32601)
        self.assertEqual(r[9]["result"], {})
        self.assertEqual(r[10]["error"]["code"], -32602)
        self.assertEqual(r[None]["error"]["code"], -32700)
        self.assertEqual(len(r), 11)              # the notification got no reply

    def test_the_agent_can_build_the_map(self):
        fresh = os.path.join(TMP, "fresh")
        make_project(fresh)
        r = self.session(fresh, [
            self.call(1, "map"),
            self.call(2, "build", paths=["../.."]),
            self.call(3, "build"),
            self.call(4, "map"),
        ])
        self.assertTrue(r[1]["result"]["isError"])
        self.assertIn("build", r[1]["result"]["content"][0]["text"])
        self.assertTrue(r[2]["result"]["isError"])                  # outside the project: refused
        self.assertFalse(r[3]["result"]["isError"])
        self.assertTrue(os.path.exists(os.path.join(fresh, ".lvs", "code_map.json")))
        self.assertIn("3 file(s)", r[4]["result"]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
