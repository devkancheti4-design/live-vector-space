"""lvs mcp - give an AI agent the map and the exact lookups over MCP (stdio, JSON-RPC 2.0).

    lvs mcp                            # the project in the current folder (store in .lvs/)
    lvs mcp --root ~/code/project      # another project folder
    lvs mcp --store /elsewhere/code    # a store kept somewhere else

Tools: map, files, find and get read the compressed store; build makes or refreshes it. The agent
reads the small map first and then asks only for the pieces it needs, so the code stays compressed
and the agent never has to decompress or read anything whole. No dependencies beyond lvs itself.
"""

import argparse
import contextlib
import io
import json
import os
import sys

from . import __version__, codemap

LATEST = "2025-06-18"     # answered when a client doesn't name a protocol version

INSTRUCTIONS = (
    "This server holds a whole codebase as numbered pieces, compressed and byte-exact. Call `map` first: a "
    "few KB that show the whole codebase (folders and files with one-line intents, and the code's shape). "
    "For details, `find` words to get piece numbers with the line that matches, then `get` those numbers "
    "for the exact code; `files` zooms into one folder. There is no need to read whole files. If there is "
    "no map yet, or the code changed, call `build`."
)

READ_ONLY = {"readOnlyHint": True, "openWorldHint": False}
TOOLS = [
    {"name": "map", "title": "Codebase map",
     "description": "The whole codebase in a few KB, however big it is: folders and files with sizes and "
                    "one-line intents, plus the shape of the code. Read this first.",
     "inputSchema": {"type": "object", "properties": {}},
     "annotations": READ_ONLY},
    {"name": "files", "title": "Zoom into a folder",
     "description": "The files under one folder (or one file), each with its size, one-line intent and "
                    "piece numbers. Use '.' for the top level.",
     "inputSchema": {"type": "object", "properties": {
         "folder": {"type": "string", "description": "a folder or file as the map names it, or '.'"},
         "page": {"type": "integer", "minimum": 1, "description": "page of a big folder (default 1)"}},
         "required": ["folder"]},
     "annotations": READ_ONLY},
    {"name": "find", "title": "Find pieces by words",
     "description": "For each word, the numbered pieces that mention it, with file, line number and the "
                    "line that matches. Names (functions, classes, constants) and plain words both work.",
     "inputSchema": {"type": "object", "properties": {
         "words": {"type": "array", "items": {"type": "string"}, "minItems": 1,
                   "description": "words to look for; each gets its own short list of hits"},
         "within": {"type": "string", "description": "only pieces inside this folder or file"}},
         "required": ["words"]},
     "annotations": READ_ONLY},
    {"name": "get", "title": "Exact pieces",
     "description": "The exact code of these numbered pieces, byte for byte, each headed by its file and lines.",
     "inputSchema": {"type": "object", "properties": {
         "numbers": {"type": "array", "items": {"type": "integer", "minimum": 1}, "minItems": 1,
                     "description": "piece numbers from map, files or find"}},
         "required": ["numbers"]},
     "annotations": READ_ONLY},
    {"name": "build", "title": "Build the map",
     "description": "Make the map and the compressed store of the project's Python code, or refresh them "
                    "after the code changed.",
     "inputSchema": {"type": "object", "properties": {
         "paths": {"type": "array", "items": {"type": "string"},
                   "description": "files or folders inside the project (default: the whole project)"},
         "exclude": {"type": "array", "items": {"type": "string"},
                     "description": "folder names to skip, e.g. tests"}}},
     "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True,
                     "openWorldHint": False}},
]
NAMES = {t["name"] for t in TOOLS}


class RpcError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.message = code, message


def strings(value, what):
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{what} must be a list of strings")
    return value


class Server:
    def __init__(self, root, store):
        self.root, self.store = root, store

    def lvs(self, argv):
        """Run one lvs command; return (ok, everything it printed)."""
        out, ok = io.StringIO(), True
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            try:
                codemap.main(argv + ["--store", self.store])
            except SystemExit as e:
                if e.code not in (None, 0):
                    ok = False
                    if isinstance(e.code, str):
                        out.write(e.code)
            except Exception as e:      # report it to the agent; the server keeps running
                ok = False
                out.write(f"{type(e).__name__}: {e}")
        return ok, out.getvalue().rstrip("\n") or "(nothing)"

    def inside_root(self, path):
        full = os.path.abspath(os.path.join(self.root, os.path.expanduser(path)))
        if os.path.commonpath([self.root, full]) != self.root:
            raise ValueError(f"{path} is outside the project folder {self.root}")
        return full

    def tool(self, name, args):
        if name == "build":
            paths = [self.inside_root(p) for p in strings(args.get("paths") or ["."], "paths")]
            exclude = strings(args.get("exclude") or [], "exclude")
            return self.lvs(["build", *paths] + (["--exclude", *exclude] if exclude else []))
        if not os.path.exists(codemap.store_paths(self.store)[0]):
            return False, f"No map yet for {self.root}: call `build` first."
        if name == "map":
            return self.lvs(["map"])
        if name == "files":
            folder, page = args.get("folder"), args.get("page", 1)
            if not isinstance(folder, str) or not isinstance(page, int) or isinstance(page, bool) or page < 1:
                raise ValueError("files needs a folder (string) and optionally a page (whole number from 1)")
            return self.lvs(["files", folder or ".", "--page", str(page)])
        if name == "find":
            words = [w.lstrip("-") for w in strings(args.get("words"), "words")]
            words = [w for w in words if w.strip()]
            if not words:
                raise ValueError("find needs at least one word")
            within = args.get("within")
            if within is not None and not isinstance(within, str):
                raise ValueError("within must be a string")
            return self.lvs(["find", *words] + (["--in", within] if within else []))
        numbers = args.get("numbers")
        if (not isinstance(numbers, list) or not numbers
                or not all(isinstance(k, int) and not isinstance(k, bool) and k >= 1 for k in numbers)):
            raise ValueError("get needs a list of piece numbers (whole numbers from 1)")
        return self.lvs(["get", *map(str, numbers)])

    def call(self, params):
        name, args = params.get("name"), params.get("arguments") or {}
        if name not in NAMES:
            raise RpcError(-32602, f"Unknown tool: {name}")
        if not isinstance(args, dict):
            raise RpcError(-32602, "arguments must be an object")
        try:
            ok, text = self.tool(name, args)
        except ValueError as e:
            ok, text = False, str(e)
        return {"content": [{"type": "text", "text": text}], "isError": not ok}

    def handle(self, msg):
        """One JSON-RPC message in; the reply, or None for notifications and stray responses."""
        if not isinstance(msg, dict):
            return error(None, -32600, "Invalid Request")
        if "method" not in msg:
            return None                 # a response; this server never sends requests
        mid, method, params = msg.get("id"), msg["method"], msg.get("params") or {}
        note = "id" not in msg
        try:
            if not isinstance(method, str) or not isinstance(params, dict):
                raise RpcError(-32600, "Invalid Request")
            if method == "initialize":
                version = params.get("protocolVersion")
                result = {"protocolVersion": version if isinstance(version, str) else LATEST,
                          "capabilities": {"tools": {"listChanged": False}},
                          "serverInfo": {"name": "lvs", "title": "lvs: codebase map and exact lookups",
                                         "version": __version__},
                          "instructions": INSTRUCTIONS}
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                result = self.call(params)
            elif note:
                return None             # notifications/initialized, notifications/cancelled, ...
            else:
                raise RpcError(-32601, f"Method not found: {method}")
        except RpcError as e:
            return None if note else error(mid, e.code, e.message)
        return None if note else {"jsonrpc": "2.0", "id": mid, "result": result}


def error(mid, code, message):
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def serve(server, rfile, wfile):
    """Newline-delimited JSON-RPC: one message per line in, one reply per line out."""
    for raw in rfile:
        if not raw.strip():
            continue
        try:
            msg = json.loads(raw)
        except ValueError:
            reply = error(None, -32700, "Parse error")
        else:
            if isinstance(msg, list):   # a batch (older protocol versions)
                reply = [r for r in map(server.handle, msg) if r is not None] or None
            else:
                reply = server.handle(msg)
        if reply is not None:
            wfile.write(json.dumps(reply, ensure_ascii=False).encode("utf-8") + b"\n")
            wfile.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lvs mcp", description="Serve the map and exact lookups of a codebase "
                                 "to an AI agent over MCP (stdio).")
    ap.add_argument("--root", default=".", help="the project folder (default: the current folder)")
    ap.add_argument("--store", help="map/store prefix, relative to the project (default: .lvs/code)")
    args = ap.parse_args(argv)
    root = os.path.abspath(os.path.expanduser(args.root))
    store = os.path.join(root, os.path.expanduser(args.store or codemap.DEFAULT_STORE))
    real_out = sys.stdout.buffer
    sys.stdout = sys.stderr             # only protocol messages may reach the real stdout
    print(f"lvs mcp {__version__}: project {root}, store {store}", file=sys.stderr, flush=True)
    serve(Server(root, store), sys.stdin.buffer, real_out)


if __name__ == "__main__":
    main()
