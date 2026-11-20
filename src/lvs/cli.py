"""The `lvs` command: build a map of a codebase, read it, look up exact pieces, watch code live,
or serve it all to an AI agent over MCP."""

import sys


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["live"]:          # these two read their own options, so they skip the map's parser
        from . import live
        return live.main(argv[1:])
    if argv[:1] == ["mcp"]:
        from . import mcp_server
        return mcp_server.main(argv[1:])
    from . import codemap
    return codemap.main(argv)


if __name__ == "__main__":
    sys.exit(main())
