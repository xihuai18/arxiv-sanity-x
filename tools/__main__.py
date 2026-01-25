"""Unified module entrypoint for tools.

Usage:
  python -m tools <command> [args...]

Commands map to existing scripts under tools/.
This avoids import-path issues when running files directly.
"""

from __future__ import annotations

import runpy
import sys

_COMMANDS: dict[str, str] = {
    "compute": "tools.compute",
    "arxiv_daemon": "tools.arxiv_daemon",
    "paper_summarizer": "tools.paper_summarizer",
    "batch_paper_summarizer": "tools.batch_paper_summarizer",
    "send_emails": "tools.send_emails",
    "daemon": "tools.daemon",
}


def _print_help() -> int:
    cmds = "\n".join(f"  - {k}" for k in sorted(_COMMANDS))
    sys.stderr.write("Usage: python -m tools <command> [args...]\n\nAvailable commands:\n" + cmds + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        return _print_help()

    cmd = argv.pop(0)
    mod = _COMMANDS.get(cmd)
    if not mod:
        sys.stderr.write(f"Unknown command: {cmd}\n\n")
        _print_help()
        return 2

    # Reconstruct argv for the target module.
    sys.argv = [cmd] + argv
    runpy.run_module(mod, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
