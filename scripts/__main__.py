"""Unified module entrypoint for scripts.

Usage:
  python -m scripts <command> [args...]

Commands map to existing scripts under scripts/.
"""

from __future__ import annotations

import runpy
import sys

_COMMANDS: dict[str, str] = {
    "cleanup_locks": "scripts.cleanup_locks",
    "cleanup_tasks": "scripts.cleanup_tasks",
}


def _print_help() -> int:
    cmds = "\n".join(f"  - {k}" for k in sorted(_COMMANDS))
    sys.stderr.write("Usage: python -m scripts <command> [args...]\n\nAvailable commands:\n" + cmds + "\n")
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

    sys.argv = [cmd] + argv
    runpy.run_module(mod, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
