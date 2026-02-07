---
name: claude-code-bridge
description: Invoke Claude Code as a read-only consultant. Use when the user asks for Claude's opinion, plan, review, or analysis. Triggers on "ask Claude", "Claude review", "Claude plan", "run claude".
---

# Claude Code Bridge

Call Claude Code CLI for read-only consultation — plans, reviews, general questions.

## Prerequisite

Claude Code CLI must be installed (`npm i -g @anthropic-ai/claude-code`) and authenticated.

## Invocation

```bash
claude -p "<your question>" --allowedTools "Read,Bash" --disallowedTools "Write,Edit"
```

- `-p` runs non-interactively; Claude prints plain text to stdout.
- `--allowedTools "Read,Bash"` lets Claude read files and run commands to understand the project.
- `--disallowedTools "Write,Edit"` prevents Claude from modifying any files.

## Typical Usage

Plan:

```bash
claude -p "Read the codebase and propose a step-by-step plan to implement <feature>" --allowedTools "Read,Bash" --disallowedTools "Write,Edit"
```

Code review:

```bash
git diff HEAD~1 | claude -p "Review this diff for bugs and improvements" --allowedTools "Read,Bash" --disallowedTools "Write,Edit"
```

General question:

```bash
claude -p "<your question>" --allowedTools "Read,Bash" --disallowedTools "Write,Edit"
```

## Notes

- Each invocation is independent; Claude does not share Codex's context.
- Add `--max-turns 3` to limit exploration on large codebases.
- Recommend `--model Opus` for best results.
