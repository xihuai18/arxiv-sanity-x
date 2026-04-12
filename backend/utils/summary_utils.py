"""
Summary utilities for extracting TL;DR and other structured content from summary files.
"""

import re
import time
from pathlib import Path

from loguru import logger

from config import settings
from tools.paper_summarizer import model_cache_key, split_pid_version


def _default_llm_name() -> str:
    return str(settings.llm.name or "")


def _summary_dir() -> str:
    return str(settings.summary_dir)


# LaTeX to Unicode mapping for email display
LATEX_TO_UNICODE = {
    # Greek letters (lowercase)
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "varepsilon": "ϵ",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "vartheta": "ϑ",
    "iota": "ι",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "pi": "π",
    "varpi": "ϖ",
    "rho": "ρ",
    "varrho": "ϱ",
    "sigma": "σ",
    "varsigma": "ς",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "φ",
    "varphi": "ϕ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    # Greek letters (uppercase)
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Xi": "Ξ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Upsilon": "Υ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
    # Math operators
    "cdot": "·",
    "cdots": "⋯",
    "ldots": "…",
    "vdots": "⋮",
    "ddots": "⋱",
    "times": "×",
    "div": "÷",
    "ast": "∗",
    "star": "★",
    "circ": "∘",
    "bullet": "•",
    "oplus": "⊕",
    "otimes": "⊗",
    "ominus": "⊖",
    "odot": "⊙",
    "cap": "∩",
    "cup": "∪",
    "wedge": "∧",
    "vee": "∨",
    "land": "∧",
    "lor": "∨",
    "bigwedge": "⋀",
    "bigvee": "⋁",
    "bigcap": "⋂",
    "bigcup": "⋃",
    "pm": "±",
    "mp": "∓",
    "setminus": "∖",
    "propto": "∝",
    "sim": "∼",
    "simeq": "≃",
    "approx": "≈",
    "cong": "≅",
    "equiv": "≡",
    "perp": "⟂",
    "parallel": "∥",
    "mid": "∣",
    # Text-style operators
    "log": "log",
    "ln": "ln",
    "exp": "exp",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "max": "max",
    "min": "min",
    "sup": "sup",
    "inf": "inf",
    "arg": "arg",
    "dim": "dim",
    "det": "det",
    "Pr": "Pr",
    "argmax": "argmax",
    "argmin": "argmin",
    # Relation symbols
    "leq": "≤",
    "le": "≤",
    "geq": "≥",
    "ge": "≥",
    "neq": "≠",
    "ne": "≠",
    "ll": "≪",
    "gg": "≫",
    "subset": "⊂",
    "supset": "⊃",
    "subseteq": "⊆",
    "supseteq": "⊇",
    "in": "∈",
    "ni": "∋",
    "notin": "∉",
    "models": "⊨",
    "vdash": "⊢",
    # Set and logic symbols
    "emptyset": "∅",
    "varnothing": "∅",
    "forall": "∀",
    "exists": "∃",
    "nexists": "∄",
    "top": "⊤",
    "bot": "⊥",
    "Re": "ℜ",
    "Im": "ℑ",
    "aleph": "ℵ",
    # Arrow symbols
    "to": "→",
    "rightarrow": "→",
    "leftarrow": "←",
    "leftrightarrow": "↔",
    "Rightarrow": "⇒",
    "Leftarrow": "⇐",
    "Leftrightarrow": "⇔",
    "mapsto": "↦",
    "longrightarrow": "⟶",
    "longleftarrow": "⟵",
    "Longrightarrow": "⟹",
    "Longleftarrow": "⟸",
    "hookrightarrow": "↪",
    "hookleftarrow": "↩",
    "uparrow": "↑",
    "downarrow": "↓",
    "implies": "⇒",
    "impliedby": "⇐",
    "iff": "⇔",
    # Common math symbols
    "infty": "∞",
    "infinity": "∞",
    "partial": "∂",
    "nabla": "∇",
    "angle": "∠",
    "triangle": "△",
    "square": "□",
    "diamond": "◇",
    "prime": "′",
    "dagger": "†",
    "ddagger": "‡",
    "ell": "ℓ",
    "hbar": "ℏ",
    # Calculus & operators
    "int": "∫",
    "iint": "∬",
    "iiint": "∭",
    "oint": "∮",
    "sum": "∑",
    "prod": "∏",
    "coprod": "∐",
    "lim": "lim",
    "limsup": "lim sup",
    "liminf": "lim inf",
    "grad": "∇",
    # Font commands for common sets
    "mathbb{R}": "ℝ",
    "mathbb{N}": "ℕ",
    "mathbb{Z}": "ℤ",
    "mathbb{Q}": "ℚ",
    "mathbb{C}": "ℂ",
    "mathbb{E}": "𝔼",
    "mathbb{P}": "ℙ",
    "mathcal{F}": "ℱ",
    "mathcal{L}": "ℒ",
    "mathcal{O}": "𝒪",
    "mathcal{H}": "ℋ",
    # Brackets and delimiters
    "langle": "⟨",
    "rangle": "⟩",
    "lceil": "⌈",
    "rceil": "⌉",
    "lfloor": "⌊",
    "rfloor": "⌋",
    "left": "",
    "right": "",
    "big": "",
    "Big": "",
    "bigg": "",
    "Bigg": "",
    # Spacing commands
    ",": " ",
    ";": " ",
    "quad": " ",
    "qquad": "  ",
    " ": " ",
    # Text-style commands (strip)
    "text": "",
    "mathrm": "",
    "mathbf": "",
    "mathit": "",
    "mathsf": "",
    "mathtt": "",
    "textbf": "",
    "textit": "",
    "emph": "",
    "bm": "",
    "boldsymbol": "",
    # Accents
    "hat": "^",
    "widehat": "^",
    "bar": "¯",
    "overline": "¯",
    "tilde": "~",
    "widetilde": "~",
    "vec": "→",
    "dot": "˙",
    "ddot": "¨",
    # Misc
    "colon": ":",
    "dots": "…",
}

# Superscript mapping for common characters
_SUPERSCRIPT_MAP = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "+": "⁺",
    "-": "⁻",
    "=": "⁼",
    "(": "⁽",
    ")": "⁾",
    "n": "ⁿ",
    "i": "ⁱ",
    "T": "ᵀ",
}

# Subscript mapping for common characters
_SUBSCRIPT_MAP = {
    "0": "₀",
    "1": "₁",
    "2": "₂",
    "3": "₃",
    "4": "₄",
    "5": "₅",
    "6": "₆",
    "7": "₇",
    "8": "₈",
    "9": "₉",
    "+": "₊",
    "-": "₋",
    "=": "₌",
    "(": "₍",
    ")": "₎",
    "a": "ₐ",
    "e": "ₑ",
    "i": "ᵢ",
    "j": "ⱼ",
    "k": "ₖ",
    "n": "ₙ",
    "o": "ₒ",
    "p": "ₚ",
    "r": "ᵣ",
    "s": "ₛ",
    "t": "ₜ",
    "u": "ᵤ",
    "v": "ᵥ",
    "x": "ₓ",
}


def _convert_script(text: str, mapping: dict) -> str:
    """Convert text to super/subscript using Unicode mapping."""
    result = []
    for ch in text:
        result.append(mapping.get(ch, ch))
    return "".join(result)


def latex_to_plaintext(text: str) -> str:
    """
    Convert LaTeX math notation to readable Unicode plaintext for email display.

    Args:
        text: Text containing LaTeX math notation (with $ delimiters)

    Returns:
        Plaintext with LaTeX converted to Unicode symbols
    """
    if not text:
        return ""

    result = text

    # Handle \frac{a}{b} -> (a)/(b)
    frac_pattern = re.compile(r"\\(?:d|t)?frac\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
    while frac_pattern.search(result):
        result = frac_pattern.sub(r"(\1)/(\2)", result)

    # Handle \sqrt{x} -> √(x)
    sqrt_pattern = re.compile(r"\\sqrt\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
    result = sqrt_pattern.sub(r"√(\1)", result)

    # Handle \sqrt[n]{x} -> ⁿ√(x)
    sqrt_n_pattern = re.compile(r"\\sqrt\s*\[([^\]]+)\]\s*\{([^{}]*)\}")
    result = sqrt_n_pattern.sub(r"\1√(\2)", result)

    # Handle font commands with braces: \mathbb{R} etc. (check dict first)
    font_cmds = [
        "mathbb",
        "mathcal",
        "mathbf",
        "mathrm",
        "mathit",
        "mathsf",
        "text",
        "textbf",
        "textit",
        "bm",
        "boldsymbol",
    ]
    for cmd in font_cmds:
        pattern = re.compile(r"\\" + cmd + r"\s*\{([^{}]*)\}")

        def replace_font(m, cmd=cmd):
            full_key = cmd + "{" + m.group(1) + "}"
            if full_key in LATEX_TO_UNICODE:
                return LATEX_TO_UNICODE[full_key]
            # Just return the content without the command
            return m.group(1)

        result = pattern.sub(replace_font, result)

    # Handle superscripts: x^{abc} or x^2
    def replace_superscript(m):
        content = m.group(1) if m.group(1) else m.group(2)
        # Try to convert all characters, fall back to ^(...) if not possible
        converted = _convert_script(content, _SUPERSCRIPT_MAP)
        if converted != content or all(c in _SUPERSCRIPT_MAP for c in content):
            return converted
        return f"^{content}" if len(content) == 1 else f"^({content})"

    result = re.sub(r"\^(?:\{([^{}]+)\}|(\w))", replace_superscript, result)

    # Handle subscripts: x_{abc} or x_2
    def replace_subscript(m):
        content = m.group(1) if m.group(1) else m.group(2)
        converted = _convert_script(content, _SUBSCRIPT_MAP)
        if converted != content or all(c in _SUBSCRIPT_MAP for c in content):
            return converted
        return f"_{content}" if len(content) == 1 else f"_({content})"

    result = re.sub(r"_(?:\{([^{}]+)\}|(\w))", replace_subscript, result)

    # Replace LaTeX commands from dictionary (longest match first)
    sorted_keys = sorted(LATEX_TO_UNICODE.keys(), key=len, reverse=True)
    for cmd in sorted_keys:
        # Skip font commands already handled
        if any(cmd.startswith(f + "{") for f in font_cmds):
            continue
        # Escape special regex characters in the command
        escaped = re.escape(cmd)
        # Match \cmd followed by word boundary or non-letter
        pattern = rf"\\{escaped}(?![a-zA-Z])"
        result = re.sub(pattern, LATEX_TO_UNICODE[cmd], result)

    # Remove remaining unknown \commands (keep content)
    result = re.sub(r"\\([a-zA-Z]+)", r"\1", result)

    # Remove $ delimiters (both inline $...$ and display $$...$$)
    result = re.sub(r"\$\$([^$]+)\$\$", r" \1 ", result)
    result = re.sub(r"\$([^$]+)\$", r"\1", result)

    # Also handle \( \) and \[ \] delimiters
    result = re.sub(r"\\\(([^)]+)\\\)", r"\1", result)
    result = re.sub(r"\\\[([^\]]+)\\\]", r" \1 ", result)

    # Clean up extra whitespace
    result = re.sub(r"[ \t]+", " ", result)
    result = result.strip()

    return result


def markdown_to_email_html(text: str) -> str:
    """
    Convert markdown formatting to HTML for email display.
    Handles bold, italic, links, lists, and converts LaTeX to plaintext.

    Args:
        text: Text with markdown formatting and LaTeX

    Returns:
        HTML-safe text with formatting preserved
    """
    if not text:
        return ""

    # First convert LaTeX to plaintext
    result = latex_to_plaintext(text)

    # Process line-by-line for block elements
    lines = result.split("\n")
    processed_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Horizontal rule: ---, ***, ___
        if re.match(r"^\s*[-*_]{3,}\s*$", line):
            processed_lines.append("<hr>")
            i += 1
            continue

        # Headings: # Title -> <strong>Title</strong>
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading_match:
            content = heading_match.group(2)
            processed_lines.append(f"<strong>{content}</strong>")
            i += 1
            continue

        # Blockquote: > text
        if line.startswith(">"):
            quote_lines = []
            while i < len(lines) and lines[i].startswith(">"):
                quote_content = re.sub(r"^>\s?", "", lines[i])
                quote_lines.append(quote_content)
                i += 1
            processed_lines.append(f"<blockquote>{'<br>'.join(quote_lines)}</blockquote>")
            continue

        # Unordered list: - item, * item, + item
        if re.match(r"^\s*[-*+]\s+", line):
            list_items = []
            while i < len(lines) and re.match(r"^\s*[-*+]\s+", lines[i]):
                item_content = re.sub(r"^\s*[-*+]\s+", "", lines[i])
                list_items.append(f"<li>{item_content}</li>")
                i += 1
            processed_lines.append(f"<ul>{''.join(list_items)}</ul>")
            continue

        # Ordered list: 1. item, 2. item
        if re.match(r"^\s*\d+\.\s+", line):
            list_items = []
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                item_content = re.sub(r"^\s*\d+\.\s+", "", lines[i])
                list_items.append(f"<li>{item_content}</li>")
                i += 1
            processed_lines.append(f"<ol>{''.join(list_items)}</ol>")
            continue

        # Regular line
        processed_lines.append(line)
        i += 1

    result = "\n".join(processed_lines)

    # Inline formatting (applied after block processing)

    # Links: [text](url) -> <a href="url">text</a>
    result = re.sub(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
        r'<a href="\2">\1</a>',
        result,
    )

    # Images: ![alt](url) -> link fallback for email
    result = re.sub(
        r"!\[([^\]]*)\]\((https?://[^)\s]+)\)",
        r'<a href="\2">[image: \1]</a>',
        result,
    )

    # Strikethrough: ~~text~~ -> <del>text</del>
    result = re.sub(r"~~([^~]+)~~", r"<del>\1</del>", result)

    # Bold: **text** or __text__ -> <strong>text</strong>
    result = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", result)
    result = re.sub(r"__([^_]+)__", r"<strong>\1</strong>", result)

    # Italic: *text* or _text_ -> <em>text</em>
    result = re.sub(r"(?<![*])\*([^*]+)\*(?![*])", r"<em>\1</em>", result)
    result = re.sub(r"(?<![_\w])_([^_]+)_(?![_\w])", r"<em>\1</em>", result)

    # Inline code: `code` -> <code>code</code>
    result = re.sub(r"`([^`]+)`", r"<code>\1</code>", result)

    # Convert remaining newlines to <br> for email
    result = result.replace("\n", "<br>")

    return result


# Pre-compiled regex patterns for TL;DR extraction
# Block pattern: TL;DR as a heading with content on following lines
_TLDR_BLOCK_PATTERN = re.compile(
    r"(?:^|\n)(?:>?\s*)?(?:#{1,6}\s*|\*{1,2})?TL;DR(?:\*{1,2})?:?\s*\n+(.*?)(?=\n#{1,6}\s+\S|\n\*{2}[^*]+\*{2}|\Z)",
    re.IGNORECASE | re.DOTALL,
)

# Inline pattern: TL;DR with content on the same line
_TLDR_INLINE_PATTERN = re.compile(
    r"(?:^|\n)(?:>?\s*)?(?:#{1,6}\s*|\*{1,2})?TL;DR(?:\*{1,2})?:?\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)


def get_summary_file(pid: str, preferred_model: str | None = None) -> Path | None:
    """
    Find the summary file for a given paper ID.

    Args:
        pid: Paper ID (may include version like "2301.00001v2")

    Returns:
        Path to the summary file if found, None otherwise
    """
    raw_pid, _ = split_pid_version(pid)
    raw_pid = raw_pid or pid

    # Try new layered structure first: summary_dir/{pid}/{model}.md
    summary_dir = Path(_summary_dir()) / raw_pid

    if summary_dir.exists() and summary_dir.is_dir():
        # Prefer the configured default model unless overridden.
        preferred = (preferred_model or _default_llm_name() or "").strip()
        if preferred:
            try:
                preferred_path = summary_dir / f"{model_cache_key(preferred)}.md"
                if preferred_path.is_file():
                    return preferred_path
            except Exception:
                pass

        # Fallback: find any .md file (sorted for stability)
        md_files = sorted(summary_dir.glob("*.md"))
        if md_files:
            return md_files[0]

    # Try legacy flat structure: summary_dir/{pid}.md
    legacy_file = Path(_summary_dir()) / f"{raw_pid}.md"
    if legacy_file.exists():
        return legacy_file

    return None


def extract_tldr_from_content(content: str, max_length: int = 500) -> str:
    """
    Extract TL;DR from summary content.

    Args:
        content: Summary markdown content
        max_length: Maximum length of TL;DR text

    Returns:
        TL;DR text, or empty string if not found
    """
    if not content:
        return ""

    # Try block pattern first (TL;DR as heading with content on following lines)
    match = _TLDR_BLOCK_PATTERN.search(content)

    # If not found, try inline pattern (TL;DR with content on same line)
    if not match:
        match = _TLDR_INLINE_PATTERN.search(content)

    if not match:
        return ""

    tldr = match.group(1).strip()

    # Clean up blockquote markers if present
    tldr = re.sub(r"^>\s?", "", tldr, flags=re.MULTILINE).strip()

    # Take first paragraph only
    first_para = tldr.split("\n\n")[0].strip()

    # Truncate if too long
    if len(first_para) > max_length:
        first_para = first_para[: max_length - 3] + "..."

    return first_para


def read_tldr_from_summary_file(pid: str) -> str:
    """
    Read and extract TL;DR from the summary file for a given paper.

    Args:
        pid: Paper ID

    Returns:
        TL;DR content string, or empty string if not found
    """
    summary_file = get_summary_file(pid)

    if not summary_file:
        return ""

    try:
        size_b = None
        try:
            size_b = int(summary_file.stat().st_size)
        except Exception:
            size_b = None

        t0 = time.time()
        content = summary_file.read_text(encoding="utf-8")
        dt = time.time() - t0
        if dt >= 0.1 or (size_b is not None and size_b >= 2 * 1024 * 1024):
            logger.trace(
                f"[BLOCKING] read_tldr_from_summary_file: pid={pid}, size={size_b}, read_time={dt:.2f}s, file={summary_file}"
            )
        return extract_tldr_from_content(content)
    except Exception as e:
        logger.debug(f"Failed to extract TL;DR for {pid}: {e}")
        return ""
