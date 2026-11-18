"""
Repository-wide portability guard.

On Windows the default text encoding is the ANSI code page (e.g. cp1252), not
UTF-8, so every text-mode ``open()`` / ``Path.read_text()`` / ``write_text()``
in the package must pass ``encoding=`` explicitly.  This test parses the source
(``ast``) and reports ``file:line`` of every offending real call.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SOURCES = sorted(
    [*(ROOT / "opticalib").rglob("*.py"), ROOT / "setup_calpy.py"],
    key=lambda p: str(p),
)

# (relative posix path, line) pairs known to be false positives. Keep empty if possible.
WHITELIST: set = set()


def _literal_str(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _open_mode(call: ast.Call):
    """Return ('literal', mode) / ('default', 'r') / ('dynamic', None)."""
    node = None
    if len(call.args) >= 2:
        node = call.args[1]
    for kw in call.keywords:
        if kw.arg == "mode":
            node = kw.value
    if node is None:
        return "default", "r"
    mode = _literal_str(node)
    return ("literal", mode) if mode is not None else ("dynamic", None)


def _has_kw(call: ast.Call, name: str) -> bool:
    return any(kw.arg == name for kw in call.keywords)


def find_missing_encoding(source: str, filename: str = "<src>"):
    """List ``(line, description)`` of text-mode I/O calls lacking ``encoding=``."""
    problems = []
    for node in ast.walk(ast.parse(source, filename=filename)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "open":
            kind, mode = _open_mode(node)
            if kind == "dynamic" or (mode is not None and "b" in mode):
                continue
            if not _has_kw(node, "encoding"):
                problems.append((node.lineno, "open() in text mode without encoding="))
        elif isinstance(func, ast.Attribute) and func.attr in ("read_text", "write_text"):
            if not _has_kw(node, "encoding"):
                problems.append((node.lineno, f".{func.attr}() without encoding="))
    return problems


class TestChecker:
    """The guard itself must detect what it claims to detect."""

    @pytest.mark.parametrize(
        "code, n",
        [
            ("open('f')", 1),
            ("open('f', 'w')", 1),
            ("open('f', mode='a')", 1),
            ("open('f', 'r', errors='replace')", 1),
            ("p.read_text()", 1),
            ("p.write_text('x')", 1),
            ("open('f', encoding='utf-8')", 0),
            ("open('f', 'rb')", 0),
            ("open('f', mode='wb')", 0),
            ("open('f', m)", 0),  # non-literal mode: ignored
            ("p.read_text(encoding='utf-8')", 0),
            ("p.write_text('x', encoding='utf-8')", 0),
            ("x = \"open('f')\"", 0),  # string content, not a call
            ("# open('f')\n", 0),
            ("os.fdopen(fd, 'w')", 0),  # other callables are out of scope
        ],
    )
    def test_detection(self, code, n):
        assert len(find_missing_encoding(code)) == n


def test_sources_found():
    assert len(SOURCES) > 50
    assert (ROOT / "setup_calpy.py") in SOURCES


def test_text_io_always_specifies_encoding():
    offenders = []
    for path in SOURCES:
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for line, what in find_missing_encoding(text, rel):
            if (rel, line) not in WHITELIST:
                offenders.append(f"{rel}:{line}: {what}")
    assert not offenders, "Missing encoding= (breaks on Windows):\n" + "\n".join(offenders)
