"""Offline publication preflight for the FSO-SC source-only release.

No packages, network requests, notebook execution, or Drive access required.
This checker catches common leaks and packaging mistakes; it does not replace
human review of scientific claims, redistribution permissions, or Colab runs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit


RELEASE_ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 10 * 1024 * 1024
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
BLOCKED_DIRS = {"data", "fso_data", "checkpoints", "runs", "downloads", "raw", "examples"}
BLOCKED_SUFFIXES = {
    ".zip", ".tar", ".tgz", ".gz", ".7z", ".rar", ".pth", ".pt", ".ckpt",
    ".safetensors", ".npy", ".npz", ".pkl", ".pickle", ".bpg", ".exe", ".dll",
    ".so", ".dylib", ".partial", ".pdf", ".tif", ".tiff",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg"}
TEXT_SUFFIXES = {".md", ".ipynb", ".py", ".json", ".yml", ".yaml", ".txt", ".ini",
                 ".csv", ".toml", ".cff", ".svg"}
WINDOWS_PATH = re.compile(r"(?:^|[\s(\"'`=])[A-Za-z]:(?:\\+|/(?!/))[^\s\"'`<>|]+")
ACCOUNT_HOME = re.compile(r"/(?:home|Users)/[^/\s\"'`<>]+/")
SECRET_PATTERNS = (
    ("private key", re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----")),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b")),
    ("Google token", re.compile(r"\b(?:AIza[0-9A-Za-z_-]{35}|ya29\.[0-9A-Za-z_-]{25,})\b")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
)
SECRET_FILE = re.compile(r"^(?:\.env(?:\..+)?|credentials.*\.json|token.*\.json|service.account.*\.json)$", re.I)
ALLOWED_MAGIC = re.compile(r"^(?:%pip|!pip|!apt-get)\s+")
MARKDOWN_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
MARKDOWN_REFERENCE = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*(.*)$")
MARKDOWN_INLINE = re.compile(r"(?<!\\)\]\(\s*")
MARKDOWN_CODE = re.compile(r"(`+).*?\1")


def notebook_errors(text, label):
    """Check the fields and Python compilation of a source-only v4 notebook."""
    try:
        book = json.loads(text)
    except json.JSONDecodeError as error:
        return [f"{label}: invalid JSON at line {error.lineno}"]
    if (not isinstance(book, dict) or book.get("nbformat") != 4
            or type(book.get("nbformat_minor")) is not int or book["nbformat_minor"] < 0
            or not isinstance(book.get("metadata"), dict) or not isinstance(book.get("cells"), list)):
        return [f"{label}: invalid notebook v4 structure"]

    issues, used_ids = [], set()
    for position, cell in enumerate(book["cells"], 1):
        where = f"{label}: cell {position}"
        if not isinstance(cell, dict) or not isinstance(cell.get("metadata"), dict):
            issues.append(f"{where}: invalid cell or metadata")
            continue
        kind = cell.get("cell_type")
        if kind not in {"code", "markdown", "raw"}:
            issues.append(f"{where}: invalid cell type")
        identifier = cell.get("id", cell["metadata"].get("id"))
        if book["nbformat_minor"] >= 5 and "id" not in cell:
            issues.append(f"{where}: missing required nbformat cell ID")
        if identifier is not None:
            if not isinstance(identifier, str) or not re.fullmatch(r"[\w-]{1,64}", identifier, re.ASCII):
                issues.append(f"{where}: malformed cell ID")
            elif identifier in used_ids:
                issues.append(f"{where}: duplicate cell ID {identifier}")
            else:
                used_ids.add(identifier)
        source = cell.get("source")
        if isinstance(source, list) and all(isinstance(line, str) for line in source):
            source = "".join(source)
        if not isinstance(source, str):
            issues.append(f"{where}: invalid source")
            continue
        if kind != "code":
            if cell.get("attachments"):
                issues.append(f"{where}: embedded attachment; use a reviewed local plot link")
            continue
        if cell.get("outputs") != [] or "execution_count" not in cell or cell["execution_count"] is not None:
            issues.append(f"{where}: clear outputs and execution count")
        # Only the explicit Colab install lines bypass CPython compilation.
        # Do not strip arbitrary shell, cell magic, or continuation operators.
        source = "\n".join("pass" if ALLOWED_MAGIC.match(line) else line for line in source.splitlines())
        try:
            compile(source, where, "exec", dont_inherit=True)
        except (SyntaxError, ValueError, OverflowError) as error:
            issues.append(f"{where}: Python syntax error: {getattr(error, 'msg', type(error).__name__)}")
    return issues


def _destination(line, start):
    """Read the ordinary or angle-delimited target, including nested parens."""
    if start >= len(line):
        return ""
    if line[start] == "<":
        end = line.find(">", start + 1)
        return line[start + 1:end] if end != -1 else ""
    depth, end = 0, start
    while end < len(line):
        char = line[end]
        if char.isspace():
            break
        if char == "\\" and end + 1 < len(line):
            end += 2
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:
                break
            depth -= 1
        end += 1
    return re.sub(r"\\([()])", r"\1", line[start:end])


def markdown_destinations(text):
    """Extract inline/image links and reference definitions outside code."""
    fence = None
    for line_number, line in enumerate(text.splitlines(), 1):
        marker = MARKDOWN_FENCE.match(line)
        if marker:
            run = marker.group(1)
            if fence is None:
                fence = run
            elif run[0] == fence[0] and len(run) >= len(fence) and not line[marker.end():].strip():
                fence = None
            continue
        if fence is not None:
            continue
        visible = MARKDOWN_CODE.sub("", line)
        definition = MARKDOWN_REFERENCE.match(visible)
        if definition:
            yield line_number, _destination(definition.group(1), 0)
        for link in MARKDOWN_INLINE.finditer(visible):
            yield line_number, _destination(visible, link.end())


def markdown_errors(path, text, root):
    issues = []
    for line_number, target in markdown_destinations(text):
        if not target or target.startswith("#"):
            continue
        where = f"{path.relative_to(root).as_posix()}:{line_number}"
        try:
            parts = urlsplit(target)
        except ValueError:
            issues.append(f"{where}: invalid link destination")
            continue
        if parts.scheme.lower() in {"http", "https", "mailto"} or (
                not parts.scheme and parts.netloc and target.startswith("//")):
            continue
        relative = unquote(parts.path)
        if not relative and not parts.scheme:
            continue
        if parts.scheme or relative.startswith("/") or "\\" in relative or "\x00" in relative:
            issues.append(f"{where}: require a repository-relative link")
            continue
        try:
            actual = Path(os.path.abspath(path.parent / relative))
            actual.resolve().relative_to(root)
            # Windows accepts wrong filename casing, GitHub's links may not.
            current = root
            for component in actual.relative_to(root).parts:
                if not current.is_dir() or component not in {item.name for item in current.iterdir()}:
                    raise FileNotFoundError(relative)
                current /= component
        except ValueError:
            issues.append(f"{where}: link escapes release root: {target}")
        except (OSError, RuntimeError):
            issues.append(f"{where}: missing or case-mismatched relative link: {target}")
    return issues


def _check_manifest(root, seen):
    name = "SOURCE_FILES_SHA256.json"
    path = root / name
    try:
        files = json.loads(path.read_text(encoding="utf-8"))["files"]
    except (OSError, ValueError, KeyError, TypeError):
        return [f"{name}: missing or invalid manifest"]
    if not isinstance(files, dict):
        return [f"{name}: files must be an object"]
    issues = []
    expected = set(seen) - {name}
    for missing in sorted(expected - set(files)):
        issues.append(f"{name}: missing entry {missing}")
    for extra in sorted(set(files) - expected):
        issues.append(f"{name}: unexpected entry {extra}")
    for relative in sorted(expected & set(files)):
        path = seen[relative]
        entry = files[relative]
        if (not isinstance(entry, dict) or entry.get("bytes") != path.stat().st_size
                or entry.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()):
            issues.append(f"{name}: size or SHA-256 mismatch: {relative}")
    return issues


def _json_strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, list):
        for item in node:
            yield from _json_strings(item)
    elif isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _json_strings(value)


def check_release(root, *, check_manifest=False):
    """Return a deterministic list of issues; never modify the package."""
    root = Path(root).resolve()
    if not root.is_dir():
        return ["Release root does not exist"]
    issues, seen = [], {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        label = relative.as_posix()
        if path.is_symlink():
            issues.append(f"{label}: symbolic link is not a self-contained release file")
            continue
        if path.is_dir():
            if path.name.lower() in BLOCKED_DIRS:
                issues.append(f"{label}: private-data or run directory")
            continue
        seen[label] = path
        suffix = path.suffix.lower()
        if suffix in BLOCKED_SUFFIXES or (SECRET_FILE.match(path.name) and path.name != ".env.example"):
            issues.append(f"{label}: excluded archive, model, binary, or credential file")
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            issues.append(f"{label}: exceeds the {MAX_FILE_BYTES // 2**20} MiB source-only size cap")
            continue
        if suffix in IMAGE_SUFFIXES and relative.parts[0] not in {"results", "docs", "assets"}:
            issues.append(f"{label}: image is outside reviewed figures/documentation folders")
        if suffix not in TEXT_SUFFIXES and path.name not in {".gitignore", "LICENSE", "NOTICE"}:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError):
            issues.append(f"{label}: expected UTF-8 text")
            continue
        # Inspect decoded notebook/config strings as well as their JSON escape
        # sequences; avoid confusing a final Python "q:" + serialized newline
        # with a drive letter.
        searchable = text
        if suffix in {".ipynb", ".json"}:
            try:
                searchable = "\n".join(_json_strings(json.loads(text)))
            except json.JSONDecodeError:
                if suffix == ".json":
                    issues.append(f"{label}: invalid JSON")
        if WINDOWS_PATH.search(searchable) or ACCOUNT_HOME.search(searchable):
            issues.append(f"{label}: private-looking absolute workstation path")
        for description, pattern in SECRET_PATTERNS:
            if pattern.search(searchable):
                issues.append(f"{label}: possible {description}; inspect without printing the value")
        if suffix == ".ipynb":
            issues.extend(notebook_errors(text, label))
        elif suffix == ".md":
            issues.extend(markdown_errors(path, text, root))
    for name in ("README.md",):
        if name not in seen:
            issues.append(f"{name}: missing required release entry point")
    if check_manifest:
        issues.extend(_check_manifest(root, seen))
    return issues


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=RELEASE_ROOT)
    parser.add_argument("--manifest", action="store_true", help="also verify the complete source hash inventory")
    args = parser.parse_args(argv)
    issues = check_release(args.root, check_manifest=args.manifest)
    if issues:
        for issue in issues:
            print(issue, file=sys.stderr)
        print(f"Release preflight failed ({len(issues)} issue(s)).", file=sys.stderr)
        return 1
    print("Release preflight passed (offline; notebooks were not run).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
