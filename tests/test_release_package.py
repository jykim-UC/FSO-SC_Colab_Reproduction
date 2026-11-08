"""Standard-library preflight tests; fixtures contain no data or credentials."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from check_release import check_release, markdown_destinations, notebook_errors


def make_notebook(*, source=None, outputs=None, execution_count=None):
    return {
        "nbformat": 4, "nbformat_minor": 5, "metadata": {},
        "cells": [{
            "id": "install", "cell_type": "code", "metadata": {},
            "source": source if source is not None else ["%pip -q install lpips\n", "x = 2\n"],
            "outputs": outputs if outputs is not None else [], "execution_count": execution_count,
        }],
    }


class ReleasePackageTests(unittest.TestCase):
    def test_actual_release_passes_offline_preflight(self):
        self.assertTrue((ROOT / "README.ko.md").is_file(), "Missing the Korean README")
        issues = check_release(ROOT)
        self.assertFalse(issues, "\n".join(issues))

    def test_links_support_fragments_angles_and_badges_but_skip_code(self):
        text = "\n".join([
            "[doc](<notes with spaces.md#results> \"optional title\")",
            "[![badge](https://example.invalid/icon.svg)](notebooks/main.ipynb)",
            "[anchor](#section)",
            "[reference]: ./docs/data.md#setup",
            "`[literal](absent-a.md)`",
            "```text",
            "[literal](absent-b.md)",
            "```",
            "![nested](docs/figure_(test).png)",
        ])
        self.assertEqual([target for _, target in markdown_destinations(text)], [
            "notes with spaces.md#results", "https://example.invalid/icon.svg",
            "notebooks/main.ipynb", "#section", "./docs/data.md#setup",
            "docs/figure_(test).png",
        ])

    def test_notebook_accepts_explicit_colab_install_lines_only(self):
        self.assertEqual(notebook_errors(json.dumps(make_notebook()), "valid.ipynb"), [])
        for source in (["!echo forbidden\n"], ["%unknown magic\n"], ["def invalid(\n"]):
            with self.subTest(source=source):
                errors = notebook_errors(json.dumps(make_notebook(source=source)), "broken.ipynb")
                self.assertTrue(any("Python syntax error" in error for error in errors), errors)

    def test_notebook_rejects_persisted_outputs_attachments_and_duplicate_ids(self):
        book = make_notebook(outputs=[{"output_type": "stream", "text": "sample"}], execution_count=3)
        book["cells"].extend([
            {"id": "install", "cell_type": "markdown", "metadata": {}, "source": ["x"],
             "attachments": {"x": {"text/plain": "example"}}},
            {"cell_type": "raw", "metadata": {}, "source": "x"},
        ])
        errors = notebook_errors(json.dumps(book), "broken.ipynb")
        for fragment in ("clear outputs", "duplicate cell ID", "embedded attachment", "missing required"):
            self.assertTrue(any(fragment in error for error in errors), errors)

    def test_leak_guard_detects_path_secrets_and_forbidden_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("# Example\n", encoding="utf-8")
            windows_path = "C:" + chr(92) + "Users" + chr(92) + "synthetic-owner" + chr(92) + "raw"
            dummy_token = "gh" + "p_" + "Z" * 36
            (root / "code.ipynb").write_text(
                json.dumps(make_notebook(source=[f"PRIVATE = {windows_path!r}\n", f"SECRET = {dummy_token!r}\n"])),
                encoding="utf-8",
            )
            (root / "weights.pth").write_bytes(b"fixture")
            (root / "train.jpg").write_bytes(b"fixture")
            (root / ".env").write_text("not_a_secret=1", encoding="utf-8")
            issues = check_release(root)
            for fragment in ("absolute workstation path", "possible GitHub token", "weights.pth:", "train.jpg:", ".env:"):
                self.assertTrue(any(fragment in issue for issue in issues), issues)
            self.assertNotIn(dummy_token, "\n".join(issues))

    def test_broken_escaping_and_case_mismatched_links_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "guide.md").write_text("# Guide", encoding="utf-8")
            (root / "README.md").write_text(
                "[valid](guide.md#setup)\n"
                "[missing](elsewhere.md)\n"
                "[case](GUIDE.md)\n"
                "[escape](%2E%2E/outside.md)\n"
                "[absolute](/outside.md)\n",
                encoding="utf-8",
            )
            issues = check_release(root)
            self.assertEqual(len(issues), 4, issues)
            self.assertTrue(any("escapes release root" in issue for issue in issues))
            self.assertTrue(any("case-mismatched" in issue for issue in issues))

    def test_manifest_catches_changed_file_and_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            readme = root / "README.md"
            readme.write_bytes(b"# Example\n")
            data = readme.read_bytes()
            manifest = {"files": {"README.md": {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}}}
            (root / "SOURCE_FILES_SHA256.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(check_release(root, check_manifest=True), [])
            readme.write_bytes(b"# Changed\n")
            (root / "extra.txt").write_bytes(b"Unexpected")
            issues = check_release(root, check_manifest=True)
            self.assertTrue(any("mismatch: README.md" in issue for issue in issues), issues)
            self.assertTrue(any("missing entry extra.txt" in issue for issue in issues), issues)


if __name__ == "__main__":
    unittest.main()
