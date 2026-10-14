"""Execute the release builder's dictionary helper without running a full build."""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


BUILDER = Path(__file__).resolve().parents[1] / "scripts/build_portable_release.ps1"
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")
TEMPLATES = {
    "STOP_TAGS.txt": "STOP_TAGS_empty.txt",
    "SEMANTIC_MAP.json": "SEMANTIC_MAP_empty.json",
    "TITLE_STOP_WORDS.txt": "TITLE_STOP_WORDS_empty.txt",
    "TITLE_SEMANTIC_MAP.json": "TITLE_SEMANTIC_MAP_empty.json",
}


@unittest.skipUnless(POWERSHELL, "PowerShell is required for Windows builder execution")
class WindowsDictionaryTemplateTests(unittest.TestCase):
    def make_templates(self, project: Path) -> None:
        (project / "dictionaries").mkdir(parents=True)
        for name, template in TEMPLATES.items():
            (project / "dictionaries" / template).write_text(
                "{}\n" if name.endswith(".json") else "", encoding="utf-8"
            )

    def invoke_helper(self, project: Path, destination: Path) -> subprocess.CompletedProcess[str]:
        def literal(value: Path) -> str:
            return "'" + str(value).replace("'", "''") + "'"

        script = f"""
$ErrorActionPreference = 'Stop'
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile({literal(BUILDER)}, [ref]$tokens, [ref]$errors)
if ($errors) {{ throw 'Builder syntax errors' }}
$definition = $ast.Find({{ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Copy-EmptyDictionaryTemplates' }}, $true)
if (-not $definition) {{ throw 'Dictionary helper not found' }}
Invoke-Expression $definition.Extent.Text
$DefaultDictionaryFiles = ConvertFrom-Json '{json.dumps(list(TEMPLATES))}'
Copy-EmptyDictionaryTemplates -SourceRoot {literal(project)} -Destination {literal(destination)}
"""
        encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
        return subprocess.run(
            [POWERSHELL, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
            capture_output=True, text=True, timeout=30,
        )

    def test_template_only_clone_and_private_local_files_both_package_empty_defaults(self) -> None:
        for with_private in (False, True):
            with self.subTest(with_private=with_private), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                project, destination = root / "project", root / "release/dictionaries"
                self.make_templates(project)
                if with_private:
                    for name in TEMPLATES:
                        (project / "dictionaries" / name).write_text("PRIVATE PREFERENCE SENTINEL", encoding="utf-8")
                result = self.invoke_helper(project, destination)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual({path.name for path in destination.iterdir()}, set(TEMPLATES))
                for name in TEMPLATES:
                    self.assertEqual((destination / name).read_bytes(), b"{}\n" if name.endswith(".json") else b"")

    def test_missing_templates_reject_normal_name_fallback(self) -> None:
        for name, template in TEMPLATES.items():
            with self.subTest(template=template), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                project, destination = root / "project", root / "release/dictionaries"
                self.make_templates(project)
                (project / "dictionaries" / template).unlink()
                (project / "dictionaries" / name).write_text("PRIVATE PREFERENCE SENTINEL", encoding="utf-8")
                result = self.invoke_helper(project, destination)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Required empty dictionary template", result.stderr)
                self.assertFalse(destination.exists())

    def test_nonempty_or_invalid_templates_are_rejected(self) -> None:
        for template, content in (
            ("SEMANTIC_MAP_empty.json", '{"private":"preference"}'),
            ("SEMANTIC_MAP_empty.json", "[]"),
            ("SEMANTIC_MAP_empty.json", "null"),
            ("SEMANTIC_MAP_empty.json", "not JSON"),
            ("TITLE_SEMANTIC_MAP_empty.json", '{"private":"preference"}'),
            ("STOP_TAGS_empty.txt", "PRIVATE PREFERENCE SENTINEL"),
            ("TITLE_STOP_WORDS_empty.txt", "PRIVATE PREFERENCE SENTINEL"),
        ):
            with self.subTest(template=template, content=content), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                project, destination = root / "project", root / "release/dictionaries"
                self.make_templates(project)
                (project / "dictionaries" / template).write_text(content, encoding="utf-8")
                result = self.invoke_helper(project, destination)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Empty dictionary template must contain", result.stderr)
                self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
