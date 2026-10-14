"""Check privacy ignore rules without creating data or changing the Git index."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class GitignorePrivacyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.git = shutil.which("git")
        if not cls.git or not (PROJECT_ROOT / ".git").exists():
            raise unittest.SkipTest("Privacy ignore checks require a Git checkout and git.")

    def assert_ignore_rules(self, paths: tuple[str, ...], *, ignored: bool) -> None:
        # --no-index also checks tracked paths: a tracked local output must have
        # an ignore rule even though that rule cannot remove it from Git history.
        # Use path strings only; never read real private files or write fixtures.
        result = subprocess.run(
            [
                self.git,
                "-c",
                f"core.excludesFile={os.devnull}",
                "check-ignore",
                "--no-index",
                "--stdin",
                "-z",
            ],
            cwd=PROJECT_ROOT,
            input=("\0".join(paths) + "\0").encode("utf-8"),
            capture_output=True,
            timeout=30,
            check=False,
        )
        self.assertIn(
            result.returncode,
            (0, 1),
            result.stderr.decode("utf-8", errors="replace"),
        )
        matched = set(result.stdout.decode("utf-8").split("\0")) - {""}
        self.assertFalse(matched - set(paths), "Git returned an unexpected path.")
        for path in paths:
            with self.subTest(path=path):
                self.assertEqual(path in matched, ignored)

    def test_credentials_and_local_settings_are_ignored(self) -> None:
        self.assert_ignore_rules(
            (
                ".env",
                ".env.production",
                ".env.local.bak",
                "web/.env.production",
                "nested/.env",
                "nested/settings.env",
                "portable-settings.env",
                ".streamlit/secrets.toml",
                "nested/secrets.toml",
                "config.local.json",
                "nested/credentials.json",
            ),
            ignored=True,
        )

    def test_history_and_preferences_are_ignored(self) -> None:
        self.assert_ignore_rules(
            (
                "datacache/recommendation_history.json",
                "datacache/ui-preferences.json",
                "nested/recommendation_history.json.tmp",
                "nested/ui-preferences.json.bak",
                "nested/llm-settings.json",
                "nested/cookies.txt",
                ".codex/session.json",
            ),
            ignored=True,
        )

    def test_proxy_pool_and_collection_state_are_ignored(self) -> None:
        self.assert_ignore_rules(
            (
                "run/clash-pool/mihomo.yaml",
                "run/clash-pool/proxy-pool.json",
                "run/clash-pool/prepared.json",
                "nested/clash-pool/process.json",
                "nested/mihomo.yaml",
                "nested/proxy-pool.json.bak",
                "nested/subscriptions.yaml",
                "logs/collection/example.state.jsonl",
                "nested/example.state.jsonl",
                "nested/example.csv.updates.jsonl.bak",
                "nested/example.collection.lock",
            ),
            ignored=True,
        )

    def test_portable_runtime_data_is_ignored(self) -> None:
        self.assert_ignore_rules(
            (
                "portable/config/portable.json",
                "portable/run/state.json",
                "portable/mysql/data/ibdata1",
                "portable/logs/app.log",
                "portable/tmp/request.json",
                "portable/data/gallery_info/example.csv",
                "portable/datacache/ui-preferences.json",
                "portable/localimgtmp/example.webp",
                "portable/onlineimgtmp/example.webp",
                "portable/library/example.txt",
            ),
            ignored=True,
        )

    def test_database_exports_and_import_archives_are_ignored(self) -> None:
        self.assert_ignore_rules(
            (
                "mysql/data/ibdata1",
                "data/gallery_info/example.csv",
                "exports/catalog.sql",
                "database_dump.sql",
                "database_backup.sql",
                "manga_gallery.sql",
                "nested/catalog.sql.gz",
                "nested/catalog.dump",
                "nested/catalog.sqlite",
                "nested/catalog.sqlite3-wal",
                "nested/catalog.db",
                "input_data.zip",
                "input_data.zip.sha256",
                "nested/input_data.zip",
                "XP-Gacha-example.zip",
                "XP-Gacha-example.zip.sha256",
            ),
            ignored=True,
        )

    def test_maintenance_outputs_and_backups_are_ignored(self) -> None:
        self.assert_ignore_rules(
            (
                "data_processing/title_translation_results.jsonl",
                "data_processing/title_translation_results.jsonl.bak",
                "data_processing/title_translation_failed_results.jsonl",
                "data_processing/title_translation_failed_results.jsonl.bak",
                "data_processing/title_words_frequency.csv",
                "data_processing/aggregated_tags.txt",
                "data_processing/111.json",
                "data_processing/222.json",
                "tools/error.json",
                "tools/localimgtmp/example.webp",
            ),
            ignored=True,
        )

    def test_runtime_dictionaries_are_private(self) -> None:
        self.assert_ignore_rules(
            (
                "dictionaries/STOP_TAGS.txt",
                "dictionaries/SEMANTIC_MAP.json",
                "dictionaries/TITLE_STOP_WORDS.txt",
                "dictionaries/TITLE_SEMANTIC_MAP.json",
                "dictionaries/SEMANTIC_MAP_old.json",
                "dictionaries/SEMANTIC_MAP.json.backup",
                "dictionaries/private_empty.json",
                "dictionaries/backup/SEMANTIC_MAP.json",
                "portable/dictionaries/SEMANTIC_MAP.json",
            ),
            ignored=True,
        )

    def test_only_the_four_empty_dictionary_templates_are_public(self) -> None:
        names = (
            "STOP_TAGS_empty.txt",
            "SEMANTIC_MAP_empty.json",
            "TITLE_STOP_WORDS_empty.txt",
            "TITLE_SEMANTIC_MAP_empty.json",
        )
        self.assert_ignore_rules(tuple(f"dictionaries/{name}" for name in names), ignored=False)
        for name in names:
            with self.subTest(template=name):
                content = (PROJECT_ROOT / "dictionaries" / name).read_text(encoding="utf-8-sig")
                if name.endswith(".json"):
                    self.assertEqual(json.loads(content), {})
                else:
                    self.assertEqual(content.strip(), "")

    def test_source_public_templates_and_fixtures_remain_visible(self) -> None:
        self.assert_ignore_rules(
            (
                "config.py",
                "config_empty.py",
                "config_docker.py",
                ".env.example",
                "portable/portable-settings.env",
                "portable/portable_launcher.py",
                "portable/macos/runtime-manifest.json",
                "docs/proxy-pool.example.json",
                "data_get/collector.py",
                "data_get/proxy_pool.py",
                "tools/clash_proxy_pool.py",
                "tests/test_gitignore_privacy.py",
                "tests/test_proxy_pool.py",
                "tests/fixtures/catalog.csv",
                "tests/fixtures/sample.json",
                "tests/fixtures/sample.yaml",
                "server/migrations/001_schema.sql",
                "web/package.json",
                "web/pnpm-lock.yaml",
                "web/pnpm-workspace.yaml",
                "docker-compose.yml",
            ),
            ignored=False,
        )


if __name__ == "__main__":
    unittest.main()
