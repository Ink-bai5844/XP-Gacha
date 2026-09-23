from __future__ import annotations

import hashlib
import importlib.util
import io
import os
import stat
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock


BUILDER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_portable_release_macos.py"
SPEC = importlib.util.spec_from_file_location("macos_portable_builder_under_test", BUILDER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load {BUILDER_PATH}")
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def make_tar(path: Path, entries: list[tuple[tarfile.TarInfo, bytes]]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for member, content in entries:
            archive.addfile(member, io.BytesIO(content) if member.isfile() else None)


def regular_member(name: str, content: bytes = b"runtime", mode: int = 0o644) -> tuple[tarfile.TarInfo, bytes]:
    member = tarfile.TarInfo(name)
    member.size = len(content)
    member.mode = mode
    return member, content


def link_member(name: str, target: str, *, hard: bool = False) -> tuple[tarfile.TarInfo, bytes]:
    member = tarfile.TarInfo(name)
    member.type = tarfile.LNKTYPE if hard else tarfile.SYMTYPE
    member.linkname = target
    return member, b""


class RuntimeArchiveTests(unittest.TestCase):
    def test_verified_archives_accept_matching_hashes_and_reject_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "python.tar.gz"
            archive.write_bytes(b"downloaded runtime")
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            self.assertEqual(builder.sha256_file(archive), digest)
            builder.verify_archive(archive, "sha256", digest)
            builder.verify_archive(archive, "md5", hashlib.md5(archive.read_bytes()).hexdigest())
            archive.write_bytes(b"tampered runtime")
            with self.assertRaises(ValueError):
                builder.verify_archive(archive, "sha256", digest)

    def test_runtime_root_is_flattened_and_internal_relative_links_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "python.tar.gz"
            make_tar(archive, [
                regular_member("python/bin/python3.13", b"#!/bin/sh\n", 0o755),
                regular_member("python/lib/libpython.dylib"),
                link_member("python/bin/python3", "python3.13"),
                link_member("python/bin/libpython.dylib", "../lib/libpython.dylib"),
            ])
            destination = root / "runtime" / "python"
            builder.extract_runtime(archive, destination, "python")

            self.assertEqual((destination / "bin/python3").read_bytes(), b"#!/bin/sh\n")
            self.assertEqual(os.readlink(destination / "bin/python3"), "python3.13")
            self.assertEqual(os.readlink(destination / "bin/libpython.dylib"), "../lib/libpython.dylib")
            self.assertTrue((destination / "bin/python3.13").stat().st_mode & stat.S_IXUSR)
            self.assertFalse((destination / "python").exists())

    def test_unsafe_runtime_members_are_rejected_without_writing_outside_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside.txt"
            device = tarfile.TarInfo("python/device")
            device.type = tarfile.CHRTYPE
            cases = {
                "parent traversal": [regular_member("python/../../outside.txt")],
                "absolute member": [regular_member(str(outside))],
                "wrong root": [regular_member("unexpected/bin/python3")],
                "absolute symlink": [link_member("python/bin/python3", str(outside))],
                "escaping symlink": [link_member("python/bin/python3", "../../../outside.txt")],
                "escaping hardlink": [link_member("python/bin/python3", "../outside.txt", hard=True)],
                "device": [(device, b"")],
                "write through symlink": [
                    link_member("python/escape", "../.."),
                    regular_member("python/escape/outside.txt"),
                ],
            }
            for index, (name, entries) in enumerate(cases.items()):
                with self.subTest(name=name):
                    archive = root / f"unsafe-{index}.tar.gz"
                    make_tar(archive, entries)
                    with self.assertRaises((ValueError, RuntimeError, tarfile.TarError)):
                        builder.extract_runtime(archive, root / f"extract-{index}", "python")
                    self.assertFalse(outside.exists())

    def test_repeated_directory_headers_are_allowed_but_duplicate_files_and_links_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = tarfile.TarInfo("python/bin")
            directory.type = tarfile.DIRTYPE
            archive = root / "repeated-directories.tar.gz"
            make_tar(archive, [
                (directory, b""),
                regular_member("python/bin/python3.12", b"runtime", 0o755),
                (directory, b""),
                link_member("python/bin/python3", "python3.12"),
                (directory, b""),
            ])
            destination = root / "valid"
            builder.extract_runtime(archive, destination, "python")
            self.assertEqual((destination / "bin/python3").read_bytes(), b"runtime")
            self.assertEqual(os.readlink(destination / "bin/python3"), "python3.12")

            cases = {
                "file": [regular_member("python/file"), regular_member("python/file", b"replacement")],
                "symlink": [link_member("python/link", "file"), link_member("python/link", "other")],
                "hardlink": [link_member("python/link", "python/file", hard=True)] * 2,
                "directory then file": [(directory, b""), regular_member("python/bin")],
                "file then directory": [regular_member("python/bin"), (directory, b"")],
            }
            for index, (name, entries) in enumerate(cases.items()):
                with self.subTest(duplicate=name):
                    archive = root / f"duplicate-{index}.tar.gz"
                    make_tar(archive, entries)
                    rejected_destination = root / f"rejected-{index}"
                    with self.assertRaisesRegex(ValueError, "duplicate"):
                        builder.extract_runtime(archive, rejected_destination, "python")
                    self.assertFalse(rejected_destination.exists())


class ApplicationCopyTests(unittest.TestCase):
    def make_project(self, root: Path) -> list[str]:
        files = {
            "app.py", "config.py", "config_docker.py", "config_empty.py", "data_pipeline.py",
            "launcher.py", "ui_data_processing.py", "utils_chat.py", "utils_charts.py",
            "utils_core.py", "utils_cv.py", "utils_history.py", "utils_nlp.py",
            "utils_online_cover.py", "requirements.txt", "README.md", "LICENSE", "VERSION",
            "server/main.py", "server/__init__.py", "server/job_tasks.py", "server/local_folders.py",
            "data_get/collector.py",
            "data_processing/addname.py", "tools/clean_cache.py", "Integration/app.py",
            "UI-imgs/logo.png", "web/dist/index.html", "web/dist/assets/app.js",
            "dictionaries/STOP_TAGS.txt", "dictionaries/SEMANTIC_MAP.json",
            "dictionaries/TITLE_STOP_WORDS.txt", "dictionaries/TITLE_SEMANTIC_MAP.json",
            "portable/portable_launcher.py", "portable/portable-settings.env",
            "portable/THIRD_PARTY_NOTICES.md", "portable/macos/Start XP-Gacha.command",
            "portable/macos/Stop XP-Gacha.command", "portable/macos/Check XP-Gacha.command",
            "portable/macos/Open XP-Gacha Folder.command", "portable/macos/README_macOS.md",
            "portable/macos/THIRD_PARTY_NOTICES.md", "portable/macos/runtime-manifest.json",
        }
        for relative in files:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"fixture: {relative}\n", encoding="utf-8")
        return sorted(files)

    def test_release_includes_application_and_templates_but_excludes_private_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            release = root / "release"
            tracked = self.make_project(project)
            private_files = {
                ".env", "database/private.sql", "datacache/preprocessed_df.pkl",
                "data_processing/export.csv", "data_get/failed.jsonl",
                "server/__pycache__/main.cpython-313.pyc", "server/.env",
                "server/untracked_debug.py", "dictionaries/personal.json",
                "portable/macos/.DS_Store",
            }
            for relative in private_files:
                path = project / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("PRIVATE DATA", encoding="utf-8")
            # Even accidentally tracked data exports and bytecode must stay out.
            tracked.extend(relative for relative in private_files if relative != "server/untracked_debug.py")
            with mock.patch.object(builder, "tracked_files", return_value=tuple(tracked)):
                builder.copy_application(project, release)

            for relative in (
                "server/main.py", "data_processing/addname.py", "UI-imgs/logo.png",
                "web/dist/index.html", "web/dist/assets/app.js", "dictionaries/SEMANTIC_MAP.json",
                "portable_launcher.py", "portable-settings.env", "Start XP-Gacha.command",
            ):
                with self.subTest(included=relative):
                    self.assertTrue((release / relative).is_file())
            for relative in private_files:
                with self.subTest(excluded=relative):
                    self.assertFalse((release / relative).exists())
            self.assertFalse((release / ".DS_Store").exists())
            self.assertTrue((release / "Start XP-Gacha.command").stat().st_mode & stat.S_IXUSR)

    def test_release_copy_fails_without_a_reliable_tracked_file_list(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_project(root / "project")
            with mock.patch.object(builder, "tracked_files", side_effect=RuntimeError("git unavailable")):
                with self.assertRaisesRegex(RuntimeError, "git unavailable"):
                    builder.copy_application(root / "project", root / "release")

    def test_release_copy_rejects_api_keys_in_the_portable_settings_template(self) -> None:
        for key, value in (("ONLINE_API_KEY", "private-key"), ("LM_STUDIO_API_KEY", '"private-local-key"')):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                project = root / "project"
                tracked = self.make_project(project)
                (project / "portable/portable-settings.env").write_text(
                    f"# Optional service settings\n{key} = {value}\n", encoding="utf-8"
                )
                release = root / "release"
                with mock.patch.object(builder, "tracked_files", return_value=tuple(tracked)):
                    with self.assertRaisesRegex(ValueError, "API Key"):
                        builder.copy_application(project, release)
                self.assertFalse((release / "portable-settings.env").exists())


class ReleaseArtifactTests(unittest.TestCase):
    def test_failed_checksum_publication_rolls_back_this_build_and_preserves_other_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = root / "stage"
            output = root / "output"
            stage.mkdir()
            output.mkdir()
            staged_release = stage / "release"
            staged_release.mkdir()
            (staged_release / "app.py").write_bytes(b"application")
            staged_archive = stage / "release.tar.gz"
            staged_archive.write_bytes(b"archive")
            staged_checksum = stage / "release.tar.gz.sha256"
            staged_checksum.write_bytes(b"checksum")
            existing_artifact = output / "previous-release.tar.gz"
            existing_artifact.write_bytes(b"previous release")
            pairs = [(source, output / source.name) for source in (staged_release, staged_archive, staged_checksum)]
            real_link = os.link

            def publish_link(source: Path, destination: Path) -> None:
                if destination == pairs[2][1]:
                    # A different build can create this path after the existence
                    # check. Its file must survive our rollback unchanged.
                    destination.write_bytes(b"another build's checksum")
                    raise FileExistsError("checksum publication failed")
                real_link(source, destination)

            with mock.patch.object(builder.os, "link", side_effect=publish_link):
                with self.assertRaisesRegex(OSError, "checksum publication failed"):
                    builder.publish_artifacts(pairs)

            self.assertEqual((staged_release / "app.py").read_bytes(), b"application")
            self.assertEqual(staged_archive.read_bytes(), b"archive")
            self.assertEqual(staged_checksum.read_bytes(), b"checksum")
            self.assertFalse(pairs[0][1].exists())
            self.assertFalse(pairs[1][1].exists())
            self.assertEqual(pairs[2][1].read_bytes(), b"another build's checksum")
            self.assertEqual(existing_artifact.read_bytes(), b"previous release")

    def test_checksums_cover_file_bytes_and_link_targets_and_are_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bin").mkdir()
            (root / "bin/python3.13").write_bytes(b"runtime bytes")
            (root / "bin/python3").symlink_to("python3.13")
            builder.write_checksums(root)
            manifest = root / "SHA256SUMS.txt"
            first = manifest.read_text(encoding="utf-8")
            checksums = dict(line.split("  ", 1)[::-1] for line in first.splitlines())
            self.assertEqual(checksums["bin/python3.13"], hashlib.sha256(b"runtime bytes").hexdigest())
            self.assertEqual(checksums["bin/python3"], hashlib.sha256(b"python3.13").hexdigest())
            self.assertNotIn("SHA256SUMS.txt", checksums)
            builder.write_checksums(root)
            self.assertEqual(manifest.read_text(encoding="utf-8"), first)

    def test_release_archive_preserves_executable_permissions_and_internal_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = root / "XP-Gacha macOS"
            executable = release / "runtime/python/bin/python3.13"
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b"#!/bin/sh\n")
            executable.chmod(0o755)
            (executable.parent / "python3").symlink_to("python3.13")
            archive = root / "XP-Gacha.tar.gz"
            builder.create_archive(release, archive)

            with tarfile.open(archive, "r:gz") as bundled:
                members = {member.name: member for member in bundled.getmembers()}
                prefix = release.name + "/runtime/python/bin/"
                self.assertEqual(stat.S_IMODE(members[prefix + "python3.13"].mode), 0o755)
                self.assertTrue(members[prefix + "python3"].issym())
                self.assertEqual(members[prefix + "python3"].linkname, "python3.13")

    def test_release_archive_rejects_links_outside_the_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = root / "release"
            release.mkdir()
            (root / "private.txt").write_text("private", encoding="utf-8")
            (release / "outside").symlink_to("../private.txt")
            archive = root / "release.tar.gz"
            with self.assertRaises(ValueError):
                builder.create_archive(release, archive)
            self.assertFalse(archive.exists())

    def test_reset_removes_verification_data_and_keeps_runtime_and_dictionaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in (
                "mysql/data/ibdata1", "config/portable.json", "logs/server.log", "run/api.pid",
                "data/recommendation_history.json", "datacache/preprocessed_df.pkl",
                "tmp/upload.png", "runtime/python/bin/python3", "dictionaries/STOP_TAGS.txt",
                "server/main.py",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture", encoding="utf-8")
            builder.reset_runtime_data(root)
            for relative in ("mysql", "config", "logs", "run", "data", "datacache", "tmp"):
                with self.subTest(cleared=relative):
                    self.assertTrue((root / relative).is_dir())
                    self.assertFalse(any(path.is_file() for path in (root / relative).rglob("*")))
            for relative in ("runtime/python/bin/python3", "dictionaries/STOP_TAGS.txt", "server/main.py"):
                self.assertEqual((root / relative).read_text(encoding="utf-8"), "fixture")


class BuildPreflightTests(unittest.TestCase):
    def test_dependency_install_upgrades_bundled_pip_before_resumable_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "release with spaces 中文"
            root.mkdir()
            python = str(root / "runtime/python/bin/python3")
            with mock.patch.object(builder, "run", side_effect=[
                "3.12.14 arm64", "", "", "", "fastapi==0.135.0",
            ]) as run:
                builder.install_dependencies(root, "3.12.14")

            calls = run.call_args_list
            self.assertEqual(len(calls), 5)
            for call in calls:
                self.assertEqual(call.args[0][0], python)
                self.assertEqual(call.kwargs["cwd"], root)
            bootstrap, dependencies = calls[1].args[0], calls[2].args[0]
            for command in (bootstrap, dependencies):
                self.assertEqual(command[:6], [python, "-I", "-m", "pip", "--isolated", "install"])
                self.assertIn("--only-binary=:all:", command)
                self.assertIn("--no-compile", command)
                self.assertIn("--disable-pip-version-check", command)
                self.assertEqual(command[command.index("--index-url") + 1], "https://pypi.org/simple")
                self.assertEqual(command[command.index("--timeout") + 1], "60")
                self.assertEqual(command[command.index("--retries") + 1], "3")
            self.assertEqual(bootstrap[-2:], ["--upgrade", "pip==26.2.1"])
            self.assertNotIn("--resume-retries", bootstrap)
            self.assertEqual(dependencies[dependencies.index("--resume-retries") + 1], "10")
            self.assertEqual(dependencies[-2:], ["-r", str(root / "requirements.txt")])
            self.assertEqual(calls[3].args[0], [python, "-I", "-m", "pip", "--isolated", "check"])
            self.assertEqual(calls[4].args[0], [python, "-I", "-m", "pip", "--isolated", "freeze"])
            self.assertTrue(calls[4].kwargs["capture"])
            self.assertEqual((root / "requirements-lock.txt").read_text(encoding="utf-8"), "fastapi==0.135.0\n")

    def test_interrupted_build_subprocess_is_stopped_and_waited_for_before_reraising(self) -> None:
        for timed_out in (False, True):
            with self.subTest(termination_timed_out=timed_out):
                process = mock.MagicMock()
                process.__enter__.return_value = process
                process.pid = 12345
                process.poll.return_value = None
                process.communicate.side_effect = KeyboardInterrupt
                if timed_out:
                    process.wait.side_effect = [builder.subprocess.TimeoutExpired("verify", 60), -9]
                else:
                    process.wait.return_value = -15
                events = mock.Mock()
                events.attach_mock(process.wait, "wait")
                with (
                    mock.patch.object(builder.subprocess, "Popen", return_value=process) as popen,
                    mock.patch.object(builder.os, "killpg", create=True) as killpg,
                ):
                    events.attach_mock(killpg, "killpg")
                    with self.assertRaises(KeyboardInterrupt):
                        builder.run(["bundled-python", "portable_launcher.py", "verify"], cwd=Path.cwd())

                self.assertTrue(popen.call_args.kwargs["start_new_session"])
                expected = [
                    mock.call.killpg(12345, builder.signal.SIGTERM),
                    mock.call.wait(timeout=60),
                ]
                if timed_out:
                    expected.extend([mock.call.killpg(12345, builder.signal.SIGKILL), mock.call.wait()])
                self.assertEqual(events.mock_calls, expected)

    def test_only_non_root_native_apple_silicon_on_supported_macos_can_build(self) -> None:
        cases = [
            ("supported", "darwin", "Darwin", "arm64", "15.0", 501, True),
            ("newer macOS", "darwin", "Darwin", "arm64", "26.0.1", 501, True),
            ("Linux", "linux", "Linux", "aarch64", "", 501, False),
            ("Intel or Rosetta", "darwin", "Darwin", "x86_64", "15.0", 501, False),
            ("older macOS", "darwin", "Darwin", "arm64", "14.7", 501, False),
            ("root", "darwin", "Darwin", "arm64", "15.0", 0, False),
        ]
        for name, sys_platform, system, machine, version, uid, allowed in cases:
            with (
                self.subTest(name=name),
                mock.patch.object(builder.sys, "platform", sys_platform),
                mock.patch.object(builder.platform, "system", return_value=system),
                mock.patch.object(builder.platform, "machine", return_value=machine),
                mock.patch.object(builder.platform, "mac_ver", return_value=(version, ("", "", ""), "")),
                mock.patch.object(builder.os, "geteuid", return_value=uid, create=True),
            ):
                if allowed:
                    builder.validate_host()
                else:
                    with self.assertRaises(RuntimeError):
                        builder.validate_host()

    def test_existing_release_database_or_artifacts_are_never_overwritten(self) -> None:
        release_name = "XP-Gacha-v0.2.8-portable-macos-arm64"
        for existing in (release_name, release_name + ".tar.gz", release_name + ".tar.gz.sha256"):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                project = root / "project"
                (project / "server").mkdir(parents=True)
                (project / "server/__init__.py").write_text('__version__ = "0.2.8"\n', encoding="utf-8")
                manifest = project / "portable/macos/runtime-manifest.json"
                manifest.parent.mkdir(parents=True)
                manifest.write_text("{}", encoding="utf-8")
                output_root = root / "releases"
                output_root.mkdir()
                existing_path = output_root / existing
                if existing == release_name:
                    protected_file = existing_path / "mysql/data/ibdata1"
                    protected_file.parent.mkdir(parents=True)
                else:
                    protected_file = existing_path
                protected_file.write_bytes(b"existing user data")
                args = builder.parser().parse_args(["--output-root", str(output_root)])
                with (
                    mock.patch.object(builder, "PROJECT_ROOT", project),
                    mock.patch.object(builder, "validate_host"),
                    mock.patch.object(builder, "download_runtime") as download,
                    mock.patch.object(builder, "run") as run,
                    mock.patch.object(builder, "reset_runtime_data") as reset,
                ):
                    with self.assertRaises(FileExistsError):
                        builder.build(args)
                download.assert_not_called()
                run.assert_not_called()
                reset.assert_not_called()
                self.assertEqual(protected_file.read_bytes(), b"existing user data")


if __name__ == "__main__":
    unittest.main()
