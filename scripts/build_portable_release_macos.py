"""Build a relocatable Apple Silicon release; build tools are never shipped.

Only the bundled interpreter receives pip installs. Each build uses a new
temporary directory; existing releases and user databases are never replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import posixpath
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    "app.py", "config.py", "config_docker.py", "config_empty.py", "data_pipeline.py",
    "launcher.py", "ui_data_processing.py", "utils_chat.py", "utils_charts.py",
    "utils_core.py", "utils_cv.py", "utils_history.py", "utils_nlp.py",
    "utils_online_cover.py", "requirements.txt", "README.md", "LICENSE",
}
PROGRAM_DIRECTORIES = {"server", "data_get", "data_processing", "tools", "Integration", "UI-imgs"}
DICTIONARIES = {"STOP_TAGS.txt", "SEMANTIC_MAP.json", "TITLE_STOP_WORDS.txt", "TITLE_SEMANTIC_MAP.json"}
DATA_DIRECTORIES = (
    "data", "datacache", "b64_cache", "b64_tmp", "localimgtmp", "onlineimgtmp",
    "library", "manga_vectors", "models", "mysql", "config", "run", "logs", "tmp", "updates",
)
COMMANDS = ("Start XP-Gacha.command", "Stop XP-Gacha.command", "Check XP-Gacha.command", "Open XP-Gacha Folder.command")

# This builder only executes on macOS, but its preflight logic is unit-tested on
# other hosts.  Keep the POSIX signal numbers available there without mutating
# the platform ``signal`` module; on macOS these resolve to the native enums.
PROCESS_GROUP_TERM_SIGNAL = getattr(signal, "SIGTERM", 15)
PROCESS_GROUP_KILL_SIGNAL = getattr(signal, "SIGKILL", 9)


def log(message: str) -> None:
    print(f"[macos-build] {message}", flush=True)


def sha256_file(path: Path) -> str:
    return file_digest(path, "sha256")


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, algorithm: str, expected: str) -> None:
    if file_digest(path, algorithm) != expected.lower():
        raise ValueError(f"{algorithm} checksum mismatch: {path}")


def validate_host() -> None:
    if sys.platform != "darwin" or platform.system() != "Darwin":
        raise RuntimeError("此脚本须在 macOS 上构建；不支持交叉打包。")
    if platform.machine().lower() != "arm64":
        raise RuntimeError("当前发行目标为原生 Apple Silicon arm64；不支持 Intel / Rosetta 构建。")
    if int(platform.mac_ver()[0].split(".")[0] or 0) < 15:
        raise RuntimeError("MySQL 8.4.11 运行时要求 macOS 15 或更新版本。")
    if os.geteuid() == 0:
        raise RuntimeError("请用普通用户构建，不要使用 sudo。")


def clean_environment() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("PYTHON", "PIP_", "DYLD_")) or key in {"VIRTUAL_ENV", "CONDA_PREFIX", "__PYVENV_LAUNCHER__"}:
            env.pop(key, None)
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PIP_CONFIG_FILE=os.devnull)
    return env


def run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None, capture: bool = False) -> str:
    with subprocess.Popen(command, cwd=cwd, env=env or clean_environment(), start_new_session=True,
                          text=True, stdout=subprocess.PIPE if capture else None) as process:
        try:
            output, _ = process.communicate()
        except KeyboardInterrupt:
            # In particular, let the verification supervisor stop MySQL before
            # TemporaryDirectory removes its data and configuration files.
            if process.poll() is None:
                os.killpg(process.pid, PROCESS_GROUP_TERM_SIGNAL)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, PROCESS_GROUP_KILL_SIGNAL)
                    process.wait()
            raise
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command, output)
        return output.strip() if capture else ""


def tracked_files(project_root: Path) -> list[str]:
    output = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=project_root)
    return [name for name in output.decode("utf-8").split("\0") if name]


def copy_application(project_root: Path, release_root: Path) -> dict[PurePosixPath, int]:
    release_root.mkdir(parents=True, exist_ok=True)
    # Data-producing code directories also contain local exports. Copy only
    # Git-visible program assets (including new, unignored source files), never
    # an entire working directory recursively.
    for name in tracked_files(project_root):
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid source path: {name}")
        selected = name in ROOT_FILES
        if relative.parts[0] in PROGRAM_DIRECTORIES:
            selected = relative.suffix.lower() in {".py", ".ps1", ".md", ".png", ".jpg", ".svg"}
            selected = selected and not any(part.startswith(".") or part == "__pycache__" for part in relative.parts)
        if not selected:
            continue
        source = project_root / name
        if source.is_symlink():
            raise ValueError(f"Program source must not be a symlink: {name}")
        destination = release_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for required in (*ROOT_FILES, "server/main.py", "server/job_tasks.py", "server/local_folders.py"):
        if not (release_root / required).is_file():
            raise RuntimeError(f"Missing application file: {required}; check source files and Git ignore rules.")
    dist = project_root / "web" / "dist"
    if not (dist / "index.html").is_file():
        raise RuntimeError("web/dist/index.html 缺失，请先构建前端。")
    for source in dist.rglob("*"):
        if source.is_symlink():
            raise ValueError(f"Frontend build must not contain symlinks: {source}")
    shutil.copytree(dist, release_root / "web" / "dist", ignore=shutil.ignore_patterns(".DS_Store"))
    dictionary_root = release_root / "dictionaries"
    dictionary_root.mkdir()
    for name in DICTIONARIES:
        shutil.copy2(project_root / "dictionaries" / name, dictionary_root / name)
    for name in ("portable_launcher.py", "portable-settings.env"):
        if name == "portable-settings.env":
            for line in (project_root / "portable" / name).read_text(encoding="utf-8-sig").splitlines():
                key, separator, value = line.strip().partition("=")
                if separator and key.strip() in {"ONLINE_API_KEY", "LM_STUDIO_API_KEY"} and value.strip().strip("\"'"):
                    raise ValueError("便携设置模板含 API Key，拒绝将个人密钥打入发行包；请清空 portable/portable-settings.env 中的 Key。")
        shutil.copy2(project_root / "portable" / name, release_root / name)
    for name in (*COMMANDS, "README_macOS.md", "THIRD_PARTY_NOTICES.md", "runtime-manifest.json"):
        shutil.copy2(project_root / "portable" / "macos" / name, release_root / name)
    for name in COMMANDS:
        (release_root / name).chmod(0o755)
    # Keep the intended modes separately as well.  Windows and some removable
    # filesystems accept chmod() but cannot represent POSIX execute bits; the
    # final tar writer uses this table instead of silently producing commands
    # that Finder cannot launch after extraction.
    return {PurePosixPath(name): 0o755 for name in COMMANDS}


def _archive_path(value: str, *, description: str) -> PurePosixPath:
    """Parse an archive path without inheriting semantics from the host OS."""
    if not value or "\0" in value or "\\" in value:
        raise ValueError(f"Invalid {description}: {value!r}")
    path = PurePosixPath(value)
    # A drive prefix is not absolute to PurePosixPath, but becomes absolute or
    # drive-relative if later handed to Windows filesystem APIs.
    if path.is_absolute() or re.match(r"^[A-Za-z]:", value):
        raise ValueError(f"Absolute {description}: {value}")
    return path


def extract_runtime(archive: Path, destination: Path, expected_root: str) -> dict[PurePosixPath, int]:
    """Extract only a single verified tree, preserving internal runtime links."""
    if destination.exists():
        raise ValueError(f"Runtime destination already exists: {destination}")
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        entries: dict[str, tarfile.TarInfo] = {}
        for member in members:
            path = _archive_path(member.name, description="archive path")
            if ".." in path.parts or not path.parts or path.parts[0] != expected_root:
                raise ValueError(f"Unsafe archive path: {member.name}")
            name = str(path)
            # Oracle's archive repeats directory headers while assembling its
            # component trees. Repeated directories are harmless; files/links
            # must still be unique to prevent overwrite and link attacks.
            if name in entries and member.isdir() and entries[name].isdir():
                continue
            if name in entries or not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                raise ValueError(f"Unsupported or duplicate archive entry: {name}")
            if member.issym() or member.islnk():
                _archive_path(member.linkname, description="archive link")
                target = posixpath.normpath(posixpath.join(str(path.parent), member.linkname) if member.issym() else member.linkname)
                if target != expected_root and not target.startswith(expected_root + "/"):
                    raise ValueError(f"Archive link escapes runtime: {name}")
            entries[name] = member
        if not entries:
            raise ValueError("Empty runtime archive")
        for name in entries:
            for parent in PurePosixPath(name).parents:
                if str(parent) in entries and not entries[str(parent)].isdir():
                    raise ValueError(f"Archive entry nested under a file or link: {name}")
        modes: dict[PurePosixPath, int] = {}
        destination.mkdir(parents=True)
        for member in members:
            member_path = PurePosixPath(member.name)
            relative = PurePosixPath(*member_path.parts[1:])
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                modes[relative] = member.mode & 0o777
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                input_stream = source.extractfile(member)
                if input_stream is None:
                    raise tarfile.ExtractError(f"Unable to read archive member: {member.name}")
                with input_stream, target.open("wb") as output:
                    shutil.copyfileobj(input_stream, output)
                target.chmod(member.mode & 0o777)
                modes[relative] = member.mode & 0o777
        for member in members:
            target = destination.joinpath(*PurePosixPath(member.name).parts[1:])
            if member.issym():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(member.linkname)
            elif member.islnk():
                link = destination.joinpath(*PurePosixPath(posixpath.normpath(member.linkname)).parts[1:])
                if not link.is_file() or link.is_symlink():
                    raise ValueError(f"Unsupported hard link target: {member.linkname}")
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(link, target)
        # Directory permissions are applied last so a read-only runtime
        # directory cannot prevent extraction of its own children.
        for relative, mode in sorted(modes.items(), key=lambda item: len(item[0].parts), reverse=True):
            target = destination.joinpath(*relative.parts)
            if target.is_dir() and not target.is_symlink():
                target.chmod(mode)
        return modes


def download_runtime(spec: dict, cache_root: Path, supplied: Path | None = None) -> Path:
    if supplied:
        verify_archive(supplied, spec["algorithm"], spec["checksum"])
        return supplied
    cache_root.mkdir(parents=True, exist_ok=True)
    archive = cache_root / spec["archive"]
    if archive.exists():
        verify_archive(archive, spec["algorithm"], spec["checksum"])
        return archive
    temporary = archive.with_suffix(archive.suffix + ".download")
    log(f"Downloading {spec['archive']}")
    try:
        run(["/usr/bin/curl", "--fail", "--location", "--retry", "4", "--connect-timeout", "30",
             "--proto", "=https", "--proto-redir", "=https", "--output", str(temporary), spec["url"]], cwd=cache_root)
        verify_archive(temporary, spec["algorithm"], spec["checksum"])
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    return archive


def reset_runtime_data(root: Path) -> None:
    for name in DATA_DIRECTORIES:
        target = root / name
        if target.is_symlink():
            raise ValueError(f"Refusing to reset a symlink: {target}")
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
    for name in ("data/gallery_info", "data/gallery_info_origin", "data/local_data", "datacache/imports", "models/cache", "mysql/data"):
        (root / name).mkdir(parents=True, exist_ok=True)


def write_checksums(root: Path) -> None:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.name == "SHA256SUMS.txt":
            continue
        if path.is_symlink():
            checksum = hashlib.sha256(os.readlink(path).encode("utf-8")).hexdigest()
        elif path.is_file():
            checksum = sha256_file(path)
        else:
            continue
        lines.append(f"{checksum}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_archive(
    root: Path,
    archive_path: Path,
    mode_overrides: dict[PurePosixPath, int] | None = None,
) -> None:
    if archive_path.exists():
        raise FileExistsError(f"Refusing to overwrite release: {archive_path}")
    for path in root.rglob("*"):
        if path.is_symlink() and not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Release symlink escapes package: {path}")
    normalized_modes: dict[PurePosixPath, int] = {}
    for relative, mode in (mode_overrides or {}).items():
        relative = PurePosixPath(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid archive mode path: {relative}")
        normalized_modes[relative] = mode & 0o777

    def portable_metadata(member: tarfile.TarInfo) -> tarfile.TarInfo:
        member.uid = member.gid = 0
        member.uname = member.gname = ""
        member_path = PurePosixPath(member.name)
        if member_path.parts and member_path.parts[0] == root.name:
            relative = PurePosixPath(*member_path.parts[1:])
            if relative in normalized_modes:
                member.mode = normalized_modes[relative]
        return member
    with tarfile.open(archive_path, "x:gz", dereference=False) as archive:
        archive.add(root, arcname=root.name, filter=portable_metadata)


def publish_artifacts(pairs: list[tuple[Path, Path]]) -> None:
    """Roll back our published paths if a later publication step fails."""
    published: list[tuple[Path, Path]] = []
    try:
        for source, destination in pairs:
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"Refusing to overwrite release: {destination}")
            if source.is_dir():
                source.rename(destination)
            else:
                # Staging lives on the same filesystem. link() atomically
                # refuses an existing file, including one created concurrently.
                os.link(source, destination)
                try:
                    source.unlink()
                except BaseException:
                    destination.unlink()
                    raise
            published.append((source, destination))
    except BaseException:
        for source, destination in reversed(published):
            try:
                destination.rename(source)
            except OSError as rollback_error:
                log(f"发布回滚失败，请检查 {destination}：{rollback_error}")
        raise


def check_frontend_tools() -> None:
    for name, required in (("node", 22), ("pnpm", 11)):
        executable = shutil.which(name)
        if not executable:
            raise RuntimeError(f"构建机缺少 {name}；安装 Node.js 22+ 与 pnpm 11，或用 --skip-frontend-build 复用已构建前端。")
        version = run([executable, "--version"], cwd=PROJECT_ROOT, capture=True)
        major = int(version.lstrip("v").split(".")[0])
        if (name == "node" and major < required) or (name == "pnpm" and major != required):
            raise RuntimeError(f"不支持的构建工具版本：{name} {version}")


def install_dependencies(root: Path, python_version: str) -> None:
    python = root / "runtime" / "python" / "bin" / "python3"
    probe = run([str(python), "-I", "-c", "import platform; print(platform.python_version(), platform.machine())"], cwd=root, capture=True)
    if probe != f"{python_version} arm64":
        raise RuntimeError(f"Unexpected bundled Python: {probe}")
    command = [str(python), "-I", "-m", "pip", "--isolated"]
    install_options = ["--only-binary=:all:", "--no-compile", "--disable-pip-version-check",
                       "--index-url", "https://pypi.org/simple", "--timeout", "60", "--retries", "3"]
    # The runtime's older pip cannot resume large dependency downloads. Upgrade
    # only this package's interpreter before using the verified resume support.
    run([*command, "install", *install_options, "--upgrade", "pip==26.2.1"], cwd=root)
    run([*command, "install", *install_options, "--resume-retries", "10",
         "-r", str(root / "requirements.txt")], cwd=root)
    run([*command, "check"], cwd=root)
    locked = run([*command, "freeze"], cwd=root, capture=True)
    (root / "requirements-lock.txt").write_text(locked + "\n", encoding="utf-8")


def verify_release(root: Path) -> None:
    python = root / "runtime" / "python" / "bin" / "python3"
    env = clean_environment()
    # Protect developer configuration even during doctor imports.
    env.update(XP_GACHA_DATA_ROOT=str(root), XP_GACHA_BASE_DIR=str(root / "library"),
               XP_GACHA_FRONTEND_DIST=str(root / "web" / "dist"),
               XP_GACHA_SETTINGS_FILE=str(root / "portable-settings.env"),
               XP_GACHA_RUNTIME_MODE="portable", ONLINE_COVER_FETCH_ENABLED="0")
    for command in ("doctor", "verify", "verify"):
        run([str(python), "-E", "-s", "-B", str(root / "portable_launcher.py"), command], cwd=root, env=env)


def build(args: argparse.Namespace) -> Path:
    validate_host()
    manifest = json.loads((PROJECT_ROOT / "portable" / "macos" / "runtime-manifest.json").read_text(encoding="utf-8"))
    version_match = re.search(r'__version__\s*=\s*[\"\']([^\"\']+)', (PROJECT_ROOT / "server" / "__init__.py").read_text())
    if not version_match or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-.][A-Za-z0-9]+)*", version_match[1]):
        raise ValueError("Invalid application version")
    version = version_match[1]
    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root = output_root.resolve()
    name = f"XP-Gacha-v{version}-portable-macos-arm64"
    release = output_root / name
    archive = output_root / f"{name}.tar.gz"
    for path in (release, archive, Path(str(archive) + ".sha256")):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"输出已存在，拒绝覆盖；请选择新的 --output-root：{path}")
    if not args.skip_frontend_build:
        check_frontend_tools()
        log("Building frontend")
        run(["pnpm", "--dir", "web", "install", "--frozen-lockfile"], cwd=PROJECT_ROOT)
        run(["pnpm", "--dir", "web", "build"], cwd=PROJECT_ROOT)
    elif not (PROJECT_ROOT / "web" / "dist" / "index.html").is_file():
        raise RuntimeError("--skip-frontend-build 要求现有 web/dist/index.html。")
    cache = PROJECT_ROOT / ".portable-cache" / "macos" / "downloads"
    archives = {key: download_runtime(manifest[key], cache, getattr(args, key + "_archive")) for key in ("python", "mysql")}
    output_root.mkdir(parents=True, exist_ok=True)
    # No existing installation is ever passed to reset_runtime_data.
    with tempfile.TemporaryDirectory(prefix=".xp-gacha-macos-build-", dir=output_root) as temporary:
        stage = Path(temporary)
        root = stage / name
        mode_overrides = copy_application(PROJECT_ROOT, root)
        for key in ("python", "mysql"):
            runtime_modes = extract_runtime(archives[key], root / "runtime" / key, manifest[key]["root"])
            for relative, mode in runtime_modes.items():
                mode_overrides[PurePosixPath("runtime") / key / relative] = mode
        reset_runtime_data(root)
        log("Installing dependencies into bundled Python")
        install_dependencies(root, manifest["python"]["version"])
        # Move the complete runtime before verification: catches absolute build
        # paths, broken links and assumptions about spaces/non-ASCII directories.
        relocated = stage / "relocation check 中文" / name
        relocated.parent.mkdir()
        root.rename(relocated)
        if not args.skip_verification:
            log("Checking relocated dependencies, first start and database restart")
            verify_release(relocated)
        reset_runtime_data(relocated)
        info = {"version": version, "platform": "macos", "architecture": "arm64", "minimumMacOS": manifest["minimumMacOS"],
                "builtAt": datetime.now(timezone.utc).isoformat(), "runtime": manifest,
                "verification": {"doctor": not args.skip_verification, "firstStart": not args.skip_verification,
                                 "restart": not args.skip_verification, "relocatedPath": not args.skip_verification},
                "archiveSha256": {key: sha256_file(path) for key, path in archives.items()},
                "sourceCommit": run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture=True),
                "sourceDirty": bool(run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture=True))}
        (relocated / "BUILD-INFO.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_checksums(relocated)
        staged_archive = stage / archive.name
        create_archive(relocated, staged_archive, mode_overrides)
        archive_hash = sha256_file(staged_archive)
        staged_checksum = stage / (archive.name + ".sha256")
        staged_checksum.write_text(f"{archive_hash}  {archive.name}\n", encoding="utf-8")
        # Recheck after the long build; never intentionally replace a release.
        if any(path.exists() or path.is_symlink() for path in (release, archive, Path(str(archive) + ".sha256"))):
            raise FileExistsError("输出在构建期间已被创建，拒绝覆盖。")
        publish_artifacts([(relocated, release), (staged_archive, archive), (staged_checksum, Path(str(archive) + ".sha256"))])
    log(f"Created {archive}")
    if args.skip_verification:
        log("WARNING: 此包跳过启动验证；请勿标记为已验证发行版。")
    return archive


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="构建 macOS 15+ Apple Silicon 完整便携包")
    result.add_argument("--output-root", type=Path, default=Path("../XP-Gacha-Releases"))
    result.add_argument("--skip-frontend-build", action="store_true")
    result.add_argument("--skip-verification", action="store_true")
    result.add_argument("--python-archive", type=Path, help="预下载的固定版本 Python 归档，仍校验哈希")
    result.add_argument("--mysql-archive", type=Path, help="预下载的固定版本 MySQL 归档，仍校验哈希")
    return result


def main() -> int:
    if sys.version_info < (3, 11):
        print("Build requires Python 3.11+.", file=sys.stderr)
        return 1
    try:
        build(parser().parse_args())
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, tarfile.TarError) as exc:
        print(f"[macos-build] ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("[macos-build] 构建已中止；若中止发生在发布阶段，请检查输出目录的产物状态。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
