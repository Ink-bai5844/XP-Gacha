from pathlib import Path


PREFIX = "NH"
TARGET_DIR = Path(__file__).resolve().parents[1] / "b64_cache"


def rename_txt_files() -> None:
    if not TARGET_DIR.exists():
        raise FileNotFoundError(f"Directory not found: {TARGET_DIR}")

    renamed_count = 0

    for file_path in TARGET_DIR.iterdir():
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() != ".txt":
            continue

        if file_path.name.startswith(PREFIX):
            continue

        if not file_path.stem[:1].isdigit():
            continue

        new_path = file_path.with_name(f"{PREFIX}{file_path.name}")
        if new_path.exists():
            print(f"Skip, target already exists: {new_path.name}")
            continue

        file_path.rename(new_path)
        renamed_count += 1
        print(f"Renamed: {file_path.name} -> {new_path.name}")

    print(f"Done. Renamed {renamed_count} txt file(s).")


if __name__ == "__main__":
    rename_txt_files()
