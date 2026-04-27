from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
PREFIX = "NH"
TARGET_DIR = Path(__file__).resolve().parent / "localimgtmp"


def rename_images() -> None:
    if not TARGET_DIR.exists():
        raise FileNotFoundError(f"Directory not found: {TARGET_DIR}")

    renamed_count = 0

    for file_path in TARGET_DIR.iterdir():
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if file_path.name.startswith(PREFIX):
            continue

        new_path = file_path.with_name(f"{PREFIX}{file_path.name}")
        if new_path.exists():
            print(f"Skip, target already exists: {new_path.name}")
            continue

        file_path.rename(new_path)
        renamed_count += 1
        print(f"Renamed: {file_path.name} -> {new_path.name}")

    print(f"Done. Renamed {renamed_count} image(s).")


if __name__ == "__main__":
    rename_images()
