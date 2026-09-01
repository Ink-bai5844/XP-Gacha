from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from data_processing.addname import process_gallery_data


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


class AddNameTitleTranslationTests(unittest.TestCase):
    def test_existing_output_translation_wins_and_input_value_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "origin.csv"
            links = root / "links.txt"
            output = root / "gallery_info" / "catalog_full.csv"

            write_csv(
                source,
                ["ID", "链接", "标题译文", "标题", "未知列", "标签", "作者", "团队", "语言", "页数", "上传日期"],
                [
                    {
                        "ID": " nh1 ",
                        "链接": "https://nhentai.net/g/1/",
                        "标题": "One",
                        "标题译文": "输入旧值",
                        "未知列": "保留, 逗号",
                    },
                    {
                        "ID": "",
                        "链接": "https://nhentai.net/g/2/",
                        "标题": "Two",
                        "标题译文": "输入链接旧值",
                        "未知列": "row-2",
                    },
                    {
                        "ID": "NH3",
                        "链接": "https://nhentai.net/g/3/",
                        "标题": "Three",
                        "标题译文": "仅输入有值",
                        "未知列": "row-3",
                    },
                    {
                        "ID": "NH4",
                        "链接": "https://nhentai.net/g/4/",
                        "标题": "Four",
                        "标题译文": "",
                        "未知列": "row-4",
                    },
                ],
            )
            write_csv(
                output,
                ["ID", "文件名", "链接", "标题", "标题译文"],
                [
                    {
                        "ID": "ＮＨ1",
                        "链接": "https://nhentai.net/g/1/",
                        "标题": "Old one",
                        "标题译文": "数据库回填优先",
                    },
                    {
                        "ID": "",
                        "链接": "https://nhentai.net/g/2",
                        "标题": "Old two",
                        "标题译文": "链接兜底译文",
                    },
                ],
            )
            links.write_text(
                "\n".join(
                    f'HREF="https://nhentai.net/g/{number}/">File {number}</A>'
                    for number in range(1, 5)
                ),
                encoding="utf-8",
            )

            process_gallery_data(source, links, output)

            fieldnames, rows = read_csv(output)
            self.assertEqual(len(rows), 4)
            self.assertEqual(fieldnames.index("标题译文"), fieldnames.index("标题") + 1)
            self.assertEqual(fieldnames[1], "文件名")
            self.assertIn("未知列", fieldnames)
            self.assertEqual(
                [row["标题译文"] for row in rows],
                ["数据库回填优先", "链接兜底译文", "仅输入有值", ""],
            )
            self.assertEqual([row["未知列"] for row in rows], ["保留, 逗号", "row-2", "row-3", "row-4"])
            self.assertEqual([row["文件名"] for row in rows], ["File 1", "File 2", "File 3", "File 4"])
            self.assertTrue(output.read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_missing_translation_column_is_added_and_parent_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "origin.csv"
            links = root / "links.txt"
            output = root / "new" / "nested" / "catalog_full.csv"
            write_csv(
                source,
                ["ID", "链接", "标题", "额外"],
                [{"ID": "NH1", "链接": "https://nhentai.net/g/1/", "标题": "One", "额外": "x"}],
            )
            links.write_text('HREF="https://nhentai.net/g/1/">One.cbz</A>', encoding="utf-8")

            process_gallery_data(source, links, output)

            fieldnames, rows = read_csv(output)
            self.assertEqual(fieldnames.index("标题译文"), fieldnames.index("标题") + 1)
            self.assertEqual(rows[0]["标题译文"], "")
            self.assertEqual(rows[0]["额外"], "x")
            self.assertEqual(rows[0]["文件名"], "One.cbz")

    def test_missing_required_columns_does_not_replace_target(self) -> None:
        for missing_column in ("链接", "标题"):
            with self.subTest(missing_column=missing_column), tempfile.TemporaryDirectory() as temporary_dir:
                root = Path(temporary_dir)
                source = root / "origin.csv"
                links = root / "links.txt"
                output = root / "catalog_full.csv"
                columns = ["ID", "链接", "标题", "额外"]
                columns.remove(missing_column)
                write_csv(source, columns, [{name: "value" for name in columns}])
                links.write_text("", encoding="utf-8")
                original = b"do-not-overwrite"
                output.write_bytes(original)

                with self.assertRaisesRegex(ValueError, f"CSV 缺少必要列: {missing_column}"):
                    process_gallery_data(source, links, output)

                self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
