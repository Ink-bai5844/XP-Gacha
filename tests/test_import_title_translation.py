from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
from sqlalchemy import create_engine, text

from server.modules.imports import DB_COLUMNS, import_dataframe


def gallery_row(
    item_id: str,
    link: str,
    title: str,
    translation: str,
    *,
    filename: str = "sample.cbz",
) -> dict[str, object]:
    return {
        "ID": item_id,
        "链接": link,
        "文件名": filename,
        "标题": title,
        "标题译文": translation,
        "标签": "测试标签",
        "作者": "测试作者",
        "团队": "测试团队",
        "语言": "中文",
        "页数": 24,
        "上传日期": "2026-09-01",
    }


class ImportTitleTranslationTests(unittest.TestCase):
    def test_sqlite_upsert_preserves_empty_translation_updates_nonempty_and_inserts_new_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            database_path = Path(temporary_dir) / "catalogue.db"
            engine = create_engine(f"sqlite+pysqlite:///{database_path.as_posix()}")
            try:
                existing = pd.DataFrame(
                    [
                        gallery_row(
                            "NH100",
                            "https://nhentai.net/g/100/",
                            "Existing NH title",
                            "保留的旧译文",
                            filename="old-nh.cbz",
                        ),
                        gallery_row(
                            "JM200",
                            "https://18comic.vip/album/200/",
                            "Existing JM title",
                            "即将更新的旧译文",
                            filename="old-jm.cbz",
                        ),
                    ],
                    columns=DB_COLUMNS,
                )
                existing.to_sql("gallery_info", engine, if_exists="replace", index=False)

                incoming = pd.DataFrame(
                    [
                        gallery_row(
                            "NH100",
                            "https://nhentai.net/g/100/",
                            "Updated NH title",
                            "",
                            filename="new-nh.cbz",
                        ),
                        gallery_row(
                            "JM200",
                            "https://18comic.vip/album/200/",
                            "Updated JM title",
                            "更新后的译文",
                            filename="new-jm.cbz",
                        ),
                        gallery_row(
                            "NH300",
                            "https://nhentai.net/g/300/",
                            "New NH title",
                            "新增条目的译文",
                            filename="new-entry.cbz",
                        ),
                    ],
                    columns=DB_COLUMNS,
                )

                with (
                    mock.patch("server.modules.imports.get_engine", return_value=engine),
                    mock.patch(
                        "server.modules.imports.migration_lock",
                        return_value=contextlib.nullcontext(),
                    ),
                ):
                    result = import_dataframe(incoming, mode="upsert")

                with engine.connect() as connection:
                    rows = connection.execute(
                        text("SELECT * FROM gallery_info ORDER BY ID")
                    ).mappings().all()

                by_id = {str(row["ID"]): dict(row) for row in rows}
                self.assertEqual(result, {"imported": 3, "total": 3, "mode": "upsert"})
                self.assertEqual(list(rows[0].keys()), DB_COLUMNS)
                self.assertEqual(set(by_id), {"NH100", "JM200", "NH300"})

                self.assertEqual(by_id["NH100"]["标题译文"], "保留的旧译文")
                self.assertEqual(by_id["NH100"]["标题"], "Updated NH title")
                self.assertEqual(by_id["NH100"]["文件名"], "new-nh.cbz")
                self.assertEqual(by_id["NH100"]["链接"], "https://nhentai.net/g/100/")

                self.assertEqual(by_id["JM200"]["标题译文"], "更新后的译文")
                self.assertEqual(by_id["JM200"]["链接"], "https://18comic.vip/album/200/")

                self.assertEqual(by_id["NH300"]["标题译文"], "新增条目的译文")
                self.assertEqual(by_id["NH300"]["链接"], "https://nhentai.net/g/300/")
                self.assertEqual(list(by_id["NH300"].keys()), DB_COLUMNS)
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
