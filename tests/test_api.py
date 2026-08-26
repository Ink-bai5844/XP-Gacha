from __future__ import annotations

import atexit
import io
import os
import shutil
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path


TEST_ROOT = Path(tempfile.mkdtemp(prefix="xp-gacha-api-test-"))
atexit.register(lambda: shutil.rmtree(TEST_ROOT, ignore_errors=True))
os.environ["XP_GACHA_DATA_ROOT"] = str(TEST_ROOT)
os.environ["XP_GACHA_BASE_DIR"] = str(TEST_ROOT / "library")
os.environ["DATABASE_URL"] = f"sqlite+pysqlite:///{(TEST_ROOT / 'test.db').as_posix()}"
os.environ["XP_GACHA_ENV"] = "test"
warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient` is deprecated.*")

from fastapi.testclient import TestClient

from config import MAX_DISPLAY
from server.main import app


def build_bundle() -> bytes:
    csv_text = "\n".join(
        [
            "ID,链接,文件名,标题,标题译文,标签,作者,团队,语言,页数,上传日期",
            'NH100,https://nhentai.net/g/100/,Sample A,雨の手紙,雨夜的信,"纯爱,日常",作者甲,团队甲,中文,24,2026-08-01',
            'JM200,https://18comic.vip/album/200/,Sample B,森の猫,森林里的猫,"兽耳,治愈",作者乙,团队乙,中文,36,2026-08-02',
        ]
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("data/gallery_info/catalog.csv", csv_text)
        archive.writestr("dictionaries/STOP_TAGS.txt", "'屏蔽词'\n")
        archive.writestr("dictionaries/SEMANTIC_MAP.json", "{}")
        archive.writestr("dictionaries/TITLE_STOP_WORDS.txt", "'の'\n")
        archive.writestr("dictionaries/TITLE_SEMANTIC_MAP.json", "{}")
    return output.getvalue()


class APISmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_end_to_end_catalogue_flow(self) -> None:
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)

        imported = self.client.post(
            "/api/import/bundle",
            files={"file": ("starter.zip", build_bundle(), "application/zip")},
            data={"mode": "replace", "include_dictionaries": "true"},
        )
        self.assertEqual(imported.status_code, 200, imported.text)
        self.assertEqual(imported.json()["imported"], 2)
        self.assertEqual(len(imported.json()["dictionaries"]), 4)

        query = self.client.post(
            "/api/library/query",
            json={
                "keyword": "雨",
                "keywordRelevance": False,
                "weights": {"tag": 1, "artist": 1, "title": 1, "history": 1},
                "page": 0,
                "pageSize": 5,
            },
        )
        self.assertEqual(query.status_code, 200, query.text)
        self.assertEqual(query.json()["total"], 1)
        self.assertEqual(query.json()["items"][0]["id"], "NH100")

        option_search = self.client.get(
            "/api/meta/options/search",
            params={"kind": "tags", "q": "纯爱", "limit": 80},
        )
        self.assertEqual(option_search.status_code, 200)
        self.assertEqual(option_search.json()["items"], ["纯爱"])
        self.assertEqual(option_search.json()["total"], 1)

        option_page = self.client.get(
            "/api/meta/options/search",
            params={"kind": "tags", "limit": 2, "offset": 2},
        )
        self.assertEqual(option_page.status_code, 200)
        self.assertEqual(option_page.json()["offset"], 2)
        self.assertEqual(len(option_page.json()["items"]), 2)
        self.assertFalse(option_page.json()["hasMore"])

        detail = self.client.get("/api/gallery/NH100")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["titleZh"], "雨夜的信")

        recorded = self.client.post(
            "/api/history",
            json={"itemId": "NH100", "action": "打开网络来源"},
        )
        self.assertEqual(recorded.status_code, 200)
        self.assertEqual(recorded.json()["entries"][0]["itemId"], "NH100")
        history_key = recorded.json()["entries"][0]["key"]
        removed = self.client.request("DELETE", "/api/history", json={"keys": [history_key]})
        self.assertEqual(removed.status_code, 200)
        self.assertEqual(removed.json()["entries"], [])

        chart = self.client.get("/api/charts/global")
        self.assertEqual(chart.status_code, 200)
        self.assertIn("tags", chart.json())

        preferences = self.client.put(
            "/api/preferences",
            json={"columnWidths": {"title": 360, "tooSmall": 1}},
        )
        self.assertEqual(preferences.status_code, 200)
        self.assertEqual(preferences.json()["columnWidths"]["tooSmall"], 40)

        scripts = self.client.get("/api/scripts")
        self.assertEqual(scripts.status_code, 200)
        self.assertEqual(len(scripts.json()["scripts"]), 26)

    def test_meta_exposes_configured_page_size(self) -> None:
        response = self.client.get("/api/meta/options")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["pageSize"], MAX_DISPLAY)
        self.assertLessEqual(len(response.json()["tags"]), 80)
        self.assertLessEqual(len(response.json()["artists"]), 80)
        self.assertLessEqual(len(response.json()["titleWords"]), 80)

    def test_zip_traversal_is_rejected(self) -> None:
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("../escape.csv", "链接\nhttps://example.com")
        response = self.client.post(
            "/api/import/bundle",
            files={"file": ("unsafe.zip", output.getvalue(), "application/zip")},
            data={"mode": "upsert", "include_dictionaries": "true"},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
