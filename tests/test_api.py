from __future__ import annotations

import atexit
import io
import json
import os
import shutil
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd


TEST_ROOT = Path(tempfile.mkdtemp(prefix="xp-gacha-api-test-"))
atexit.register(lambda: shutil.rmtree(TEST_ROOT, ignore_errors=True))
os.environ["XP_GACHA_DATA_ROOT"] = str(TEST_ROOT)
os.environ["XP_GACHA_BASE_DIR"] = str(TEST_ROOT / "library")
os.environ["DATABASE_URL"] = f"sqlite+pysqlite:///{(TEST_ROOT / 'test.db').as_posix()}"
os.environ["XP_GACHA_ENV"] = "test"
warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient` is deprecated.*")

from fastapi.testclient import TestClient

from config import MAX_DISPLAY
import server.main as server_main
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
        cls.client = TestClient(app, client=("127.0.0.1", 50000))

    def test_00_empty_catalogue_reports_zero_base_metrics(self) -> None:
        response = self.client.post(
            "/api/library/query",
            json={
                "weights": {"tag": 1, "artist": 1, "title": 1, "history": 1},
                "page": 0,
                "pageSize": 5,
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["total"], 0)
        self.assertEqual(response.json()["metrics"]["items"], 0)
        self.assertIn("数据库尚未导入", " ".join(response.json()["warnings"]))

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

        no_match = self.client.post(
            "/api/library/query",
            json={
                "keyword": "不存在的库存关键词",
                "keywordRelevance": False,
                "weights": {"tag": 1, "artist": 1, "title": 1, "history": 1},
                "page": 0,
                "pageSize": 5,
            },
        )
        self.assertEqual(no_match.status_code, 200, no_match.text)
        self.assertEqual(no_match.json()["total"], 0)
        self.assertEqual(no_match.json()["metrics"]["items"], 2)

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

    def test_health_does_not_request_full_system_status(self) -> None:
        database = {
            "available": True,
            "table_ready": True,
            "row_count": 12,
            "error": None,
        }
        with (
            patch.object(server_main.system, "health_status", return_value={"database": database}) as health,
            patch.object(server_main.system, "status", side_effect=AssertionError("full status must not run")) as status,
        ):
            response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["database"], database)
        health.assert_called_once_with()
        status.assert_not_called()

    def test_system_status_refresh_is_opt_in(self) -> None:
        payload = {"counts": {"csv": 3}}
        with patch.object(server_main.system, "status", return_value=payload) as status:
            normal = self.client.get("/api/system/status")
            refreshed = self.client.get("/api/system/status?refresh=true")

        self.assertEqual(normal.status_code, 200, normal.text)
        self.assertEqual(refreshed.status_code, 200, refreshed.text)
        self.assertEqual(status.call_args_list[0].kwargs, {"refresh": False})
        self.assertEqual(status.call_args_list[1].kwargs, {"refresh": True})

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

    def test_llm_settings_are_local_only_and_never_return_keys(self) -> None:
        settings_file = TEST_ROOT / "llm-settings.env"
        payload = {
            "localApiBase": "http://127.0.0.1:1234/v1",
            "localModel": "local-model",
            "localApiKey": "local-test-secret",
            "clearLocalApiKey": False,
            "onlineApiBase": "https://api.example.com/v1",
            "onlineModel": "online-model",
            "onlineApiKey": "online-test-secret",
            "clearOnlineApiKey": False,
        }
        with patch.dict(os.environ, {"XP_GACHA_SETTINGS_FILE": str(settings_file)}, clear=False):
            missing_marker = self.client.put(
                "/api/chat/settings",
                json=payload,
                headers={"host": "127.0.0.1"},
            )
            self.assertEqual(missing_marker.status_code, 403)

            saved = self.client.put(
                "/api/chat/settings",
                json=payload,
                headers={"host": "127.0.0.1", "x-xp-gacha-settings": "same-origin"},
            )
            self.assertEqual(saved.status_code, 200, saved.text)
            self.assertNotIn("local-test-secret", saved.text)
            self.assertNotIn("online-test-secret", saved.text)
            self.assertTrue(saved.json()["local"]["apiKeyConfigured"])
            self.assertTrue(saved.json()["online"]["apiKeyConfigured"])

            rejected_secret = "must-not-echo\nthis-value"
            invalid = self.client.put(
                "/api/chat/settings",
                json={**payload, "onlineApiKey": rejected_secret},
                headers={"host": "127.0.0.1", "x-xp-gacha-settings": "same-origin"},
            )
            self.assertEqual(invalid.status_code, 422)
            self.assertNotIn(rejected_secret, invalid.text)
            self.assertNotIn("must-not-echo", invalid.text)

            remote = self.client.get("/api/chat/settings", headers={"host": "192.168.1.20"})
            self.assertEqual(remote.status_code, 403)

            with TestClient(app, client=("192.168.1.20", 50000)) as remote_client:
                spoofed_host = remote_client.get(
                    "/api/chat/settings",
                    headers={"host": "127.0.0.1"},
                )
                self.assertEqual(spoofed_host.status_code, 403)

            foreign_origin = self.client.get(
                "/api/chat/settings",
                headers={"host": "127.0.0.1", "origin": "https://evil.example"},
            )
            self.assertEqual(foreign_origin.status_code, 403)

    def test_chat_stream_emits_typed_events_and_truncates_context_meta(self) -> None:
        context_frame = pd.DataFrame({"ID": ["NH100", "JM200"]})
        upstream_events = iter(
            [
                {"type": "reasoning", "content": "先思考"},
                {"type": "content", "content": "再回答"},
            ]
        )
        with (
            patch("server.main.library.rows_for_ids", return_value=context_frame) as rows_for_ids,
            patch("server.main.get_ai_response_events", return_value=upstream_events) as response_events,
        ):
            response = self.client.post(
                "/api/chat/stream",
                json={
                    "query": "给我推荐一本",
                    "apiMode": "线上 API",
                    "temperature": 0.7,
                    "maxTokens": 4096,
                    "contextIds": ["NH100", "JM200", "IGNORED300"],
                    "contextCount": 2,
                    "deepThinking": True,
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual(
            [event["type"] for event in events],
            ["meta", "reasoning", "reasoning_done", "content", "done"],
        )
        self.assertEqual(events[0]["contextIds"], ["NH100", "JM200"])
        rows_for_ids.assert_called_once_with(["NH100", "JM200"])
        response_events.assert_called_once()
        call = response_events.call_args
        self.assertEqual(call.args[0], "给我推荐一本")
        self.assertIs(call.args[1], context_frame)
        self.assertEqual(
            call.kwargs,
            {
                "api_mode": "线上 API",
                "temperature": 0.7,
                "max_tokens": 4096,
                "deep_thinking": True,
            },
        )

    def test_chat_stream_empty_context_does_not_fall_back_to_library_query(self) -> None:
        context_frame = pd.DataFrame()
        with (
            patch("server.main.library.query") as query_library,
            patch("server.main.library.rows_for_ids", return_value=context_frame) as rows_for_ids,
            patch("server.main.get_ai_response_events", return_value=iter(())) as response_events,
        ):
            response = self.client.post(
                "/api/chat/stream",
                json={
                    "query": "不使用库存回答",
                    "contextIds": [],
                    "contextCount": 10,
                    "deepThinking": False,
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual([event["type"] for event in events], ["meta", "done"])
        self.assertEqual(events[0]["contextIds"], [])
        query_library.assert_not_called()
        rows_for_ids.assert_called_once_with([])
        response_events.assert_called_once()
        call = response_events.call_args
        self.assertEqual(call.args[0], "不使用库存回答")
        self.assertIs(call.args[1], context_frame)
        self.assertEqual(
            call.kwargs,
            {
                "api_mode": "本地 (LM Studio)",
                "temperature": 0.7,
                "max_tokens": 4096,
                "deep_thinking": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
