from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pydantic import ValidationError

from server.modules.llm_settings import LLMSettingsModule, runtime_llm_connection
from server.schemas import ChatRequest, LLMSettingsRequest
from utils_chat import get_ai_response_events, get_ai_response_stream


class StreamingResponseStub:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self):
        return iter(self._lines)


def completion_chunk(delta: dict[str, str]) -> bytes:
    payload = {"choices": [{"delta": delta}]}
    return f"data: {json.dumps(payload, ensure_ascii=False)}".encode("utf-8")


class LLMSettingsModuleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="xp-gacha-llm-settings-")
        self.settings_file = Path(self.temp_dir.name) / ".env"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _environment(self):
        return patch.dict(
            os.environ,
            {
                "XP_GACHA_SETTINGS_FILE": str(self.settings_file),
                "XP_GACHA_RUNTIME_MODE": "source",
            },
            clear=True,
        )

    def test_status_never_returns_api_key(self) -> None:
        secret = "sk-never-return-this"
        self.settings_file.write_text(
            f'LM_STUDIO_API_BASE="http://127.0.0.1:1234/v1"\n'
            f'LM_STUDIO_API_KEY="{secret}"\n'
            'LM_STUDIO_MODEL="local-model"\n'
            'ONLINE_API_BASE="https://api.example.com/v1"\n'
            'ONLINE_API_KEY="online-secret"\n'
            'ONLINE_MODEL="online-model"\n',
            encoding="utf-8",
        )
        with self._environment():
            status = LLMSettingsModule().status()

        self.assertTrue(status["local"]["apiKeyConfigured"])
        self.assertTrue(status["online"]["apiKeyConfigured"])
        self.assertNotIn(secret, json.dumps(status, ensure_ascii=False))
        self.assertNotIn("online-secret", json.dumps(status, ensure_ascii=False))

    def test_update_preserves_unrelated_values_deduplicates_and_applies_immediately(self) -> None:
        self.settings_file.write_text(
            "# keep this comment\n"
            "MYSQL_PASSWORD=database-secret\n"
            "ONLINE_API_KEY=old-key\n"
            "ONLINE_API_KEY=duplicate-key\n",
            encoding="utf-8",
        )
        with self._environment():
            status = LLMSettingsModule().update(
                local_api_base="http://127.0.0.1:1234/v1",
                local_model="qwen-local",
                local_api_key="local#key=value",
                clear_local_api_key=False,
                online_api_base="https://api.example.com/v1",
                online_model="online-model",
                online_api_key="new-online-key",
                clear_online_api_key=False,
            )
            self.assertEqual(os.environ["LM_STUDIO_MODEL"], "qwen-local")
            self.assertEqual(os.environ["ONLINE_API_KEY"], "new-online-key")

        saved = self.settings_file.read_text(encoding="utf-8")
        self.assertIn("# keep this comment", saved)
        self.assertIn("MYSQL_PASSWORD=database-secret", saved)
        self.assertEqual(saved.count("ONLINE_API_KEY="), 1)
        self.assertIn('LM_STUDIO_API_KEY="local#key=value"', saved)
        self.assertTrue(status["local"]["apiKeyConfigured"])

    def test_empty_key_keeps_existing_and_clear_is_explicit(self) -> None:
        self.settings_file.write_text('ONLINE_API_KEY="keep-me"\n', encoding="utf-8")
        with self._environment():
            module = LLMSettingsModule()
            kept = module.update(
                local_api_base="http://127.0.0.1:1234/v1",
                local_model="local-model",
                local_api_key=None,
                clear_local_api_key=False,
                online_api_base="https://api.example.com/v1",
                online_model="online-model",
                online_api_key=None,
                clear_online_api_key=False,
            )
            self.assertTrue(kept["online"]["apiKeyConfigured"])
            cleared = module.update(
                local_api_base="http://127.0.0.1:1234/v1",
                local_model="local-model",
                local_api_key=None,
                clear_local_api_key=False,
                online_api_base="https://api.example.com/v1",
                online_model="online-model",
                online_api_key=None,
                clear_online_api_key=True,
            )
            self.assertFalse(cleared["online"]["apiKeyConfigured"])

    def test_environment_only_key_is_kept_in_memory_but_not_written_to_disk(self) -> None:
        self.settings_file.write_text("MYSQL_USER=xp_gacha\n", encoding="utf-8")
        with patch.dict(
            os.environ,
            {
                "XP_GACHA_SETTINGS_FILE": str(self.settings_file),
                "XP_GACHA_RUNTIME_MODE": "source",
                "ONLINE_API_KEY": "environment-only-secret",
            },
            clear=True,
        ):
            status = LLMSettingsModule().update(
                local_api_base="http://127.0.0.1:1234/v1",
                local_model="local-model",
                local_api_key=None,
                clear_local_api_key=False,
                online_api_base="https://api.example.com/v1",
                online_model="online-model",
                online_api_key=None,
                clear_online_api_key=False,
            )
            self.assertEqual(os.environ["ONLINE_API_KEY"], "environment-only-secret")
            self.assertTrue(status["online"]["apiKeyConfigured"])

        saved = self.settings_file.read_text(encoding="utf-8")
        self.assertNotIn("environment-only-secret", saved)
        self.assertNotIn("ONLINE_API_KEY=", saved)

    def test_replace_fallback_updates_a_bind_mounted_style_file(self) -> None:
        self.settings_file.write_text('ONLINE_MODEL="old-model"\n', encoding="utf-8")
        with self._environment(), patch(
            "server.modules.llm_settings.os.replace",
            side_effect=OSError("simulated bind mount"),
        ):
            LLMSettingsModule().update(
                local_api_base="http://127.0.0.1:1234/v1",
                local_model="local-model",
                local_api_key=None,
                clear_local_api_key=False,
                online_api_base="https://api.example.com/v1",
                online_model="new-model",
                online_api_key=None,
                clear_online_api_key=False,
            )

        self.assertIn('ONLINE_MODEL="new-model"', self.settings_file.read_text(encoding="utf-8"))

    def test_runtime_snapshot_falls_back_to_custom_settings_file(self) -> None:
        self.settings_file.write_text(
            'ONLINE_API_BASE="https://custom.example/v1"\n'
            'ONLINE_API_KEY="custom-key"\n'
            'ONLINE_MODEL="custom-model"\n',
            encoding="utf-8",
        )
        with self._environment():
            snapshot = runtime_llm_connection("线上 API")
        self.assertEqual(snapshot, ("https://custom.example/v1", "custom-key", "custom-model"))

    def test_schema_rejects_unknown_fields_and_unsafe_urls(self) -> None:
        base = {
            "localApiBase": "http://127.0.0.1:1234/v1",
            "localModel": "local-model",
            "onlineApiBase": "https://api.example.com/v1",
            "onlineModel": "online-model",
        }
        with self.assertRaises(ValidationError):
            LLMSettingsRequest.model_validate({**base, "MYSQL_PASSWORD": "do-not-touch"})
        with self.assertRaises(ValidationError):
            LLMSettingsRequest.model_validate({**base, "onlineApiBase": "file:///tmp/key"})
        with self.assertRaises(ValidationError):
            LLMSettingsRequest.model_validate(
                {**base, "onlineApiBase": "https://api.example.com/v1?api_key=secret"}
            )
        with self.assertRaises(ValidationError):
            LLMSettingsRequest.model_validate({**base, "onlineModel": "bad\tmodel"})
        with self.assertRaises(ValidationError):
            LLMSettingsRequest.model_validate(
                {**base, "onlineApiKey": "new-key", "clearOnlineApiKey": True}
            )


class RuntimeLLMSettingsTest(unittest.TestCase):
    def test_chat_request_accepts_deep_thinking_alias_and_defaults_off(self) -> None:
        enabled = ChatRequest.model_validate({"query": "hello", "deepThinking": True})
        defaulted = ChatRequest.model_validate({"query": "hello"})

        self.assertTrue(enabled.deep_thinking)
        self.assertFalse(defaulted.deep_thinking)

    def test_deepseek_payload_explicitly_enables_and_disables_thinking(self) -> None:
        for deep_thinking, expected_type in ((True, "enabled"), (False, "disabled")):
            with self.subTest(deep_thinking=deep_thinking):
                captured: dict = {}

                def fake_post(_url, **kwargs):
                    captured["payload"] = kwargs["json"]
                    return StreamingResponseStub([b"data: [DONE]"])

                with (
                    patch(
                        "utils_chat.runtime_llm_connection",
                        return_value=("https://api.deepseek.com", "deepseek-key", "deepseek-v4-flash"),
                    ),
                    patch("utils_chat.requests.post", side_effect=fake_post),
                ):
                    list(
                        get_ai_response_events(
                            "hello",
                            pd.DataFrame(),
                            api_mode="线上 API",
                            deep_thinking=deep_thinking,
                        )
                )

                self.assertEqual(captured["payload"]["thinking"], {"type": expected_type})
                if deep_thinking:
                    self.assertEqual(captured["payload"]["reasoning_effort"], "high")
                else:
                    self.assertNotIn("reasoning_effort", captured["payload"])

    def test_generic_compatible_endpoint_does_not_receive_provider_specific_thinking_fields(self) -> None:
        generic_endpoints = (
            "https://provider.example/v1",
            "https://api.deepseek.com.evil.example/v1",
        )
        for api_base in generic_endpoints:
            with self.subTest(api_base=api_base):
                captured: dict = {}

                def fake_post(_url, **kwargs):
                    captured["payload"] = kwargs["json"]
                    return StreamingResponseStub([b"data: [DONE]"])

                with (
                    patch(
                        "utils_chat.runtime_llm_connection",
                        return_value=(api_base, "generic-key", "deepseek-v4-flash"),
                    ),
                    patch("utils_chat.requests.post", side_effect=fake_post),
                ):
                    list(
                        get_ai_response_events(
                            "hello",
                            pd.DataFrame(),
                            api_mode="线上 API",
                            deep_thinking=True,
                        )
                    )

                self.assertNotIn("thinking", captured["payload"])
                self.assertNotIn("enable_thinking", captured["payload"])
                self.assertNotIn("reasoning_effort", captured["payload"])

    def test_transport_failures_are_emitted_as_typed_error_events(self) -> None:
        with (
            patch(
                "utils_chat.runtime_llm_connection",
                return_value=("https://provider.example/v1", "key", "model"),
            ),
            patch("utils_chat.requests.post", side_effect=RuntimeError("simulated upstream failure")),
        ):
            events = list(
                get_ai_response_events(
                    "hello",
                    pd.DataFrame(),
                    api_mode="线上 API",
                )
            )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "error")
        self.assertIn("simulated upstream failure", events[0]["content"])

    def test_stream_events_separate_reasoning_content_reasoning_and_answer_content(self) -> None:
        response = StreamingResponseStub(
            [
                completion_chunk({"reasoning_content": "先分析"}),
                completion_chunk({"reasoning": "，再判断"}),
                completion_chunk({"content": "最终答案"}),
                b"data: [DONE]",
            ]
        )
        with (
            patch(
                "utils_chat.runtime_llm_connection",
                return_value=("https://provider.example/v1", "key", "model"),
            ),
            patch("utils_chat.requests.post", return_value=response),
        ):
            events = list(
                get_ai_response_events(
                    "hello",
                    pd.DataFrame(),
                    api_mode="线上 API",
                    deep_thinking=True,
                )
            )

        self.assertEqual(
            events,
            [
                {"type": "reasoning", "content": "先分析"},
                {"type": "reasoning", "content": "，再判断"},
                {"type": "content", "content": "最终答案"},
            ],
        )

    def test_think_tags_split_across_upstream_chunks_are_normalized(self) -> None:
        response = StreamingResponseStub(
            [
                completion_chunk({"content": "<thi"}),
                completion_chunk({"content": "nk>步骤"}),
                completion_chunk({"content": "一</th"}),
                completion_chunk({"content": "ink>最终"}),
                completion_chunk({"content": "答案"}),
                b"data: [DONE]",
            ]
        )
        with (
            patch(
                "utils_chat.runtime_llm_connection",
                return_value=("https://provider.example/v1", "key", "model"),
            ),
            patch("utils_chat.requests.post", return_value=response),
        ):
            events = list(
                get_ai_response_events(
                    "hello",
                    pd.DataFrame(),
                    api_mode="线上 API",
                    deep_thinking=True,
                )
            )

        reasoning = "".join(event["content"] for event in events if event["type"] == "reasoning")
        content = "".join(event["content"] for event in events if event["type"] == "content")
        self.assertEqual(reasoning, "步骤一")
        self.assertEqual(content, "最终答案")
        self.assertNotIn("<think>", reasoning + content)
        self.assertNotIn("</think>", reasoning + content)

    def test_local_key_is_used_by_the_next_request(self) -> None:
        captured: dict = {}

        class Response:
            def raise_for_status(self) -> None:
                return None

            def iter_lines(self):
                return [b"data: [DONE]"]

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs["headers"]
            return Response()

        with (
            patch.dict(
                os.environ,
                {
                    "LM_STUDIO_API_BASE": "http://127.0.0.1:9999/v1",
                    "LM_STUDIO_API_KEY": "runtime-local-key",
                    "LM_STUDIO_MODEL": "runtime-model",
                },
                clear=False,
            ),
            patch("utils_chat.requests.post", side_effect=fake_post),
        ):
            list(get_ai_response_stream("hello", pd.DataFrame(), api_mode="本地 (LM Studio)"))

        self.assertEqual(captured["url"], "http://127.0.0.1:9999/v1/chat/completions")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer runtime-local-key")

    def test_chat_reads_provider_configuration_as_one_snapshot(self) -> None:
        captured: dict = {}

        class Response:
            def raise_for_status(self) -> None:
                return None

            def iter_lines(self):
                return [b"data: [DONE]"]

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs["headers"]
            captured["model"] = kwargs["json"]["model"]
            return Response()

        with (
            patch(
                "utils_chat.runtime_llm_connection",
                return_value=("https://provider.example/v1", "snapshot-key", "snapshot-model"),
            ) as snapshot,
            patch("utils_chat.requests.post", side_effect=fake_post),
        ):
            list(get_ai_response_stream("hello", pd.DataFrame(), api_mode="线上 API"))

        snapshot.assert_called_once_with("线上 API")
        self.assertEqual(captured["url"], "https://provider.example/v1/chat/completions")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer snapshot-key")
        self.assertEqual(captured["model"], "snapshot-model")


if __name__ == "__main__":
    unittest.main()
