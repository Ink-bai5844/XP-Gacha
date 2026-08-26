from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import server.modules.system as system_module


class SearchCapabilityStatusTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="xp-gacha-search-status-")
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _write_model(path: Path, *, clip: bool = False) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
        (path / "model.safetensors").write_bytes(b"model")
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")
        if clip:
            (path / "preprocessor_config.json").write_text("{}", encoding="utf-8")
        else:
            (path / "modules.json").write_text("[]", encoding="utf-8")
            (path / "1_Pooling").mkdir()
            (path / "1_Pooling" / "config.json").write_text("{}", encoding="utf-8")

    @staticmethod
    def _write_vector(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"vector")

    def _status(self) -> dict:
        semantic_model = self.root / "models" / "Qwen3-Embedding-0.6B"
        semantic_vector = self.root / "manga_vectors" / "manga_vectors_Qwen3.pkl"
        clip_model = self.root / "models" / "clip-vit-base-patch32"
        clip_vector = self.root / "manga_vectors" / "clip_image_index.pkl"
        with (
            patch.object(system_module.config, "LOCAL_MODEL_PATH", str(semantic_model)),
            patch.object(system_module.config, "VECTOR_FILE", str(semantic_vector)),
            patch.object(system_module.config, "CLIP_MODEL_PATH", str(clip_model)),
            patch.object(system_module.config, "IMG_VECTOR_FILE", str(clip_vector)),
            patch.object(system_module, "database_status", return_value={"available": True, "table_ready": True, "row_count": 0}),
        ):
            return system_module.SystemModule().status()

    def test_reports_model_vector_and_all_missing_states(self) -> None:
        status = self._status()
        self.assertEqual(status["searchCapabilities"]["semantic"]["missing"], ["model", "vector"])
        self.assertEqual(status["searchCapabilities"]["cover"]["missing"], ["model", "vector"])

        self._write_model(self.root / "models" / "Qwen3-Embedding-0.6B")
        self._write_vector(self.root / "manga_vectors" / "clip_image_index.pkl")
        status = self._status()
        self.assertEqual(status["searchCapabilities"]["semantic"]["missing"], ["vector"])
        self.assertEqual(status["searchCapabilities"]["cover"]["missing"], ["model"])
        self.assertTrue(status["searchCapabilities"]["cover"]["idReady"])
        self.assertFalse(status["searchCapabilities"]["cover"]["uploadReady"])

        self._write_vector(self.root / "manga_vectors" / "manga_vectors_Qwen3.pkl")
        self._write_model(self.root / "models" / "clip-vit-base-patch32", clip=True)
        status = self._status()
        self.assertTrue(status["searchCapabilities"]["semantic"]["ready"])
        self.assertTrue(status["searchCapabilities"]["cover"]["ready"])
        self.assertTrue(status["searchCapabilities"]["cover"]["idReady"])
        self.assertTrue(status["searchCapabilities"]["cover"]["uploadReady"])
        self.assertEqual(status["searchCapabilities"]["semantic"]["missing"], [])
        self.assertEqual(status["searchCapabilities"]["cover"]["missing"], [])

    def test_exposes_paths_downloads_and_build_actions(self) -> None:
        status = self._status()
        semantic = status["searchCapabilities"]["semantic"]
        cover = status["searchCapabilities"]["cover"]

        self.assertEqual(semantic["dependencies"]["model"]["path"], str(self.root / "models" / "Qwen3-Embedding-0.6B"))
        self.assertEqual(semantic["dependencies"]["vector"]["path"], str(self.root / "manga_vectors" / "manga_vectors_Qwen3.pkl"))
        self.assertEqual(semantic["setup"]["scriptId"], "text-vector")
        self.assertEqual(cover["setup"]["scriptId"], "clip-vector")
        self.assertTrue(semantic["dependencies"]["model"]["downloadUrl"].startswith("https://huggingface.co/"))
        self.assertTrue(cover["dependencies"]["model"]["downloadUrl"].startswith("https://huggingface.co/"))

    def test_reports_incomplete_qwen_pooling_configuration(self) -> None:
        model_path = self.root / "models" / "Qwen3-Embedding-0.6B"
        self._write_model(model_path)
        (model_path / "1_Pooling" / "config.json").unlink()

        semantic = self._status()["searchCapabilities"]["semantic"]

        self.assertEqual(semantic["dependencies"]["model"]["state"], "incomplete")
        self.assertIn("model", semantic["missing"])
        self.assertFalse(semantic["ready"])


if __name__ == "__main__":
    unittest.main()
