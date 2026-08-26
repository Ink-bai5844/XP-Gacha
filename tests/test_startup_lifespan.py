from __future__ import annotations

import asyncio
import threading
import time
import unittest
from unittest.mock import patch

from server import main as server_main


class StartupLifespanTests(unittest.TestCase):
    def test_library_metadata_warmup_does_not_block_application_startup(self) -> None:
        warmup_started = threading.Event()
        release_warmup = threading.Event()
        warmup_finished = threading.Event()
        entered_after = 0.0

        def slow_metadata_warmup() -> None:
            warmup_started.set()
            release_warmup.wait(timeout=1.5)
            warmup_finished.set()

        async def exercise_lifespan() -> None:
            nonlocal entered_after
            started_at = time.perf_counter()
            async with server_main.app_lifespan(None):
                entered_after = time.perf_counter() - started_at
                warmup_started.wait(timeout=1.0)
                release_warmup.set()
                warmup_finished.wait(timeout=1.0)

        with patch.object(server_main, "_warm_library_metadata", side_effect=slow_metadata_warmup):
            asyncio.run(exercise_lifespan())

        self.assertTrue(warmup_started.is_set())
        self.assertTrue(warmup_finished.is_set())
        self.assertLess(
            entered_after,
            0.3,
            "Slow catalogue warm-up must run outside the ASGI startup critical path.",
        )


if __name__ == "__main__":
    unittest.main()
