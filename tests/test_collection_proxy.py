from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


if importlib.util.find_spec("curl_cffi") is None:
    curl_cffi_stub = types.ModuleType("curl_cffi")
    curl_cffi_stub.requests = types.SimpleNamespace(get=lambda *_args, **_kwargs: None)
    sys.modules["curl_cffi"] = curl_cffi_stub

if importlib.util.find_spec("bs4") is None:
    bs4_stub = types.ModuleType("bs4")

    class _EmptySoup:
        def find_all(self, *_args, **_kwargs):
            return []

    bs4_stub.BeautifulSoup = lambda *_args, **_kwargs: _EmptySoup()
    sys.modules["bs4"] = bs4_stub

from data_get import NH_get_info_online


class _ListPageResponse:
    status_code = 200
    text = "<html><body></body></html>"


class CollectionProxyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_base_url = NH_get_info_online.BASE_URL
        self.original_image_dir = NH_get_info_online.IMG_DIR
        self.original_workers = NH_get_info_online.MAX_WORKERS
        self.original_proxies = NH_get_info_online.PROXIES

    def tearDown(self) -> None:
        NH_get_info_online.BASE_URL = self.original_base_url
        NH_get_info_online.IMG_DIR = self.original_image_dir
        NH_get_info_online.MAX_WORKERS = self.original_workers
        NH_get_info_online.PROXIES = self.original_proxies

    def test_module_proxy_is_initialized_from_project_configuration(self) -> None:
        self.assertEqual(
            NH_get_info_online.PROXIES,
            NH_get_info_online.build_proxies(NH_get_info_online.ONLINE_COVER_PROXY),
        )

    def test_nh_collection_does_not_force_a_loopback_proxy(self) -> None:
        direct_proxies = NH_get_info_online.build_proxies("")
        with (
            mock.patch.object(NH_get_info_online, "PROXIES", direct_proxies),
            mock.patch.object(NH_get_info_online.requests, "get", return_value=_ListPageResponse()) as request_get,
            mock.patch("sys.stdout", new=io.StringIO()),
        ):
            self.assertEqual(NH_get_info_online.get_page_urls(1, retries=1), [])

        self.assertEqual(
            request_get.call_args.kwargs["proxies"],
            {"http": "", "https": ""},
        )

    def test_docker_honors_an_explicit_host_gateway_proxy(self) -> None:
        self.assertEqual(
            NH_get_info_online.build_proxies("http://host.docker.internal:7890"),
            {
                "http": "http://host.docker.internal:7890",
                "https": "http://host.docker.internal:7890",
            },
        )

    def test_all_list_page_failures_fail_the_collection_job(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with (
                mock.patch.object(NH_get_info_online, "get_page_urls", return_value=False),
                mock.patch.object(NH_get_info_online.time, "sleep"),
                mock.patch("sys.stdout", new=io.StringIO()),
                self.assertRaisesRegex(RuntimeError, "列表页请求全部失败"),
            ):
                NH_get_info_online.main(
                    max_page=1,
                    output_csv=str(root / "output.csv"),
                    image_dir=str(root / "images"),
                    error_log=str(root / "errors.log"),
                    max_workers=1,
                    loop=False,
                )

    def test_portable_keeps_a_host_loopback_proxy(self) -> None:
        self.assertEqual(
            NH_get_info_online.build_proxies("http://127.0.0.1:7890"),
            {
                "http": "http://127.0.0.1:7890",
                "https": "http://127.0.0.1:7890",
            },
        )


if __name__ == "__main__":
    unittest.main()
