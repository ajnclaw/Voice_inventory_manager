"""
Tests for the per-product phone photo upload (POST /item/photo) and the
separate cost / sale price handling (POST /inventory/set-price, GET /inventory).

Everything runs against a throwaway SQLite file and a throwaway image folder:
the real inventory.db, product_images/, logs/ and monitoring.db are never
written. Pictures dropped into artifacts/ (see artifacts/README.md) are
uploaded as extra real-world cases.

Run from the project root:
    .venv/bin/python3 -m unittest discover -s tests -v
"""

import base64
import http.client
import json
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import HTTPServer
from pathlib import Path
from unittest import mock

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from _isolation import data_patches

TOKEN = "test-bearer-token"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

# Smallest valid-looking headers; the server only inspects the first bytes.
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
WEBP = b"RIFF\x24\x00\x00\x00WEBPVP8 " + b"\x00" * 64


def _b64(data):
    return base64.b64encode(data).decode()


class SniffImageExtensionTests(unittest.TestCase):
    def setUp(self):
        import agent_server

        self.sniff = agent_server._sniff_image_extension

    def test_recognises_the_three_formats(self):
        self.assertEqual(self.sniff(JPEG), ".jpg")
        self.assertEqual(self.sniff(PNG), ".png")
        self.assertEqual(self.sniff(WEBP), ".webp")

    def test_rejects_everything_else(self):
        for junk in [b"", b"GIF89a....", b"<svg></svg>", b"<html>", b"%PDF-1.7", b"RIFF....WAVE", b"\x00" * 20]:
            self.assertIsNone(self.sniff(junk), junk)


class PhotoAndPriceServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import agent_server
        from agent import inventory_db

        cls.agent_server = agent_server
        cls.inventory_db = inventory_db
        cls.tmp = Path(tempfile.mkdtemp(prefix="shop-test-"))
        cls.image_dir = cls.tmp / "product_images"
        cls.db_path = cls.tmp / "inventory.db"

        # Guard: never let this suite touch the real data by accident.
        assert cls.db_path != PROJECT_DIR / "inventory.db"

        # data_patches also points INVENTORY_DB at cls.db_path and keeps
        # sessions/monitoring/usage/transcripts out of the real files.
        cls.patches = data_patches(cls.tmp) + [
            mock.patch.object(agent_server, "IMAGE_DIR", cls.image_dir),
            mock.patch.object(agent_server, "AGENT_API_TOKEN", TOKEN),
            mock.patch.object(agent_server, "GOOGLE_AUTH_ENABLED", False),
            # Keep run traces and monitoring counters out of the real files.
            mock.patch.object(agent_server, "RunLogger", mock.MagicMock()),
            mock.patch.object(agent_server.monitoring, "record_request_outcome"),
        ]
        for patch in cls.patches:
            patch.start()

        cls.server = HTTPServer(("127.0.0.1", 0), agent_server.Handler)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        for patch in cls.patches:
            patch.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        # Fresh table content for every test.
        if self.db_path.exists():
            self.db_path.unlink()
        if self.image_dir.exists():
            shutil.rmtree(self.image_dir)

        self.inventory_db.add_item("Air Cleaner", category="Filters", initial_quantity=5, cost_price=380)
        self.inventory_db.add_item("Clutch Wire", category="Cables", initial_quantity=2, cost_price=90, sale_price=140)

    # -- helpers ---------------------------------------------------------

    def call(self, method, path, body=None, auth=True):
        headers = {"Content-Type": "application/json"}
        if auth:
            headers["Authorization"] = f"Bearer {TOKEN}"

        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request(method, path, body=json.dumps(body).encode() if body is not None else None, headers=headers)
        response = conn.getresponse()
        raw = response.read()
        conn.close()

        try:
            return response.status, json.loads(raw) if raw else None
        except ValueError:
            return response.status, raw

    def upload(self, item, data, **extra):
        return self.call("POST", "/item/photo", {"item": item, "content_base64": _b64(data), **extra})

    def item(self, name):
        return next(i for i in self.inventory_db.list_inventory() if i["name"] == name)

    def stored_files(self):
        return sorted(self.image_dir.iterdir()) if self.image_dir.exists() else []

    # -- photo upload ----------------------------------------------------

    def test_photo_is_stored_linked_and_served_back(self):
        status, body = self.upload("Air Cleaner", JPEG)
        self.assertEqual(status, 200, body)

        image_path = body["output"]["image_path"]
        self.assertRegex(image_path, r"^product_images/[0-9a-f]{32}\.jpg$")
        self.assertEqual(self.item("Air Cleaner")["image_path"], image_path)
        self.assertEqual([f.name for f in self.stored_files()], [image_path.split("/")[1]])

        status, served = self.call("GET", "/" + image_path)
        self.assertEqual(status, 200)
        self.assertEqual(served, JPEG)

    def test_png_and_webp_get_matching_extensions(self):
        _, png = self.upload("Air Cleaner", PNG)
        _, webp = self.upload("Clutch Wire", WEBP)
        self.assertTrue(png["output"]["image_path"].endswith(".png"))
        self.assertTrue(webp["output"]["image_path"].endswith(".webp"))

    def test_client_filename_and_mime_type_are_ignored(self):
        status, body = self.upload("Air Cleaner", JPEG, filename="../../evil.html", mime_type="text/html")
        self.assertEqual(status, 200, body)
        self.assertRegex(body["output"]["image_path"], r"^product_images/[0-9a-f]{32}\.jpg$")
        self.assertEqual(len(self.stored_files()), 1)

    def test_replacing_a_photo_points_the_item_at_the_new_one(self):
        _, first = self.upload("Air Cleaner", JPEG)
        _, second = self.upload("Air Cleaner", PNG)
        self.assertNotEqual(first["output"]["image_path"], second["output"]["image_path"])
        self.assertEqual(self.item("Air Cleaner")["image_path"], second["output"]["image_path"])

    def test_photo_does_not_change_stock_or_prices(self):
        before = self.item("Clutch Wire")
        self.upload("Clutch Wire", JPEG)
        after = self.item("Clutch Wire")

        for key in ("current_quantity", "cost_price", "sale_price", "category"):
            self.assertEqual(before[key], after[key], key)

    def test_unknown_item_is_rejected_and_leaves_no_file_behind(self):
        status, body = self.upload("No Such Item", JPEG)
        self.assertEqual(status, 400)
        self.assertIn("error", body)
        self.assertEqual(self.stored_files(), [])

    def test_non_image_bytes_are_rejected(self):
        for junk in [b"<html>hi</html>", b"%PDF-1.7 ....", b"GIF89a....", b"just text"]:
            status, body = self.upload("Air Cleaner", junk)
            self.assertEqual(status, 400, junk)
            self.assertIn("JPEG, PNG or WebP", body["error"])
        self.assertEqual(self.stored_files(), [])
        self.assertIsNone(self.item("Air Cleaner")["image_path"])

    def test_bad_base64_and_missing_fields(self):
        cases = [
            ({"item": "Air Cleaner", "content_base64": "!!!not base64!!!"}, 400),
            ({"item": "Air Cleaner", "content_base64": ""}, 400),
            ({"item": "Air Cleaner"}, 400),
            ({"item": "", "content_base64": _b64(JPEG)}, 400),
            ({"content_base64": _b64(JPEG)}, 400),
        ]
        for body, expected in cases:
            status, response = self.call("POST", "/item/photo", body)
            self.assertEqual(status, expected, body)
            self.assertIn("error", response)
        self.assertEqual(self.stored_files(), [])

    def test_oversized_photo_is_rejected(self):
        with mock.patch.object(self.agent_server, "MAX_PHOTO_BYTES", 100):
            status, body = self.upload("Air Cleaner", JPEG + b"\x00" * 200)
        self.assertEqual(status, 413)
        self.assertEqual(self.stored_files(), [])

    def test_requires_authentication(self):
        status, _ = self.call("POST", "/item/photo", {"item": "Air Cleaner", "content_base64": _b64(JPEG)}, auth=False)
        self.assertEqual(status, 401)
        self.assertEqual(self.stored_files(), [])

    def test_image_route_refuses_path_traversal(self):
        status, _ = self.call("GET", "/product_images/..%2F..%2Fetc%2Fpasswd")
        self.assertIn(status, (400, 404))
        status, _ = self.call("GET", "/product_images/../inventory.db")
        self.assertIn(status, (400, 404))

    # -- cost / sale price -----------------------------------------------

    def test_setting_sale_price_keeps_cost_price(self):
        status, body = self.call("POST", "/inventory/set-price", {"item": "Air Cleaner", "sale_price": 520})
        self.assertEqual(status, 200, body)

        row = self.item("Air Cleaner")
        self.assertEqual(row["cost_price"], 380)
        self.assertEqual(row["sale_price"], 520)

    def test_setting_cost_price_keeps_sale_price(self):
        status, body = self.call("POST", "/inventory/set-price", {"item": "Clutch Wire", "cost_price": 100})
        self.assertEqual(status, 200, body)

        row = self.item("Clutch Wire")
        self.assertEqual(row["cost_price"], 100)
        self.assertEqual(row["sale_price"], 140)

    def test_set_price_needs_a_price_and_a_known_item(self):
        status, _ = self.call("POST", "/inventory/set-price", {"item": "Air Cleaner"})
        self.assertEqual(status, 400)
        status, _ = self.call("POST", "/inventory/set-price", {"item": "No Such Item", "sale_price": 5})
        self.assertEqual(status, 400)

    def test_item_without_sale_price_reports_none_not_cost(self):
        self.assertIsNone(self.item("Air Cleaner")["sale_price"])
        self.assertEqual(self.item("Air Cleaner")["cost_price"], 380)

    def test_inventory_overview_lists_both_prices(self):
        status, body = self.call("GET", "/inventory")
        self.assertEqual(status, 200)

        rows = json.dumps(body)
        for expected in ("cost_price", "sale_price", "380", "140"):
            self.assertIn(expected, rows)

    def test_item_list_payload_carries_both_prices(self):
        items = self.agent_server._serialize_item_list(self.inventory_db.list_inventory())
        by_name = {i["name"]: i for i in items}

        self.assertEqual((by_name["Air Cleaner"]["cost_price"], by_name["Air Cleaner"]["sale_price"]), (380, None))
        self.assertEqual((by_name["Clutch Wire"]["cost_price"], by_name["Clutch Wire"]["sale_price"]), (90, 140))

    # -- real pictures from artifacts/ -----------------------------------

    def test_pictures_dropped_in_artifacts_folder(self):
        pictures = sorted(p for p in ARTIFACTS_DIR.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)

        if not pictures:
            self.skipTest("no pictures in artifacts/ -- drop some there to include them")

        for picture in pictures:
            with self.subTest(picture=picture.name):
                data = picture.read_bytes()

                if len(data) > self.agent_server.MAX_PHOTO_BYTES:
                    status, _ = self.upload("Air Cleaner", data)
                    self.assertEqual(status, 413)
                    continue

                status, body = self.upload("Air Cleaner", data)
                self.assertEqual(status, 200, body)

                status, served = self.call("GET", "/" + body["output"]["image_path"])
                self.assertEqual(status, 200)
                self.assertEqual(served, data)


if __name__ == "__main__":
    unittest.main()
