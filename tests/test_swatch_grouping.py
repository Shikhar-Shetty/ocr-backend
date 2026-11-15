"""Grouping related product records (preview, close-up, faces) into one swatch."""

import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)  # PyMuPDF's `fitz` alias notice

import fitz  # noqa: E402
from PIL import Image  # noqa: E402

from prestige_erp_backend.extractor import catalogue_extractor as ce  # noqa: E402

EXTRACTOR = os.path.join(
    os.path.dirname(__file__), "..", "src", "prestige_erp_backend", "extractor", "catalogue_extractor.py"
)


def product(name, sku=None, page=1, image="a", size="800x1600 mm", surface=None, conf=0.8, **extra):
    img = {"id": image, "url": f"/img/{image}.png", "sha256": image, "isPrimary": True, "confidence": conf}
    record = {
        "sourcePage": page, "sourcePages": [page], "productCode": sku, "productName": name,
        "size": size, "surface": surface, "confidence": conf, "primaryImageUrl": img["url"],
        "images": [img], "attributes": [], "reviewWarnings": [], "rawText": name,
        "confidenceBreakdown": {"name": conf}, "skuStatus": "EXTRACTED" if sku else "NOT_PRESENT",
    }
    record.update(extra)
    return record


class StripViewQualifierTests(unittest.TestCase):
    def test_view_words_are_removed(self):
        for name in ("Calacatta Gold Close-up", "Calacatta Gold (Close Up)", "Calacatta Gold - Room View",
                     "Calacatta Gold Preview", "Calacatta Gold Face 2", "Calacatta Gold Zoomed In"):
            self.assertEqual(ce.strip_view_qualifiers(name), "Calacatta Gold", name)

    def test_real_name_words_are_kept(self):
        for name in ("Wood Texture", "Mood Grey", "Detail Beige", "Calacatta Gold"):
            self.assertEqual(ce.strip_view_qualifiers(name), name)

    def test_never_empties_a_name(self):
        self.assertEqual(ce.strip_view_qualifiers("Preview"), "Preview")


class GroupingTests(unittest.TestCase):
    def test_preview_and_close_up_become_one_swatch(self):
        kept, removed = ce.dedupe_products([
            product("Calacatta Gold", image="preview"),
            product("Calacatta Gold Close-up", image="closeup"),
        ])
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(removed), 1)
        swatch = kept[0]
        self.assertEqual(swatch["productName"], "Calacatta Gold")
        self.assertEqual([img["id"] for img in swatch["images"]], ["preview", "closeup"])
        self.assertEqual(sum(1 for img in swatch["images"] if img["isPrimary"]), 1)

    def test_same_sku_is_one_swatch_even_with_different_names(self):
        kept, _ = ce.dedupe_products([
            product("Statuario", sku="KJ-8016", image="a"),
            product("Marble Collection Hero", sku="KJ8016", image="b", page=2),
        ])
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(kept[0]["images"]), 2)
        self.assertEqual(kept[0]["sourcePages"], [1, 2])
        self.assertTrue(any("different names" in w for w in kept[0]["reviewWarnings"]))

    def test_different_skus_stay_separate(self):
        kept, _ = ce.dedupe_products([
            product("Onyx Blue", sku="ONX-01", image="a"),
            product("Onyx Blue", sku="ONX-02", image="b"),
        ])
        self.assertEqual(len(kept), 2)

    def test_different_sizes_or_surfaces_stay_separate(self):
        kept, _ = ce.dedupe_products([
            product("Calacatta Gold", image="a", size="600x1200 mm"),
            product("Calacatta Gold", image="b", size="800x1600 mm"),
        ])
        self.assertEqual(len(kept), 2)
        kept, _ = ce.dedupe_products([
            product("Calacatta Gold", image="a", surface="Glossy"),
            product("Calacatta Gold", image="b", surface="Matt"),
        ])
        self.assertEqual(len(kept), 2)

    def test_name_link_cannot_bridge_two_skus(self):
        kept, _ = ce.dedupe_products([
            product("Onyx", sku="ONX-01", image="a"),
            product("Onyx Close-up", image="b"),
            product("Onyx", sku="ONX-02", image="c"),
        ])
        self.assertEqual(len(kept), 2)
        self.assertEqual(sorted(p["productCode"] for p in kept), ["ONX-01", "ONX-02"])

    def test_missing_details_are_filled_from_the_group(self):
        kept, _ = ce.dedupe_products([
            product("Calacatta Gold", image="a", size=None, conf=0.9,
                    reviewWarnings=["Size/dimension not detected in text."]),
            product("Calacatta Gold Close-up", sku="CAL-01", image="b", conf=0.7, surface="Glossy",
                    ocrSkuCandidates=[{"value": "CAL-01"}]),
        ])
        swatch = kept[0]
        self.assertEqual(swatch["productCode"], "CAL-01")
        self.assertEqual(swatch["size"], "800x1600 mm")
        self.assertEqual(swatch["surface"], "Glossy")
        self.assertNotIn("Size/dimension not detected in text.", swatch["reviewWarnings"])

    def test_member_alternates_are_not_carried_over(self):
        closeup = product("Calacatta Gold Close-up", image="b")
        closeup["images"].append({"id": "guess", "url": "/img/guess.png", "sha256": "guess",
                                  "isPrimary": False, "isAlternate": True})
        kept, _ = ce.dedupe_products([product("Calacatta Gold", image="a"), closeup])
        self.assertEqual([img["id"] for img in kept[0]["images"]], ["a", "b"])

    def test_identical_images_are_not_duplicated(self):
        kept, _ = ce.dedupe_products([
            product("Calacatta Gold", image="same"),
            product("Calacatta Gold Preview", image="same"),
        ])
        self.assertEqual(len(kept[0]["images"]), 1)

    def test_low_confidence_names_never_link_records(self):
        # Regression: a misread name ('1200mm') repeated on every page merged 26 designs into one.
        kept, _ = ce.dedupe_products([
            product("Matt Finish", image="a", conf=0.3, confidenceBreakdown={"name": 0.3}),
            product("Matt Finish", image="b", conf=0.3, confidenceBreakdown={"name": 0.3}),
        ])
        self.assertEqual(len(kept), 2)

    def _code_only_pages(self, preview_name=None):
        """A swatch page (3 faces, code, specs) and a room-preview page that shares its code."""
        unresolved = dict(conf=0.6, confidenceBreakdown={"name": 0.0})
        swatch_page = product(ce.UNRESOLVED_NAME, sku="20001", image="face1", page=10, **unresolved)
        swatch_page["images"] += [
            {"id": f, "url": f"/img/{f}.png", "sha256": f, "isPrimary": False, "isSwatchImage": True}
            for f in ("face2", "face3")
        ]
        if preview_name:
            preview_page = product(preview_name, sku="20001", image="room", page=9, size=None, conf=0.5,
                                   confidenceBreakdown={"name": 0.9})
        else:
            preview_page = product(ce.UNRESOLVED_NAME, sku="20001", image="room", page=9, size=None,
                                   **unresolved)
        return ce.dedupe_products([preview_page, swatch_page])

    def test_pages_sharing_an_sku_become_one_swatch_without_a_name(self):
        kept, _ = self._code_only_pages()
        self.assertEqual(len(kept), 1)
        swatch = kept[0]
        self.assertEqual(swatch["productCode"], "20001")
        self.assertEqual(swatch["productName"], ce.UNRESOLVED_NAME)  # the SKU is never the name
        self.assertEqual(swatch["primaryImageUrl"], "/img/face1.png")  # a flat tile face, not the room shot
        self.assertEqual([i["id"] for i in swatch["images"]], ["face1", "face2", "face3", "room"])

    def test_name_from_the_other_page_with_the_same_sku_is_used(self):
        kept, _ = self._code_only_pages(preview_name="Desert Brown")
        swatch = kept[0]
        self.assertEqual(swatch["productName"], "Desert Brown")
        self.assertEqual(swatch["productCode"], "20001")
        self.assertEqual(swatch["primaryImageUrl"], "/img/face1.png")
        self.assertEqual(len(swatch["images"]), 4)
        self.assertTrue(any("shares this SKU" in w for w in swatch["reviewWarnings"]))
        self.assertFalse(any("name could not" in w for w in swatch["reviewWarnings"]))

    def test_unrelated_products_are_untouched(self):
        kept, removed = ce.dedupe_products([
            product("Calacatta Gold", image="a"), product("Statuario", image="b"),
        ])
        self.assertEqual(len(kept), 2)
        self.assertEqual(removed, [])


def _noise_png(path, seed):
    rng = random.Random(seed)
    img = Image.new("RGB", (240, 240))
    img.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(240 * 240)])
    img.save(path)


class EndToEndTests(unittest.TestCase):
    """Runs the real extractor CLI on a generated two-row catalogue grid."""

    def test_catalogue_grid(self):
        cards = [  # (name, code line, size line)
            ("Calacatta Gold", "Code: CAL-01", "Size: 800x"),
            ("Calacatta Gold Close-up", "", "1600 mm"),
            ("Onyx Blue", "Code: ONX-01", "Size: 800 x 1600 mm"),
            ("Statuario Matt", "Code: STM-22", "Size: 600x1200 mm"),
            ("Statuario Matt Room View", "Code: STM-22", "Size: 600x1200 mm"),
            ("Onyx Blue", "Code: ONX-02", "Size: 800 x 1600 mm"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = os.path.join(tmp, "catalogue.pdf")
            doc = fitz.open()
            page = doc.new_page(width=842, height=842)
            for i, (name, code, size) in enumerate(cards):
                col, row = i % 3, i // 3
                x, y = 60 + col * 260, 60 + row * 380
                img_path = os.path.join(tmp, f"tile{i}.png")
                _noise_png(img_path, i)
                page.insert_image(fitz.Rect(x, y, x + 200, y + 200), filename=img_path)
                page.insert_text((x, y + 225), name, fontsize=14, fontname="hebo")
                if code:
                    page.insert_text((x, y + 245), code, fontsize=9)
                page.insert_text((x, y + 260), size, fontsize=9)
            doc.save(pdf_path)

            out_dir = os.path.join(tmp, "out")
            completed = subprocess.run(
                [sys.executable, "-W", "ignore", EXTRACTOR, "--pdf", pdf_path, "--output-dir", out_dir,
                 "--job-id", "t1", "--no-ocr"],
                capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with open(os.path.join(out_dir, "extraction_result.json"), encoding="utf-8") as f:
                result = json.load(f)

        by_code = {p["productCode"]: p for p in result["products"]}
        codes = sorted(c for c in by_code if c)
        self.assertEqual(codes, ["CAL-01", "ONX-01", "ONX-02", "STM-22"])
        self.assertNotIn("800X", by_code)  # the size fragment never becomes an SKU

        self.assertEqual(by_code["CAL-01"]["productName"], "Calacatta Gold")
        self.assertEqual(len(by_code["CAL-01"]["images"]), 2)      # preview + close-up
        self.assertEqual(len(by_code["STM-22"]["images"]), 2)      # product + room view
        self.assertEqual(by_code["STM-22"]["productName"], "Statuario Matt")
        self.assertEqual(len(by_code["ONX-01"]["images"]), 1)      # same name, different SKU
        self.assertEqual(len(result["products"]), 4)
        for p in result["products"]:
            self.assertEqual(sum(1 for img in p["images"] if img.get("isPrimary")), 1, p["productName"])


class CodeOnlyCatalogueTests(unittest.TestCase):
    """Preview page + swatch page per design, linked only by a numeric code (800X1200 MATT layout)."""

    def run_catalogue(self, preview_names):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = os.path.join(tmp, "catalogue.pdf")
            doc = fitz.open()
            for i, (code, preview_name) in enumerate(zip(("20001", "20002"), preview_names)):
                room = os.path.join(tmp, f"room{i}.png")
                _noise_png(room, 100 + i)
                page = doc.new_page(width=1190, height=842)          # room preview
                page.insert_image(fitz.Rect(85, 85, 1105, 757), filename=room)
                page.insert_text((1005, 715), code, fontsize=20, fontname="hebo")
                if preview_name:
                    page.insert_text((85, 790), preview_name, fontsize=24, fontname="hebo")
                page = doc.new_page(width=1190, height=842)          # swatch: 3 faces + spec strip
                for f, x in enumerate((112, 436, 761)):
                    face = os.path.join(tmp, f"face{i}{f}.png")
                    _noise_png(face, 200 + 10 * i + f)
                    page.insert_image(fitz.Rect(x, 133, x + 317, 609), filename=face)
                page.insert_text((113, 685), code, fontsize=30, fontname="hebo")
                for x, top, bottom in ((114, "800x", "1200mm"), (222, "SURFACE", "MATT"), (347, "RANDOM", "06")):
                    page.insert_text((x, 760), top, fontsize=18)
                    page.insert_text((x, 777), bottom, fontsize=18, fontname="hebo")
                page.insert_text((926, 762), "TAP TO", fontsize=13)
                page.insert_text((926, 778), "INDEX", fontsize=15)
            doc.save(pdf_path)
            out_dir = os.path.join(tmp, "out")
            completed = subprocess.run(
                [sys.executable, "-W", "ignore", EXTRACTOR, "--pdf", pdf_path, "--output-dir", out_dir,
                 "--job-id", "c1", "--no-ocr"],
                capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with open(os.path.join(out_dir, "extraction_result.json"), encoding="utf-8") as f:
                return {p["productCode"]: p for p in json.load(f)["products"]}

    def test_code_is_the_sku_and_never_the_name(self):
        products = self.run_catalogue([None, None])
        self.assertEqual(sorted(products), ["20001", "20002"])
        for code, p in products.items():
            self.assertEqual(p["productName"], ce.UNRESOLVED_NAME, code)
            self.assertEqual(p["size"], "800x1200 mm")
            self.assertEqual(len(p["images"]), 4)                        # 3 faces + room preview
            self.assertNotIn("room", p["primaryImageUrl"])
            self.assertIn(f"_p{2 if code == '20001' else 4}_", p["primaryImageUrl"])  # a face is primary
            self.assertIn(("Random Faces", "6 Faces"), [(a["name"], a["value"]) for a in p["attributes"]])

    def test_name_on_the_preview_page_is_used_for_the_swatch(self):
        products = self.run_catalogue(["DESERT BROWN", None])
        self.assertEqual(products["20001"]["productName"], "DESERT BROWN")
        self.assertIn("_p2_", products["20001"]["primaryImageUrl"])     # still the flat tile face
        self.assertEqual(len(products["20001"]["images"]), 4)
        self.assertEqual(products["20002"]["productName"], ce.UNRESOLVED_NAME)


if __name__ == "__main__":
    unittest.main()
