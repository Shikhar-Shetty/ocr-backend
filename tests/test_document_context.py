"""Catalogue-level brand and category detection, and spec parts in product names."""

import unittest
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)  # PyMuPDF's `fitz` alias notice

import fitz  # noqa: E402

from prestige_erp_backend.extractor import catalogue_extractor as ce  # noqa: E402

BRANDS = ["Gold", "Jaquar", "Rock", "Kajaria", "DC & Nano"]
CATEGORIES = ["Floor Tiles", "Wall Tiles", "Sanitaryware", "Faucets"]


def make_pdf(pages, title=""):
    """pages: a list of pages, each a list of (text, font size)."""
    doc = fitz.open()
    for lines in pages:
        page = doc.new_page()
        y = 72
        for text, size in lines:
            page.insert_text((72, y), text, fontsize=size)
            y += size * 1.8
    if title:
        doc.set_metadata({"title": title})
    return doc


class NamePartsTests(unittest.TestCase):
    def test_size_and_finish_parts_are_not_part_of_the_name(self):
        self.assertEqual(ce.clean_name_text("Statuario Gold \\ 600x1500 \\ Glossy"), "Statuario Gold")
        self.assertEqual(ce.clean_name_text("Onyx Blue | 800x1600 | Matt"), "Onyx Blue")
        self.assertEqual(ce.clean_name_text("Crema • Polished Vitrified"), "Crema")

    def test_names_without_separators_are_untouched(self):
        self.assertEqual(ce.clean_name_text("Rustic Wood"), "Rustic Wood")
        self.assertEqual(ce.clean_name_text("Ivory/Beige 600x600"), "Ivory/Beige")


class DocumentBrandTests(unittest.TestCase):
    def test_brand_from_the_cover_heading(self):
        doc = make_pdf([[("JAQUAR", 40), ("Bath fittings collection 2026", 14)], [("Product list", 10)]])
        brand, _ = ce.detect_document_context(doc, "catalogue.pdf", BRANDS, CATEGORIES)
        self.assertEqual(brand, "Jaquar", "spelled as the ERP spells it")

    def test_brand_from_the_file_name_or_title(self):
        doc = make_pdf([[("Price list", 12)]], title="Kajaria Eternity 2026")
        self.assertEqual(ce.detect_document_context(doc, "x.pdf", BRANDS, CATEGORIES)[0], "Kajaria")
        doc = make_pdf([[("Price list", 12)]])
        self.assertEqual(ce.detect_document_context(doc, "DC_Nano_2026.pdf", BRANDS, [])[0], "DC & Nano")

    def test_a_brand_word_inside_a_product_name_is_not_the_brand(self):
        doc = make_pdf([[("Statuario Gold", 30), ("Rock finish, 600x1200 mm", 10)]])
        brand, _ = ce.detect_document_context(doc, "tiles.pdf", BRANDS, CATEGORIES)
        self.assertIsNone(brand)

    def test_a_page_logo_line_sets_that_page_brand(self):
        state = {"lines": [{"text": "JAQUAR"}, {"text": "Statuario Gold"}], "totalText": ""}
        ctx = {"brands": ce.brand_names(BRANDS)}
        self.assertEqual(ce.page_brand_for(state, "Kajaria", ctx), "Jaquar")
        state = {"lines": [{"text": "Statuario Gold"}], "totalText": ""}
        self.assertEqual(ce.page_brand_for(state, "Kajaria", ctx), "Kajaria", "else the catalogue brand")


class DocumentCategoryTests(unittest.TestCase):
    def test_category_from_the_cover(self):
        doc = make_pdf([[("Sanitary Products", 36), ("Wall hung WC and wash basins", 12)]])
        _, category = ce.detect_document_context(doc, "c.pdf", BRANDS, CATEGORIES)
        self.assertEqual(category, "Sanitaryware")

    def test_synonyms(self):
        doc = make_pdf([[("Bath Fittings", 36), ("Single lever mixers and taps", 12)]])
        self.assertEqual(ce.detect_document_context(doc, "c.pdf", BRANDS, CATEGORIES)[1], "Faucets")

    def test_no_list_or_no_match_gives_no_category(self):
        doc = make_pdf([[("Welcome", 36)]])
        self.assertIsNone(ce.detect_document_context(doc, "c.pdf", BRANDS, CATEGORIES)[1])
        self.assertIsNone(ce.detect_document_context(doc, "c.pdf", BRANDS, [])[1])

    def test_resolve_category(self):
        # A non-tile catalogue: its category for every product.
        self.assertEqual(ce.resolve_category("Floor Tiles", "Sanitaryware", CATEGORIES), "Sanitaryware")
        # A tile catalogue: the per-tile guess, when the ERP has it.
        self.assertEqual(ce.resolve_category("Wall Tiles", "Floor Tiles", CATEGORIES), "Wall Tiles")
        self.assertEqual(ce.resolve_category("Slab Tiles", "Floor Tiles", CATEGORIES), "Floor Tiles")
        self.assertEqual(ce.resolve_category("Slab Tiles", None, CATEGORIES), "Floor Tiles",
                         "a tile kind the ERP lacks goes to its floor tiles")
        self.assertIsNone(ce.resolve_category("Wall Tiles", None, ["Faucets"]), "no tile category at all")
        # No list from the ERP: the built-in guess as before.
        self.assertEqual(ce.resolve_category("Slab Tiles", None, None), "Slab Tiles")


if __name__ == "__main__":
    unittest.main()
