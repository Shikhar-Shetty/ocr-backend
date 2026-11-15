"""Dimension detection and SKU validation in the catalogue extractor."""

import unittest
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)  # PyMuPDF's `fitz` alias notice

from prestige_erp_backend.extractor import catalogue_extractor as ce  # noqa: E402


class ParseDimensionTests(unittest.TestCase):
    def assertSize(self, text, size, thickness=None, hint=None):
        result = ce.parse_dimension(text, hint)
        self.assertIsNotNone(result, text)
        self.assertEqual(result["size"], size, text)
        self.assertEqual(result["thickness"], thickness, text)

    def test_common_tile_sizes(self):
        self.assertSize("Size: 800x1600", "800x1600 mm")
        self.assertSize("800 X 1600 MM", "800x1600 mm")
        self.assertSize("800×1600mm", "800x1600 mm")
        self.assertSize("800 * 1600", "800x1600 mm")
        self.assertSize("800mm x 1600mm", "800x1600 mm")

    def test_size_split_across_lines(self):
        self.assertSize("Calacatta Gold\n800x\n1600 mm", "800x1600 mm")

    def test_centimetres_are_converted_to_mm(self):
        self.assertSize("80x160 cm", "800x1600 mm")
        self.assertSize("60 x 120 cm", "600x1200 mm")

    def test_unit_hint_resolves_bare_sizes(self):
        self.assertSize("60x120", "600x1200 mm", hint="cm")

    def test_thickness(self):
        self.assertSize("800x1600x9mm", "800x1600 mm", thickness="9 mm")
        self.assertSize("600 x 1200 x 8.5 mm", "600x1200 mm", thickness="8.5 mm")

    def test_feet_and_inches(self):
        self.assertSize("2x2 ft", "2x2 ft")
        self.assertSize("2x4", "2x4 ft")
        self.assertSize('12"x18"', "12x18 inch")

    def test_bare_two_digit_size_has_unknown_unit(self):
        result = ce.parse_dimension("30x60")
        self.assertEqual(result["size"], "30x60")
        self.assertIsNone(result["unit"])

    def test_implausible_values_are_not_sizes(self):
        self.assertIsNone(ce.parse_dimension("8000x16000 mm"))
        self.assertIsNone(ce.parse_dimension("no size here"))

    def test_unit_hint_detection(self):
        self.assertEqual(ce.detect_unit_hint("Size (cm)"), "cm")
        self.assertEqual(ce.detect_unit_hint("All sizes in mm"), "mm")
        self.assertIsNone(ce.detect_unit_hint("Calacatta Gold"))

    def test_specs_use_the_parser(self):
        specs = ce.extract_tile_specifications("Calacatta Gold\nSize: 80x160 cm x 0.9 cm\nGlossy")
        self.assertEqual(specs["size"], "800x1600 mm")
        self.assertEqual(specs["thickness"], "9 mm")
        self.assertTrue(specs["sizeUnitKnown"])


class SkuValidationTests(unittest.TestCase):
    def skus(self, line):
        return [token.upper() for token, _ in ce.sku_tokens_from_line(line)]

    def test_size_fragment_is_never_an_sku(self):
        # The reported bug: '800x' from '800x1600' was taken as the SKU.
        self.assertEqual(self.skus("800x 1600"), [])
        self.assertEqual(self.skus("Size 800x"), [])
        self.assertEqual(self.skus("x1600 mm"), [])
        self.assertEqual(self.skus("800X"), [])

    def test_full_sizes_and_measurements_are_not_skus(self):
        for line in ("800x1600", "800X1600MM", "800x1600x9mm", "2x2", "9MM", "800mm x 1600mm"):
            self.assertEqual(self.skus(line), [], line)

    def test_labeled_size_is_not_an_sku(self):
        self.assertEqual(self.skus("Code: 800X1600"), [])

    def test_real_codes_are_kept(self):
        self.assertEqual(self.skus("Code: VZ-1234"), ["VZ-1234"])
        self.assertEqual(self.skus("CAL-GLD-01  800x1600"), ["CAL-GLD-01"])
        self.assertEqual(self.skus("SKU KJ8016"), ["KJ8016"])

    def test_code_that_contains_a_size_is_kept(self):
        self.assertEqual(self.skus("VZ-800X1600-CAL"), ["VZ-800X1600-CAL"])

    def test_is_valid_sku(self):
        self.assertFalse(ce.is_valid_sku("800X"))
        self.assertFalse(ce.is_valid_sku("X1600"))
        self.assertFalse(ce.is_valid_sku("800X1600X9"))
        self.assertTrue(ce.is_valid_sku("KJ8016"))
        self.assertTrue(ce.is_valid_sku("X100A"))

    def test_name_cleaning_drops_size_fragments(self):
        self.assertEqual(ce.clean_name_text("Calacatta Gold 800x"), "Calacatta Gold")


class NumericDesignCodeTests(unittest.TestCase):
    """Catalogues such as '800X1200 MATT 2026' identify designs only by a number ('20001')."""

    def line(self, text, size=10.0, bold=False):
        return {"text": text, "bbox": [0, 0, 50, 10], "size": size, "bold": bold, "conf": 1.0,
                "source": "native", "template": False}

    def test_standalone_number_is_a_code(self):
        self.assertEqual(ce.numeric_code_from_line("20001"), "20001")
        self.assertEqual(ce.numeric_code_from_line("# 20001"), "20001")

    def test_years_and_short_numbers_are_not_codes(self):
        self.assertIsNone(ce.numeric_code_from_line("2026"))
        self.assertIsNone(ce.numeric_code_from_line("06"))
        self.assertIsNone(ce.numeric_code_from_line("Call 98450 12345"))

    def test_numeric_code_must_be_prominent(self):
        lines = [self.line("20001", size=30.0, bold=True), self.line("12345", size=10.0)]
        codes = [c["text"] for c in ce.extract_sku_candidates(lines, body=11.0)]
        self.assertEqual(codes, ["20001"])

    def test_measurements_are_never_names(self):
        for text in ("1200mm", "800x", "9 MM", "1200 mm"):
            self.assertFalse(ce.is_name_eligible(text), text)
        self.assertTrue(ce.is_name_eligible("SEA SAND BEIGE"))

    def test_stacked_spec_strip(self):
        lines = [
            {"text": "SURFACE", "bbox": [222, 746, 309, 764]}, {"text": "RANDOM", "bbox": [347, 746, 433, 764]},
            {"text": "MATT", "bbox": [222, 763, 273, 781]}, {"text": "06", "bbox": [347, 763, 369, 781]},
        ]
        self.assertEqual(ce.stacked_label_value(lines, "random"), "06")
        self.assertEqual(ce.stacked_label_value(lines, "surface"), "MATT")
        self.assertIsNone(ce.stacked_label_value(lines, "pcs"))


if __name__ == "__main__":
    unittest.main()
