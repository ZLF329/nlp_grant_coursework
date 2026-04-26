from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from src.all_type_parser.all_type_parser import parse


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("pdfplumber"), "pdfplumber is required for PDF parser tests")
class FellowshipApplicantSectionTests(unittest.TestCase):
    def test_examples_parse_applicant_cv_and_research_background(self):
        cases = [
            ("IC00001", PROJECT_ROOT / "Data/examples/IC00001_DF_Doctoral.pdf"),
            ("IC00091", PROJECT_ROOT / "Data/examples/IC00091_AF_Postdoctoral.pdf"),
        ]

        for label, pdf_path in cases:
            with self.subTest(label=label):
                parsed = parse(str(pdf_path))
                app_details = parsed.get("APPLICATION DETAILS", {})

                self.assertIn("Applicant CV", app_details)
                self.assertGreater(len(app_details["Applicant CV"].split()), 100)

                self.assertIn("Applicant Research Background", app_details)
                self.assertGreater(len(app_details["Applicant Research Background"].split()), 100)
