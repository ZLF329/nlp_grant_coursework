from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.scoring.pipeline import build_section_index, score_application_base
from src.verify.faithfulness import FallbackJudge


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CRITERIA_PATH = PROJECT_ROOT / "criteria_points.json"


class FakeStage1Client:
    model_name = "fake-stage1"

    def __init__(self, payload: dict):
        self.payload = payload

    def generate_json(self, messages, *, schema, max_tokens):  # noqa: ANN001
        del messages, schema, max_tokens
        return json.dumps(self.payload, ensure_ascii=False)


def sample_application() -> dict:
    return {
        "SUMMARY INFORMATION": {
            "Contracting Organisation": "King's College London",
            "Application Title": "Test application",
        },
        "APPLICATION DETAILS": {
            "Plain English Summary of Research": "Clear patient summary.",
            "Detailed Research Plan": "Detailed plan and methods.",
            "Patient & Public Involvement": "PPI strategy.",
        },
        "LEAD APPLICANT & RESEARCH TEAM": {
            "Lead Applicant": {
                "Full Name": "Dr Example",
                "Organisation": "King's College London",
            }
        },
        "TRAINING": {
            "Training & Development and Research Support": "Training plan and supervisors.",
        },
        "BUDGET": {
            "SUMMARY BUDGET": "Budget justification.",
        },
    }


class PipelineTests(unittest.TestCase):
    def test_section_index_is_stable(self):
        section_data = build_section_index(sample_application())
        self.assertEqual(
            section_data["section_index"],
            {
                "S01": "Contracting Organisation",
                "S02": "Application Title",
                "S03": "Plain English Summary of Research",
                "S04": "Detailed Research Plan",
                "S05": "Patient & Public Involvement",
                "S06": "Lead Applicant",
                "S07": "Training & Development and Research Support",
                "S08": "SUMMARY BUDGET",
            },
        )

    def test_new_section_object_response_scores_with_section_ids(self):
        payload = {
            "general": {
                "g.1": {
                    "signals": {"g.1.a": 2},
                    "needed_section_ids": ["S04", "S07"],
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            result = score_application_base(
                application=sample_application(),
                criteria_path=CRITERIA_PATH,
                doc_id="test_doc",
                stage1_client=FakeStage1Client(payload),
                judge=FallbackJudge(),
                artifacts_dir=tmpdir,
            )

        general = result["features"]["general"]["sub_criteria"][0]
        self.assertEqual(general["needed_section_ids"], ["S04", "S07"])
        self.assertGreater(general["score_10"], 0)
        self.assertEqual(
            [item["section_id"] for item in general["evidence"]],
            ["S04", "S07"],
        )
        self.assertIn("section_index", result)
        self.assertEqual(result["pool_lookup"], {})
        self.assertEqual(result["section_chunk_ids"], {})

    def test_invalid_section_ids_zero_out_positive_scores(self):
        payload = {
            "general": {
                "g.1": {
                    "signals": {"g.1.a": 2},
                    "needed_section_ids": ["S99"],
                }
            }
        }
        result = score_application_base(
            application=sample_application(),
            criteria_path=CRITERIA_PATH,
            doc_id="test_doc",
            stage1_client=FakeStage1Client(payload),
            judge=FallbackJudge(),
        )
        general = result["features"]["general"]["sub_criteria"][0]
        self.assertEqual(general["needed_section_ids"], [])
        self.assertEqual(general["signals"][0]["score_0to2_raw"], 0)
        self.assertEqual(general["score_10"], 0)

    def test_legacy_subcriteria_payload_is_still_accepted(self):
        payload = {
            "sub_criteria": [
                {
                    "sub_id": "g.1",
                    "score": 2,
                    "needed_section_ids": ["S04", "S07"],
                }
            ]
        }
        result = score_application_base(
            application=sample_application(),
            criteria_path=CRITERIA_PATH,
            doc_id="legacy_doc",
            stage1_client=FakeStage1Client(payload),
            judge=FallbackJudge(),
        )
        general = result["features"]["general"]["sub_criteria"][0]
        self.assertEqual(general["needed_section_ids"], ["S04", "S07"])
        self.assertEqual(general["signals"][0]["score_0to2_raw"], 2)

    def test_stage1_artifacts_are_written(self):
        payload = {
            "general": {
                "g.1": {
                    "signals": {"g.1.a": 1},
                    "needed_section_ids": ["S04"],
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            result = score_application_base(
                application=sample_application(),
                criteria_path=CRITERIA_PATH,
                doc_id="artifact_doc",
                stage1_client=FakeStage1Client(payload),
                judge=FallbackJudge(),
                artifacts_dir=tmpdir,
            )
            artifacts = result["debug"]["artifacts"]
            for artifact_path in artifacts.values():
                self.assertTrue(Path(artifact_path).exists(), artifact_path)


if __name__ == "__main__":
    unittest.main()
