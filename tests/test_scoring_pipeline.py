from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.pool.build_pool import build_chunk_pool
from src.scoring.pipeline import build_evidence_text, load_rubric, score_application_base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CRITERIA_PATH = PROJECT_ROOT / "criteria_points.json"


class FakeClient:
    def __init__(self, *, payloads: list[dict], model_name: str):
        self.payloads = list(payloads)
        self.model_name = model_name
        self.calls: list[dict[str, object]] = []

    def generate_json(self, messages, *, schema, max_tokens):  # noqa: ANN001
        self.calls.append({
            "messages": messages,
            "schema": schema,
            "max_tokens": max_tokens,
        })
        if not self.payloads:
            raise AssertionError(f"{self.model_name} ran out of payloads")
        return json.dumps(self.payloads.pop(0), ensure_ascii=False)


def sample_application() -> dict:
    return {
        "SUMMARY INFORMATION": {
            "Contracting Organisation": "King's College London",
            "Application Title": "Test application",
        },
        "APPLICATION DETAILS": {
            "Plain English Summary of Research": "Clear patient summary for a public audience.",
            "Detailed Research Plan": (
                "Alpha evidence paragraph. " * 80
                + "\n\n"
                + "Beta methods paragraph. " * 80
                + "\n\n"
                + "Gamma implementation paragraph. " * 80
            ),
            "Patient & Public Involvement": "PPI strategy with advisory group and co-design feedback.",
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
            "SUMMARY BUDGET": "Budget justification and costed support.",
        },
    }


def build_payloads_for_application(application: dict) -> tuple[dict[str, list[str]], list[dict], list[dict]]:
    rubric = load_rubric(CRITERIA_PATH)
    pool = build_chunk_pool(application)
    pool_lookup = pool["pool_lookup"]

    general_chunks = list(pool_lookup)[:2]
    proposed_chunks = list(pool_lookup)[2:5]

    retrieval_payload = {
        "general": [general_chunks[0], "missing", general_chunks[0], general_chunks[1]],
        "proposed_research": [proposed_chunks[2], proposed_chunks[0], proposed_chunks[1]],
        "training_development": [],
        "sites_support": [],
        "wpcc": [],
        "application_form": [],
    }

    scorer_a_payloads: list[dict] = []
    scorer_b_payloads: list[dict] = []
    for section in rubric:
        payload_a: dict[str, dict] = {}
        payload_b: dict[str, dict] = {}
        for sub in section["sub_criteria"]:
            payload_a[sub["sub_id"]] = {
                "signals": {signal["sid"]: 0 for signal in sub["signals"]},
                "used_chunk_ids": [],
            }
            payload_b[sub["sub_id"]] = {
                "signals": {signal["sid"]: 0 for signal in sub["signals"]},
                "used_chunk_ids": [],
            }

        if section["section_key"] == "general":
            first_sub = section["sub_criteria"][0]
            sid = first_sub["signals"][0]["sid"]
            payload_a[first_sub["sub_id"]] = {
                "signals": {sid: 2},
                "used_chunk_ids": [general_chunks[0]],
            }
            payload_b[first_sub["sub_id"]] = {
                "signals": {sid: 1},
                "used_chunk_ids": [general_chunks[1]],
            }
        elif section["section_key"] == "proposed_research":
            first_sub = section["sub_criteria"][0]
            payload_a[first_sub["sub_id"]] = {
                "signals": {signal["sid"]: 2 for signal in first_sub["signals"]},
                "used_chunk_ids": [proposed_chunks[2]],
            }
            payload_b[first_sub["sub_id"]] = {
                "signals": {signal["sid"]: 0 for signal in first_sub["signals"]},
                "used_chunk_ids": [proposed_chunks[0]],
            }
        elif section["section_key"] == "training_development":
            first_sub = section["sub_criteria"][0]
            sid = first_sub["signals"][0]["sid"]
            payload_a[first_sub["sub_id"]] = {
                "signals": {sid: 2},
                "used_chunk_ids": [],
            }
            payload_b[first_sub["sub_id"]] = {
                "signals": {sid: 2},
                "used_chunk_ids": [],
            }

        scorer_a_payloads.append(payload_a)
        scorer_b_payloads.append(payload_b)

    return retrieval_payload, scorer_a_payloads, scorer_b_payloads


class PipelineTests(unittest.TestCase):
    def test_load_rubric_expands_grouped_general_structure(self):
        rubric = load_rubric(CRITERIA_PATH)
        general = next(section for section in rubric if section["section_key"] == "general")
        self.assertEqual(len(general["sub_criteria"]), 12)
        self.assertEqual(general["sub_criteria"][0]["sub_id"], "g.1")
        self.assertEqual(general["sub_criteria"][0]["group_name"], "Common Characteristics Of Good Applications")
        self.assertEqual(general["sub_criteria"][4]["sub_id"], "g.5")
        self.assertEqual(general["sub_criteria"][4]["group_name"], "Tell Us Why You Need This Award")
        self.assertEqual(general["sub_criteria"][8]["sub_id"], "g.9")
        self.assertEqual(general["sub_criteria"][8]["group_name"], "Applicant")

    def test_score_application_uses_chunk_pool_and_dual_model_ensemble(self):
        application = sample_application()
        retrieval_payload, scorer_a_payloads, scorer_b_payloads = build_payloads_for_application(application)
        retrieval_client = FakeClient(payloads=[retrieval_payload], model_name="retrieval-model")
        scorer_a = FakeClient(payloads=scorer_a_payloads, model_name="model-a")
        scorer_b = FakeClient(payloads=scorer_b_payloads, model_name="model-b")

        result = score_application_base(
            application=application,
            criteria_path=CRITERIA_PATH,
            doc_id="dual_doc",
            retrieval_client=retrieval_client,
            scorer_client_a=scorer_a,
            scorer_client_b=scorer_b,
        )

        self.assertGreater(result["pool_size"], 0)
        self.assertTrue(result["pool_lookup"])
        self.assertTrue(result["section_chunk_ids"])
        self.assertNotIn("section_index", result)

        general = result["features"]["general"]["sub_criteria"][0]
        self.assertEqual(general["signals"][0]["model_a_score"], 2)
        self.assertEqual(general["signals"][0]["model_b_score"], 1)
        self.assertEqual(general["signals"][0]["score_0to2_raw"], 1.5)
        self.assertEqual(general["confidence_label"], "medium_confidence")
        self.assertEqual(general["confidence_gap"], 1.0)
        self.assertEqual(general["evidence_status"], "ok")
        self.assertEqual(general["evidence_count"], 2)
        self.assertEqual(len(general["evidence"]), 2)

    def test_retrieval_filters_invalid_and_duplicate_chunk_ids(self):
        application = sample_application()
        retrieval_payload, scorer_a_payloads, scorer_b_payloads = build_payloads_for_application(application)
        retrieval_client = FakeClient(payloads=[retrieval_payload], model_name="retrieval-model")
        scorer_a = FakeClient(payloads=scorer_a_payloads, model_name="model-a")
        scorer_b = FakeClient(payloads=scorer_b_payloads, model_name="model-b")

        result = score_application_base(
            application=application,
            criteria_path=CRITERIA_PATH,
            doc_id="retrieval_doc",
            retrieval_client=retrieval_client,
            scorer_client_a=scorer_a,
            scorer_client_b=scorer_b,
        )

        retrieved = result["debug"]["retrieved_chunks"]["general"]
        self.assertEqual(len(retrieved), 2)
        self.assertEqual(len(set(retrieved)), 2)

    def test_scoring_message_uses_original_order_with_ellipsis(self):
        application = sample_application()
        retrieval_payload, scorer_a_payloads, scorer_b_payloads = build_payloads_for_application(application)
        retrieval_client = FakeClient(payloads=[retrieval_payload], model_name="retrieval-model")
        scorer_a = FakeClient(payloads=scorer_a_payloads, model_name="model-a")
        scorer_b = FakeClient(payloads=scorer_b_payloads, model_name="model-b")

        score_application_base(
            application=application,
            criteria_path=CRITERIA_PATH,
            doc_id="message_doc",
            retrieval_client=retrieval_client,
            scorer_client_a=scorer_a,
            scorer_client_b=scorer_b,
        )

        proposed_user = scorer_a.calls[1]["messages"][1]["content"]
        self.assertIn("evidence_text:", proposed_user)
        self.assertIn("<", proposed_user)
        self.assertIn("...", proposed_user)

    def test_positive_scores_without_valid_evidence_are_zeroed(self):
        application = sample_application()
        retrieval_payload, scorer_a_payloads, scorer_b_payloads = build_payloads_for_application(application)
        retrieval_client = FakeClient(payloads=[retrieval_payload], model_name="retrieval-model")
        scorer_a = FakeClient(payloads=scorer_a_payloads, model_name="model-a")
        scorer_b = FakeClient(payloads=scorer_b_payloads, model_name="model-b")

        result = score_application_base(
            application=application,
            criteria_path=CRITERIA_PATH,
            doc_id="invalid_evidence_doc",
            retrieval_client=retrieval_client,
            scorer_client_a=scorer_a,
            scorer_client_b=scorer_b,
        )

        training = result["features"]["training_development"]["sub_criteria"][0]
        self.assertEqual(training["signals"][0]["score_0to2_raw"], 0.0)
        self.assertEqual(training["evidence_status"], "ok")
        self.assertEqual(training["evidence_count"], 0)

    def test_subcriterion_confidence_is_computed_from_mean_signal_gap(self):
        application = sample_application()
        retrieval_payload, scorer_a_payloads, scorer_b_payloads = build_payloads_for_application(application)
        retrieval_client = FakeClient(payloads=[retrieval_payload], model_name="retrieval-model")
        scorer_a = FakeClient(payloads=scorer_a_payloads, model_name="model-a")
        scorer_b = FakeClient(payloads=scorer_b_payloads, model_name="model-b")

        result = score_application_base(
            application=application,
            criteria_path=CRITERIA_PATH,
            doc_id="confidence_doc",
            retrieval_client=retrieval_client,
            scorer_client_a=scorer_a,
            scorer_client_b=scorer_b,
        )

        proposed = result["features"]["proposed_research"]["sub_criteria"][0]
        self.assertEqual(proposed["confidence_gap"], 2.0)
        self.assertEqual(proposed["confidence_label"], "low_confidence")

        overall = result["overall"]
        self.assertIn("avg_confidence_0to2", overall)
        self.assertIn("avg_plausibility_0to5", overall)
        self.assertIn("weak_signal_count", overall)

    def test_build_evidence_text_inserts_ellipsis_between_non_adjacent_chunks(self):
        pool = build_chunk_pool(sample_application())
        pool_lookup = pool["pool_lookup"]
        chunk_ids = list(pool_lookup)
        chunk_order = {chunk_id: idx for idx, chunk_id in enumerate(chunk_ids)}
        text = build_evidence_text([chunk_ids[0], chunk_ids[2]], pool_lookup, chunk_order)
        self.assertIn(f"<{chunk_ids[0]}>", text)
        self.assertIn(f"<{chunk_ids[2]}>", text)
        self.assertIn("\n...\n", text)

    def test_build_evidence_text_inserts_section_dividers(self):
        pool = build_chunk_pool(sample_application())
        pool_lookup = pool["pool_lookup"]
        chunk_ids = list(pool_lookup)
        chunk_order = {chunk_id: idx for idx, chunk_id in enumerate(chunk_ids)}
        text = build_evidence_text([chunk_ids[0], chunk_ids[-1]], pool_lookup, chunk_order)
        self.assertIn("=====", text)

    def test_artifacts_are_written(self):
        application = sample_application()
        retrieval_payload, scorer_a_payloads, scorer_b_payloads = build_payloads_for_application(application)
        retrieval_client = FakeClient(payloads=[retrieval_payload], model_name="retrieval-model")
        scorer_a = FakeClient(payloads=scorer_a_payloads, model_name="model-a")
        scorer_b = FakeClient(payloads=scorer_b_payloads, model_name="model-b")

        with tempfile.TemporaryDirectory() as tmpdir:
            result = score_application_base(
                application=application,
                criteria_path=CRITERIA_PATH,
                doc_id="artifact_doc",
                retrieval_client=retrieval_client,
                scorer_client_a=scorer_a,
                scorer_client_b=scorer_b,
                artifacts_dir=tmpdir,
            )
            for artifact_path in result["debug"]["artifacts"].values():
                self.assertTrue(Path(artifact_path).exists(), artifact_path)


if __name__ == "__main__":
    unittest.main()
