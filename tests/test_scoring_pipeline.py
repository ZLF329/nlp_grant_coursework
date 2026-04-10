from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.pool.build_pool import build_chunk_pool
from src.scoring.pipeline import (
    build_evidence_text,
    load_rubric,
    rule_based_retrieval,
    score_application_base,
)


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


class RawFakeClient(FakeClient):
    def generate_json(self, messages, *, schema, max_tokens):  # noqa: ANN001
        self.calls.append({
            "messages": messages,
            "schema": schema,
            "max_tokens": max_tokens,
        })
        if not self.payloads:
            raise AssertionError(f"{self.model_name} ran out of payloads")
        return self.payloads.pop(0)


def sample_application() -> dict:
    return {
        "SUMMARY INFORMATION": {
            "Contracting Organisation": "King's College London",
            "Application Title": "Test application",
        },
        "APPLICATION DETAILS": {
            "Plain English Summary of Research": "Clear patient summary for a public audience.",
            "Scientific Abstract": "Mechanistic abstract with objectives, methods, and outcomes.",
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
    retrieved = rule_based_retrieval(rubric, pool["section_chunk_ids"], pool["pool_lookup"])

    general_chunks = retrieved["general"][:2]
    proposed_chunks = retrieved["proposed_research"][:3]

    scorer_a_payloads: list[dict] = []
    scorer_b_payloads: list[dict] = []
    for section in rubric:
        payload_a: dict[str, dict] = {}
        payload_b: dict[str, dict] = {}
        for sub in section["sub_criteria"]:
            payload_a[sub["sub_id"]] = {
                "signals": {signal["sid"]: 0 for signal in sub["signals"]},
                "used_chunk_ids": [],
                "rationale": f"Model A found limited support for {sub['name']}.",
            }
            payload_b[sub["sub_id"]] = {
                "signals": {signal["sid"]: 0 for signal in sub["signals"]},
                "used_chunk_ids": [],
                "rationale": f"Model B found limited support for {sub['name']}.",
            }

        if section["section_key"] == "general":
            first_sub = section["sub_criteria"][0]
            payload_a[first_sub["sub_id"]] = {
                "signals": {signal["sid"]: 2 for signal in first_sub["signals"]},
                "used_chunk_ids": [general_chunks[0]],
                "rationale": "Model A found clear evidence across the grouped general signals.",
            }
            payload_b[first_sub["sub_id"]] = {
                "signals": {signal["sid"]: 1 for signal in first_sub["signals"]},
                "used_chunk_ids": [general_chunks[1]],
                "rationale": "Model B found partial support across the grouped general signals.",
            }
        elif section["section_key"] == "proposed_research":
            first_sub = section["sub_criteria"][0]
            payload_a[first_sub["sub_id"]] = {
                "signals": {signal["sid"]: 2 for signal in first_sub["signals"]},
                "used_chunk_ids": [proposed_chunks[2]],
                "rationale": "Model A found a strong plain-English summary with clear beneficiaries and purpose.",
            }
            payload_b[first_sub["sub_id"]] = {
                "signals": {signal["sid"]: 0 for signal in first_sub["signals"]},
                "used_chunk_ids": [proposed_chunks[0]],
                "rationale": "Model B found little convincing plain-English support in the cited chunk.",
            }
        elif section["section_key"] == "training_development":
            first_sub = section["sub_criteria"][0]
            payload_a[first_sub["sub_id"]] = {
                "signals": {signal["sid"]: 2 for signal in first_sub["signals"]},
                "used_chunk_ids": [],
                "rationale": "Model A attempted a positive training score without grounding it to any chunk.",
            }
            payload_b[first_sub["sub_id"]] = {
                "signals": {signal["sid"]: 2 for signal in first_sub["signals"]},
                "used_chunk_ids": [],
                "rationale": "Model B also attempted a positive training score without evidence.",
            }

        scorer_a_payloads.append(payload_a)
        scorer_b_payloads.append(payload_b)

    return retrieved, scorer_a_payloads, scorer_b_payloads


class PipelineTests(unittest.TestCase):
    def test_load_rubric_general_has_six_subcriteria(self):
        rubric = load_rubric(CRITERIA_PATH)
        general = next(section for section in rubric if section["section_key"] == "general")

        self.assertEqual(len(general["sub_criteria"]), 6)
        self.assertEqual(general["sub_criteria"][0]["sub_id"], "g.1")
        self.assertEqual(general["sub_criteria"][0]["group_name"], "Common Characteristics of Good Applications")
        self.assertEqual(len(general["sub_criteria"][0]["signals"]), 4)
        self.assertEqual(general["sub_criteria"][1]["sub_id"], "g.2")
        self.assertEqual(general["sub_criteria"][1]["group_name"], "Tell Us Why You Need This Award")
        self.assertEqual(len(general["sub_criteria"][1]["signals"]), 4)
        self.assertEqual(general["sub_criteria"][2]["sub_id"], "g.3")
        self.assertEqual(general["sub_criteria"][2]["group_name"], "Applicant")

    def test_rule_based_retrieval_maps_sections_to_expected_chunks(self):
        application = sample_application()
        rubric = load_rubric(CRITERIA_PATH)
        pool = build_chunk_pool(application)
        pool_lookup = pool["pool_lookup"]

        retrieved = rule_based_retrieval(rubric, pool["section_chunk_ids"], pool_lookup)
        all_chunk_ids = list(pool_lookup)

        self.assertEqual(retrieved["general"], all_chunk_ids)
        self.assertEqual(retrieved["application_form"], all_chunk_ids)

        training_sections = {pool_lookup[chunk_id]["parser_section"] for chunk_id in retrieved["training_development"]}
        self.assertEqual(training_sections, {"Training & Development and Research Support"})

        sites_sections = {pool_lookup[chunk_id]["parser_section"] for chunk_id in retrieved["sites_support"]}
        self.assertEqual(sites_sections, {"Contracting Organisation", "Lead Applicant"})

        proposed_sections = {pool_lookup[chunk_id]["parser_section"] for chunk_id in retrieved["proposed_research"]}
        self.assertEqual(
            proposed_sections,
            {
                "Plain English Summary of Research",
                "Scientific Abstract",
                "Detailed Research Plan",
                "Patient & Public Involvement",
            },
        )

    def test_score_application_uses_rule_based_retrieval_and_dual_model_rationales(self):
        application = sample_application()
        retrieved, scorer_a_payloads, scorer_b_payloads = build_payloads_for_application(application)
        scorer_a = FakeClient(payloads=scorer_a_payloads, model_name="model-a")
        scorer_b = FakeClient(payloads=scorer_b_payloads, model_name="model-b")

        result = score_application_base(
            application=application,
            criteria_path=CRITERIA_PATH,
            doc_id="dual_doc",
            scorer_client_a=scorer_a,
            scorer_client_b=scorer_b,
        )

        self.assertGreater(result["pool_size"], 0)
        self.assertTrue(result["pool_lookup"])
        self.assertTrue(result["section_chunk_ids"])
        self.assertEqual(result["run_info"]["retrieval_method"], "rule_based")
        self.assertEqual(result["debug"]["retrieved_chunks"], retrieved)

        general = result["features"]["general"]["sub_criteria"][0]
        self.assertEqual(general["signals"][0]["model_a_score"], 2)
        self.assertEqual(general["signals"][0]["model_b_score"], 1)
        self.assertEqual(general["signals"][0]["score_0to2_raw"], 1.5)
        self.assertEqual(general["confidence_label"], "medium_confidence")
        self.assertEqual(general["confidence_gap"], 1.0)
        self.assertEqual(general["evidence_status"], "ok")
        self.assertEqual(general["evidence_count"], 2)
        self.assertEqual(len(general["evidence"]), 2)
        self.assertEqual(general["rationale_model_a"], "Model A found clear evidence across the grouped general signals.")
        self.assertEqual(general["rationale_model_b"], "Model B found partial support across the grouped general signals.")
        self.assertFalse(general["missing_evidence"])
        self.assertEqual(general["missing_evidence_models"], [])

    def test_scoring_message_uses_original_order_with_ellipsis(self):
        application = sample_application()
        _, scorer_a_payloads, scorer_b_payloads = build_payloads_for_application(application)
        scorer_a = FakeClient(payloads=scorer_a_payloads, model_name="model-a")
        scorer_b = FakeClient(payloads=scorer_b_payloads, model_name="model-b")

        score_application_base(
            application=application,
            criteria_path=CRITERIA_PATH,
            doc_id="message_doc",
            scorer_client_a=scorer_a,
            scorer_client_b=scorer_b,
        )

        proposed_user = scorer_a.calls[1]["messages"][1]["content"]
        self.assertIn("evidence_text:", proposed_user)
        self.assertIn("<", proposed_user)
        self.assertIn("...", proposed_user)

    def test_positive_scores_without_valid_evidence_are_zeroed_and_flagged(self):
        application = sample_application()
        _, scorer_a_payloads, scorer_b_payloads = build_payloads_for_application(application)
        scorer_a = FakeClient(payloads=scorer_a_payloads, model_name="model-a")
        scorer_b = FakeClient(payloads=scorer_b_payloads, model_name="model-b")

        result = score_application_base(
            application=application,
            criteria_path=CRITERIA_PATH,
            doc_id="invalid_evidence_doc",
            scorer_client_a=scorer_a,
            scorer_client_b=scorer_b,
        )

        training = result["features"]["training_development"]["sub_criteria"][0]
        self.assertEqual(training["signals"][0]["score_0to2_raw"], 0.0)
        self.assertEqual(training["evidence_status"], "ok")
        self.assertEqual(training["evidence_count"], 0)
        self.assertTrue(training["missing_evidence"])
        self.assertEqual(training["missing_evidence_models"], ["a", "b"])
        self.assertEqual(training["rationale_model_a"], "Model A attempted a positive training score without grounding it to any chunk.")
        self.assertEqual(training["rationale_model_b"], "Model B also attempted a positive training score without evidence.")

    def test_subcriterion_confidence_is_computed_from_mean_signal_gap(self):
        application = sample_application()
        _, scorer_a_payloads, scorer_b_payloads = build_payloads_for_application(application)
        scorer_a = FakeClient(payloads=scorer_a_payloads, model_name="model-a")
        scorer_b = FakeClient(payloads=scorer_b_payloads, model_name="model-b")

        result = score_application_base(
            application=application,
            criteria_path=CRITERIA_PATH,
            doc_id="confidence_doc",
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
        _, scorer_a_payloads, scorer_b_payloads = build_payloads_for_application(application)
        scorer_a = FakeClient(payloads=scorer_a_payloads, model_name="model-a")
        scorer_b = FakeClient(payloads=scorer_b_payloads, model_name="model-b")

        with tempfile.TemporaryDirectory() as tmpdir:
            result = score_application_base(
                application=application,
                criteria_path=CRITERIA_PATH,
                doc_id="artifact_doc",
                scorer_client_a=scorer_a,
                scorer_client_b=scorer_b,
                artifacts_dir=tmpdir,
            )
            for artifact_path in result["debug"]["artifacts"].values():
                self.assertTrue(Path(artifact_path).exists(), artifact_path)

    def test_invalid_json_writes_raw_debug_files(self):
        application = sample_application()
        _, scorer_a_payloads, scorer_b_payloads = build_payloads_for_application(application)
        scorer_a = FakeClient(payloads=scorer_a_payloads, model_name="model-a")
        scorer_b = RawFakeClient(
            payloads=[json.dumps(scorer_b_payloads[0], ensure_ascii=False), "```json\n{\"broken\": \n```"],
            model_name="model-b",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError) as ctx:
                score_application_base(
                    application=application,
                    criteria_path=CRITERIA_PATH,
                    doc_id="broken_doc",
                    scorer_client_a=scorer_a,
                    scorer_client_b=scorer_b,
                    artifacts_dir=tmpdir,
                )

            message = str(ctx.exception)
            self.assertIn("Raw outputs written to:", message)
            self.assertIn("broken_doc_proposed_research_model_a_raw.txt", message)
            self.assertIn("broken_doc_proposed_research_model_b_raw.txt", message)
            self.assertTrue((Path(tmpdir) / "broken_doc_proposed_research_model_a_raw.txt").exists())
            self.assertTrue((Path(tmpdir) / "broken_doc_proposed_research_model_b_raw.txt").exists())
            self.assertTrue((Path(tmpdir) / "broken_doc_model_a_raw.json").exists())
            self.assertTrue((Path(tmpdir) / "broken_doc_model_b_raw.json").exists())


if __name__ == "__main__":
    unittest.main()
