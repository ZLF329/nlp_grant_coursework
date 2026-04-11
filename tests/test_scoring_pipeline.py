from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.pool.build_pool import APPLICATION_CONTEXT_SECTION, build_chunk_pool
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
        self.last_response_body: dict[str, object] | None = None

    def generate_json(self, messages, *, schema, max_tokens):  # noqa: ANN001
        self.calls.append({
            "messages": messages,
            "schema": schema,
            "max_tokens": max_tokens,
        })
        if not self.payloads:
            raise AssertionError(f"{self.model_name} ran out of payloads")
        payload = self.payloads.pop(0)
        self.last_response_body = {"message": {"content": json.dumps(payload, ensure_ascii=False)}}
        return json.dumps(payload, ensure_ascii=False)


class RawFakeClient(FakeClient):
    def generate_json(self, messages, *, schema, max_tokens):  # noqa: ANN001
        self.calls.append({
            "messages": messages,
            "schema": schema,
            "max_tokens": max_tokens,
        })
        if not self.payloads:
            raise AssertionError(f"{self.model_name} ran out of payloads")
        payload = self.payloads.pop(0)
        self.last_response_body = {"message": {"content": str(payload)}}
        return payload


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
            "Training & Development and Research Support": "Training plan and supervisors.",
        },
        "LEAD APPLICANT & RESEARCH TEAM": {
            "Lead Applicant": {
                "Full Name": "Dr Example",
                "Organisation": "King's College London",
            }
        },
        "BUDGET": {
            "SUMMARY BUDGET": "Budget justification and costed support.",
        },
    }


def build_payloads_for_application(application: dict) -> tuple[dict[str, list[str]], list[dict], list[dict]]:
    rubric = load_rubric(CRITERIA_PATH)
    pool = build_chunk_pool(application)
    chunk_ids_by_section = pool["section_chunk_ids"]

    proposed = next(section for section in rubric if section["section_key"] == "proposed_research")
    training = next(section for section in rubric if section["section_key"] == "training_development")
    proposed_first_sub = proposed["sub_criteria"][0]
    proposed_second_sub = proposed["sub_criteria"][1]
    training_first_sub = training["sub_criteria"][0]

    stage1_payloads: list[dict] = []
    for section_name in chunk_ids_by_section:
        application_section = {
            "section_name": section_name,
            "section_content": {
                chunk_id: pool["pool_lookup"][chunk_id]["text"]
                for chunk_id in chunk_ids_by_section[section_name]
            },
        }
        if section_name == "Plain English Summary of Research":
            first_chunk = chunk_ids_by_section[section_name][0]
            stage1_payloads.append({
                "section_name": section_name,
                "findings": [
                    {
                        "sub_id": proposed_first_sub["sub_id"],
                        "signals": {
                            proposed_first_sub["signals"][0]["sid"]: {
                                "evidence": {
                                    "good_evidence_ids": [first_chunk],
                                    "bad_evidence_ids": [],
                                },
                                "implication": "clearly states the unmet need in lay terms.",
                            }
                        },
                    }
                ],
                "resolved_signals": [proposed_first_sub["signals"][0]["sid"]],
            })
        elif section_name == "Scientific Abstract":
            section_chunks = chunk_ids_by_section[section_name]
            stage1_payloads.append({
                "section_name": section_name,
                "findings": [
                    {
                        "sub_id": proposed_second_sub["sub_id"],
                        "signals": {
                            proposed_second_sub["signals"][0]["sid"]: {
                                "evidence": {
                                    "good_evidence_ids": [section_chunks[0]],
                                    "bad_evidence_ids": [],
                                },
                                "implication": "provides concrete methodological support.",
                            }
                        },
                    }
                ],
                "resolved_signals": [proposed_second_sub["signals"][0]["sid"]],
            })
        elif section_name == "Detailed Research Plan":
            section_chunks = chunk_ids_by_section[section_name]
            good_ids = section_chunks[:1]
            bad_ids = section_chunks[1:2]
            stage1_payloads.append({
                "section_name": section_name,
                "findings": [
                    {
                        "sub_id": proposed_second_sub["sub_id"],
                        "signals": {
                            proposed_second_sub["signals"][1]["sid"]: {
                                "evidence": {
                                    "good_evidence_ids": good_ids,
                                    "bad_evidence_ids": bad_ids,
                                },
                                "implication": "adds delivery detail but leaves feasibility risk unresolved.",
                            }
                        },
                    }
                ],
                "resolved_signals": [proposed_second_sub["signals"][1]["sid"]],
            })
        elif section_name == "Training & Development and Research Support":
            first_chunk = chunk_ids_by_section[section_name][0]
            stage1_payloads.append({
                "section_name": section_name,
                "findings": [
                    {
                        "sub_id": training_first_sub["sub_id"],
                        "signals": {
                            training_first_sub["signals"][0]["sid"]: {
                                "evidence": {
                                    "good_evidence_ids": [first_chunk],
                                    "bad_evidence_ids": [],
                                },
                                "implication": "shows a credible training environment.",
                            }
                        },
                    }
                ],
                "resolved_signals": [training_first_sub["signals"][0]["sid"]],
            })
        else:
            stage1_payloads.append({
                "section_name": section_name,
                "findings": [],
                "resolved_signals": [],
            })

    stage2_payloads: list[dict] = []
    for section in rubric:
        payload: dict[str, dict] = {}
        for sub in section["sub_criteria"]:
            payload[sub["sub_id"]] = {
                "signals": {signal["sid"]: 0 for signal in sub["signals"]},
                "used_chunk_ids": [],
                "drawbacks": f"Limited support for {sub['name']}.",
            }

        if section["section_key"] == "proposed_research":
            pesr_chunk = chunk_ids_by_section["Plain English Summary of Research"][0]
            abstract_chunk = chunk_ids_by_section["Scientific Abstract"][0]
            drp_chunks = chunk_ids_by_section["Detailed Research Plan"]
            payload[proposed_first_sub["sub_id"]] = {
                "signals": {signal["sid"]: 2 for signal in proposed_first_sub["signals"]},
                "used_chunk_ids": [pesr_chunk],
                "drawbacks": "The application states an important unmet need, but the evidence remains brief.",
            }
            payload[proposed_second_sub["sub_id"]] = {
                "signals": {
                    proposed_second_sub["signals"][0]["sid"]: 2,
                    proposed_second_sub["signals"][1]["sid"]: 1,
                },
                "used_chunk_ids": [abstract_chunk, *drp_chunks[:2]],
                "drawbacks": "Methods are clear, but feasibility remains only partially supported.",
            }
        elif section["section_key"] == "training_development":
            payload[training_first_sub["sub_id"]] = {
                "signals": {signal["sid"]: 2 for signal in training_first_sub["signals"]},
                "used_chunk_ids": [],
                "drawbacks": "Training looks strong but this response is intentionally ungrounded.",
            }

        stage2_payloads.append(payload)

    return chunk_ids_by_section, stage1_payloads, stage2_payloads


class PipelineTests(unittest.TestCase):
    def test_load_rubric_general_has_six_subcriteria(self):
        rubric = load_rubric(CRITERIA_PATH)
        general = next(section for section in rubric if section["section_key"] == "general")

        self.assertEqual(len(general["sub_criteria"]), 6)
        self.assertEqual(general["sub_criteria"][0]["sub_id"], "g.1")
        self.assertEqual(general["sub_criteria"][0]["group_name"], "Common Characteristics of Good Applications")
        self.assertEqual(len(general["sub_criteria"][0]["signals"]), 4)

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
        self.assertEqual(training_sections, {APPLICATION_CONTEXT_SECTION, "Training & Development and Research Support"})

    def test_score_application_base_builds_belief_state_and_scores(self):
        application = sample_application()
        chunk_ids_by_section, stage1_payloads, stage2_payloads = build_payloads_for_application(application)
        scorer = FakeClient(payloads=[*stage1_payloads, *stage2_payloads], model_name="single-model")

        result = score_application_base(
            application=application,
            criteria_path=CRITERIA_PATH,
            doc_id="single_doc",
            scorer_client=scorer,
        )

        self.assertEqual(result["run_info"]["scorer_model"], "single-model")
        self.assertEqual(result["belief_state"]["processed_sections"], list(chunk_ids_by_section))
        self.assertIn("proposed_research", result["features"])

        proposed = result["features"]["proposed_research"]["sub_criteria"][0]
        self.assertEqual(proposed["signals"][0]["score"], 2)
        self.assertIn("brief", proposed["drawbacks"])
        self.assertEqual(proposed["rationale"], proposed["drawbacks"])
        self.assertEqual(proposed["confidence_gap"], 0.0)
        self.assertEqual(proposed["confidence_label"], "high_confidence")
        self.assertGreater(result["overall"]["final_score_0to100"], 0)

        pr_belief = result["belief_state"]["subcriteria_beliefs"]["pr.2"]["signals"]["pr.2.b"]
        self.assertIn("Detailed Research Plan:", pr_belief["implications"][0])

    def test_stage1_message_has_three_blocks_and_no_weights(self):
        application = sample_application()
        _, stage1_payloads, stage2_payloads = build_payloads_for_application(application)
        scorer = FakeClient(payloads=[*stage1_payloads, *stage2_payloads], model_name="single-model")

        score_application_base(
            application=application,
            criteria_path=CRITERIA_PATH,
            doc_id="message_doc",
            scorer_client=scorer,
        )

        first_user = scorer.calls[0]["messages"][1]["content"]
        self.assertIn('"application_section"', first_user)
        self.assertIn('"criteria"', first_user)
        self.assertIn('"current_belief_state"', first_user)
        self.assertIn('"signal_implications"', first_user)
        self.assertNotIn('"Application Form"', first_user)
        self.assertNotIn('"af.1.a"', first_user)
        self.assertNotIn('"weight"', first_user)
        self.assertNotIn('"good_evidence_ids": ["pesr__001"]', first_user)

        first_schema = scorer.calls[0]["schema"]
        schema_text = json.dumps(first_schema)
        self.assertNotIn("af.1", schema_text)

    def test_stage2_message_uses_full_text_belief_and_no_weights(self):
        application = sample_application()
        chunk_ids_by_section, stage1_payloads, stage2_payloads = build_payloads_for_application(application)
        scorer = FakeClient(payloads=[*stage1_payloads, *stage2_payloads], model_name="single-model")

        score_application_base(
            application=application,
            criteria_path=CRITERIA_PATH,
            doc_id="stage2_doc",
            scorer_client=scorer,
        )

        first_stage2_call = scorer.calls[len(stage1_payloads)]
        user = first_stage2_call["messages"][1]["content"]
        self.assertIn('"application_text"', user)
        self.assertIn('"final_belief_state"', user)
        self.assertIn("`drawbacks` must describe missing evidence", user)
        self.assertNotIn('"weight"', user)

    def test_missing_signals_is_monotonic(self):
        application = sample_application()
        _, stage1_payloads, stage2_payloads = build_payloads_for_application(application)
        scorer = FakeClient(payloads=[*stage1_payloads, *stage2_payloads], model_name="single-model")

        result = score_application_base(
            application=application,
            criteria_path=CRITERIA_PATH,
            doc_id="missing_doc",
            scorer_client=scorer,
        )

        updates = result["debug"]["stage1_section_updates"]
        self.assertFalse(any(sid.startswith("af.") for sid in result["belief_state"]["missing_signals"]))
        previous: set[str] | None = None
        for update in updates:
            current = set(update["missing_signals_after"])
            if previous is not None:
                self.assertTrue(current.issubset(previous))
                self.assertLessEqual(len(current), len(previous))
            previous = current

    def test_positive_scores_without_valid_evidence_are_zeroed_and_flagged(self):
        application = sample_application()
        _, stage1_payloads, stage2_payloads = build_payloads_for_application(application)
        scorer = FakeClient(payloads=[*stage1_payloads, *stage2_payloads], model_name="single-model")

        result = score_application_base(
            application=application,
            criteria_path=CRITERIA_PATH,
            doc_id="invalid_evidence_doc",
            scorer_client=scorer,
        )

        training = result["features"]["training_development"]["sub_criteria"][0]
        self.assertEqual(training["signals"][0]["score_0to5_raw"], 0.0)
        self.assertEqual(training["evidence_count"], 0)
        self.assertTrue(training["missing_evidence"])

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
        _, stage1_payloads, stage2_payloads = build_payloads_for_application(application)
        scorer = FakeClient(payloads=[*stage1_payloads, *stage2_payloads], model_name="single-model")

        with tempfile.TemporaryDirectory() as tmpdir:
            result = score_application_base(
                application=application,
                criteria_path=CRITERIA_PATH,
                doc_id="artifact_doc",
                scorer_client=scorer,
                artifacts_dir=tmpdir,
            )
            for artifact_path in result["debug"]["artifacts"].values():
                self.assertTrue(Path(artifact_path).exists(), artifact_path)

    def test_invalid_json_writes_raw_debug_files(self):
        application = sample_application()
        chunk_ids_by_section, stage1_payloads, _ = build_payloads_for_application(application)
        raw_payloads = [json.dumps(payload, ensure_ascii=False) for payload in stage1_payloads]
        raw_payloads.append("```json\n{\"broken\": \n```")
        scorer = RawFakeClient(payloads=raw_payloads, model_name="single-model")

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError) as ctx:
                score_application_base(
                    application=application,
                    criteria_path=CRITERIA_PATH,
                    doc_id="broken_doc",
                    scorer_client=scorer,
                    artifacts_dir=tmpdir,
                )

            message = str(ctx.exception)
            self.assertIn("Stage 2 returned invalid JSON", message)
            self.assertTrue((Path(tmpdir) / "broken_doc_stage2_general_raw.txt").exists())
            self.assertTrue((Path(tmpdir) / "broken_doc_stage1_raw.json").exists())
            self.assertEqual(len(stage1_payloads), len(chunk_ids_by_section))


if __name__ == "__main__":
    unittest.main()
