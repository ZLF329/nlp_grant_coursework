import os
import sys

# Ensure imports work both from project root (server) and standalone
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.feature_eng.sentence_extract import (
    load_json,
    get_sections,
    merge_sections,
    split_text_into_sentences,
    sentence_words_count,
    build_sentence_segment,
    default_structure,
    save_features
)
from src.feature_eng.evaluator import PlainEnglishEvaluator
from src.feature_eng.dependency import analyze_single_file
from src.feature_eng.application import extract_applicant_features
from src.feature_eng.budget import extract_budget_features

def extract_nlp_features(raw_data):
    """Extract NLP features from parsed JSON data dict. Returns a dict of all NLP metrics."""
    nlp = build_sentence_segment()
    evaluator = PlainEnglishEvaluator()

    sections = get_sections(raw_data, default_structure)

    # Global Metrics (Based on ALL sections)
    full_text = merge_sections(sections, default_structure)
    all_sentences = split_text_into_sentences(full_text, nlp)
    all_word_counts = sentence_words_count(all_sentences)

    total_sentences = len(all_sentences)
    total_words_all = sum(all_word_counts)
    avg_sentence_length = total_words_all / total_sentences if total_sentences > 0 else 0

    long_sents = [count for count in all_word_counts if count >= 30]
    long_sentence_ratio = len(long_sents) / total_sentences if total_sentences > 0 else 0

    # Plain English Specific Analysis
    summary_text = (
        sections.get("Plain English Summary of Research")
        or sections.get("Plain English Summary")
        or ""
    )
    readability_results = {}
    if summary_text:
        summary_sentences = split_text_into_sentences(summary_text, nlp)
        summary_total_words = sum(sentence_words_count(summary_sentences))
        readability_results = evaluator.analyze_text(summary_text, summary_total_words)

    # Dependency Parse Analysis
    max_d, avg_d = analyze_single_file(sections)

    # Applicant Features
    applicant_features = extract_applicant_features(raw_data)

    # Budget Features
    budget_features = extract_budget_features(raw_data)

    return {
        "total_sentences": total_sentences,
        "avg_sentence_length": round(avg_sentence_length, 2),
        "long_sentence_ratio": round(long_sentence_ratio, 4),
        "plain_english_analysis": readability_results,
        "max_dependency_depth": max_d,
        "avg_dependency_depth": round(avg_d, 2),
        "applicant_features": applicant_features,
        "budget_features": budget_features,
    }


def main():
    # --- 1. Path Configurations ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    input_dir = os.path.join(project_root, "data", "samples", "json_data")
    output_dir = os.path.join(current_dir, "feature_data")

    filename = "IC00494_after.json"

    raw_data = load_json(input_dir, filename)
    output_data = extract_nlp_features(raw_data)

    # --- Export ---
    save_features(output_data, output_dir, filename)

if __name__ == "__main__":
    main()
