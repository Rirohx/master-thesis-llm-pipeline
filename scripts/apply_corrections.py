"""
Applies manual changes from the review process as described in:
> Yilmaz, I. (2026). AI Affordances in Education: A Scoping Review on Use Cases, Constraints, and Alignment with Educational Purpose. University of Cologne.
  Appendix B.3: Cluster Refinement Process
"""

from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
INPUT_FILE = ROOT / "data" / "output" / "corpus_final.csv"
OUTPUT_FILE = ROOT / "data" / "output" / "corpus_final_corrected.csv"


def load_csv(path: Path) -> tuple[list[dict], list[str] | None, str]:
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(1000)
        f.seek(0)
        delim = ";" if sample.count(";") > sample.count(",") else ","
        reader = csv.DictReader(f, delimiter=delim)
        rows = list(reader)
        fieldnames = reader.fieldnames
    return rows, fieldnames, delim


def save_csv(rows: list[dict], fieldnames: list[str], path: Path, delim: str = ";"):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delim)
        writer.writeheader()
        writer.writerows(rows)



def apply_label_moves(rows: list[dict], label_moves: dict, step_name: str) -> list[dict]:
    updated = 0
    for r in rows:
        try:
            # label - cluster pairs
            pairs = json.loads(r.get("use_case_label_cluster_pairs", "[]"))
            # current papers cluster list (set)
            clusters = set(json.loads(r.get("use_case_clusters", "[]")))
        except Exception:
            continue

        # lookup: label - cluster pairs
        paper_labels = {p["label"]: p["cluster"] for p in pairs}
        changed = False

        #  label  move
        for label, new_cluster in label_moves.items():
            if label in paper_labels:
                old_cluster = paper_labels[label]
                if old_cluster != new_cluster:
                    clusters.add(new_cluster)
                    changed = True

        if changed:
            # Remove old cluster only if no other label is assigned
            moved_original_clusters = set(
                paper_labels[label]
                for label in label_moves
                if label in paper_labels and paper_labels[label] != label_moves[label]
            )
            for old_cl in moved_original_clusters:
                still_needed = any(
                    p["label"] not in label_moves and p["cluster"] == old_cl
                    for p in pairs
                )
                if not still_needed:
                    clusters.discard(old_cl)
            r["use_case_clusters"] = json.dumps(sorted(clusters))
            updated += 1

    print(f"  [{step_name}] Rows updated: {updated}")
    return rows


# Step 1: Top-50 spotcheck corrections
STEP1_MOVES = {
    "feedback_generation": "assessment_and_grading",
    "lesson_planning_support": "teacher_and_instructor_support",
}

# Step 2: tiny cluster merges
STEP2_MOVES = {
    "bibliometric_analysis_support": "research_support",
    "counterfactual_explanation": "learning_analytics_and_student_modeling",
    "counterfactual_suggestion": "learning_analytics_and_student_modeling",
    "dance_feedback_support": "tutoring_and_learning_support",
    "head_noun_identification_support": "writing_and_feedback_support",
    "learning_portfolio_generation": "content_and_curriculum_generation",
    "media_literacy_support": "ai_literacy_support",
    "mentor_matching": "administrative_and_institutional_support",
    "mentor_matching_support": "administrative_and_institutional_support",
    "named_entity_recognition": "administrative_and_institutional_support",
    "natural_language_processing_tool": "writing_and_feedback_support",
    "text_summarization": "writing_and_feedback_support",
    "educational_orientation_recommendation": "career_guidance_support",
    "estimation_accuracy_improvement": "learning_analytics_and_student_modeling",
    "procedure_step_detection": "simulation_and_experiential_learning",
    "resource_availability_prediction": "administrative_and_institutional_support",
    "social_emotional_learning_support": "wellbeing_support",
    "student_motivation_support": "wellbeing_support",
    "social_norm_nudge": "simulation_and_experiential_learning",
    "speech_recognition_partner": "tutoring_and_learning_support",
    "embodied_agent_trust_evaluation": "chatbot_and_conversational_learning",
}

# Step 3: mid-tier cluster corrections
STEP3_MOVES = {
    "ageism_detection": "bias_detection_and_mitigation",
    "algorithmic_fairness_mitigation": "bias_detection_and_mitigation",
    "assessment_bias_detection": "bias_detection_and_mitigation",
    "assessment_bias_mitigation": "bias_detection_and_mitigation",
    "bias_analysis": "bias_detection_and_mitigation",
    "bias_awareness_support": "bias_detection_and_mitigation",
    "bias_detection": "bias_detection_and_mitigation",
    "bias_detection_support": "bias_detection_and_mitigation",
    "bias_mitigation": "bias_detection_and_mitigation",
    "equitable_recommendation_postprocessing": "bias_detection_and_mitigation",
    "fair_peer_recommendation": "bias_detection_and_mitigation",
    "fairness_analysis": "bias_detection_and_mitigation",
    "fairness_assessment": "bias_detection_and_mitigation",
    "fairness_aware_prediction": "bias_detection_and_mitigation",
    "fairness_aware_risk_prediction": "bias_detection_and_mitigation",
    "fairness_evaluation": "bias_detection_and_mitigation",
    "gender_awareness_support": "bias_detection_and_mitigation",
    "grading_bias_detection": "bias_detection_and_mitigation",
    "groupthink_mitigation": "bias_detection_and_mitigation",
    "hate_speech_detection": "bias_detection_and_mitigation",
    "intergroup_bias_detection": "bias_detection_and_mitigation",
    "pronunciation_bias_analysis": "bias_detection_and_mitigation",
    "recommendation_fairness_monitoring": "bias_detection_and_mitigation",
    "recruitment_bias_analysis": "bias_detection_and_mitigation",
    "visual_bias_detection": "bias_detection_and_mitigation",
    "visual_content_bias_detection": "bias_detection_and_mitigation",
    "medical_image_generation": "content_and_curriculum_generation",
    "synthetic_image_generation": "content_and_curriculum_generation",
    "educational_quality_evaluation": "assessment_and_grading",
    "neural_network_assessment": "assessment_and_grading",
    "persuasiveness_prediction": "learning_analytics_and_student_modeling",
    "music_quality_evaluation": "music_performance_feedback",
    "feedback_evaluation_automation": "assessment_and_grading",
    "feedback_hallucination_detection": "assessment_and_grading",
    "program_evaluation_support": "administrative_and_institutional_support",
    "peer_review_problem_detection": "assessment_and_grading",
    "model_interpretation": "learning_analytics_and_student_modeling",
    "model_interpretability_support": "learning_analytics_and_student_modeling",
    "model_explainability_support": "learning_analytics_and_student_modeling",
    "model_explanation": "learning_analytics_and_student_modeling",
    "model_explanation_support": "learning_analytics_and_student_modeling",
    "ml_model_training_support": "ai_literacy_support",
    "model_building_learning_support": "ai_literacy_support",
    "model_testing_support": "ai_literacy_support",
    "ml_modeling_support": "teacher_and_instructor_support",
    "nlp_assistance": "writing_and_feedback_support",
    "nlp_support": "writing_and_feedback_support",
    "pragmatic_language_analysis": "writing_and_feedback_support",
    "symbolic_text_prediction": "writing_and_feedback_support",
    "text_analysis_support": "writing_and_feedback_support",
    "text_transformation_support": "writing_and_feedback_support",
    "synthetic_instructor_generation": "simulation_and_experiential_learning",
    "text_to_image_generation": "content_and_curriculum_generation",
    "educational_media_metadata_enrichment": "music_content_analysis",
    "music_emotion_recognition": "music_content_analysis",
    "music_feature_extraction": "music_content_analysis",
    "research_idea_generation": "research_support",
    "idea_detection": "learning_analytics_and_student_modeling",
}

# Step 4: design_support cluster breakdown
STEP4_MOVES = {
    "instructional_design_support": "teacher_and_instructor_support",
    "lesson_planning_assistance": "teacher_and_instructor_support",
    "instructional_design_assistance": "teacher_and_instructor_support",
    "lesson_design_support": "teacher_and_instructor_support",
    "lesson_plan_generation_support": "teacher_and_instructor_support",
    "lesson_design_assistance": "teacher_and_instructor_support",
    "instructional_design_suggestion": "teacher_and_instructor_support",
    "instructional_design_feedback": "teacher_and_instructor_support",
    "lesson_plan_design_assistance": "teacher_and_instructor_support",
    "design_automation": "teacher_and_instructor_support",
    "program_planning_support": "teacher_and_instructor_support",
    "learning_design_support": "teacher_and_instructor_support",
    "prompt_design_support": "teacher_and_instructor_support",
    "prompt_design_assistance": "teacher_and_instructor_support",
    "design_ideation_support": "creative_design_support",
    "design_idea_generation": "creative_design_support",
    "design_assistance": "creative_design_support",
    "design_image_generation": "creative_design_support",
    "design_scaffolding_support": "creative_design_support",
    "design_guidance_support": "creative_design_support",
    "intelligent_painting_assistance": "creative_design_support",
    "maker_project_design_support": "creative_design_support",
    "design_art_support": "creative_design_support",
    "drawing_assistance": "creative_design_support",
    "design_partner_support": "creative_design_support",
    "generative_design_support": "creative_design_support",
    "design_assistance_support": "creative_design_support",
    "design_visualization_support": "creative_design_support",
    "drawing_instruction_support": "creative_design_support",
    "engineering_design_feedback": "tutoring_and_learning_support",
    "design_factor_analysis": "tutoring_and_learning_support",
    "progressive_prompt_scaffolding": "simulation_and_experiential_learning",
    "schema_creation_support": "learning_analytics_and_student_modeling",
    "structure_recommendation": "writing_and_feedback_support",
}

# Step 5: content_curation and content_retrieval merge
STEP5_MOVES = {
    "educational_content_curation": "content_retrieval_and_curation",
    "educational_content_query": "content_retrieval_and_curation",
    "educational_content_retrieval": "content_retrieval_and_curation",
    "educational_resource_classification": "content_retrieval_and_curation",
    "educational_search_reranking": "content_retrieval_and_curation",
    "exploratory_search_support": "content_retrieval_and_curation",
    "generative_ai_information_seeking": "content_retrieval_and_curation",
    "handwritten_text_recognition": "content_retrieval_and_curation",
    "hidden_curriculum_access_support": "content_retrieval_and_curation",
    "historical_question_answering": "content_retrieval_and_curation",
    "idea_organisation_support": "content_retrieval_and_curation",
    "idea_organization_support": "content_retrieval_and_curation",
    "image_quality_filtering": "content_retrieval_and_curation",
    "information_extraction": "content_retrieval_and_curation",
    "information_retrieval": "content_retrieval_and_curation",
    "information_retrieval_assistance": "content_retrieval_and_curation",
    "information_retrieval_support": "content_retrieval_and_curation",
    "information_search": "content_retrieval_and_curation",
    "information_search_support": "content_retrieval_and_curation",
    "instant_information_access": "content_retrieval_and_curation",
    "keyphrase_extraction_support": "content_retrieval_and_curation",
    "keyword_extraction": "content_retrieval_and_curation",
    "keyword_extraction_support": "content_retrieval_and_curation",
    "knowledge_retrieval_assistance": "content_retrieval_and_curation",
    "knowledge_retrieval_support": "content_retrieval_and_curation",
    "learning_material_retrieval": "content_retrieval_and_curation",
    "learning_objective_revision": "content_retrieval_and_curation",
    "multilabel_text_classification": "content_retrieval_and_curation",
    "multimedia_resource_retrieval": "content_retrieval_and_curation",
    "multimodal_content_alignment": "content_retrieval_and_curation",
    "oer_adaptation": "content_retrieval_and_curation",
    "pedagogical_filtering": "content_retrieval_and_curation",
    "personalized_content_sequencing": "content_retrieval_and_curation",
    "personalized_example_retrieval": "content_retrieval_and_curation",
    "reading_list_management": "content_retrieval_and_curation",
    "reading_management_support": "content_retrieval_and_curation",
    "sample_selection_support": "content_retrieval_and_curation",
    "sdg_tagging_automation": "content_retrieval_and_curation",
    "semantic_annotation_generation": "content_retrieval_and_curation",
    "semantic_search_support": "content_retrieval_and_curation",
    "source_integration_support": "content_retrieval_and_curation",
    "supplementary_learning_resource": "content_retrieval_and_curation",
    "testimony_preservation": "content_retrieval_and_curation",
    "topic_selection_support": "content_retrieval_and_curation",
    "vocabulary_lookup": "content_retrieval_and_curation",
    "writing_category_coding": "content_retrieval_and_curation",
    # content_delivery merge, added on top of the content_curation batch above
    "ai_content_delivery": "content_retrieval_and_curation",
    "ai_enhanced_textbook": "content_retrieval_and_curation",
    "ai_gamified_learning": "content_retrieval_and_curation",
    "ai_information_source": "content_retrieval_and_curation",
    "ai_voice_synthesis": "content_retrieval_and_curation",
    "anatomy_image_display": "content_retrieval_and_curation",
    "animated_content_delivery": "content_retrieval_and_curation",
    "ar_visualization": "content_retrieval_and_curation",
    "art_content_recommendation": "content_retrieval_and_curation",
    "automated_content_delivery": "content_retrieval_and_curation",
    "automated_instruction_delivery": "content_retrieval_and_curation",
    "content_delivery": "content_retrieval_and_curation",
    "content_delivery_support": "content_retrieval_and_curation",
    "content_provision": "content_retrieval_and_curation",
    "definition_lookup_support": "content_retrieval_and_curation",
    "e_textbook_content_delivery": "content_retrieval_and_curation",
    "health_information_provision": "content_retrieval_and_curation",
    "historical_content_provision": "content_retrieval_and_curation",
    "information_provision": "content_retrieval_and_curation",
    "information_provision_support": "content_retrieval_and_curation",
    "knowledge_presentation": "content_retrieval_and_curation",
    "offline_ai_content_support": "content_retrieval_and_curation",
    "personalized_content_delivery": "content_retrieval_and_curation",
    "political_information_provision": "content_retrieval_and_curation",
    "practice_problem_provision": "content_retrieval_and_curation",
    "resource_sharing_support": "content_retrieval_and_curation",
    "smart_content_provision": "content_retrieval_and_curation",
    "task_instruction_translation": "content_retrieval_and_curation",
    "text_to_speech_synthesis": "content_retrieval_and_curation",
    "virtual_learning_environment": "content_retrieval_and_curation",
    "virtual_learning_environment_integration": "content_retrieval_and_curation",
    "voice_synthesis": "content_retrieval_and_curation",
}

# Step 6: final small cluster merges
STEP6_MOVES = {
    "artwork_classification_support": "creative_design_support",
    "lifelong_learning_platform": "tutoring_and_learning_support",
}

# Order of changes
CORRECTION_STEPS = [
    ("Step 1", STEP1_MOVES),
    ("Step 2", STEP2_MOVES),
    ("Step 3", STEP3_MOVES),
    ("Step 4", STEP4_MOVES),
    ("Step 5", STEP5_MOVES),
    ("Step 6", STEP6_MOVES),
]

# Correct Lookup Dic.: Every step's moves merged
LABEL_CORRECTIONS = {}
for _step_name, _moves in CORRECTION_STEPS:
    LABEL_CORRECTIONS.update(_moves)

def main():
    print("Loading corpus...")
    rows, fieldnames, delim = load_csv(INPUT_FILE)
    print(f"Rows loaded: {len(rows)}")

    for step_name, moves in CORRECTION_STEPS:
        print(f"\nApplying {step_name}...")
        apply_label_moves(rows, moves, step_name)

    print("\nSaving output...")
    save_csv(rows, fieldnames, OUTPUT_FILE, delim)
    print(f"Output saved: {OUTPUT_FILE}")
    print("\nDone.")


if __name__ == "__main__":
    main()
