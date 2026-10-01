"""
The script reads corpus_final_corrected.csv and metadata.csv to generate supporting tables and figures.

Data Output is structured and processed as follows:
1. corpus_final.csv is the raw, unedited final output
2. corpus_final_corrected.csv is the adjusted output with the script "apply_corrections.py"
3. manually created Cluster Tree. This produces:
   cluster_tree_distribution.svg and aied_landscape_overview.csv
"""

from __future__ import annotations
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from apply_corrections import LABEL_CORRECTIONS

# Chart Syntax
plt.rcParams["font.family"] = "Times New Roman"
TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10
CHART_DPI = 300

ROOT = Path(__file__).parent.parent

CORPUS_CSV = ROOT / "data" / "output" / "corpus_final_corrected.csv"
METADATA_CSV = ROOT / "data" / "output" / "metadata.csv"
TABLES_DIR = ROOT / "data" / "output" / "tables"
CHARTS_DIR = ROOT / "data" / "output" / "charts"

SEED_TAXONOMY = {
    "tutoring_and_learning_support",
    "writing_and_feedback_support",
    "assessment_and_grading",
    "content_and_curriculum_generation",
    "chatbot_and_conversational_learning",
    "academic_integrity_and_monitoring",
    "learning_analytics_and_student_modeling",
    "teacher_and_instructor_support",
    "administrative_and_institutional_support",
    "accessibility_and_inclusive_learning",
    "simulation_and_experiential_learning",
}

# Cluster Tree Hierarchy: Root cluster with sub-category and leaf clusters
# Every leaf/subcategory name must match an actual cluster in use_case_clusters.
TAXONOMY_TREE = {
    "tutoring_and_learning_support": {
        "content_personalization": {},
        "collaborative_learning_support": {},
        "learner_autonomy_support": {"goal_setting_support": {}},
        "software_development_support": {},
        "ai_literacy_support": {},
        "learning_activity_support": {},
        "caregiver_support": {},
        "expert_system": {},
        "creative_design_support": {},
    },
    "content_and_curriculum_generation": {
        "video_content_processing": {},
        "music_content_analysis": {},
        "content_evaluation": {},
    },
    "teacher_and_instructor_support": {},
    "assessment_and_grading": {
        "feedback_support": {
            "visual_feedback_support": {},
            "audio_feedback_support": {},
            "music_performance_feedback": {},
            "pronunciation_feedback": {},
        },
    },
    "writing_and_feedback_support": {},
    "learning_analytics_and_student_modeling": {
        "affective_state_monitoring": {"wellbeing_support": {}},
        "engagement_support": {},
        "reflection_support": {},
        "perspective_taking_support": {},
        "knowledge_analytics_support": {},
        "explainability_support": {},
        "knowledge_graph_support": {},
    },
    "administrative_and_institutional_support": {
        "academic_integrity_and_monitoring": {"content_moderation": {}},
        "ethical_education_support": {},
        "bias_detection_and_mitigation": {},
        "career_guidance_support": {},
        "decision_support": {},
        "cybersecurity_threat_detection": {},
        "structure_classification": {},
    },
    "chatbot_and_conversational_learning": {
        "chatbot_development": {},
    },
    "content_retrieval_and_curation": {},
    "research_support": {
        "data_augmentation": {},
    },
    "accessibility_and_inclusive_learning": {},
    "simulation_and_experiential_learning": {},
    "other": {
        "computer_vision_analysis": {},
        "scientific_property_prediction": {},
    },
    "no_use_case": {},
}


# Maps each node to its list of parent clusters.
# The tree is nested, therefore this method finds any parent node.
def flatten_taxonomy_tree(tree, path=()):
    flat = {}
    for name, children in tree.items():
        flat[name] = path
        flat.update(flatten_taxonomy_tree(children, path + (name,)))
    return flat

# Builds the Cluster Tree based on flatten_taxonomy_tree()
# Takes the cluster assignments and assigns papers to the root clusters from the Cluster Tree
def resolve_tree_clusters(df: pd.DataFrame) -> pd.Series:
    tree_map = {node: (path[0] if path else node) for node, path in flatten_taxonomy_tree(TAXONOMY_TREE).items()}
    return df["use_case_clusters"].apply(lambda clusters: sorted({tree_map.get(c, c) for c in clusters}))


def load_corpus() -> pd.DataFrame:
    df = pd.read_csv(CORPUS_CSV, sep=None, engine="python")
    for col in ("use_cases", "use_case_clusters"):
        df[col] = df[col].apply(
            lambda x: [v.lower() for v in json.loads(x)] if isinstance(x, str) else []
        )
    before = len(df)
    df = df[~df["use_case_clusters"].apply(lambda c: c == ["no_use_case"])].reset_index(drop=True)
    print(f"  Excluded {before - len(df)} no_use_case papers, {len(df)} remaining")
    return df


def save_table(df: pd.DataFrame, filename: str):
    path = TABLES_DIR / filename
    df.to_csv(path, index=False)
    print(f"  Table: {filename} ({len(df)} rows)")


def save_chart(filename: str):
    path = CHARTS_DIR / filename
    plt.savefig(path, bbox_inches="tight", dpi=CHART_DPI)
    plt.close()
    print(f"  Chart: {filename}")


def bar_chart(df, label_col, title, filename, color="#4C72B0", width=10.0, height=None):
    h = height if height is not None else max(4.0, len(df) * 0.5)
    fig, ax = plt.subplots(figsize=(width, h))
    bars = ax.barh(df[label_col], df["papers"], color=color)
    ax.set_xlabel("Number of Papers", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="normal")
    ax.tick_params(labelsize=TICK_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if "percentage" in df.columns:
        for bar, pct in zip(bars, df["percentage"]):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{pct}%", va="center", fontsize=TICK_SIZE)
    plt.tight_layout()
    save_chart(filename)


def png_bar_chart(counts, label_col, title, filename, color):
    ordered = counts.sort_values("count", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, len(ordered) * 0.4)))
    ax.barh(ordered[label_col], ordered["count"], color=color)
    ax.set_xlabel("Number of Papers", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="normal")
    ax.tick_params(labelsize=TICK_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = CHARTS_DIR / filename
    plt.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Chart: {filename}")


# 1. corpus_final_corrected.csv
def year_distribution(df: pd.DataFrame):
    counts = df["Year"].dropna().astype(int).value_counts().sort_index()
    result = counts.reset_index()
    result.columns = ["year", "papers"]
    result["papers"] = result["papers"].astype(int)
    result["percentage"] = (result["papers"] / result["papers"].sum() * 100).round(1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(result["year"].astype(str), result["papers"], color="#4C72B0")
    ax.set_xlabel("Publication Year", fontsize=LABEL_SIZE)
    ax.set_ylabel("Number of Papers", fontsize=LABEL_SIZE)
    ax.set_title("Temporal Distribution of AIED", fontsize=TITLE_SIZE, fontweight="normal")
    ax.tick_params(labelsize=TICK_SIZE)
    for i, row in result.iterrows():
        ax.text(i, row["papers"] + 15, str(int(row["papers"])), ha="center", fontsize=TICK_SIZE)
    plt.tight_layout()
    save_chart("temporal_distribution.svg")


def publication_type_distribution(df: pd.DataFrame):
    counts = df["Publication Type"].dropna().value_counts()
    result = counts.reset_index()
    result.columns = ["publication_type", "papers"]
    result["percentage"] = (result["papers"] / counts.sum() * 100).round(1)

    bar_chart(result.sort_values("papers", ascending=True),
              "publication_type", "Publication Type Distribution", "publication_type.svg")


def cluster_frequency(df: pd.DataFrame):
    counts = Counter()
    for clusters in df["use_case_clusters"]:
        counts.update(clusters)

    total_papers = len(df)
    result = pd.DataFrame(
        [{"cluster": c, "papers": n, "percentage": round(n / total_papers * 100, 1)}
         for c, n in counts.most_common()]
    )
    result["is_seed"] = result["cluster"].apply(lambda x: x in SEED_TAXONOMY)

    seed_df = result[result["is_seed"]].copy()
    seed_df = seed_df.rename(columns={"cluster": "seed_cluster"})
    seed_df = seed_df[["seed_cluster", "papers", "percentage"]].reset_index(drop=True)

    bar_chart(seed_df.sort_values("papers", ascending=True), "seed_cluster",
              "Seed Taxonomy Distribution",
              "cluster_frequency_seed.svg", color="#4C72B0", height=5.0)

    novel_df = result[~result["is_seed"]].copy()
    novel_df = novel_df.rename(columns={"cluster": "novel_cluster"})
    novel_df = novel_df[["novel_cluster", "papers", "percentage"]].reset_index(drop=True)

    novel_top10 = novel_df.head(10)
    bar_chart(novel_top10.sort_values("papers", ascending=True), "novel_cluster",
              "Top 10 Novel Clusters",
              "cluster_frequency_novel_top10.svg", color="#4C72B0", height=5.0)


def top_use_cases(df: pd.DataFrame):
    counts = Counter()
    for labels in df["use_cases"]:
        counts.update(labels)

    top25 = pd.DataFrame(
        [{"use_case": l, "papers": n} for l, n in counts.most_common(25)]
    ).sort_values("papers", ascending=True)
    bar_chart(top25, "use_case", "Top 25 Use Case labels",
              "top_use_cases_25.svg", width=14.0, height=9.0)


WORDCLOUD_TOP_N = 50
WORDCLOUD_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]


def wordcloud_color(word, font_size, position, orientation, random_state=None, **kwargs):
    return random.choice(WORDCLOUD_COLORS)


def use_case_wordcloud(df: pd.DataFrame):
    counts = Counter()
    for labels in df["use_cases"]:
        counts.update(labels)

    top = counts.most_common(WORDCLOUD_TOP_N)
    print(f"  {sum(counts.values())} label instances, {sum(n for _, n in top)} shown in top {WORDCLOUD_TOP_N}")
    save_table(pd.DataFrame(top, columns=["label", "count"]), "use_case_wordcloud_frequencies.csv")

    # Scale word size by the square root of the count for better visual separation
    sizes = {label.replace("_", " "): math.sqrt(count) for label, count in top}

    random.seed(42)  # keep the colours identical between runs
    cloud = WordCloud(
        width=2400,
        height=1200,
        background_color="white",
        color_func=wordcloud_color,
        prefer_horizontal=1.0,
        margin=8,
        max_font_size=90,
        random_state=42,
    ).generate_from_frequencies(sizes)

    fig, ax = plt.subplots(figsize=(24, 12))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout()
    path = CHARTS_DIR / "use_case_wordcloud.png"
    plt.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Chart: {path.name}")


def multi_label_distribution(df: pd.DataFrame, column: str, label: str, stem: str, title: str = None):
    counts = df[column].apply(len)
    dist = counts.value_counts().sort_index().reset_index()
    dist.columns = ["cluster_count", "papers"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(dist["cluster_count"].astype(str), dist["papers"], color="#4C72B0")
    ax.set_xlabel(f"Number of {label} Assigned per Paper", fontsize=LABEL_SIZE)
    ax.set_ylabel("Number of Papers", fontsize=LABEL_SIZE)
    ax.set_title(title or f"Multi-Label Distribution of {label} Assigned per Paper", fontsize=TITLE_SIZE, fontweight="normal")
    ax.tick_params(labelsize=TICK_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, count in zip(bars, dist["papers"]):
        pct = round(count / len(df) * 100, 1)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f"{int(count)}\n({pct}%)", ha="center", fontsize=TICK_SIZE)
    plt.tight_layout()
    save_chart(f"{stem}.svg")

# 2. metadata.csv
def metadata_analysis(df: pd.DataFrame, meta):
    merged = df.merge(meta[["paper_id", "educational_level", "ai_technology"]],
                      on="paper_id", how="left")

    counts = merged["educational_level"].value_counts()
    ed_level = counts.reset_index()
    ed_level.columns = ["educational_level", "papers"]
    ed_level["percentage"] = (ed_level["papers"] / counts.sum() * 100).round(1)
    bar_chart(ed_level.sort_values("papers", ascending=True), "educational_level",
              "Educational Level Distribution", "educational_level.svg", "#4C72B0")

    counts = merged["ai_technology"].value_counts()
    ai_tech = counts.reset_index()
    ai_tech.columns = ["ai_technology", "papers"]
    ai_tech["percentage"] = (ai_tech["papers"] / counts.sum() * 100).round(1)
    bar_chart(ai_tech.sort_values("papers", ascending=True), "ai_technology",
              "AI Technology Distribution", "ai_technology.svg", "#4C72B0")


def subject_domain_distribution(meta):
    total_papers = len(meta)
    missing = meta["subject_domain"].isna().sum()
    print(f"  Papers missing a subject domain value: {missing}")

    counts = meta["subject_domain"].dropna().value_counts().reset_index()
    counts.columns = ["subject_domain", "count"]
    counts["percentage"] = (counts["count"] / total_papers * 100).round(1)

    png_bar_chart(counts, "subject_domain", "Subject Domain Distribution",
                  "subject_domain_distribution.png", "#4C72B0")


def empirical_validation_type_distribution(meta):
    total_papers = len(meta)
    missing = meta["empirical_validation_type"].isna().sum()
    print(f"  Papers missing an empirical validation type value: {missing}")

    counts = meta["empirical_validation_type"].dropna().value_counts().reset_index()
    counts.columns = ["empirical_validation_type", "count"]
    counts["percentage"] = (counts["count"] / total_papers * 100).round(1)

    png_bar_chart(counts, "empirical_validation_type", "Empirical Validation Type Distribution",
                  "empirical_validation_type_distribution.png", "#4C72B0")

# 3. cluster tree restructuring
def cluster_tree_distribution(df: pd.DataFrame):
    total_papers = len(df)
    counts = Counter()
    for clusters in df["tree_clusters"]:
        counts.update(clusters)

    result = pd.DataFrame(
        [{"tree_cluster": c, "papers": n, "percentage": round(n / total_papers * 100, 1)}
         for c, n in counts.most_common()]
    )

    bar_chart(result.sort_values("papers", ascending=True), "tree_cluster",
              "Cluster Distribution",
              "cluster_tree_distribution.svg", color="#4C72B0", height=6.0)


def aied_landscape_overview(df: pd.DataFrame):
    node_path = flatten_taxonomy_tree(TAXONOMY_TREE)
    children_map = defaultdict(list)
    for node, path in node_path.items():
        if path:
            children_map[path[-1]].append(node)

    node_labels = defaultdict(Counter)
    for cell in df["use_case_label_cluster_pairs"].dropna():
        try:
            pairs = json.loads(cell)
        except (json.JSONDecodeError, TypeError):
            continue
        for pair in pairs:
            label = pair["label"].lower()
            cluster = LABEL_CORRECTIONS.get(label, pair["cluster"].lower())
            node_labels[cluster][label] += 1

    def subtree_labels(node):
        labels = Counter(node_labels.get(node, {}))
        for child in children_map.get(node, []):
            labels.update(subtree_labels(child))
        return labels

    total_papers = len(df)
    counts = Counter()
    for clusters in df["tree_clusters"]:
        counts.update(clusters)

    rows = []
    for cluster, papers in counts.most_common():
        top5 = subtree_labels(cluster).most_common(5)
        top5_str = "; ".join(f"{label} ({round(n / papers * 100, 1)}%)" for label, n in top5)
        rows.append({
            "cluster": cluster,
            "papers": papers,
            "percent_of_corpus": round(papers / total_papers * 100, 1),
            "top_5_use_case_labels": top5_str,
        })

    save_table(pd.DataFrame(rows), "aied_landscape_overview.csv")


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading corpus from {CORPUS_CSV.name}")
    df = load_corpus()
    df["tree_clusters"] = resolve_tree_clusters(df)
    meta = pd.read_csv(METADATA_CSV) if METADATA_CSV.exists() else None
    if meta is not None:
        # keep only the papers the analysis actually covers, so the metadata figures
        # use the same corpus as every other figure
        meta = meta[meta["paper_id"].isin(df["paper_id"])]
        for col in ("ai_technology", "subject_domain"):
            meta[col] = meta[col].str.lower().str.strip()
    print(f"Loaded {len(df)} papers")
    if meta is not None:
        print(f"Loaded metadata for {len(meta)} papers")
    else:
        print("No metadata.csv found, run step 4 of the pipeline to get educational level and AI technology")
    print()

    # 1. corpus_final_corrected.csv
    print("Year distribution")
    year_distribution(df)

    print("\nPublication type")
    publication_type_distribution(df)

    print("\nCluster frequency")
    cluster_frequency(df)

    print("\nTop use cases")
    top_use_cases(df)

    print("\nWord cloud")
    use_case_wordcloud(df)

    print("\nMulti-label distribution")
    multi_label_distribution(df, "use_case_clusters", "Clusters", "multi_label_distribution",
                             title="Multi-label Distribution")

    # 2. metadata.csv
    if meta is not None:
        print("\nMetadata analysis")
        metadata_analysis(df, meta)

        print("\nSubject domain distribution")
        subject_domain_distribution(meta)

        print("\nEmpirical validation type distribution")
        empirical_validation_type_distribution(meta)

    # 3. cluster tree restructuring
    print("\nCluster tree distribution")
    cluster_tree_distribution(df)

    print("\nAIED landscape overview")
    aied_landscape_overview(df)


if __name__ == "__main__":
    main()
