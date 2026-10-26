"""
Main screening pipeline for the AI in education literature review.
The pipeline consists of  four steps:
1) Screening: Literature screening based on inclusion and exlusion criteria
2) Use-Case-labeling: Use-case labeling
3) Clustering: Use-case labels are clustered to the seed taxonomy and novel clusters
4) Meta-data Extraction: Extracting metadata
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
import pandas as pd

def _load_dotenv():
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
_load_dotenv()

# Add the project root to the path to find the utils folder
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_client import LLMClient, LLMError
from utils import checkpoint

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config" / "llm_config.json"
PROMPTS_DIR = ROOT / "prompts"
OUTPUT_DIR = ROOT / "data" / "output"

INCLUDED_CSV  = OUTPUT_DIR / "included.csv"
EXCLUDED_CSV  = OUTPUT_DIR / "excluded.csv"
LABELED_CSV   = OUTPUT_DIR / "labeled.csv"
TAXONOMY_JSON = OUTPUT_DIR / "taxonomy_mapping.json"
FINAL_CSV     = OUTPUT_DIR / "corpus_final.csv"
FINAL_JSON    = OUTPUT_DIR / "corpus_final.json"
METADATA_CSV  = OUTPUT_DIR / "metadata.csv"

VALID_DECISIONS = {"INCLUDE", "EXCLUDE"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    if not path.exists():
        sys.exit(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def load_input(input_file: str, limit: int | None) -> pd.DataFrame:
    path = Path(input_file)
    if not path.exists():
        sys.exit(f"Input file not found: {path}")

    # detect comma vs semicolon delimiter
    sep = ","
    with path.open(encoding="utf-8") as f:
        first_line = f.readline()
    if first_line.count(";") > first_line.count(","):
        sep = ";"

    df = pd.read_csv(path, sep=sep, quotechar='"', on_bad_lines="skip")

    for col in ("Title", "Abstract"):
        if col not in df.columns:
            sys.exit(f"Input CSV is missing required column: {col}")

    before = len(df)
    df = df.dropna(subset=["Title", "Abstract"])
    df = df[df["Title"].str.strip().ne("") & df["Abstract"].str.strip().ne("")]
    dropped = before - len(df)
    if dropped:
        log.warning("Dropped %d rows with empty Title or Abstract", dropped)

    if df.empty:
        sys.exit("Input File is empty")

    if limit:
        df = df.head(limit)
        log.info("Limiting to first %d papers", limit)

    # use existing paper_id or assign from index
    if "paper_id" not in df.columns:
        df = df.reset_index(drop=True)
        df.insert(0, "paper_id", df.index)
    else:
        df["paper_id"] = df["paper_id"].astype(int)
        df = df.reset_index(drop=True)

    log.info("Loaded %d papers from %s", len(df), path)
    return df


def batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def normalise_labels(df):
    freq = Counter()
    for cell in df["use_cases"].dropna():
        try:
            freq.update(json.loads(cell))
        except json.JSONDecodeError:
            pass

    by_words = {}
    for label in freq:
        key = frozenset(label.split("_"))
        if key not in by_words or freq[label] > freq[by_words[key]]:
            by_words[key] = label

    remap = {
        label: by_words[frozenset(label.split("_"))]
        for label in freq
        if label != by_words[frozenset(label.split("_"))]
    }

    if not remap:
        return df

    def fix(cell):
        try:
            labels = json.loads(cell)
            return json.dumps([remap.get(label, label) for label in labels], ensure_ascii=True)
        except (json.JSONDecodeError, TypeError):
            return cell

    df = df.copy()
    df["use_cases"] = df["use_cases"].apply(fix)
    log.info("Label normalisation: consolidated %d word-order variant(s)", len(remap))
    for orig, norm in list(remap.items())[:10]:
        log.info("  %s → %s", orig, norm)
    return df


def run_screening(df: pd.DataFrame, client: LLMClient, batch_size: int, resume: bool) -> dict:
    log.info("Step 1: screening %d papers", len(df))
    tmpl = load_prompt("screening_prompt.txt")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    done: dict[int, dict] = {}
    if resume:
        for r in checkpoint.load():
            done[r["paper_id"]] = r

    papers = df.to_dict(orient="records")
    remaining = [p for p in papers if p["paper_id"] not in done]
    total_batches = -(-len(remaining) // batch_size)  # ceiling division
    log.info("%d already done, %d remaining (%d batches)", len(done), len(remaining), total_batches)

    for batch_num, batch in enumerate(batches(remaining, batch_size), start=1):
        batch_input = [
            {"paper_id": p["paper_id"], "title": p["Title"], "abstract": p["Abstract"]}
            for p in batch
        ]
        prompt = tmpl.replace("{papers_json}", json.dumps(batch_input, ensure_ascii=True))

        log.info("Screening batch %d/%d (papers %s)", batch_num, total_batches, [p["paper_id"] for p in batch])
        raw = client.call(prompt)
        results = client.parse_json_array(raw)

        for item in results:
            pid = item.get("paper_id")
            if pid is None:
                log.warning("Result missing paper_id, skipping: %s", item)
                continue
            done[int(pid)] = item

        checkpoint.save(list(done.values()))

    by_id = {r["paper_id"]: r for r in done.values()}
    rows = []
    for p in papers:
        r = by_id.get(p["paper_id"], {})
        row = dict(p)
        decision = r.get("screening_decision")
        if isinstance(decision, str):
            decision = decision.strip().upper()
        if decision not in VALID_DECISIONS:
            row["screening_decision"] = "ERROR"
            row["screening_error"] = f"Invalid decision value: {decision}"
        else:
            row["screening_decision"] = decision
        row["fit_score"] = r.get("fit_score", "")
        row["fit_justification"] = r.get("fit_justification", "")
        rows.append(row)

    result_df = pd.DataFrame(rows)
    included = result_df[result_df["screening_decision"] == "INCLUDE"].copy()
    excluded = result_df[result_df["screening_decision"] == "EXCLUDE"].copy()
    errors = result_df[result_df["screening_decision"] == "ERROR"].copy()

    included.to_csv(INCLUDED_CSV, index=False)
    excluded.to_csv(EXCLUDED_CSV, index=False)
    if not errors.empty:
        errors.to_csv(OUTPUT_DIR / "screening_errors.csv", index=False)
    checkpoint.clear()

    total = len(included) + len(excluded) + len(errors)
    rate = (len(included) / total * 100) if total else 0
    if not errors.empty:
        log.warning("Step 1 flagged %d papers as ERROR, see screening_errors.csv", len(errors))
    log.info(
        "Step 1 done: %d included, %d excluded, %d errors, %.0f%% inclusion rate",
        len(included), len(excluded), len(errors), rate,
    )
    return {"included": len(included), "excluded": len(excluded), "errors": len(errors), "total": total}


def run_labeling(client: LLMClient, batch_size: int, resume: bool) -> dict:
    if not INCLUDED_CSV.exists():
        sys.exit(f"Missing input for Step 2: {INCLUDED_CSV}")

    df = pd.read_csv(INCLUDED_CSV)
    log.info("Step 2: labeling %d papers", len(df))
    prompt_template = load_prompt("use_case_labeling_prompt.txt")

    done = {}
    if resume:
        for r in checkpoint.load():
            done[r["paper_id"]] = r

    papers = df.to_dict(orient="records")
    remaining = [p for p in papers if int(p["paper_id"]) not in done]
    total_batches = -(-len(remaining) // batch_size)
    log.info("%d already done, %d remaining (%d batches)", len(done), len(remaining), total_batches)

    for batch_num, batch in enumerate(batches(remaining, batch_size), start=1):
        batch_input = [
            {"paper_id": int(p["paper_id"]), "title": p["Title"], "abstract": p["Abstract"]}
            for p in batch
        ]
        prompt = prompt_template.replace("{papers_json}", json.dumps(batch_input, ensure_ascii=True))

        log.info("Labeling batch %d/%d (papers %s)", batch_num, total_batches, [p["paper_id"] for p in batch])
        raw = client.call(prompt)
        results = client.parse_json_array(raw)

        for item in results:
            pid = item.get("paper_id")
            if pid is None:
                log.warning("Result missing paper_id, skipping: %s", item)
                continue
            done[int(pid)] = item

        checkpoint.save(list(done.values()))


    labeling_map = {r["paper_id"]: r for r in done.values()}
    rows = []
    for p in papers:
        r = labeling_map.get(int(p["paper_id"]), {})
        row = dict(p)
        row["use_cases"] = json.dumps(r.get("use_cases", []), ensure_ascii=True)
        row["labeling_fit_justification"] = r.get("fit_justification", "")
        rows.append(row)

    labeled_df = pd.DataFrame(rows)
    labeled_df = normalise_labels(labeled_df)
    labeled_df.to_csv(LABELED_CSV, index=False)
    checkpoint.clear()

    for _, row in labeled_df.iterrows():
        try:
            labels = json.loads(row["use_cases"])
        except (json.JSONDecodeError, TypeError):
            labels = []
        if not labels:
            log.warning("Paper %s got no use-case labels, check model response", row["paper_id"])

    unique_labels = set()
    for cell in labeled_df["use_cases"].dropna():
        try:
            unique_labels.update(json.loads(cell))
        except json.JSONDecodeError:
            pass

    log.info(
        "Step 2 done: %d papers labeled, %d unique use-case labels",
        len(labeled_df), len(unique_labels),
    )
    return {"labeled": len(labeled_df), "unique_labels": len(unique_labels)}


def consolidate_clusters(mapping: list) -> list:
    # merges duplicate cluster names that differ only by case or minor word overlap
    cluster_counts = Counter(m["cluster"] for m in mapping)

    norm_map = {}
    for cluster in cluster_counts:
        norm = cluster.lower().strip()
        if norm not in norm_map:
            norm_map[norm] = cluster
        elif cluster_counts[cluster] > cluster_counts[norm_map[norm]]:
            norm_map[norm] = cluster

    # word-overlap merge: 80% shared words
    names = list(set(norm_map.values()))
    word_sets = {c: set(c.split("_")) for c in names}
    merge_map = {}
    for i, c1 in enumerate(names):
        if c1 in merge_map:
            continue
        for c2 in names[i + 1:]:
            if c2 in merge_map:
                continue
            w1, w2 = word_sets[c1], word_sets[c2]
            overlap = len(w1 & w2) / max(len(w1 | w2), 1)
            if overlap >= 0.65:
                keep = c1 if cluster_counts[c1] >= cluster_counts[c2] else c2
                drop = c2 if keep == c1 else c1
                merge_map[drop] = keep

    def resolve(cluster: str) -> str:
        canon = norm_map.get(cluster.lower().strip(), cluster)
        return merge_map.get(canon, canon)

    consolidated = [{**m, "cluster": resolve(m["cluster"])} for m in mapping]
    n_merged = sum(1 for a, b in zip(mapping, consolidated) if a["cluster"] != b["cluster"])
    if n_merged:
        log.info("Consolidation: merged %d duplicate cluster names", n_merged)
    return consolidated


def run_clustering(client: LLMClient, label_batch_size: int) -> dict:
    if not LABELED_CSV.exists():
        sys.exit(f"Missing input for Step 3: {LABELED_CSV}")

    df = pd.read_csv(LABELED_CSV)
    log.info("Step 3: clustering labels")
    prompt_template = load_prompt("clustering_prompt.txt")

    all_labels = []
    for cell in df["use_cases"].dropna():
        try:
            labels = json.loads(cell)
            if isinstance(labels, list):
                all_labels.extend(labels)
        except json.JSONDecodeError:
            pass

    unique_labels = sorted(set(all_labels), key=lambda label: (label.split("_")[0], label))
    log.info("Unique raw labels to cluster: %d", len(unique_labels))

    # cluster in batches to avoid time out
    mapping = []
    seen_novel = []
    total_batches = -(-len(unique_labels) // label_batch_size)


    for batch_num, label_batch in enumerate(batches(unique_labels, label_batch_size), start=1):
        prompt = prompt_template.replace(
            "{raw_labels_json}", json.dumps(label_batch, ensure_ascii=True)
        ).replace(
            "{existing_clusters_json}", json.dumps(seen_novel, ensure_ascii=True)
        )
        log.info("Clustering batch %d/%d (%d labels, %d known novel clusters)",
                 batch_num, total_batches, len(label_batch), len(seen_novel))
        raw = client.call(prompt)
        result = client.parse_json_array(raw)

        for item in result:
            if not isinstance(item, dict) or "raw_label" not in item or "cluster" not in item:
                raise LLMError(f"Malformed clustering response item: {item}")

        mapping.extend(result)
        seen_novel = sorted(set(
            m["cluster"] for m in mapping if m.get("is_novel_cluster")
        ))

    if not mapping:
        raise LLMError("Clustering produced an empty mapping")

    mapping = consolidate_clusters(mapping)

    TAXONOMY_JSON.parent.mkdir(parents=True, exist_ok=True)
    TAXONOMY_JSON.write_text(
        json.dumps(mapping, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    log.info("Taxonomy mapping saved → %s", TAXONOMY_JSON.name)

    novel = [m for m in mapping if m.get("is_novel_cluster")]
    if novel:
        log.info("%d novel clusters created: %s", len(novel), [m["cluster"] for m in novel])

    lut = {m["raw_label"].lower(): m["cluster"] for m in mapping}


    def resolve_clusters(cell: str) -> str:
        try:
            labels = json.loads(cell)
            clusters = list({lut.get(l.lower(), "unmapped") for l in labels})
            return json.dumps(sorted(clusters), ensure_ascii=True)
        except Exception:
            return "[]"

    def pair_labels(cell: str) -> str:
        try:
            labels = json.loads(cell)
            pairs = [{"label": l, "cluster": lut.get(l.lower(), "unmapped")} for l in labels]
            return json.dumps(pairs, ensure_ascii=True)
        except Exception:
            return "[]"

    df["use_case_clusters"] = df["use_cases"].apply(resolve_clusters)
    df["use_case_label_cluster_pairs"] = df["use_cases"].apply(pair_labels)
    df.to_csv(FINAL_CSV, index=False)

    records = []
    for row in df.where(pd.notna(df), None).to_dict(orient="records"):
        try:
            use_cases = json.loads(row.get("use_cases") or "[]")
        except (json.JSONDecodeError, TypeError):
            use_cases = []
        try:
            use_case_clusters = json.loads(row.get("use_case_clusters") or "[]")
        except (json.JSONDecodeError, TypeError):
            use_case_clusters = []
        try:
            use_case_label_cluster_pairs = json.loads(row.get("use_case_label_cluster_pairs") or "[]")
        except (json.JSONDecodeError, TypeError):
            use_case_label_cluster_pairs = []
        records.append({
            "paper_id": int(row["paper_id"]),
            "title": row.get("Title") or "",
            "authors": row.get("Authors") or "",
            "year": row.get("Year"),
            "journal": row.get("Journal") or "",
            "publication_type": row.get("Publication Type") or "",
            "screening_decision": row.get("screening_decision") or "",
            "fit_score": row.get("fit_score"),
            "fit_justification": row.get("fit_justification") or "",
            "use_cases": use_cases,
            "use_case_clusters": use_case_clusters,
            "use_case_label_cluster_pairs": use_case_label_cluster_pairs,
            "labeling_fit_justification": row.get("labeling_fit_justification") or "",
        })
    FINAL_JSON.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Final corpus saved → %s and %s", FINAL_CSV.name, FINAL_JSON.name)

    all_clusters = {m["cluster"] for m in mapping}
    novel_clusters = [m["cluster"] for m in mapping if m.get("is_novel_cluster")]

    log.info(
        "Step 3 done: %d labels mapped to %d clusters, %d novel",
        len(unique_labels), len(all_clusters), len(novel_clusters),
    )
    if novel_clusters:
        log.info("Novel clusters: %s", novel_clusters)
    return {"unique_labels": len(unique_labels), "clusters": len(all_clusters), "novel": len(novel_clusters)}


# Step 4: extract per-paper metadata (educational level, subject domain, AI technology)
def run_metadata_extraction(client: LLMClient, batch_size: int, limit: int | None, resume: bool):
    df = pd.read_csv(FINAL_CSV)
    if limit:
        df = df.head(limit)
        log.info("Limiting to first %d papers", limit)

    log.info("Step 4: extracting metadata for %d papers", len(df))
    prompt_template = load_prompt("metadata_prompt.txt")

    done = {}
    if resume:
        for r in checkpoint.load():
            done[r["paper_id"]] = r
        log.info("Checkpoint loaded: %d records", len(done))

    papers = df.to_dict(orient="records")
    remaining = [p for p in papers if int(p["paper_id"]) not in done]
    total_batches = -(-len(remaining) // batch_size)
    log.info("%d already done, %d remaining (%d batches)", len(done), len(remaining), total_batches)

    for batch_num, batch in enumerate(batches(remaining, batch_size), start=1):
        batch_input = [
            {"paper_id": int(p["paper_id"]), "title": p["Title"], "abstract": p["Abstract"]}
            for p in batch
        ]
        prompt = prompt_template.replace("{papers_json}", json.dumps(batch_input, ensure_ascii=True))


        log.info("Batch %d/%d (papers %s)", batch_num, total_batches, [p["paper_id"] for p in batch_input])
        raw = client.call(prompt)
        results = client.parse_json_array(raw)

        for item in results:
            pid = item.get("paper_id")
            if pid is None:
                log.warning("Result missing paper_id, skipping: %s", item)
                continue
            done[int(pid)] = item

        checkpoint.save(list(done.values()))

    metadata_map = {r["paper_id"]: r for r in done.values()}
    rows = []
    for p in papers:
        pid = int(p["paper_id"])
        r = metadata_map.get(pid, {})
        rows.append({
            "paper_id": pid,
            "Title": p["Title"],
            "Abstract": p.get("Abstract", ""),
            "Year": p.get("Year", ""),
            "educational_level": r.get("educational_level", "not_specified"),
            "subject_domain": r.get("subject_domain", "not_specified"),
            "ai_technology": r.get("ai_technology", "not_specified"),
            "empirical_validation": r.get("empirical_validation", "unclear"),
            "empirical_validation_type": r.get("empirical_validation_type", "unclear"),
            "empirical_validation_justification": r.get("empirical_validation_justification", ""),
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(METADATA_CSV, index=False)
    checkpoint.clear()

    log.info("Step 4 done: metadata saved for %d papers to %s", len(result_df), METADATA_CSV.name)
    log.info("Educational levels: %s", result_df["educational_level"].value_counts().to_dict())
    log.info("AI technologies: %s", result_df["ai_technology"].value_counts().to_dict())
    log.info("Top subject domains: %s", result_df["subject_domain"].value_counts().head(10).to_dict())
    log.info("Empirical validation: %s", result_df["empirical_validation"].value_counts().to_dict())
    log.info("Empirical validation types: %s", result_df["empirical_validation_type"].value_counts().to_dict())


def main():
    parser = argparse.ArgumentParser(description="AI literature screening pipeline")
    parser.add_argument("--input-file", required=True, help="Path to input CSV (Title, Abstract)")
    parser.add_argument("--batch-size", type=int, default=5, help="Papers per API call for steps 1, 2, 4 (default 5)")
    parser.add_argument("--label-batch-size", type=int, default=20, help="Labels per API call for step 3 clustering (default 20)")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of input papers (for testing)")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing outputs and rerun from Step 1")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted step from checkpoint")
    parser.add_argument("--stop-after", type=int, default=3, choices=[1, 2, 3, 4], help="Stop after step 1, 2, 3, or 4 (metadata extraction)")
    args = parser.parse_args()
    batch_size = max(1, args.batch_size)
    label_batch_size = max(1, args.label_batch_size)

    api_key = os.environ.get("KICONNECT_API_KEY", "").strip()
    if not api_key:
        sys.exit(
            "KICONNECT_API_KEY is not set.\n"
            "Export it in your shell:  export KICONNECT_API_KEY=your_key_here\n"
            "Or add it to a .env file and source it."
        )

    if not CONFIG_PATH.exists():
        sys.exit(f"Config not found: {CONFIG_PATH}")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    client = LLMClient(config, api_key)

    if args.fresh:
        log.info("--fresh: removing existing outputs")
        for path in (INCLUDED_CSV, EXCLUDED_CSV, LABELED_CSV, TAXONOMY_JSON, FINAL_CSV, METADATA_CSV):
            if path.exists():
                path.unlink()
        checkpoint.clear()

    if FINAL_CSV.exists() and METADATA_CSV.exists() and not args.fresh:
        log.info("corpus_final.csv and metadata.csv already exist - pipeline complete. Use --fresh to rerun.")
        return

    s1_stats = {}
    s3_stats = {}

    if LABELED_CSV.exists() and not args.fresh:
        log.info("labeled.csv exists - skipping Steps 1 and 2")
    elif INCLUDED_CSV.exists() and EXCLUDED_CSV.exists() and not args.fresh:
        log.info("included.csv exists - skipping Step 1")
    else:
        df = load_input(args.input_file, args.limit)
        s1_stats = run_screening(df, client, batch_size, resume=args.resume)

    if args.stop_after == 1:
        return

    if not LABELED_CSV.exists():
        run_labeling(client, batch_size, resume=args.resume)

    if args.stop_after == 2:
        return

    # if taxonomy exists, skip the LLM call and just re-apply the mapping
    if TAXONOMY_JSON.exists() and not args.fresh:
        log.info("Taxonomy mapping already exists, skipping API call and re-applying mapping")
        df = pd.read_csv(LABELED_CSV)
        mapping = json.loads(TAXONOMY_JSON.read_text(encoding="utf-8"))
        for item in mapping:
            if not isinstance(item, dict) or "raw_label" not in item or "cluster" not in item:
                raise LLMError(f"Malformed clustering response item: {item}")
        lut = {m["raw_label"].lower(): m["cluster"] for m in mapping}

        def resolve_clusters(cell: str) -> str:
            try:
                labels = json.loads(cell)
                clusters = list({lut.get(l.lower(), "unmapped") for l in labels})
                return json.dumps(sorted(clusters), ensure_ascii=True)
            except Exception:
                return "[]"

        def pair_labels(cell: str) -> str:
            try:
                labels = json.loads(cell)
                pairs = [{"label": l, "cluster": lut.get(l.lower(), "unmapped")} for l in labels]
                return json.dumps(pairs, ensure_ascii=True)
            except Exception:
                return "[]"

        df["use_case_clusters"] = df["use_cases"].apply(resolve_clusters)
        df["use_case_label_cluster_pairs"] = df["use_cases"].apply(pair_labels)
        df.to_csv(FINAL_CSV, index=False)
        log.info("Final corpus saved → %s", FINAL_CSV.name)
    else:
        s3_stats = run_clustering(client, label_batch_size)

    total = s1_stats.get("total") or (
        len(pd.read_csv(INCLUDED_CSV)) + len(pd.read_csv(EXCLUDED_CSV))
        if INCLUDED_CSV.exists() and EXCLUDED_CSV.exists() else 0
    )
    included = s1_stats.get("included") or (len(pd.read_csv(INCLUDED_CSV)) if INCLUDED_CSV.exists() else 0)
    clusters = s3_stats.get("clusters", 0)
    novel = s3_stats.get("novel", 0)
    rate = (included / total * 100) if total else 0

    log.info(
        "Pipeline done: %d screened, %d included (%.0f%%), %d clusters, %d novel",
        total, included, rate, clusters, novel,
    )

    if args.stop_after == 3:
        return

    if METADATA_CSV.exists() and not args.fresh:
        log.info("metadata.csv exists - skipping Step 4. Use --fresh to rerun.")
    else:
        run_metadata_extraction(client, batch_size, args.limit, args.resume)


if __name__ == "__main__":
    main()
