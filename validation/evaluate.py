"""Evaluate ChromaDB retrieval against labelled SP01 interaction cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import chromadb


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import get_settings  # noqa: E402
from ingest import embed_texts  # noqa: E402


def metadata_matches(metadata: dict, predicate: dict) -> bool:
    return all(metadata.get(key) == value for key, value in predicate.items())


def metrics_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> dict:
    top_ids = retrieved_ids[:k]
    hit_count = sum(chunk_id in relevant_ids for chunk_id in top_ids)
    first_rank = next(
        (rank for rank, chunk_id in enumerate(top_ids, start=1) if chunk_id in relevant_ids),
        None,
    )
    return {
        "hit": int(hit_count > 0),
        "precision": hit_count / k,
        "recall": hit_count / len(relevant_ids),
        "reciprocal_rank": 1 / first_rank if first_rank else 0.0,
        "first_relevant_rank": first_rank,
        "relevant_retrieved": hit_count,
    }


def evaluate(data_path: Path, output_path: Path, k_values: list[int] | None = None) -> dict:
    dataset = json.loads(data_path.read_text(encoding="utf-8"))
    k_values = sorted(set(k_values or dataset.get("k_values", [3, 5, 8])))
    if not k_values or min(k_values) < 1:
        raise ValueError("k values must be positive integers.")

    settings = get_settings()
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    collection = client.get_collection(settings.chroma_collection, embedding_function=None)
    collection_data = collection.get(include=["metadatas", "documents"])
    catalogue = list(
        zip(
            collection_data["ids"],
            collection_data["metadatas"],
            collection_data["documents"],
            strict=True,
        )
    )

    cases = dataset["cases"]
    embeddings = embed_texts([case["transcript"] for case in cases], settings)
    query_result = collection.query(
        query_embeddings=embeddings,
        n_results=min(max(k_values), collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    case_results = []
    for index, case in enumerate(cases):
        predicates = case["relevant_metadata"]
        relevant_ids = {
            chunk_id
            for chunk_id, metadata, _ in catalogue
            if any(metadata_matches(metadata, predicate) for predicate in predicates)
        }
        if not relevant_ids:
            raise ValueError(f"Case {case['id']} has no relevant chunks in the current index.")

        retrieved_ids = query_result["ids"][index]
        per_k = {
            str(k): metrics_at_k(retrieved_ids, relevant_ids, k)
            for k in k_values
        }
        retrieved = []
        for rank, (chunk_id, document, metadata, distance) in enumerate(
            zip(
                retrieved_ids,
                query_result["documents"][index],
                query_result["metadatas"][index],
                query_result["distances"][index],
                strict=True,
            ),
            start=1,
        ):
            retrieved.append(
                {
                    "rank": rank,
                    "id": chunk_id,
                    "relevant": chunk_id in relevant_ids,
                    "score": round(1.0 - float(distance), 6),
                    "source_file": metadata.get("source_file"),
                    "sheet": metadata.get("sheet"),
                    "transcript": metadata.get("transcript"),
                    "topic": metadata.get("topic"),
                    "row_start": metadata.get("row_start"),
                    "excerpt": document[:400],
                }
            )
        case_results.append(
            {
                "id": case["id"],
                "title": case["title"],
                "relevant_chunk_count": len(relevant_ids),
                "metrics": per_k,
                "retrieved": retrieved,
            }
        )

    aggregate = {}
    for k in k_values:
        values = [case["metrics"][str(k)] for case in case_results]
        aggregate[str(k)] = {
            "hit_rate": statistics.fmean(value["hit"] for value in values),
            "precision": statistics.fmean(value["precision"] for value in values),
            "recall": statistics.fmean(value["recall"] for value in values),
            "mrr": statistics.fmean(value["reciprocal_rank"] for value in values),
        }

    result = {
        "dataset": str(data_path.relative_to(ROOT)),
        "embedding_model": settings.openai_embedding_model,
        "collection": settings.chroma_collection,
        "collection_size": collection.count(),
        "case_count": len(case_results),
        "metric_definitions": {
            "hit_rate": "Fraction of cases with at least one labelled relevant chunk in the top k.",
            "precision": "Mean labelled relevant chunks retrieved divided by k.",
            "recall": "Mean labelled relevant chunks retrieved divided by all labelled relevant chunks for that case.",
            "mrr": "Mean reciprocal rank of the first labelled relevant chunk; zero when absent from the top k."
        },
        "aggregate": aggregate,
        "cases": case_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def print_summary(result: dict) -> None:
    print(
        f"Collection: {result['collection']} ({result['collection_size']} chunks) | "
        f"Cases: {result['case_count']} | Model: {result['embedding_model']}"
    )
    print("k  Hit Rate  Precision  Recall   MRR")
    for k, metrics in result["aggregate"].items():
        print(
            f"{int(k):<2} {metrics['hit_rate']:>8.2%} "
            f"{metrics['precision']:>9.2%} {metrics['recall']:>7.2%} "
            f"{metrics['mrr']:>6.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "validation" / "validation_data.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "validation" / "results.json",
    )
    parser.add_argument("--k", type=int, nargs="+", help="Override k values, e.g. --k 3 5 8")
    arguments = parser.parse_args()
    evaluation = evaluate(arguments.data.resolve(), arguments.output.resolve(), arguments.k)
    print_summary(evaluation)
    print(f"Detailed results: {arguments.output.resolve()}")
