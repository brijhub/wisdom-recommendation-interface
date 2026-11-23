"""Extract source files, create embeddings, and store/search them in ChromaDB."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Iterable

import chromadb
from openai import OpenAI
from openpyxl import load_workbook

from config import DATA_DIR, Settings, get_settings


MAX_CHARS = 5_000
OVERLAP_CHARS = 300
PACK_SHEETS = {
    "situation",
    "persona",
    "hidden state",
    "objection arcs",
    "objective arc",
    "objective arcs",
    "win conditions",
}
CALIBRATION_SHEETS = {
    "transcript 1",
    "transcript 2",
    "transcript 3",
    "line flags",
    "part 2 single moments",
}


@dataclass(frozen=True)
class KnowledgeChunk:
    id: str
    content: str
    metadata: dict


def _text(value: object) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value).strip()


def _split_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split long records on line boundaries with a small character overlap."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + max_chars // 2, end)
            if boundary > start:
                end = boundary
        parts.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - OVERLAP_CHARS, start + 1)
    return [part for part in parts if part]


def _make_chunks(content: str, metadata: dict) -> list[KnowledgeChunk]:
    chunks = []
    for part_index, part in enumerate(_split_text(content), start=1):
        part_metadata = {**metadata, "part": part_index}
        identity = json.dumps(part_metadata, sort_keys=True, default=str) + part
        chunks.append(
            KnowledgeChunk(
                id=sha256(identity.encode("utf-8")).hexdigest(),
                content=part,
                metadata=part_metadata,
            )
        )
    return chunks


def _reviewer_from_filename(path: Path) -> str | None:
    match = re.search(
        r"calibration\s*-\s*(.+?)(?:\s*\(|\.xlsx$)",
        path.name,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    if "panel" in path.name.lower():
        return "Nagendra and Shubham"
    return None


def _allowed_sheets(path: Path, sheet_names: set[str]) -> set[str]:
    """Return the strict sheet allowlist for a supported workbook type."""
    normalized = {name.casefold() for name in sheet_names}
    filename = path.name.casefold()
    if "pack feedback" in filename:
        return PACK_SHEETS
    if "calibration" in filename or normalized.intersection(CALIBRATION_SHEETS):
        return CALIBRATION_SHEETS
    return set()


def extract_excel(path: Path) -> list[KnowledgeChunk]:
    """Extract only explicitly approved Pack or Calibration workbook sheets."""
    workbook = load_workbook(path, data_only=True, read_only=False)
    output: list[KnowledgeChunk] = []
    reviewer = _reviewer_from_filename(path)
    allowed_sheets = _allowed_sheets(path, set(workbook.sheetnames))
    if not allowed_sheets:
        return output

    for sheet in workbook.worksheets:
        if sheet.sheet_state != "visible" or sheet.title.casefold() not in allowed_sheets:
            continue
        rows = []
        for row in sheet.iter_rows():
            values = [cell.value for cell in row]
            if any(value not in (None, "") for value in values):
                rows.append((row[0].row, values))
        if not rows:
            continue

        candidates = rows[: min(12, len(rows))]
        header_number, header_values = max(
            candidates,
            key=lambda item: sum(value not in (None, "") for value in item[1]),
        )
        has_header = sum(value not in (None, "") for value in header_values) >= 3
        headers = [
            _text(value) if value not in (None, "") else f"Column {index}"
            for index, value in enumerate(header_values, start=1)
        ]

        if has_header:
            preamble = [
                " | ".join(_text(value) for value in values if value not in (None, ""))
                for row_number, values in rows
                if row_number < header_number
            ]
            if preamble:
                output.extend(
                    _make_chunks(
                        f"Sheet context: {sheet.title}\n" + "\n".join(preamble),
                        {
                            "source_file": path.name,
                            "source_type": "excel",
                            "sheet": sheet.title,
                            "record_type": "sheet_context",
                            "row_start": rows[0][0],
                            "row_end": header_number - 1,
                            **({"reviewer": reviewer} if reviewer else {}),
                        },
                    )
                )

            for row_number, values in rows:
                if row_number <= header_number:
                    continue
                fields = []
                for index, value in enumerate(values):
                    if value in (None, ""):
                        continue
                    label = headers[index] if index < len(headers) else f"Column {index + 1}"
                    fields.append(f"{label}: {_text(value)}")
                if not fields:
                    continue
                content = f"Sheet: {sheet.title}\n" + "\n".join(fields)
                record_type = "calibration_feedback" if sheet.title.startswith("Transcript") else "workbook_row"
                metadata = {
                    "source_file": path.name,
                    "source_type": "excel",
                    "sheet": sheet.title,
                    "record_type": record_type,
                    "row_start": row_number,
                    "row_end": row_number,
                    "scenario_id": "SP01",
                    **({"reviewer": reviewer} if reviewer else {}),
                }
                if sheet.title.startswith("Transcript"):
                    metadata["transcript"] = sheet.title
                if values and values[0] not in (None, ""):
                    metadata["topic"] = _text(values[0])[:200]
                output.extend(_make_chunks(content, metadata))
        else:
            lines = [
                f"Row {row_number}: "
                + " | ".join(_text(value) for value in values if value not in (None, ""))
                for row_number, values in rows
            ]
            output.extend(
                _make_chunks(
                    f"Sheet: {sheet.title}\n" + "\n".join(lines),
                    {
                        "source_file": path.name,
                        "source_type": "excel",
                        "sheet": sheet.title,
                        "record_type": "sheet_text",
                        "row_start": rows[0][0],
                        "row_end": rows[-1][0],
                        "scenario_id": "SP01",
                        **({"reviewer": reviewer} if reviewer else {}),
                    },
                )
            )
    return output


def extract_all(data_dir: Path = DATA_DIR) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    for path in sorted(data_dir.glob("*.xlsx")):
        chunks.extend(extract_excel(path))
    if not chunks:
        raise FileNotFoundError(
            f"No approved Pack or Calibration workbook sheets found in {data_dir}."
        )
    return chunks


def _client(settings: Settings) -> OpenAI:
    settings.require_openai_key()
    return OpenAI(api_key=settings.openai_api_key, timeout=60.0, max_retries=2)


def embed_texts(texts: Iterable[str], settings: Settings | None = None) -> list[list[float]]:
    settings = settings or get_settings()
    response = _client(settings).embeddings.create(
        model=settings.openai_embedding_model,
        input=list(texts),
        dimensions=settings.openai_embedding_dimensions,
    )
    return [item.embedding for item in response.data]


def _chroma_metadata(metadata: dict) -> dict:
    """Chroma metadata values must be scalar, so encode lists as JSON strings."""
    return {
        key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
        for key, value in metadata.items()
        if value is not None
    }


def ingest(data_dir: Path = DATA_DIR, settings: Settings | None = None) -> int:
    settings = settings or get_settings()
    chunks = extract_all(data_dir)
    embeddings: list[list[float]] = []
    for start in range(0, len(chunks), 64):
        embeddings.extend(embed_texts((chunk.content for chunk in chunks[start : start + 64]), settings))

    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    collection_names = {collection.name for collection in client.list_collections()}
    if settings.chroma_collection in collection_names:
        client.delete_collection(settings.chroma_collection)
    collection = client.create_collection(
        name=settings.chroma_collection,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={
            "embedding_model": settings.openai_embedding_model,
            "embedding_dimensions": settings.openai_embedding_dimensions,
        },
    )
    collection.add(
        ids=[chunk.id for chunk in chunks],
        documents=[chunk.content for chunk in chunks],
        metadatas=[_chroma_metadata(chunk.metadata) for chunk in chunks],
        embeddings=embeddings,
    )
    if collection.count() != len(chunks):
        raise RuntimeError("Stored ChromaDB chunk count does not match extracted chunks.")
    return len(chunks)


def search(query: str, settings: Settings | None = None) -> list[dict]:
    settings = settings or get_settings()
    vector = embed_texts([query], settings)[0]
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    try:
        collection = client.get_collection(settings.chroma_collection, embedding_function=None)
    except Exception as error:
        raise ValueError("The knowledge base is missing. Run: python src/ingest.py") from error
    collection_metadata = collection.metadata or {}
    if collection_metadata.get("embedding_model") != settings.openai_embedding_model:
        raise ValueError("The embedding model changed. Run ingestion again.")
    results = collection.query(
        query_embeddings=[vector],
        n_results=min(settings.retrieval_limit, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    documents = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]
    ids = results.get("ids") or [[]]
    return [
        {
            "id": chunk_id,
            "content": content,
            "metadata": metadata,
            "score": 1.0 - float(distance),
        }
        for chunk_id, content, metadata, distance in zip(
            ids[0], documents[0], metadatas[0], distances[0], strict=True
        )
    ]


if __name__ == "__main__":
    count = ingest()
    print(f"Ingested {count} knowledge chunks into local ChromaDB.")

