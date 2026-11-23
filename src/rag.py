"""Retrieval-augmented wisdom recommendations for one supplied transcript."""

from openai import OpenAI

from config import Settings, get_settings
from ingest import search


SYSTEM_PROMPT = """You are an internal interaction coach. Review only the supplied conversation
using the retrieved SME evidence. Make the feedback specific to what the speakers actually said,
not a generic narrative. Never invent a timestamp, name, number, policy, promise, or event.

Write in the conversation's main language and register (use natural Hinglish when it is mixed).
Keep the complete response under 220 words and follow this exact Markdown structure:

### What worked
- Give one or two bullets only. Start each with a timestamp when one exists, then identify the
  exact move and why it helped. Cite the supporting evidence as [1], [2], etc.

### Improve next time
- Give the one or two highest-impact issues only. Start with the exact timestamp when available.
  State what happened and its practical consequence in no more than two short sentences.
  On the next line write **Try this:** followed by one brief, ready-to-use alternative response.
  Cite supporting evidence.

### Next action
Give one concrete action in a single sentence.

Do not add an introduction, conclusion, rubric summary, repeated explanation, or separate evidence
section. If timestamps are absent, identify the moment by speaker and short quote instead.
"""


def recommend_wisdom(transcript: str, settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    transcript = transcript.strip()
    if not transcript:
        raise ValueError("Conversation transcript cannot be empty.")
    settings.require_openai_key()
    matches = search(transcript, settings)
    if not matches:
        raise ValueError("The knowledge base is empty. Run the ingestion command first.")

    evidence = "\n\n".join(
        f"[{index}] {match['content']}"
        for index, match in enumerate(matches, start=1)
    )
    prompt = f"""Conversation to review:
<conversation>
{transcript}
</conversation>

Retrieved knowledge-base evidence:
<evidence>
{evidence}
</evidence>

Provide evidence-grounded coaching for this conversation."""
    client = OpenAI(api_key=settings.openai_api_key, timeout=90.0, max_retries=2)
    response = client.responses.create(
        model=settings.openai_chat_model,
        instructions=SYSTEM_PROMPT,
        input=prompt,
        temperature=0.2,
        max_output_tokens=400,
    )
    answer = response.output_text.strip()
    if not answer:
        raise RuntimeError("OpenAI returned an empty recommendation.")

    sources = []
    for index, match in enumerate(matches, start=1):
        metadata = match["metadata"]
        sources.append(
            {
                "citation": index,
                "source_file": metadata.get("source_file", "Unknown"),
                "sheet": metadata.get("sheet"),
                "section": metadata.get("section") or metadata.get("transcript"),
                "row_start": metadata.get("row_start"),
                "row_end": metadata.get("row_end"),
                "reviewer": metadata.get("reviewer"),
                "score": round(match["score"], 4),
                "excerpt": match["content"],
            }
        )
    return {"recommendation": answer, "sources": sources}

