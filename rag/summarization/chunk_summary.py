# rag/chunk_summary.py
import ollama

SUMMARY_MODEL = "phi3:mini"

MAP_PROMPT = """You are summarizing a document chunk for retrieval.
Write a concise summary that preserves important facts, entities, numbers, definitions, and relationships.
Keep it short but information-dense.

Chunk:
{text}

Summary:"""

REDUCE_PROMPT = """You are combining multiple chunk summaries into one coherent document summary.
Merge overlapping points, remove repetition, and preserve important facts, names, dates, and numbers.
Write a clear, structured summary.

Summaries:
{text}

Combined Summary:"""


def summarize_text(text: str, model: str = SUMMARY_MODEL) -> str:
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": text}]
    )
    return response["message"]["content"].strip()


def map_summarize_chunks(chunks: list[str], model: str = SUMMARY_MODEL) -> list[str]:
    summaries = []
    for i, chunk in enumerate(chunks):
        prompt = MAP_PROMPT.format(text=chunk)
        summary = summarize_text(prompt, model=model)
        summaries.append(summary)
        if i % 10 == 0:
            print(f"  -> summarized {i}/{len(chunks)} chunks")
    ollama.generate(model=model, prompt="", keep_alive=0)
    return summaries


def reduce_summaries(summaries: list[str], model: str = SUMMARY_MODEL) -> str:
    combined_text = "\n\n".join(
        f"Summary {i+1}:\n{s}" for i, s in enumerate(summaries)
    )
    prompt = REDUCE_PROMPT.format(text=combined_text)
    final_summary = summarize_text(prompt, model=model)
    ollama.generate(model=model, prompt="", keep_alive=0)
    return final_summary


def map_reduce_summary(chunks: list[str], model: str = SUMMARY_MODEL) -> tuple[list[str], str]:
    map_summaries = map_summarize_chunks(chunks, model=model)
    final_summary = reduce_summaries(map_summaries, model=model)
    return map_summaries, final_summary