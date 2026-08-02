import json
import re
import time
from pathlib import Path

import requests
from datasets import Dataset, load_from_disk

INPUT_DATASET = "output_data/federalist_chunked"
OUTPUT_JSONL = "output_data/publius_qa_local_model.jsonl"
OUTPUT_DATASET = "output_data/publius_qa_local_model_ds"
FAIL_LOG = "output_data/publius_qa_failures.jsonl"


API_BASE = "http://localhost:1234/v1"
# MODEL = "google/gemma-4-26b-a4b-qat"
MODEL = "phi-4-mini-instruct"
MAX_CHUNKS = 700
TIMEOUT = 300
MAX_RETRIES = 3
SLEEP_BETWEEN_RETRIES = 1
RESUME = True

SYSTEM_PROMPT = """You are creating supervised fine-tuning data.
Return ONLY valid JSON.
Create exactly 3 grounded question-answer pairs from the source passage.
Rules:
- Every answer must be supported by the passage.
- Do not add outside facts.
- Keep answers in clear modern English.
- Escape all internal double quotes inside JSON strings.
- Do not include markdown fences.
- Write 2 to 4 complete sentences, usually 45 to 110 words.
- State the answer directly, then explain why it follows from the passage.
- Do not answer with a fragment, a quotation alone, or a single word.
Output schema:
{
  \"qas\": [
    {\"question\": \"...\", \"answer\": \"...\"},
    {\"question\": \"...\", \"answer\": \"...\"},
    {\"question\": \"...\", \"answer\": \"...\"}
  ]
}
"""

REPAIR_PROMPT = """Your previous output was invalid JSON.
Rewrite it as valid JSON only.
Do not change the meaning.
Do not add commentary.
Return exactly this schema:
{
  \"qas\": [
    {\"question\": \"...\", \"answer\": \"...\"},
    {\"question\": \"...\", \"answer\": \"...\"},
    {\"question\": \"...\", \"answer\": \"...\"}
  ]
}
"""

TRAINING_SYSTEM = (
    "You are a careful constitutional assistant writing in clear modern English with "
    "a restrained Publius-like tone. Answer using only ideas supported by the source text."
)


def chat(messages, temperature=0.2):
    r = requests.post(
        f"{API_BASE}/chat/completions",
        json={
            "model": MODEL,
            "messages": messages,
            "temperature": temperature,
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def extract_json_blob(text):
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("No JSON object found in output")
    return match.group(0)


def try_parse_json(text):
    blob = extract_json_blob(text)
    return json.loads(blob)


def normalize_qa_blob(blob):
    qas = blob.get("qas", [])
    cleaned = []
    for qa in qas:
        q = str(qa.get("question", "")).strip()
        a = str(qa.get("answer", "")).strip()
        if q and a:
            cleaned.append({"question": q, "answer": a})
    if len(cleaned) < 1:
        raise ValueError("No usable Q&A pairs after normalization")
    return {"qas": cleaned[:3]}


def load_existing_questions(path):
    seen = set()
    if not RESUME or not Path(path).exists():
        return seen
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            msgs = row.get("messages", [])
            if len(msgs) > 1:
                seen.add(msgs[1].get("content", ""))
    return seen


def build_training_rows(chunk, qa_blob):
    rows = []
    for qa in qa_blob["qas"]:
        q = qa["question"]
        a = qa["answer"]
        rows.append({
            "messages": [
                {"role": "system", "content": TRAINING_SYSTEM},
                {"role": "user", "content": f"Source passage:\n\n{chunk}\n\nQuestion: {q}"},
                {"role": "assistant", "content": a},
            ]
        })
    return rows


def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def log_failure(chunk_id, chunk, raw, error_text):
    Path(FAIL_LOG).parent.mkdir(parents=True, exist_ok=True)
    row = {
        "chunk_id": chunk_id,
        "error": error_text,
        "raw_output": raw,
        "source_excerpt": chunk[:1200],
    }
    with open(FAIL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_dataset_from_jsonl(jsonl_path, output_dataset):
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("No rows available to save as dataset")
    Dataset.from_list(rows).save_to_disk(output_dataset)


def generate_for_chunk(chunk_id, chunk):
    user_prompt = f"SOURCE PASSAGE:\n\n{chunk}"
    raw = chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ], temperature=0.2)

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return normalize_qa_blob(try_parse_json(raw))
        except Exception as e:
            last_error = str(e)
            if attempt == MAX_RETRIES:
                break
            try:
                raw = chat([
                    {"role": "system", "content": REPAIR_PROMPT},
                    {"role": "user", "content": raw},
                ], temperature=0.0)
            except Exception as repair_e:
                last_error = f"repair_failed: {repair_e}"
            time.sleep(SLEEP_BETWEEN_RETRIES)

    log_failure(chunk_id, chunk, raw, last_error or "unknown_error")
    return None


def main():
    Path(OUTPUT_JSONL).parent.mkdir(parents=True, exist_ok=True)
    seen_user_messages = load_existing_questions(OUTPUT_JSONL)

    ds = load_from_disk(INPUT_DATASET)
    rows = ds.to_list()
    if not rows:
        raise ValueError("Input dataset is empty")

    text_key = None
    for candidate in ["text", "chunk", "content"]:
        if candidate in rows[0]:
            text_key = candidate
            break
    if text_key is None:
        raise ValueError("Could not find text column")

    chunks = []
    for row in rows:
        text = (row.get(text_key) or "").strip()
        if len(text.split()) >= 120:
            chunks.append(text)
    chunks = chunks[:MAX_CHUNKS]

    total_written = 0
    for i, chunk in enumerate(chunks, 1):
        print(f"Processing chunk {i}/{len(chunks)}")
        qa_blob = generate_for_chunk(i, chunk)
        if qa_blob is None:
            print(f"  Skipped chunk {i} after invalid output; logged failure.")
            continue

        new_rows = []
        for row in build_training_rows(chunk, qa_blob):
            user_msg = row["messages"][1]["content"]
            if user_msg not in seen_user_messages:
                new_rows.append(row)
                seen_user_messages.add(user_msg)

        if new_rows:
            append_jsonl(OUTPUT_JSONL, new_rows)
            total_written += len(new_rows)
            print(f"  Wrote {len(new_rows)} examples")
        else:
            print("  No new rows written (likely resume run).")

    save_dataset_from_jsonl(OUTPUT_JSONL, OUTPUT_DATASET)
    print(f"Done. Wrote {total_written} new examples.")
    print(f"Dataset saved to {OUTPUT_DATASET}")


if __name__ == "__main__":
    main()