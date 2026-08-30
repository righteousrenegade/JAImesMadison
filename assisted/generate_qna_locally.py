import json
import re
import time
from pathlib import Path

import requests
from datasets import Dataset, load_from_disk

INPUT_DATASET = "output_data/federalist_chunked"
OUTPUT_JSONL = "output_data/publius_qa_local_model_v2.jsonl"
OUTPUT_DATASET = "output_data/publius_qa_local_model_v2_ds"
FAIL_LOG = "output_data/publius_qa_failures_v2.jsonl"


API_BASE = "http://localhost:1234/v1"
# MODEL = "google/gemma-4-26b-a4b-qat"
MODEL = "google/gemma-4-12b"
MAX_CHUNKS = 1000
TIMEOUT = 300
MAX_RETRIES = 3
SLEEP_BETWEEN_RETRIES = 1
RESUME = True

### Old sysprompt that didn't seem to output well.
SYSTEM_PROMPT = """You are creating supervised fine-tuning data. Return ONLY valid JSON.
Create exactly 3 grounded question-answer pairs from the source passage.

Write each answer in Publius's voice: the answerer is Publius speaking directly to the reader. Do not describe Publius, imitate Publius from a distance, or explain how Publius would answer.

Rules:
- Each question should be a single, clear, and specific question that can be answered using only the source passage.
- Each question should ask directly about the passage. For example, "What is the main argument of this passage?" is valid. Do not mention Publius, the author, the speaker, or the persona in a question.
- Questions should be purely conceptual and related to the content of the passage alone.
- Questions must never mention Publius, the author, the speaker, the persona, or how anyone would answer.
- Every answer must be supported by the passage.
- Begin with the substantive answer, never with a description of who is speaking.
- Write in first-person rhetorical voice where natural: "I contend...", "It is...", "The reason is...".
- Never use phrases such as "Publius would say", "Publius argues", "the author says", "the speaker believes", "in Publius's view", or "this passage explains".
- The answer must stand alone as something Publius could have written to answer the reader. Do not discuss style, role-play, imitation, or the generation task.
- Do not add outside facts.
- Escape all internal double quotes inside JSON strings.
- Do not include markdown fences.
- Write 2 to 4 complete sentences, usually 45 to 150 words.
- State the answer directly, then explain why it follows from the passage.
- Do not answer with a fragment, a quotation alone, or a single word.
Example of the required voice:
Question: Why is an energetic government necessary?
Answer: An energetic government is necessary because the purposes of government cannot be fulfilled by a system that wants either the power or the steadiness to act. The safety of the Union, the regular administration of justice, and the protection of the public interests require authority proportioned to the ends entrusted to it.
Output schema:
{
  \"qas\": [
    {\"question\": \"...\", \"answer\": \"...\"},
    {\"question\": \"...\", \"answer\": \"...\"},
    {\"question\": \"...\", \"answer\": \"...\"}
  ]
}
"""


SYSTEM_PROMPT = """You are generating supervised Q&A data.

Answer the reader directly in the voice of the person who wrote the
source passage. The answer must sound like something that person personally
wrote, not like an explanation of what that person believes.

Do not describe the writer or the writer's position.
Do not say:
- "Publius would say"
- "Publius argues"
- "the author believes"
- "according to Publius"
- "the passage explains"

Bad:
"Publius would argue that an energetic government is necessary."

Good:
"An energetic government is necessary because the purposes entrusted to
government cannot be fulfilled by a system wanting either the power or the
steadiness to act."

Create exactly 3 grounded question-answer pairs.
Questions must ask directly about the source passage and must not mention
Publius, the author, the speaker, or the persona.

Every answer must:
- answer the question directly
- use only information from the passage
- use a direct first-person rhetorical voice where natural
- contain 2 to 4 complete sentences
- avoid commentary about the writer or the writing task

Return only valid JSON with this schema:
{
  "qas": [
    {"question": "...", "answer": "..."},
    {"question": "...", "answer": "..."},
    {"question": "...", "answer": "..."}
  ]
}
"""


REPAIR_PROMPT = """Your previous output was invalid JSON or violated the voice requirements.
Rewrite it as valid JSON only.
Keep every answer grounded in the source passage, but rewrite it as Publius speaking directly to the reader.
Never mention Publius, the author, the speaker, the persona, role-play, or how Publius would answer.
Never begin an answer with "Publius would", "Publius argues", "the author", or similar meta-language.
Start each answer with its substantive claim.
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
    "You are a clone of the original Publius persona, writing in clear modern English with "
    "a restrained Publius-like tone. Answer using only ideas supported by the source text."
)

def chat(messages, temperature=0.4):
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
    forbidden_question = re.compile(
        r"\b(publius|the author|the speaker|the persona)\b|"
        r"\bhow .* would\b",
        re.IGNORECASE,
    )
    forbidden_answer = re.compile(
        r"\b(publius would|publius argues|the author says|the speaker believes|"
        r"in publius['’]?s view|how publius would)\b",
        re.IGNORECASE,
    )
    for qa in qas:
        q = str(qa.get("question", "")).strip()
        a = str(qa.get("answer", "")).strip()
        if not q or not a:
            continue
        if forbidden_question.search(q) or forbidden_answer.search(a):
            raise ValueError("Q&A contains forbidden meta-persona language")
        cleaned.append({"question": q, "answer": a})
    if len(cleaned) != 3:
        raise ValueError(f"Expected exactly 3 Q&A pairs, got {len(cleaned)}")
    return {"qas": cleaned}

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