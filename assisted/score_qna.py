import json
import math
import re
from collections import Counter
from pathlib import Path

INPUT_JSONL = "output_data/publius_qa_local_model.jsonl"
OUTPUT_SCORED_JSONL = "output_data/publius_qa_scored.jsonl"
OUTPUT_FILTERED_JSONL = "output_data/publius_qa_filtered.jsonl"
OUTPUT_REPORT = "output_data/publius_qa_score_report.md"

MIN_ANSWER_WORDS = 25
MAX_QUESTION_WORDS = 50
MIN_SOURCE_OVERLAP = 0.08
MAX_DUPLICATE_SIMILARITY = 0.9
KEEP_TOP_PERCENT = 0.8

STOPWORDS = {
    "the","a","an","and","or","but","if","then","than","that","this","these","those",
    "is","are","was","were","be","been","being","of","to","in","on","for","with","as",
    "by","at","from","it","its","into","their","there","here","which","who","whom","what",
    "when","where","why","how","do","does","did","can","could","should","would","may","might",
    "not","no","so","such","about","through","over","under","between","among"
}


def tokenize(text):
    return re.findall(r"[a-zA-Z']+", text.lower())


def content_words(text):
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 2]


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def safe_get_messages(row):
    msgs = row.get("messages", [])
    system = msgs[0]["content"] if len(msgs) > 0 else ""
    user = msgs[1]["content"] if len(msgs) > 1 else ""
    assistant = msgs[2]["content"] if len(msgs) > 2 else ""
    return system, user, assistant


def extract_source_and_question(user_text):
    source = user_text
    question = ""
    if "Question:" in user_text:
        left, right = user_text.rsplit("Question:", 1)
        source = left.replace("Source passage:", "").strip()
        question = right.strip()
    return source, question


def score_row(row, prior_answers):
    _, user_text, assistant = safe_get_messages(row)
    source, question = extract_source_and_question(user_text)

    source_tokens = content_words(source)
    answer_tokens = content_words(assistant)
    question_tokens = content_words(question)

    answer_word_count = len(tokenize(assistant))
    question_word_count = len(tokenize(question))
    overlap = 0.0
    if answer_tokens and source_tokens:
        overlap = len(set(answer_tokens) & set(source_tokens)) / max(1, len(set(answer_tokens)))

    duplicate_similarity = 0.0
    for prev in prior_answers:
        duplicate_similarity = max(duplicate_similarity, jaccard(answer_tokens, prev))

    score = 100.0
    reasons = []

    if answer_word_count < MIN_ANSWER_WORDS:
        penalty = min(25, (MIN_ANSWER_WORDS - answer_word_count) * 0.8)
        score -= penalty
        reasons.append(f"short_answer(-{penalty:.1f})")

    if question_word_count > MAX_QUESTION_WORDS:
        penalty = min(10, (question_word_count - MAX_QUESTION_WORDS) * 0.4)
        score -= penalty
        reasons.append(f"long_question(-{penalty:.1f})")

    if overlap < MIN_SOURCE_OVERLAP:
        penalty = 30 * (1 - (overlap / MIN_SOURCE_OVERLAP if MIN_SOURCE_OVERLAP else 1))
        score -= penalty
        reasons.append(f"low_grounding(-{penalty:.1f})")

    if duplicate_similarity > MAX_DUPLICATE_SIMILARITY:
        penalty = 20 * duplicate_similarity
        score -= penalty
        reasons.append(f"near_duplicate(-{penalty:.1f})")

    if len(set(answer_tokens)) < 20:
        score -= 12
        reasons.append("low_answer_variety(-12.0)")

    question_type_bonus = 0
    q = question.lower()
    for marker in ["why", "how", "what", "which", "compare", "explain", "summarize"]:
        if marker in q:
            question_type_bonus = 4
            break
    score += question_type_bonus
    if question_type_bonus:
        reasons.append(f"good_prompt_shape(+{question_type_bonus:.1f})")

    hallucination_markers = ["according to history", "scholars", "historians", "as we know", "in america today"]
    if any(m in assistant.lower() for m in hallucination_markers):
        score -= 18
        reasons.append("possible_outside_knowledge(-18.0)")

    score = max(0.0, min(100.0, score))
    return {
        **row,
        "score": round(score, 2),
        "question": question,
        "source_passage": source,
        "answer": assistant,
        "metrics": {
            "answer_word_count": answer_word_count,
            "question_word_count": question_word_count,
            "source_overlap": round(overlap, 4),
            "duplicate_similarity": round(duplicate_similarity, 4),
        },
        "reasons": reasons,
    }, answer_tokens


def main():
    path = Path(INPUT_JSONL)
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_JSONL}")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    scored = []
    prior_answers = []
    for row in rows:
        srow, answer_tokens = score_row(row, prior_answers)
        scored.append(srow)
        prior_answers.append(answer_tokens)

    scored.sort(key=lambda x: x["score"], reverse=True)
    keep_n = max(1, math.floor(len(scored) * KEEP_TOP_PERCENT))
    filtered = [r for r in scored[:keep_n] if r["score"] >= 60]

    Path(OUTPUT_SCORED_JSONL).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_SCORED_JSONL, "w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(OUTPUT_FILTERED_JSONL, "w", encoding="utf-8") as f:
        for row in filtered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    reason_counts = Counter(r for row in scored for r in row["reasons"])
    avg_score = sum(r["score"] for r in scored) / max(1, len(scored))
    avg_overlap = sum(r["metrics"]["source_overlap"] for r in scored) / max(1, len(scored))

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("# Q&A dataset score report\n\n")
        f.write(f"- Total examples: {len(scored)}\n")
        f.write(f"- Kept examples: {len(filtered)}\n")
        f.write(f"- Average score: {avg_score:.2f}\n")
        f.write(f"- Average source overlap: {avg_overlap:.4f}\n\n")
        f.write("## Most common scoring notes\n\n")
        for reason, count in reason_counts.most_common(15):
            f.write(f"- {reason}: {count}\n")
        f.write("\n## Top 10 examples\n\n")
        for row in scored[:10]:
            f.write(f"### Score {row['score']}\n")
            f.write(f"**Question:** {row['question']}\n\n")
            f.write(f"**Answer:** {row['answer'][:500]}\n\n")
            f.write(f"**Reasons:** {', '.join(row['reasons']) or 'none'}\n\n")

    print(f"Scored {len(scored)} examples; kept {len(filtered)}")
    print(f"Wrote {OUTPUT_SCORED_JSONL}, {OUTPUT_FILTERED_JSONL}, and {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
