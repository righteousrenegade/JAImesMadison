import re, glob
from datasets import Dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-4-31B-it")
MAX_TOKENS = 4096
DATA_DIR = "source_data/federalist_papers/"          # folder with your 85 .txt files

def essay_label(path):
    m = re.search(r"(\d+)", path)
    return f"The Federalist No. {int(m.group(1))}" if m else path.split("/")[-1][:-4]

def chunk_file(path):
    header = essay_label(path) + "\n\n"
    text = open(path, encoding="utf-8").read().strip()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks, current = [], header
    for para in paragraphs:
        # oversized single paragraph? hard-split on sentences
        while len(tokenizer.encode(current + "\n\n" + para)) > MAX_TOKENS:
            room = MAX_TOKENS - len(tokenizer.encode(current))
            sentences = re.split(r"(?<=[.!?])\s+", para)
            fit, i = [], 0
            while i < len(sentences) and len(tokenizer.encode(" ".join(fit + [sentences[i]]))) < room:
                fit.append(sentences[i]); i += 1
            if not fit:  # single sentence too long — force it through
                chunks.append(current.strip())
                current = header + sentences[0]
                para = " ".join(sentences[1:])
                continue
            chunks.append(current.strip() + "\n\n" + " ".join(fit))
            current = header + " ".join(sentences[i:])
            para = ""
        if para:
            current = (current + "\n\n" + para) if current != header else header + para
    if current != header:
        chunks.append(current.strip())
    return [{"text": c} for c in chunks]

rows = []
for path in sorted(glob.glob(DATA_DIR + "*.txt")):
    rows.extend(chunk_file(path))

dataset = Dataset.from_list(rows)
print(dataset)          # expect ~250-350 examples for all 85 essays
dataset.save_to_disk("output_data/federalist_chunked")

# autocomplete test prompt: The Federalist No. 86\n\nTo the People of the State of Bahamas: 