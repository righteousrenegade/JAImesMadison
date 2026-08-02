# Publius (The Federalist) style LoRA — Gemma 4 31B, 4-bit QLoRA, single 24GB RTX 3090
from unsloth import FastLanguageModel
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
import torch

# ---------------------------------------------------------------- config
MODEL_NAME = "unsloth/gemma-4-12b-it"
MAX_SEQ    = 4096          # back up — you have room now
DATASET_PATH  = "output_data/federalist_chunked/"        # output of chunk_dataset.py (rerun with MAX_TOKENS = 2048)
OUTPUT_DIR    = "publius_style/"
ADAPTER_DIR   = "publius_style_lora/"
#MAX_SEQ       = 2048

# ---------------------------------------------------------------- model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ,
    dtype          = None,          # unsloth picks bf16 on the 3090
    load_in_4bit   = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r              = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha     = 32,
    lora_dropout   = 0,             # required for unsloth's optimized Gemma kernels
    bias           = "none",
    use_gradient_checkpointing = "unsloth",   # the memory saver on a 24GB card
    random_state   = 3407,
)

# ---------------------------------------------------------------- data
dataset = load_from_disk(DATASET_PATH)
print(f"Training on {len(dataset)} chunks")

# ---------------------------------------------------------------- train
trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        packing            = True,
        max_seq_length     = MAX_SEQ,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,    # effective batch = 4
        num_train_epochs   = 3,
        learning_rate      = 1e-4,          # Gemma diverges at Llama-default 2e-4
        lr_scheduler_type  = "cosine",
        warmup_ratio       = 0.05,
        optim              = "paged_adamw_8bit",  # pages optimizer state to CPU RAM under pressure
        logging_steps      = 5,
        save_strategy      = "epoch",
        output_dir         = OUTPUT_DIR,
        seed               = 3407,
        report_to          = "none",        # no wandb prompts
    ),
)
trainer.train()

# ---------------------------------------------------------------- save
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"Adapter saved to {ADAPTER_DIR}/")