from datasets import load_from_disk
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "unsloth/gemma-4-12b-it"
DATASET_PATH = "output_data/publius_qa_local_model_v2_ds"
OUTPUT_DIR = "output_data/publius_chat_lora_v2"
MAX_SEQ_LENGTH = 4096

def format_messages(examples, tokenizer):
    return {
        "text": [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            for messages in examples["messages"]
        ]
    }


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    dataset = load_from_disk(DATASET_PATH)
    dataset = dataset.map(
        lambda examples: format_messages(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=5,
            num_train_epochs=2,
            learning_rate=1e-4,
            logging_steps=1,
            save_strategy="epoch",
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            report_to="none",
            bf16=True,
            fp16=False,
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_num_proc=1,
            packing=False,
        ),
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved adapter to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()