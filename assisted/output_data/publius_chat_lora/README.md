---
base_model: unsloth/gemma-4-12b-it
library_name: transformers
pipeline_tag: text-generation
tags:
- lora
- peft
- unsloth
- gemma-4
- constitutional-ai
- federalist-papers
---

# Publius Chat LoRA

A LoRA adapter fine-tuned from [unsloth/gemma-4-12b-it](https://huggingface.co/unsloth/gemma-4-12b-it)
for a restrained Publius-like constitutional assistant.

## Usage

This repository contains adapter weights, not the base model. Install the project
requirements and load the base model before attaching this adapter:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-4-12b-it",
    max_seq_length=2048,
    load_in_4bit=True,
)
model.load_adapter("YOUR_USERNAME/publius-chat-lora")
FastLanguageModel.for_inference(model)
```

Use the tokenizer's `apply_chat_template` with `system`, `user`, and `assistant`
messages. The training data contains 376 supervised conversations grounded in the
Federalist Papers.

## Training details

- LoRA rank: 8
- LoRA alpha: 16
- Maximum sequence length: 2048
- Base model: `unsloth/gemma-4-12b-it`
- Adapter type: PEFT LoRA

## Limitations

This adapter is not a substitute for historical or legal research. It may produce
anachronisms, inaccurate attributions, or unsupported answers. Verify quotations
and historical claims against primary sources.
