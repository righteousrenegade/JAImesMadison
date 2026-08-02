# test_publius.py
from unsloth import FastLanguageModel
from transformers import TextStreamer
import os

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)      # assumes this file lives in assisted/
ADAPTER_DIR = os.path.join(PROJECT_ROOT, "assisted/publius_style_lora/").replace("\\", "/")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/gemma-4-12b-it",
    max_seq_length = 4096,
    load_in_4bit   = True,
)
model.load_adapter(ADAPTER_DIR)
FastLanguageModel.for_inference(model)
prompt1 = "The Federalist No. 51\n\nAmbition must be made to counteract ambition."
promt2 = "The Federalist No. 86\n\nTo the People of the State of Bahamas:"

prompt = "The Federalist No. 86\n\nTo the People of the Bahamas seeking Statehood:\n\nAmong the objections to your petition,"
inputs = tokenizer(text=prompt, return_tensors="pt").to("cuda")
model.generate(
    **inputs,
    max_new_tokens = 400,
    temperature    = 0.3,
    repetition_penalty = 1.1,
    min_p          = 0.1,
    streamer       = TextStreamer(tokenizer, skip_prompt=True),
)