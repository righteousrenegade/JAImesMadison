import torch
from unsloth import FastLanguageModel


BASE_MODEL = "unsloth/gemma-4-12b-it"
ADAPTER_DIR = "output_data/publius_chat_lora"
MAX_SEQ_LENGTH = 2048
MAX_NEW_TOKENS = 220
SYSTEM_PROMPT = (
    "You are a helpful constitutional assistant with a restrained Publius-like tone. "
    "Be concise, grounded, and avoid making up facts."
)


def load_chatbot():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    model.load_adapter(ADAPTER_DIR)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_reply(model, tokenizer, messages):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        text=prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH - MAX_NEW_TOKENS,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.12,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    print("Loading Publius...")
    model, tokenizer = load_chatbot()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Ready. Type /reset to clear the conversation or /quit to exit.")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"/quit", "/exit"}:
            print("Goodbye.")
            break
        if user_input.lower() == "/reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("Conversation reset.")
            continue

        messages.append({"role": "user", "content": user_input})
        reply = generate_reply(model, tokenizer, messages)
        messages.append({"role": "assistant", "content": reply})
        print(f"\nPublius: {reply}")


if __name__ == "__main__":
    main()