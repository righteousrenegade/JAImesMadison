import json
import torch
from pathlib import Path
from unsloth import FastLanguageModel

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

BASE_MODEL = "unsloth/gemma-4-12b-it"
ADAPTER_DIR = "output_data/publius_chat_lora"
MAX_SEQ_LENGTH = 2048
MAX_NEW_TOKENS = 220
SYSTEM_PROMPT = (
    "You are a helpful constitutional assistant with a restrained Publius-like tone. "
    "Be concise, grounded, and avoid making up facts."
)

console = Console() if RICH_AVAILABLE else None


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


def print_welcome():
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            "[bold]Publius Chatbot[/bold]\n"
            "Type your message to chat.\n"
            "Type /help for commands.",
            title="Welcome",
            border_style="blue"
        ))
    else:
        print("Loading Publius...")
        print("Ready. Type /help for commands.")


def print_history(messages):
    if not messages:
        return
    if RICH_AVAILABLE:
        console.print("\n[bold]Conversation History[/bold]")
        for i, m in enumerate(messages):
            role = m["role"]
            content = m["content"]
            style = "green" if role == "user" else "cyan" if role == "assistant" else "yellow"
            console.print(f"\n[{style}]{role.upper()}[/{style}]:")
            console.print(content)
    else:
        print("\n--- History ---")
        for m in messages:
            print(f"{m['role'].upper()}: {m['content'][:200]}")


def save_history(messages, path):
    Path(path).write_text(json.dumps(messages, indent=2))
    if RICH_AVAILABLE:
        console.print(f"[green]Saved conversation to {path}[/green]")
    else:
        print(f"Saved to {path}")


def load_history(path):
    data = json.loads(Path(path).read_text())
    if RICH_AVAILABLE:
        console.print(f"[green]Loaded conversation from {path}[/green]")
    else:
        print(f"Loaded from {path}")
    return data


def print_help():
    help_text = """
Commands:
/help      - Show this help
/history   - Show current conversation
/reset     - Clear conversation, keep system prompt
/stats     - Show message count and token estimate
/save <file> - Save conversation to JSON file
/load <file> - Load conversation from JSON file
/quit      - Exit
"""
    if RICH_AVAILABLE:
        console.print(Panel(help_text, title="Commands"))
    else:
        print(help_text)


def print_stats(messages, tokenizer):
    count = len([m for m in messages if m["role"] != "system"])
    # rough token estimate
    text = " ".join(m["content"] for m in messages)
    tokens = len(tokenizer.encode(text)) if hasattr(tokenizer, "encode") else len(text.split())
    if RICH_AVAILABLE:
        console.print(f"[bold]Messages:[/bold] {count}  [bold]Approx tokens:[/bold] {tokens}")
    else:
        print(f"Messages: {count}, Approx tokens: {tokens}")


def main():
    print_welcome()
    model, tokenizer = load_chatbot()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            if RICH_AVAILABLE:
                user_input = console.input("[bold magenta]You[/bold magenta]: ")
            else:
                user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        cmd = user_input.lower().strip()

        if cmd in {"/quit", "/exit"}:
            print("Goodbye.")
            break
        if cmd == "/help":
            print_help()
            continue
        if cmd == "/reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("Conversation reset.") if not RICH_AVAILABLE else console.print("[yellow]Conversation reset.[/yellow]")
            continue
        if cmd == "/history":
            print_history(messages)
            continue
        if cmd == "/stats":
            print_stats(messages, tokenizer)
            continue
        if cmd.startswith("/save "):
            path = user_input[6:].strip()
            save_history(messages, path)
            continue
        if cmd.startswith("/load "):
            path = user_input[6:].strip()
            try:
                loaded = load_history(path)
                # keep system prompt
                messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [m for m in loaded if m["role"] != "system"]
            except Exception as e:
                print(f"Failed to load: {e}")
            continue

        messages.append({"role": "user", "content": user_input})
        reply = generate_reply(model, tokenizer, messages)
        messages.append({"role": "assistant", "content": reply})

        if RICH_AVAILABLE:
            console.print(Panel(Markdown(reply), title="Publius", border_style="cyan"))
        else:
            print(f"\nPublius: {reply}")


if __name__ == "__main__":
    main()