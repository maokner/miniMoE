"""
Sample from a trained miniMoE checkpoint.

miniMoE is a *base* language model trained on FineWeb-Edu web text - it was not
instruction-tuned, so it continues text rather than following commands. The
default mode reflects that: you type some text and the model continues it. The
--chat mode is a thin wrapper that formats your turns as User/Assistant so it
feels conversational, but expect completion-style behavior, not a real assistant.

Examples:
    # interactive text completion
    python src/sample.py -c minimoe_step_0019073.pt

    # interactive "chat"
    python src/sample.py -c minimoe_step_0019073.pt --chat

    # one-shot, good for quick tests / scripting
    python src/sample.py -c minimoe_step_0019073.pt -p "The mitochondria is"
"""
import argparse
import glob
import os
import sys

import tiktoken
import torch

from chat_template import append_reply, build_inference_prompt
from checkpoint import load_checkpoint as load_checkpoint_with_metadata

EOS_TOKEN_ID = 50256


def get_device(requested):
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_checkpoint(path, device):
    model, config, metadata = load_checkpoint_with_metadata(path, device)
    step = metadata.get("step", "?")
    tokens_seen = metadata.get("tokens_seen") or 0
    print(
        f"Loaded {path} (step {step}, {tokens_seen:,} tokens seen) on {device}",
        file=sys.stderr,
    )
    return model, config


class TokenStreamer:
    """Decodes tokens to text incrementally, holding back partial multi-byte
    characters so we never print a stray replacement glyph."""

    def __init__(self, enc):
        self.enc = enc
        self.tokens = []
        self.printed = ""

    def push(self, token):
        self.tokens.append(token)
        text = self.enc.decode(self.tokens)
        if text.endswith("�"):  # incomplete UTF-8 char, wait for more
            return ""
        delta = text[len(self.printed):]
        self.printed = text
        return delta


@torch.no_grad()
def generate(
    model,
    config,
    device,
    enc,
    prompt_ids,
    max_new_tokens,
    temperature,
    top_k,
    stop_ids=None,
    stream=False,
):
    """Autoregressively sample a continuation of prompt_ids.

    Stops on the EOS token, on the stop_ids subsequence (dropped from output),
    or after max_new_tokens. Returns the decoded continuation text; if stream
    is set, also prints tokens as they are produced. A hold-back buffer the
    size of stop_ids keeps the stop marker itself from ever being streamed.
    """
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = []
    holdback = len(stop_ids) if stop_ids else 0
    streamer = TokenStreamer(enc) if stream else None
    committed = 0

    def flush_to(up_to):
        nonlocal committed
        while committed < up_to:
            sys.stdout.write(streamer.push(out[committed]))
            committed += 1
        sys.stdout.flush()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -config.max_seq_length:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, k)
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        token = int(next_id.item())
        if token == EOS_TOKEN_ID:
            break

        out.append(token)
        idx = torch.cat([idx, next_id], dim=1)

        if stop_ids and out[-len(stop_ids):] == stop_ids:
            out = out[:-len(stop_ids)]
            break

        if stream:
            flush_to(len(out) - holdback)

    if stream:
        flush_to(len(out))
        sys.stdout.write("\n")
        sys.stdout.flush()

    return enc.decode(out)


def completion_repl(model, config, device, enc, args):
    print(
        "miniMoE - text completion (base model). Type some text and it will "
        "continue it.\nCommands: /exit to quit.\n",
        file=sys.stderr,
    )
    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt.strip() in {"/exit", "/quit"}:
            return
        if not prompt:
            continue
        ids = enc.encode(prompt)
        if args.bos:
            ids = [EOS_TOKEN_ID] + ids
        sys.stdout.write(prompt)  # echo so the continuation reads as one text
        sys.stdout.flush()
        generate(
            model, config, device, enc, ids,
            args.max_new_tokens, args.temperature, args.top_k,
            stop_ids=None, stream=True,
        )


def chat_repl(model, config, device, enc, args):
    print(
        "miniMoE - chat mode (base model, so expect completion-style replies).\n"
        "Commands: /reset to clear history, /exit to quit.\n",
        file=sys.stderr,
    )
    preamble = (
        "The following is a conversation between a User and a helpful "
        "Assistant.\n\n"
    )
    stop_ids = enc.encode("\nUser:")

    def fresh_history():
        return [EOS_TOKEN_ID] + enc.encode(preamble)

    history = fresh_history()
    while True:
        try:
            user = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if user in {"/exit", "/quit"}:
            return
        if user == "/reset":
            history = fresh_history()
            print("(history cleared)", file=sys.stderr)
            continue
        if not user:
            continue

        history += enc.encode(f"User: {user}\nAssistant:")
        history = history[-config.max_seq_length:]  # keep within context window

        sys.stdout.write("Assistant:")
        sys.stdout.flush()
        reply = generate(
            model, config, device, enc, history,
            args.max_new_tokens, args.temperature, args.top_k,
            stop_ids=stop_ids, stream=True,
        )
        history += enc.encode(f" {reply.strip()}\n")


def sft_chat_repl(model, config, device, enc, args):
    """Chat with an SFT'd checkpoint using the canonical training template.

    Unlike --chat (a thin wrapper around the base model), this assumes the model
    was fine-tuned with chat_template.py: it formats turns identically and relies
    on the model emitting EOS to end its turn, which generate() already honors.
    """
    print(
        "miniMoE - SFT chat mode. Type a message; the assistant will reply.\n"
        "Commands: /reset to clear history, /exit to quit.\n",
        file=sys.stderr,
    )
    history = []
    while True:
        try:
            user = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if user in {"/exit", "/quit"}:
            return
        if user == "/reset":
            history = []
            print("(history cleared)", file=sys.stderr)
            continue
        if not user:
            continue

        prompt_ids = build_inference_prompt(history, user, enc)
        prompt_ids = prompt_ids[-config.max_seq_length:]  # keep within context

        sys.stdout.write("Assistant:")
        sys.stdout.flush()
        reply = generate(
            model, config, device, enc, prompt_ids,
            args.max_new_tokens, args.temperature, args.top_k,
            stop_ids=None, stream=True,
        )
        history = append_reply(history, user, reply, enc)
        history = history[-config.max_seq_length:]


def parse_args():
    p = argparse.ArgumentParser(description="Sample from a miniMoE checkpoint.")
    p.add_argument("-c", "--checkpoint", default="minimoe_step_0019073.pt",
                   help="Path to a .pt checkpoint.")
    p.add_argument("-p", "--prompt", default=None,
                   help="Run once on this prompt and exit (non-interactive).")
    p.add_argument("--chat", action="store_true",
                   help="Conversational wrapper instead of raw completion.")
    p.add_argument("--sft", action="store_true",
                   help="Chat with an SFT'd checkpoint using the training template.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8,
                   help="0 = greedy/deterministic; higher = more random.")
    p.add_argument("--top-k", type=int, default=50,
                   help="Sample from the top-k tokens (0 to disable).")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--bos", action=argparse.BooleanOptionalAction, default=True,
                   help="Prepend the document-start token to the prompt.")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        found = sorted(glob.glob("*.pt")) + sorted(glob.glob("checkpoints/*.pt"))
        message = f"Checkpoint not found: {args.checkpoint}"
        if found:
            message += "\nAvailable checkpoints: " + ", ".join(found)
        else:
            message += "\nNo .pt files found in . or ./checkpoints"
        sys.exit(message)

    device = get_device(args.device)
    enc = tiktoken.get_encoding("gpt2")
    model, config = load_checkpoint(args.checkpoint, device)

    if args.prompt is not None:
        ids = enc.encode(args.prompt)
        if args.bos:
            ids = [EOS_TOKEN_ID] + ids
        sys.stdout.write(args.prompt)
        generate(
            model, config, device, enc, ids,
            args.max_new_tokens, args.temperature, args.top_k,
            stop_ids=None, stream=True,
        )
        return

    if args.sft:
        sft_chat_repl(model, config, device, enc, args)
    elif args.chat:
        chat_repl(model, config, device, enc, args)
    else:
        completion_repl(model, config, device, enc, args)


if __name__ == "__main__":
    main()
