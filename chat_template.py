"""
Canonical chat template for miniMoE SFT.

This is the single source of truth for how conversations are formatted, shared
by the SFT data pipeline (sft_data.py) and inference (sample.py --sft) so the
two can never drift apart. Getting these to match is the whole game in SFT: the
model only behaves at inference if it sees exactly the format it was trained on.

Design notes for a GPT-2 BPE tokenizer with no special chat tokens:

  - We reuse the plain-text role markers "User:" / "Assistant:" that already
    tokenize cleanly, rather than inventing special tokens the base model has
    never seen.
  - A sequence starts with EOS (the document-start convention the base model
    was pretrained with) and every assistant turn *ends* with EOS. That trailing
    EOS is what teaches the model to stop; at inference we halt on it.
  - The invariant is: an assistant turn is always preceded by "Assistant:" and
    always followed by EOS; a user turn always follows an EOS (either the leading
    one or the previous turn's terminator). This makes single- and multi-turn
    rendering identical, so multi-turn inference is just "append and continue".

A message is a dict {"role": "system"|"user"|"assistant", "content": str}.
"""

EOS_TOKEN_ID = 50256


def _user_prompt(content):
    # The prefix the model is conditioned on (masked out of the SFT loss). The
    # trailing "Assistant:" with no space is exactly what we generate from.
    return f"User: {content.strip()}\nAssistant:"


def _system_prefix(content):
    return f"System: {content.strip()}\n"


def render_conversation(messages, enc):
    """Render one conversation into aligned (tokens, is_target) lists.

    tokens[i] is a token id; is_target[i] is 1 if position i should contribute
    to the loss (assistant content and its terminating EOS) and 0 otherwise
    (leading EOS, system text, user turns, the "Assistant:" prefix). The two
    lists have the same length; the caller shifts them into inputs/labels.

    Returns (tokens, is_target), or (None, None) if the conversation has no
    usable assistant turn.
    """
    tokens = [EOS_TOKEN_ID]
    is_target = [0]

    # A leading system message becomes a masked "System: ..." preamble.
    idx = 0
    if messages and messages[0]["role"] == "system":
        for tok in enc.encode(_system_prefix(messages[0]["content"])):
            tokens.append(tok)
            is_target.append(0)
        idx = 1

    saw_assistant = False
    pending_user = None
    for message in messages[idx:]:
        role = message["role"]
        content = message["content"]
        if role == "user":
            pending_user = content
        elif role == "assistant":
            if pending_user is None:
                # Assistant turn with no preceding user turn: skip, it would
                # teach the model to speak unprompted.
                continue
            for tok in enc.encode(_user_prompt(pending_user)):
                tokens.append(tok)
                is_target.append(0)
            # The leading space is part of the completion so the model learns
            # to produce it; content + terminal EOS are the supervised targets.
            for tok in enc.encode(f" {content.strip()}"):
                tokens.append(tok)
                is_target.append(1)
            tokens.append(EOS_TOKEN_ID)
            is_target.append(1)
            saw_assistant = True
            pending_user = None

    if not saw_assistant:
        return None, None
    return tokens, is_target


def build_inference_prompt(history, user_message, enc):
    """Token ids for a fresh generation, given prior turns and a new user turn.

    `history` is the running list of token ids from previous turns (start it as
    []); this returns the full context ending in "Assistant:", ready to feed to
    generate(). Mirrors render_conversation exactly.
    """
    ids = list(history) if history else [EOS_TOKEN_ID]
    ids += enc.encode(_user_prompt(user_message))
    return ids


def append_reply(history, user_message, reply, enc):
    """Fold a completed turn back into history for the next round.

    Produces the same token stream render_conversation would for this turn:
    the user prompt, a leading-space reply, and the terminating EOS.
    """
    ids = list(history) if history else [EOS_TOKEN_ID]
    ids += enc.encode(_user_prompt(user_message))
    ids += enc.encode(f" {reply.strip()}")
    ids.append(EOS_TOKEN_ID)
    return ids
