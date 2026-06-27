import tiktoken
from model import Model, device
import torch
tokenizer = tiktoken.get_encoding("gpt2")

train_data = "data/shakespeare.txt"
text = open(train_data, "r").read()
tokens = tokenizer.encode(text)

def get_batch(tokens, batch_size, start_idx, block_size):
    end_idx = start_idx + block_size
    if end_idx >= len(tokens):
        end_idx = len(tokens) - 1
        start_idx = end_idx - block_size
    x = tokens[start_idx:end_idx]
    y = tokens[start_idx + 1:end_idx + 1]
    return torch.tensor(x, dtype=torch.long).unsqueeze(0), torch.tensor(y, dtype=torch.long).unsqueeze(0)