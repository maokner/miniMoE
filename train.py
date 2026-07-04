from model import Model, ModelConfig
import math
import time

import torch
import tiktoken
from torch.utils.data import DataLoader, TensorDataset


def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    return device


def make_loader(tokens, block_size, batch_size, shuffle=True):
    num_blocks = len(tokens) // (block_size + 1)
    tokens = tokens[:num_blocks * (block_size + 1)]
    sequences = tokens.view(num_blocks, block_size + 1)

    x = sequences[:, :-1]
    y = sequences[:, 1:]

    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size, shuffle=shuffle)


def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            total_loss += loss.item()
    model.train()
    return total_loss / len(loader)


def sync_device(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


device = get_device()
print(f"Using device: {device}")

tokenizer = tiktoken.get_encoding("gpt2")
text = open("data/shakespeare.txt", "r").read()
tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)

train = tokens[:int(0.9 * len(tokens))]
test = tokens[int(0.9 * len(tokens)):]

block_size = 1024
batch_size = 16

loader = make_loader(train, block_size, batch_size, shuffle=True)
loader_test = make_loader(test, block_size, batch_size, shuffle=False)

warmup_steps = 10
max_lr = 6e-4
min_lr = max_lr * 0.1
max_steps = 50

model = Model(ModelConfig())
model.to(device)
model = torch.compile(model)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4, device=device)
torch.set_float32_matmul_precision('high')

num_epochs = 1
i = 0
for epoch in range(num_epochs):
    for x, y in loader:
        t0 = time.time()
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(i)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()

        if i == 10:
            break
        sync_device(device)
        t1 = time.time()
        print(f"Epoch {epoch+1}, Step {i+1}, Loss: {loss.item():.4f},norm: {norm.item():.4f},  dt: {1000 * (t1-t0):.2f} ms tok/sec: {batch_size * block_size / (t1-t0):.2f}")

        i += 1

exit()
prompt = torch.tensor(tokenizer.encode("First"), dtype=torch.long)
out = model.generate(prompt, 30, temperature=1, top_k=None)
print(tokenizer.decode(out[0].tolist()))
