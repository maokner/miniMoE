import torch
import torch.nn.functional as F
import tiktoken
from torch.utils.data import DataLoader, Dataset, TensorDataset

from model import Model, ModelConfig

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
print(f"Using device: {device}")


tokenizer = tiktoken.get_encoding("gpt2")
text = open("data/shakespeare.txt", "r").read()
tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)

block_size = 128

num_blocks = len(tokens) // (block_size + 1)
tokens = tokens[:num_blocks * (block_size + 1)]

sequences = tokens.view(num_blocks, block_size + 1)

x = sequences[:, :-1]  # [num_blocks, 128]
y = sequences[:, 1:]   # [num_blocks, 128]

dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = Model(ModelConfig())
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
i = 0
for x, y in loader:
    if i == 3:
        break
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"Step {i+1}, Loss: {loss.item()}")
    i += 1


prompt = torch.tensor(tokenizer.encode("First"), dtype=torch.long)
out = model.generate(prompt, 30, temperature=1, top_k=None)
print(tokenizer.decode(out[0].tolist()))
