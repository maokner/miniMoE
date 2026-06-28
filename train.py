import torch
import torch.nn.functional as F
import tiktoken
from torch.utils.data import DataLoader, Dataset

from model import Model


tokenizer = tiktoken.get_encoding("gpt2")
device = "cpu"


class TokenWindowDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def batch_loss(model, x, y, aux_loss_weight):
    x = x.to(device)
    y = y.to(device)

    logits, aux_loss = model(x)
    lm_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
    )
    loss = lm_loss + aux_loss_weight * aux_loss
    return loss, lm_loss, aux_loss


@torch.no_grad()
def evaluate(model, loader, max_batches, aux_loss_weight):
    model.eval()
    losses = []

    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= max_batches:
            break

        loss, _, _ = batch_loss(model, x, y, aux_loss_weight)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


train_data = "data/shakespeare.txt"
with open(train_data, "r") as f:
    text = f.read()

tokens = tokenizer.encode(text)

# CPU smoke-test settings. Increase these after the loop is working.
block_size = 64
batch_size = 1
train_batches = 4
eval_batches = 4
hidden_dim = 128
aux_loss_weight = 0.01

dataset = TokenWindowDataset(tokens, block_size)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

model = Model(50257, hidden_dim, max_seq_length=1024).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

initial_eval_loss = evaluate(model, eval_loader, eval_batches, aux_loss_weight)
print(f"initial eval loss over {eval_batches} batches: {initial_eval_loss:.4f}")

model.train()
for step, (x, y) in enumerate(train_loader):
    if step >= train_batches:
        break

    optimizer.zero_grad(set_to_none=True)
    loss, lm_loss, aux_loss = batch_loss(model, x, y, aux_loss_weight)
    loss.backward()
    optimizer.step()

    print(
        f"train step {step + 1}: "
        f"loss={loss.item():.4f}, "
        f"lm_loss={lm_loss.item():.4f}, "
        f"aux_loss={aux_loss.item():.4f}"
    )

final_eval_loss = evaluate(model, eval_loader, eval_batches, aux_loss_weight)
print(f"final eval loss over {eval_batches} batches: {final_eval_loss:.4f}")

prompt = torch.tensor(tokenizer.encode("Hi I am Matt, "), dtype=torch.long)
out = model.generate(prompt, 30, temperature=1, top_k=None)
print(tokenizer.decode(out[0].tolist()))
