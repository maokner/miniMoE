import torch
import torch.nn as nn
from torch.nn import functional as F

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

class Expert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
    def forward(self, x):
        return self.net(x)


class MoEFeedForward(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts=8, top_k=2):
        super().__init__()
        self.top_k = top_k

        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim) for _ in range(num_experts)])
        self.router = nn.Linear(input_dim, num_experts, bias=False)
    
    def forward(self, x):
        #x = [batch_size, sequence_length, input_dim]
        B, T, D = x.shape
        num_tokens = B * T
        num_experts = len(self.experts)

        router_logits = self.router(x)  #[batch_size, sequence_length, num_experts] 

        router_probs = F.softmax(router_logits, dim=-1) # [B, T, E]

        topk_logits, topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1)

        flat_x = x.reshape(num_tokens, D) # [batch_size * sequence_length, input_dim]

        flat_topk_indices = topk_indices.reshape(num_tokens, self.top_k)  # [batch_size * sequence_length, top_k]
        flat_topk_weights = topk_weights.reshape(num_tokens, self.top_k)  #[batch_size * sequence_length, top_k]
        
        flat_output = torch.zeros_like(flat_x)  # [batch_size * sequence_length, input_dim]

        for expert_id, expert in enumerate(self.experts):
            selected = (flat_topk_indices == expert_id)
            token_ids, topk_slots = torch.where(selected)  

            if token_ids.numel() == 0:
                continue

            expert_input = flat_x[token_ids]

            expert_output = expert(expert_input)

            routing_weights = flat_topk_weights[token_ids, topk_slots].unsqueeze(-1) #number of tokens routed here , 1
            weighted_output =  routing_weights * expert_output

            flat_output = flat_output.index_add(0,token_ids, weighted_output)
        
        selected_mask = F.one_hot(topk_indices, num_classes=num_experts).float() #[B, T, K, E] : for each expert K selected add one-hot encoding
        load = selected_mask.sum(dim=2).mean(dim=(0, 1)) / self.top_k # [E] containing relative percentages of each expert being selected
        importance = router_probs.mean(dim=(0, 1)) # [E] average probability assigned to each expert
        aux_loss = num_experts * torch.sum(load * importance)
        output = flat_output.reshape(B, T, D)  # [batch_size, sequence_length, input_dim]

        return output, aux_loss 
    
class TransformerMoEBlock(nn.Module):
    def __init__(self, hidden_dim, num_experts=8, top_k=2):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.moe = MoEFeedForward(hidden_dim, hidden_dim * 4, num_experts=num_experts, top_k=top_k)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.moe_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, causal_mask=None):
        seq_len = x.size(1)
        if causal_mask is None:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)

        x_norm = self.attn_norm(x)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm, attn_mask=causal_mask, need_weights=False)

        x = x + attn_output  # Residual connection
        moe_output, aux_loss = self.moe(self.moe_norm(x))
        x = x + moe_output
        return x, aux_loss






class Model(torch.nn.Module):
    def __init__(self, vocab_size, hidden_dim, max_seq_length=1024):
        super(Model, self).__init__()
        self.max_seq_length = max_seq_length
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.positional_embedding = nn.Embedding(max_seq_length, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.MoEBlocks = nn.ModuleList([TransformerMoEBlock(hidden_dim) for _ in range(8)])
        self.final_norm = nn.LayerNorm(hidden_dim)




    def forward(self, x):
        # X = [Batch size, Sequence length ] need to convert to [batch size, Sequence length, hidden_dim] for attention
        x = x.long()  # Ensure input is of type long for embedding
        x = self.token_embedding(x) # [Batch size, Sequence length, hidden_dim]
        B, T, D = x.shape
        pos = torch.arange(T, device=x.device)
    
        x = x + self.positional_embedding(pos)  # [Batch size, Sequence length, hidden_dim]

        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)

        aux_losses = []

        for block in self.MoEBlocks:
            x, aux_loss = block(x, causal_mask=causal_mask)
            aux_losses.append(aux_loss)
        moe_aux_loss = torch.stack(aux_losses).mean()

        x = self.final_norm(x)

        logits = self.output_projection(x)
        return logits, moe_aux_loss

    @torch.no_grad()
    def generate(self, x, max_new_tokens, temperature=1.0, top_k=None):
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")

        was_training = self.training
        self.eval()

        try:
            if x.dim() == 1:
                x = x.unsqueeze(0)

            x = x.to(next(self.parameters()).device).long()

            for _ in range(max_new_tokens):
                x_context = x[:, -self.max_seq_length:]
                logits, _ = self(x_context)
                next_token_logits = logits[:, -1, :]

                if temperature == 0:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    next_token_logits = next_token_logits / temperature

                    if top_k is not None:
                        sample_top_k = min(top_k, next_token_logits.size(-1))
                        values, _ = torch.topk(next_token_logits, sample_top_k)
                        next_token_logits[next_token_logits < values[:, [-1]]] = -float("inf")

                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                if next_token == 50256:   #EOS token
                    return x

                x = torch.cat((x, next_token), dim=1)

            return x

        finally:
            if was_training:
                self.train()
