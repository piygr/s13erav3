# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, eps, activation_fn):
        super(TransformerBlock, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        assert self.head_dim * num_attention_heads == hidden_size, "Hidden size must be divisible by the number of attention heads."

        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.layer_norm_1 = nn.LayerNorm(hidden_size, eps=eps)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            activation_fn,  # Use the activation function specified
            nn.Linear(intermediate_size, hidden_size)
        )

        self.layer_norm_2 = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, hidden_states, attention_mask=None):
        # Layer normalization
        normed_hidden_states = self.layer_norm_1(hidden_states)

        # Compute Q, K, V
        query = self.query_proj(normed_hidden_states)
        key = self.key_proj(normed_hidden_states)
        value = self.value_proj(normed_hidden_states)

        # Reshape for multi-head attention
        batch_size, seq_length, _ = query.size()
        query = query.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)

        # Apply scaled dot-product attention with causal masking ###FlashAttention
        attention_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, is_causal=True)

        # Reshape back to [batch_size, seq_length, hidden_size]
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)

        # Output projection
        attention_output = self.out_proj(attention_output)

        # Residual connection
        hidden_states = hidden_states + attention_output

        # Feed-forward
        normed_hidden_states = self.layer_norm_2(hidden_states)
        feed_forward_output = self.feed_forward(normed_hidden_states)
        hidden_states = hidden_states + feed_forward_output

        return hidden_states


class SmollM(nn.Module):
    def __init__(self, config):
        super(SmollM, self).__init__()
        self.vocab_size = config['vocab_size']      #power of 2
        self.hidden_size = config['hidden_size']
        self.num_hidden_layers = config['num_hidden_layers']
        self.num_attention_heads = config['num_attention_heads']
        self.max_position_embeddings = config['max_position_embeddings']
        self.intermediate_size = config['intermediate_size']
        self.initializer_range = config['initializer_range']
        self.eps = config['rms_norm_eps']

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.position_embeddings = nn.Embedding(self.max_position_embeddings, self.hidden_size)

        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
                eps=self.eps,
                activation_fn=F.silu  # Activation function specified in config
            ) for _ in range(self.num_hidden_layers)
        ])

        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=self.eps)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        self.embedding.weight = self.lm_head.weight #### Weight Sharing

        self._init_weights()

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.size()
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        embeddings = self.embedding(input_ids) + self.position_embeddings(position_ids)
        hidden_states = embeddings

        for layer in self.layers:
            hidden_states = layer(hidden_states, src_key_padding_mask=attention_mask)

        hidden_states = self.layer_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)    #### SmoLLM Std applied as per the config
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)    #### SmoLLM Std applied as per the config
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

