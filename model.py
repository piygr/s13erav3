# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    LlamaRMSNorm,
)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class CausalAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads

        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        #self.attention_dropout = attention_dropout
        self.is_causal = True

        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_size, self.head_dim * num_attention_heads, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.head_dim * num_key_value_heads, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.head_dim * num_key_value_heads, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None):
        batch, seq_len = hidden_states.shape[:-1]
        hidden_shape = (batch, seq_len, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        y = F.scaled_dot_product_attention(query_states,
                                           key_states,
                                           value_states,
                                           is_causal=True,
                                           enable_gqa=True)  # Flash attention

        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)  # re-assemble all head outputs side by side
        # output projection
        y = self.o_proj(y)
        return y



class MLP(nn.Module):   ###Inspired from LLamaMLP
    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads, intermediate_size, eps, activation_fn):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = activation_fn

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads, intermediate_size, eps, activation_fn):
        super(TransformerBlock, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        assert self.head_dim * num_attention_heads == hidden_size, "Hidden size must be divisible by the number of attention heads."
        assert self.hidden_size % self.num_key_value_heads == 0, "hidden_size must be divisible by num_key_value_heads"

        self.layer_norm_1 = LlamaRMSNorm(self.hidden_size, eps=eps)

        self.attn = CausalAttention(hidden_size, num_attention_heads, num_key_value_heads)

        # Feedforward layer
        self.feed_forward = MLP(hidden_size, num_attention_heads, num_key_value_heads, intermediate_size, eps, activation_fn)
        self.layer_norm_2 = LlamaRMSNorm(self.hidden_size, eps=eps)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None):
        # Layer normalization
        residual = hidden_states
        hidden_states = self.layer_norm_1(hidden_states)

        '''
        # Query projection
        query = self.query_proj(hidden_states)
        query = query.view(hidden_states.size(0), hidden_states.size(1), self.num_attention_heads,
                           self.head_dim).transpose(1, 2)

        # Key and Value projections with shared num_key_value_heads
        key = self.key_proj(hidden_states)
        value = self.value_proj(hidden_states)

        key = key.view(hidden_states.size(0), hidden_states.size(1), self.num_key_value_heads,
                       self.head_dim).transpose(1, 2)
        value = value.view(hidden_states.size(0), hidden_states.size(1), self.num_key_value_heads,
                           self.head_dim).transpose(1, 2)

        # Expand keys and values to match num_attention_heads
        key = key.repeat_interleave(self.num_attention_heads // self.num_key_value_heads, dim=1)
        value = value.repeat_interleave(self.num_attention_heads // self.num_key_value_heads, dim=1)

        # Apply rotary embeddings to query and key
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Scaled dot-product attention
        attention_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, is_causal=True)

        # Reshape back to [batch_size, seq_length, hidden_size]
        attention_output = attention_output.transpose(1, 2).contiguous().view(hidden_states.size(0), -1,
                                                                              self.hidden_size)

        # Output projection
        attention_output = self.out_proj(attention_output)
        '''
        attention_output = self.attn(hidden_states, position_embeddings=position_embeddings)

        # Residual connection
        hidden_states = residual + attention_output

        # Feedforward layer
        residual = hidden_states

        # Feed-forward
        hidden_states = self.layer_norm_2(hidden_states)
        feed_forward_output = self.feed_forward(hidden_states)

        hidden_states = residual + feed_forward_output

        return hidden_states


class SmollM(nn.Module):
    def __init__(self, config):
        super(SmollM, self).__init__()
        self.vocab_size = config['vocab_size']
        self.hidden_size = config['hidden_size']
        self.num_hidden_layers = config['num_hidden_layers']
        self.num_attention_heads = config['num_attention_heads']
        self.num_key_value_heads = config['num_key_value_heads']
        self.max_position_embeddings = config['max_position_embeddings']
        self.intermediate_size = config['intermediate_size']
        self.initializer_range = config['initializer_range']
        self.eps = config['rms_norm_eps']

        self.head_dim = self.hidden_size // self.num_attention_heads

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim)

        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                intermediate_size=self.intermediate_size,
                eps=self.eps,
                activation_fn=F.silu  # Activation function specified in config
            ) for _ in range(self.num_hidden_layers)
        ])

        self.layer_norm = LlamaRMSNorm(self.hidden_size, eps=self.eps)

        # Language modeling head
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        # Share weights between embedding and lm_head
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.size()
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        embeddings = self.embedding(input_ids)

        hidden_states = embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings)

        hidden_states = self.layer_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

