"""
Qwen3 Model Implementation - From Scratch
Using numpy only for matrix operations
"""

import numpy as np
from config import cfg
from tensor_ops import softmax, silu, rms_norm, apply_rope, causal_mask


class Linear:
    def __init__(self, in_features, out_features, bias=True):
        scale = np.sqrt(2.0 / in_features)
        self.weight = (
            np.random.randn(in_features, out_features).astype(np.float32) * scale
        )
        self.bias = np.zeros(out_features, dtype=np.float32) if bias else None

    def __call__(self, x):
        if self.bias is not None:
            return x @ self.weight + self.bias
        return x @ self.weight

    def parameters(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params


class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.embedding = (
            np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.02
        )

    def __call__(self, x):
        return self.embedding[x]

    def parameters(self):
        return [self.embedding]


class GroupedQueryAttention:
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.q_proj = Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, x, position_ids, attention_mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )
        v = self.v_proj(x).reshape(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )

        q = np.transpose(q, (0, 2, 1, 3))
        k = np.transpose(k, (0, 2, 1, 3))
        v = np.transpose(v, (0, 2, 1, 3))

        q, k = apply_rope(q, k, position_ids, cfg.rope_theta)

        num_kv_groups = self.num_heads // self.num_kv_heads
        if num_kv_groups > 1:
            k = np.repeat(k, num_kv_groups, axis=1)
            v = np.repeat(v, num_kv_groups, axis=1)

        attn_weights = np.matmul(q, np.transpose(k, (0, 1, 3, 2))) / np.sqrt(
            self.head_dim
        )

        mask = causal_mask(seq_len)
        attn_weights = attn_weights + np.where(mask, 0, -1e9)

        attn_probs = softmax(attn_weights, axis=-1)
        attn_output = np.matmul(attn_probs, v)

        attn_output = np.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)

        return self.o_proj(attn_output)

    def parameters(self):
        return (
            self.q_proj.parameters()
            + self.k_proj.parameters()
            + self.v_proj.parameters()
            + self.o_proj.parameters()
        )


class SwiGLUFFN:
    def __init__(self, hidden_size, intermediate_size):
        self.w1 = Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))

    def parameters(self):
        return self.w1.parameters() + self.w3.parameters() + self.w2.parameters()


class TransformerBlock:
    def __init__(
        self, hidden_size, num_heads, num_kv_heads, intermediate_size, rms_norm_eps
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.attention = GroupedQueryAttention(
            hidden_size, num_heads, num_kv_heads, hidden_size // num_heads
        )
        self.ffn = SwiGLUFFN(hidden_size, intermediate_size)

        self.input_layernorm_weight = np.ones(hidden_size, dtype=np.float32)
        self.post_attention_layernorm_weight = np.ones(hidden_size, dtype=np.float32)

    def forward(self, x, position_ids, attention_mask=None):
        residual = x
        x = rms_norm(x, self.input_layernorm_weight, cfg.rms_norm_eps)
        x = self.attention.forward(x, position_ids, attention_mask)
        x = x + residual

        residual = x
        x = rms_norm(x, self.post_attention_layernorm_weight, cfg.rms_norm_eps)
        x = self.ffn(x)
        x = x + residual

        return x

    def parameters(self):
        params = self.attention.parameters() + self.ffn.parameters()
        params.extend(
            [self.input_layernorm_weight, self.post_attention_layernorm_weight]
        )
        return params


class Qwen3Model:
    def __init__(self, config):
        self.config = config

        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)

        self.layers = []
        for _ in range(config.num_hidden_layers):
            self.layers.append(
                TransformerBlock(
                    config.hidden_size,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.intermediate_size,
                    config.rms_norm_eps,
                )
            )

        self.norm = np.ones(config.hidden_size, dtype=np.float32)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        position_ids = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]

        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer.forward(hidden_states, position_ids, attention_mask)

        hidden_states = rms_norm(hidden_states, self.norm, self.cfg_rms_norm_eps)

        logits = self.lm_head(hidden_states)
        return logits

    @property
    def cfg_rms_norm_eps(self):
        return self.config.rms_norm_eps

    def parameters(self):
        params = self.embed_tokens.parameters()
        for layer in self.layers:
            params.extend(layer.parameters())
        params.extend([self.norm])
        params.extend(self.lm_head.parameters())
        return params


def cross_entropy_loss(logits, targets):
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    logits_max = np.max(logits_flat, axis=-1, keepdims=True)
    logits_exp = np.exp(logits_flat - logits_max)
    probs = logits_exp / np.sum(logits_exp, axis=-1, keepdims=True)

    target_probs = probs[np.arange(batch_size * seq_len), targets_flat]
    loss = -np.mean(np.log(target_probs + 1e-9))

    return loss


def test_model():
    model = Qwen3Model(cfg)
    input_ids = np.random.randint(0, cfg.vocab_size, (2, 10))
    logits = model.forward(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Expected shape: {(2, 10, {cfg.vocab_size})}")


if __name__ == "__main__":
    test_model()
