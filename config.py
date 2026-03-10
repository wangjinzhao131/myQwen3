"""
Qwen3 Model Configuration
==========================
For learning purposes - using a tiny model scale
"""


# Model architecture (tiny version for learning)
class Config:
    # Model dimensions - small for learning (original Qwen3-14B has 5120 hidden, 48 layers)
    vocab_size = 32000  # Vocabulary size
    hidden_size = 512  # Hidden dimension (scaled down)
    intermediate_size = 1376  # FFN intermediate dim (approx 2.67x hidden)
    num_hidden_layers = 8  # Number of transformer layers
    num_attention_heads = 16  # Attention heads
    num_key_value_heads = 2  # GQA: 16/2 = 8 groups

    # Maximum context length
    max_position_embeddings = 512

    # Activation
    hidden_act = "silu"  # SwiGLU uses SiLU

    # Normalization
    rms_norm_eps = 1e-6  # RMSNorm epsilon

    # RoPE
    rope_theta = 10000.0  # Base frequency for RoPE

    # Training
    gradient_accumulation_steps = 4
    learning_rate = 1e-3
    weight_decay = 0.01
    warmup_steps = 100
    max_steps = 1000
    log_interval = 10

    # Optimizer
    betas = (0.9, 0.95)  # Adam betas
    eps = 1e-8

    # For debugging
    use_flash_attention = False

    @classmethod
    def get_attention_scale(cls):
        return (cls.hidden_size // cls.num_attention_heads) ** -0.5


# Alias
cfg = Config()
