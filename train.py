"""
Training Loop - From Scratch
Implements forward pass, backward pass (numerical gradients), and optimization
"""

import numpy as np
from config import cfg
from model import Qwen3Model, cross_entropy_loss
from optimizer import AdamW
from tokenizer import SimpleTokenizer


class Dataset:
    def __init__(self, text, tokenizer, seq_len=64):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        tokenizer.train(text)
        self.ids = np.array(tokenizer.encode(text), dtype=np.int64)

    def __len__(self):
        return max(1, len(self.ids) - self.seq_len)

    def get_batch(self, batch_size=4):
        indices = np.random.randint(0, len(self), batch_size)

        x = np.stack([self.ids[i : i + self.seq_len] for i in indices])
        y = np.stack([self.ids[i + 1 : i + self.seq_len + 1] for i in indices])

        return x, y


def forward_pass(model, input_ids, targets=None):
    logits = model.forward(input_ids)
    loss = None
    if targets is not None:
        loss = cross_entropy_loss(logits, targets)
    return logits, loss


def compute_loss_with_params(model, params, param_indices, input_ids, targets):
    original_params = []
    for idx, param_idx in enumerate(param_indices):
        original_params.append(params[idx].copy())

    for idx, param_idx in enumerate(param_indices):
        model_param = model.parameters()[param_idx]
        model_param[:] = params[idx]

    _, loss = forward_pass(model, input_ids, targets)
    return loss


def backward_pass(model, input_ids, targets, eps=1e-5):
    grads = []
    params = model.parameters()

    _, base_loss = forward_pass(model, input_ids, targets)

    for param in params:
        grad = np.zeros_like(param)
        it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])

        while not it.finished:
            idx = it.multi_index
            old_val = param[idx]

            param[idx] = old_val + eps
            _, loss_plus = forward_pass(model, input_ids, targets)

            param[idx] = old_val - eps
            _, loss_minus = forward_pass(model, input_ids, targets)

            param[idx] = old_val
            grad[idx] = (loss_plus - loss_minus) / (2 * eps)
            it.iternext()

        grads.append(grad)

    return grads, base_loss


def train(model, dataset, optimizer, num_steps=100, log_interval=10):
    losses = []

    for step in range(num_steps):
        input_ids, targets = dataset.get_batch(batch_size=2)

        grads, loss = backward_pass(model, input_ids, targets)

        optimizer.step(grads)

        losses.append(loss)

        if step % log_interval == 0:
            avg_loss = (
                np.mean(losses[-log_interval:])
                if len(losses) >= log_interval
                else np.mean(losses)
            )
            print(f"Step {step:4d} | Loss: {loss:.4f} | Avg Loss: {avg_loss:.4f}")

    return losses


def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0):
    input_ids = np.array([tokenizer.encode(prompt)], dtype=np.int64)

    for _ in range(max_new_tokens):
        if input_ids.shape[1] > cfg.max_position_embeddings:
            input_ids = input_ids[:, -cfg.max_position_embeddings :]

        logits = model.forward(input_ids)
        next_token_logits = logits[0, -1, :] / temperature

        probs = np.exp(next_token_logits - np.max(next_token_logits))
        probs = probs / np.sum(probs)

        next_token = np.random.choice(len(probs), p=probs)

        input_ids = np.concatenate([input_ids, [[next_token]]], axis=1)

    return tokenizer.decode(input_ids[0].tolist())


def main():
    print("=" * 50)
    print("Qwen3 From Scratch - Training Demo")
    print("=" * 50)

    corpus = (
        """
    Once upon a time, there was a little girl named Alice.
    She lived in a small house near a big forest.
    One day, she decided to explore the forest.
    She walked and walked until she found a rabbit hole.
    The rabbit hole was deep and dark.
    Alice was curious, so she jumped into the hole.
    She fell down, down, down into a magical world.
    In this world, she met many strange creatures.
    There was a white rabbit with a pocket watch.
    There was a smiling cat that could disappear.
    There was a mad hatter who loved tea parties.
    Alice had many adventures in this wonderland.
    She grew tall and she grew small.
    She played croquet with the Queen of Hearts.
    Finally, she woke up and realized it was all a dream.
    """
        * 10
    )

    tokenizer = SimpleTokenizer(vocab_size=1000)
    dataset = Dataset(corpus, tokenizer, seq_len=32)

    print(f"\nCorpus size: {len(corpus)} chars")
    print(f"Tokenized: {len(dataset.ids)} tokens")
    print(f"Vocab size: {tokenizer.vocab_size_actual}")
    print(f"Dataset batches: {len(dataset)}")

    model = Qwen3Model(cfg)
    params = model.parameters()
    print(f"\nModel parameters: {sum(p.size for p in params):,}")

    optimizer = AdamW(params, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)

    print("\nStarting training...")
    print("-" * 50)

    losses = train(model, dataset, optimizer, num_steps=50, log_interval=10)

    print("-" * 50)
    print(f"Final loss: {losses[-1]:.4f}")

    print("\nGenerating text...")
    prompt = "once upon a time"
    generated = generate(model, tokenizer, prompt, max_new_tokens=30, temperature=0.8)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")


if __name__ == "__main__":
    main()
