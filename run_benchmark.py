"""
HTP vs Standard Embedding — Full Benchmark

Trains both models on the same data under identical conditions,
then evaluates them on 20 prompts and prints a side-by-side comparison.

Usage:
    python run_benchmark.py

No GPU required. Runs in ~30 minutes on a modern CPU.
"""

import os
import glob
import math
import json
import torch
from torch.utils.data import Dataset, DataLoader

from models import MicroLLM_HTP, MicroLLM_Standard
from htp_embedding import WordTokenizer


# ─── Configuration ──────────────────────────────────────────────

DATA_DIR = "data"
SEQ_LEN = 128
BATCH_SIZE = 32
EPOCHS = 15
LR = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

PROMPTS = [
    # Structured Q&A (10)
    "Pergunta: Qual é o maior bioma do Brasil? Resposta:",
    "Pergunta: Qual é o maior rio do Brasil? Resposta:",
    "Pergunta: Quantos estados tem o Brasil? Resposta:",
    "Pergunta: O que é o Cerrado? Resposta:",
    "Pergunta: O que é o Pantanal? Resposta:",
    "Pergunta: Qual é o ponto mais alto do Brasil? Resposta:",
    "Pergunta: Quais países fazem fronteira com o Brasil? Resposta:",
    "Pergunta: O que é a Caatinga? Resposta:",
    "Pergunta: Qual é o clima predominante no Brasil? Resposta:",
    "Pergunta: O que é a Mata Atlântica? Resposta:",
    # Free-form completions (7)
    "A região nordeste do Brasil",
    "A Amazônia é",
    "O Pantanal é uma região",
    "O clima do Brasil",
    "Os rios do Brasil",
    "A costa do Brasil",
    "O relevo brasileiro",
    # Direct questions (3)
    "Qual é a capital do Brasil",
    "O Brasil tem quantos estados",
    "O rio Amazonas",
]


# ─── Dataset ────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.seq_len = seq_len
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.n_samples = max(0, len(tokens) - seq_len)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (self.tokens[idx : idx + self.seq_len],
                self.tokens[idx + 1 : idx + self.seq_len + 1])


# ─── Training ───────────────────────────────────────────────────

def train_model(model, dataloader, epochs=EPOCHS):
    """Train a model and return loss history."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history = []

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss, n_batches = 0.0, 0
        for bx, by in dataloader:
            optimizer.zero_grad()
            _, loss = model(bx, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        history.append(avg_loss)
        print(f"    Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f}")

    return history


# ─── Generation ─────────────────────────────────────────────────

def generate_responses(model, tokenizer, prompts, max_tokens=50, temperature=0.7, top_k=30):
    """Generate responses for a list of prompts."""
    model.eval()
    results = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special=False)
        tensor = torch.tensor([ids], dtype=torch.long)
        out = model.generate(tensor, max_new_tokens=max_tokens,
                             temperature=temperature, top_k=top_k)
        new_tokens = out[0, len(ids):].tolist()
        response = tokenizer.decode(new_tokens)
        results.append({"prompt": prompt, "response": response})
    return results


# ─── Main ───────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  HTP vs Standard Embedding — Micro-LLM Benchmark")
    print("=" * 70)

    # 1. Load data
    print("\n[1/6] Loading data...")
    texts = []
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "*.txt"))):
        with open(f, "r", encoding="utf-8") as fh:
            text = fh.read().strip()
            if text:
                texts.append(text)
                print(f"  {f} ({len(text):,} chars)")

    if not texts:
        print(f"ERROR: No .txt files found in {DATA_DIR}/")
        return

    # 2. Tokenize
    print("\n[2/6] Building tokenizer...")
    tokenizer = WordTokenizer()
    tokenizer.build(texts)

    full_text = "\n\n".join(texts)
    all_tokens = tokenizer.encode(full_text, add_special=False)
    print(f"  Tokens: {len(all_tokens):,} | Vocab: {tokenizer.vocab_size}")

    while len(all_tokens) < SEQ_LEN + 1:
        all_tokens = all_tokens + all_tokens

    dataset = TextDataset(all_tokens, SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"  Samples: {len(dataset):,} | Batches/epoch: {len(dataloader)}")

    vocab_size = tokenizer.vocab_size

    # 3. Train HTP model
    print("\n[3/6] Training HTP model...")
    htp_model = MicroLLM_HTP(vocab_size=vocab_size)
    htp_model.htp.build_table(tokenizer)
    print(f"  Trainable parameters: {htp_model.count_trainable_parameters():,}")
    htp_history = train_model(htp_model, dataloader)

    # 4. Train Standard model
    print("\n[4/6] Training Standard model...")
    std_model = MicroLLM_Standard(vocab_size=vocab_size)
    print(f"  Trainable parameters: {std_model.count_trainable_parameters():,}")
    std_history = train_model(std_model, dataloader)

    # 5. Generate responses
    print("\n[5/6] Generating responses...")
    print("  HTP model...")
    htp_results = generate_responses(htp_model, tokenizer, PROMPTS)
    print("  Standard model...")
    std_results = generate_responses(std_model, tokenizer, PROMPTS)

    # 6. Print comparison
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    print(f"\n  HTP model:      {htp_model.count_trainable_parameters():,} params | Final loss: {htp_history[-1]:.4f}")
    print(f"  Standard model: {std_model.count_trainable_parameters():,} params | Final loss: {std_history[-1]:.4f}")

    print("\n  Training loss comparison:")
    print(f"  {'Epoch':<8} {'HTP':<12} {'Standard':<12}")
    print(f"  {'-'*32}")
    for ep in [0, 4, 9, 14]:
        if ep < len(htp_history):
            print(f"  {ep+1:<8} {htp_history[ep]:<12.4f} {std_history[ep]:<12.4f}")

    print("\n" + "-" * 70)
    print("  GENERATION COMPARISON (20 prompts)")
    print("-" * 70)

    for i, (h, s) in enumerate(zip(htp_results, std_results)):
        print(f"\n  [{i+1:2d}] Prompt: {h['prompt']}")
        print(f"       HTP: {h['response'][:120]}...")
        print(f"       Std: {s['response'][:120]}...")

    # Save results
    output = {
        "htp_params": htp_model.count_trainable_parameters(),
        "std_params": std_model.count_trainable_parameters(),
        "htp_loss_history": htp_history,
        "std_loss_history": std_history,
        "htp_responses": htp_results,
        "std_responses": std_results,
    }
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  Full results saved to benchmark_results.json")
    print("=" * 70)
    print("  Benchmark complete. Review responses and score manually.")
    print("  Scoring rubric: Correct / Wrong / Incoherent")
    print("=" * 70)


if __name__ == "__main__":
    main()
