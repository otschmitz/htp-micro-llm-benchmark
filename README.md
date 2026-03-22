# HTP vs Standard Embedding — Micro-LLM Benchmark

A controlled experiment comparing **Harmonic Token Projection (HTP)** against **standard learned embeddings** (`nn.Embedding`) as the embedding layer in micro-scale language models.

Both models share the same transformer backbone and are trained on the same data under identical conditions. The only difference is how words are mapped to vectors.

> **Paper**: Schmitz, T. (2025). *Harmonic Token Projection as Embedding Layer for Micro-Scale Language Models: A Preliminary Comparison Against Learned Embeddings*. Included as `paper.pdf`.
>
> **HTP Reference**: arXiv:2511.20665 (Schmitz, 2025)

---

## What is this?

This repository contains everything needed to reproduce a comparison between two embedding strategies for small language models:

- **HTP-LLM**: Uses HTP deterministic embeddings (0 learnable parameters in the embedding layer) plus a small 24-dim learned complement. Total: ~221K parameters.
- **Std-LLM**: Uses a standard `nn.Embedding` layer (58,880 learnable parameters). Total: ~254K parameters.

Both models use the same transformer architecture:
- 2 decoder layers
- 4 attention heads (head dim = 20)
- SwiGLU feed-forward (hidden dim = 176)
- RoPE position encoding
- RMSNorm (pre-normalization)
- Model dimension: 80

They are trained on the same Portuguese-language corpus (~2,800 word-level tokens about Brazilian geography) and evaluated on the same 20 prompts.

---

## Our observations (preliminary)

In our single-run experiment:

| | HTP-LLM | Std-LLM |
|---|---|---|
| Parameters | 220,944 | 253,840 |
| Final loss (15 epochs) | 0.087 | 0.090 |
| Correct responses (of 20) | 9 | 7 |
| Incoherent responses (of 20) | 6 | 8 |

These results are from a small corpus with manual evaluation. They suggest patterns worth investigating but are not sufficient for strong claims. We encourage you to run the benchmark yourself and draw your own conclusions.

---

## Quick start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/htp-micro-llm-benchmark.git
cd htp-micro-llm-benchmark

# 2. Create a virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Run the full benchmark (trains both models + generates responses)
python run_benchmark.py
```

No GPU required. The full benchmark takes approximately 30 minutes on a modern CPU.

---

## Repository structure

```
htp-micro-llm-benchmark/
├── README.md                 # This file
├── requirements.txt          # Python dependencies (torch, numpy)
├── run_benchmark.py          # Main script: trains both models, runs 20 prompts
├── models.py                 # Both model architectures (shared transformer)
├── htp_embedding.py          # HTP embedding + word tokenizer
├── data/
│   └── brasil_geografia_+2.5k_tokens.txt   # Training corpus (~2,800 tokens)
├── paper.pdf                 # Full paper with methodology and analysis
├── paper.tex                 # LaTeX source of the paper
└── references.bib            # Bibliography
```

---

## How it works

### The embedding bottleneck

In micro-scale models (<500K parameters), the embedding layer can consume a disproportionate share of the parameter budget. With a vocabulary of 736 words and dimension 80, the standard embedding requires 58,880 parameters — about 23% of the total model.

HTP eliminates this cost by computing embeddings deterministically from each word's Unicode characters, through modular arithmetic with prime numbers and sine/cosine functions. The result: zero learnable parameters in the embedding layer.

### What `run_benchmark.py` does

1. Loads the training data from `data/`
2. Builds a word-level tokenizer
3. Trains the HTP model for 15 epochs
4. Trains the Standard model for 15 epochs (same data, same hyperparameters)
5. Generates responses for 20 prompts using both models
6. Prints a side-by-side comparison
7. Saves full results to `benchmark_results.json`

### Scoring

After running the benchmark, manually score each response as:

- **Correct**: Factually accurate, directly addresses the prompt, grammatically fluent
- **Wrong**: Coherent but factually incorrect or answers a different question
- **Incoherent**: Broken grammar, repetitive loops, or nonsensical text

---

## Training details

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3 × 10⁻⁴ |
| Weight decay | 0.1 |
| Batch size | 32 |
| Sequence length | 128 |
| Epochs | 15 |
| Gradient clipping | 1.0 (max norm) |
| Initialization | Xavier uniform |

---

## Extending the experiment

We encourage researchers to extend this work. Some ideas:

- **More data**: Add more `.txt` files to `data/` to increase corpus size
- **Multiple seeds**: Run the benchmark multiple times with different random seeds
- **Automated evaluation**: Add perplexity on held-out data, BLEU scores, or factual accuracy classifiers
- **Different languages**: Test with corpora in other languages
- **Larger models**: Modify `models.py` to increase layers, heads, or dimension
- **Subword tokenization**: Replace the word-level tokenizer with BPE or SentencePiece

To add your own training data, simply place `.txt` files in the `data/` directory. The tokenizer will automatically build a vocabulary from all files.

---

## What HTP is

Harmonic Token Projection (HTP) converts any word into a fixed-size vector in three steps:

1. **Integer encoding**: The word's Unicode characters are combined into a large integer (base-65536)
2. **Modular residues**: The integer is divided by each of the first K prime numbers; only remainders are kept
3. **Harmonic projection**: Each remainder is mapped to sine/cosine coordinates

The result is a deterministic, unique embedding that requires no training. The trade-off is that semantically similar words (e.g., "hot" and "warm") have no inherent proximity in HTP space — the transformer must learn all semantic relationships from context.

For the full mathematical description, see [arXiv:2511.20665](https://arxiv.org/abs/2511.20665).

---

## Important caveats

- The training corpus is very small (~2,800 tokens). Both models are likely memorizing rather than generalizing.
- Evaluation is manual and based on 20 prompts. The sample size is too small for statistical significance.
- Results are from a single training run per model. Stochastic variation in initialization may affect outcomes.
- All data is in Portuguese and concerns a single domain (geography).
- HTP embeddings carry no semantic information. This may be a significant limitation in other contexts.

If HTP's apparent advantages disappear under more rigorous testing, that is a valuable finding. We are sharing this work precisely so it can be challenged.

---

## Citation

If you use this benchmark or build on this work, please cite:

```bibtex
@article{schmitz2025htp,
  author  = {Schmitz, T.},
  title   = {Harmonic Token Projection: Deterministic Embeddings via
             Modular Arithmetic and Trigonometric Functions},
  journal = {arXiv preprint arXiv:2511.20665},
  year    = {2025}
}
```

---

## License

MIT
