"""
Harmonic Token Projection (HTP) — Embedding determinístico por PALAVRA.

Baseado em: arXiv:2511.20665 (Schmitz, 2025)

Cada palavra/token é convertida num inteiro grande N via base-65536
dos seus codepoints Unicode, e N é projetado em R^D via resíduos
modulares + sin/cos harmônico.
"""

import math
import re
import json
import torch
import torch.nn as nn
from pathlib import Path


def _first_n_primes(n: int) -> list[int]:
    """Retorna os primeiros n números primos."""
    primes = []
    candidate = 2
    while len(primes) < n:
        is_prime = all(candidate % p != 0 for p in primes if p * p <= candidate)
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


# ─── Tokenizador por palavra ───────────────────────────────────

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


class WordTokenizer:
    """
    Tokenizador por palavra para uso com HTP.

    Constrói vocabulário de palavras a partir dos dados.
    Cada palavra recebe um ID inteiro grande via HTP (base-65536),
    mas para o modelo usamos um mapeamento local palavra→índice.
    """

    def __init__(self):
        self.word_to_id: dict[str, int] = {}
        self.id_to_word: dict[int, str] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.word_to_id)

    def build(self, texts: list[str]):
        """Constrói vocabulário a partir dos textos."""
        # Tokens especiais primeiro
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.word_to_id[tok] = i
            self.id_to_word[i] = tok

        next_id = len(SPECIAL_TOKENS)
        words = set()
        for text in texts:
            for w in _tokenize_text(text):
                words.add(w)

        for w in sorted(words):
            if w not in self.word_to_id:
                self.word_to_id[w] = next_id
                self.id_to_word[next_id] = w
                next_id += 1

        print(f"WordTokenizer: {self.vocab_size} palavras")

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        words = _tokenize_text(text)
        ids = []
        if add_special:
            ids.append(self.word_to_id[BOS_TOKEN])
        for w in words:
            ids.append(self.word_to_id.get(w, self.word_to_id[UNK_TOKEN]))
        if add_special:
            ids.append(self.word_to_id[EOS_TOKEN])
        return ids

    def decode(self, ids: list[int]) -> str:
        special_ids = {self.word_to_id[t] for t in SPECIAL_TOKENS}
        words = []
        for id_ in ids:
            if id_ in special_ids:
                continue
            words.append(self.id_to_word.get(id_, UNK_TOKEN))
        return _reconstruct_text(words)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {"word_to_id": self.word_to_id}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"WordTokenizer salvo em {path}")

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.word_to_id = data["word_to_id"]
        # Converter chaves string de volta para int
        self.id_to_word = {int(v): k for k, v in self.word_to_id.items()}
        # word_to_id values são int do JSON
        self.word_to_id = {k: int(v) for k, v in self.word_to_id.items()}
        print(f"WordTokenizer carregado: {self.vocab_size} palavras")


def _tokenize_text(text: str) -> list[str]:
    """Divide texto em palavras e pontuação (lowercase)."""
    return re.findall(r"[a-zA-ZÀ-ÿ]+|[0-9]+(?:\.[0-9]+)?|[^\s]", text.lower())


def _reconstruct_text(words: list[str]) -> str:
    """Reconstrói texto a partir de lista de palavras."""
    text = ""
    for i, w in enumerate(words):
        if i > 0 and w[0].isalnum() and text and text[-1].isalnum():
            text += " "
        text += w
    return text


# ─── HTP Embedding por palavra ─────────────────────────────────

def word_to_htp_integer(word: str, max_chars: int = 16) -> int:
    """
    Converte palavra em inteiro grande N via base-65536.

    N = sum_{j=0}^{L-1} ord(c_j) * (2^16)^(L-1-j)

    Para tokens especiais, usa valores fixos pequenos.
    """
    if word == PAD_TOKEN:
        return 0
    if word == BOS_TOKEN:
        return 1
    if word == EOS_TOKEN:
        return 2
    if word == UNK_TOKEN:
        return 3

    chars = list(word[:max_chars])
    N = 0
    base = 2**16
    for ch in chars:
        N = N * base + ord(ch)
    return N


class HTPWordEmbedding(nn.Module):
    """
    HTP embedding por palavra.

    Cada palavra do vocabulário é convertida num inteiro grande N,
    e N é projetado em R^D via resíduos modulares + sin/cos.

    Os embeddings são pré-computados para todo o vocabulário (lookup table
    determinística, sem parâmetros aprendidos).
    """

    def __init__(self, dim: int = 128, max_chars: int = 16):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.k = dim // 2
        self.max_chars = max_chars

        primes = _first_n_primes(self.k)
        # Guardar como Python ints para aritmética de precisão arbitrária
        self.prime_list = primes
        self.register_buffer("moduli", torch.tensor(primes, dtype=torch.float64))
        self.register_buffer("freq", (2.0 * math.pi) / self.moduli)

        # Tabela de embeddings pré-computados (preenchida por build_table)
        self.register_buffer("embedding_table", torch.zeros(1, dim))

    def build_table(self, tokenizer: WordTokenizer):
        """Pré-computa embeddings HTP para todo o vocabulário."""
        vocab_size = tokenizer.vocab_size
        table = torch.zeros(vocab_size, self.dim)

        for word, idx in tokenizer.word_to_id.items():
            N = word_to_htp_integer(word, self.max_chars)
            emb = self._embed_integer(N)
            table[idx] = emb

        self.embedding_table = table
        print(f"HTP table: {vocab_size} embeddings de dim {self.dim} (0 parâmetros)")

    def _embed_integer(self, N: int) -> torch.Tensor:
        """Projeta um inteiro N em R^D via HTP."""
        residues = torch.tensor(
            [N % p for p in self.prime_list], dtype=torch.float64
        )
        theta = residues * self.freq.cpu()
        sin_part = torch.sin(theta)
        cos_part = torch.cos(theta)
        return torch.cat([sin_part, cos_part]).float()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Lookup na tabela pré-computada.

        Args:
            idx: (B, T) índices do vocabulário

        Returns:
            (B, T, dim) embeddings
        """
        return self.embedding_table[idx]


if __name__ == "__main__":
    tok = WordTokenizer()
    tok.build(["O Brasil é o maior país da América do Sul."])

    htp = HTPWordEmbedding(dim=128)
    htp.build_table(tok)

    ids = tok.encode("O Brasil é o maior", add_special=False)
    print(f"Tokens: {[tok.id_to_word[i] for i in ids]}")
    print(f"IDs: {ids}")

    tensor = torch.tensor([ids], dtype=torch.long)
    emb = htp(tensor)
    print(f"Embedding shape: {emb.shape}")
    print(f"Norms: {emb[0].norm(dim=-1)}")

    # Verificar que palavras diferentes têm embeddings diferentes
    e_brasil = emb[0, 1]
    e_maior = emb[0, 4]
    cos_sim = torch.dot(e_brasil, e_maior) / (e_brasil.norm() * e_maior.norm())
    print(f"Cos sim 'brasil' vs 'maior': {cos_sim:.4f}")
