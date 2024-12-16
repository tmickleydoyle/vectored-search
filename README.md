**WIP**

## Vector Search Over Code

### About

Uses ChromaDB to index code snippets and perform similarity searches. The code snippets are tokenized and embedded using a transformer model trained on a large codebase, `nomic-embed-text-v1`. The code is parsed with the `tree-sitter` library and the embeddings are generated using the `transformers` library. Natural langauge searches are tokenized and embedded in the same way, and the cosine similarity between the query and the code snippets is used to rank the results. The top `k` results are then returned pretty-printed and displayed in the terminal. To convert this into a RAG model, the code snippets could be used as the `context` and the query could be used as the `question`.

### Getting Started

1. Clone the repository
2. Run `docker run -p 8000:8000 chromadb/chroma` to start the ChromaDB server
3. Install dependencies: `pip install -r requirements.txt`
4. Run the main script: `python main.py`
5. :pizza:

### Files Structure

```text
project/
│
├── main.py               # Entry point for executing the queries
├── config.py             # Configuration variables
├── db_client.py          # Handles ChromaDB client setup and interactions
├── code_loader.py        # Handles file loading and processing
├── snippet_printer.py    # Pretty-printing snippets
├── query_engine.py       # Encapsulates query logic
└── requirements.txt      # Dependency list
```

### CLI Commands

Training on the data
```text
python main.py --train
```

Generate embeddings for the code snippets. Example questions are in the main.py file.
```text
python main.py --generate
```

Train and genrate embeddings
```text
python main.py --train --generate
```

### Example Output

Embeddings trained on my [language_model](https://github.com/tmickleydoyle/language_model) repo

```text
Question: What is the context length for the model?
File: /Users/thomasmickley-doyle/repos/language_model/config.py
class Config:
    """
    Configuration class to store hyperparameters and other settings.

    Attributes:
        batch_size: Number of independent sequences to process in parallel.
        block_size: Maximum context length for predictions.
        n_embd: Embedding dimension.
        n_head: Number of attention heads.
        n_layer: Number of transformer blocks.
        dropout: Dropout rate.
        vocab_size: Size of the vocabulary.
        max_iters: Maximum number of training iterations.
        eval_interval: Number of iterations between evaluation runs.
        learning_rate: Learning rate for the optimizer.
        eval_iters: Number of iterations to run evaluation for.
        device: Device to run the model on.
        seed: Random seed for reproducibility.
    """
    # Model hyperparameters
    batch_size = 64  # how many independent sequences will we process in parallel?
    block_size = 128   # maximum context length for predictions
    n_embd = 384    # embedding dimension
    n_head = 6      # number of attention heads
    n_layer = 6     # number of transformer blocks
    dropout = 0.2   # dropout rate
    vocab_size = 512  # Added this parameter for BPE tokenizer

    # Training hyperparameters
    max_iters = 2500
    eval_interval = 500
    learning_rate = 3e-4
    eval_iters = 200

    # System
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 1337

    def __init__(self):
        torch.manual_seed(self.seed)
---------------
Question: GPTLanguageModel
File: /Users/thomasmickley-doyle/repos/language_model/model.py
class GPTLanguageModel(nn.Module):
    """
    GPT language model with transformer blocks. Generates text autoregressively.
    """
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config

        self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        forward pass for the model

        Args:
            idx: Input tensor of token indices
            targets: Target tensor of token indices
        """
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.config.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        generate new tokens autoregressively given a prompt

        Args:
            idx: Input tensor of token indices
            max_new_tokens: Maximum number of new tokens to generate
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
---------------
Question: How can I load the decoder from the tokenizer?
File: /Users/thomasmickley-doyle/repos/language_model/bpe.py
    def load_decoder(self, path: str = "decoder.txt") -> None:
        """
        Load the decoder from a file.

        Args:
            path (str): The path to load the decoder from.
        """
        self.vocab = {}
        with open(path, "r") as file:
            for line in file:
                idx, token_hex = line.strip().split()
                self.vocab[int(idx)] = bytes.fromhex(token_hex)
```
