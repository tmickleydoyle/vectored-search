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
---------------
Question: What is the default device for the model?
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
Question: Where can I find the byte pair encoder?
File: /Users/thomasmickley-doyle/repos/language_model/bpe.py
class BPE:
    """
    The Byte Pair Encoding (BPE) class.

    Attributes:
        merges (Dict[Tuple[int, int], int]): The merges of token ids.
        vocab (Dict[int, bytes]): The vocabulary of token ids.
    """
    def __init__(self) -> None:
        super().__init__()

    def get_pairs(self, tokens: List[int]) -> Counter:
        """
        Get the pairs of characters in the tokens.

        Args:
            tokens (List[int]): The list of token ids.

        Returns:
            Counter: The pairs of characters.
        """
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pairs[tokens[i], tokens[i + 1]] += 1
        return pairs

    def train(self, text: str, max_vocab_size: int = 100, verbose: bool = False, pattern_merge_percent: int = 2, char_len: int = 10000) -> None:
        """
        Train the BPE model on the given text.

        Args:
            text (str): The text to train the model on.
            max_vocab_size (int): The maximum vocabulary size.
            verbose (bool): Whether to print verbose output.
            pattern_merge_percent (int): The percentage threshold for including a merge. Defaul is 95% for the top 5% of pairs.
            char_len (int): The total number of characters from the text to consider.
        """
        assert max_vocab_size >= 0
        num_merges = max_vocab_size
        include_merge = int((pattern_merge_percent / 100) * char_len)
        assert include_merge >= 0

        # Input text preprocessing
        text_bytes = text.encode("utf-8")  # raw bytes
        ids = list(text_bytes)  # list of integers in range 0..255

        # Iteratively merge the most common pairs to create new tokens
        merges = {}  # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)}  # int -> bytes
        for i in range(num_merges):
            # Count up the number of times every consecutive pair appears
            stats = self.get_pairs(ids)
            stats = {pair: count for pair, count in stats.items() if count >= include_merge}
            if not stats:
                break  # no more pairs to merge

            # Find the pair with the highest count
            pair = max(stats, key=stats.get)
            # Mint a new token: assign it the next available id
            idx = 256 + i
            # Replace all occurrences of pair in ids with idx
            ids = self.merge(ids, pair, idx)
            # Save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # Prints
            if verbose:
                print(f"merge: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # Save class variables
        self.merges = merges  # used in encode()
        self.vocab = vocab    # used in decode()

    def merge(self, ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """
        Merge all occurrences of the given pair in the ids list.

        Args:
            ids (List[int]): The list of token ids.
            pair (Tuple[int, int]): The pair of token ids to merge.
            new_id (int): The new token id to replace the pair with.

        Returns:
            List[int]: The updated list of token ids.
        """
        i = 0
        while i < len(ids) - 1:
            if ids[i] == pair[0] and ids[i + 1] == pair[1]:
                ids = ids[:i] + [new_id] + ids[i + 2:]
            else:
                i += 1
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode the given list of token ids into a string.

        Args:
            ids (List[int]): The list of token ids.

        Returns:
            str: The decoded string.
        """
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text: str) -> List[int]:
        """
        Encode the given string into a list of token ids.

        Args:
            text (str): The string to encode.

        Returns:
            List[int]: The list of token ids.
        """
        text_bytes = text.encode("utf-8")  # raw bytes
        ids = list(text_bytes)  # list of integers in range 0..255
        while len(ids) >= 2:
            # Find the pair with the lowest merge index
            stats = self.get_pairs(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # Subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # We can detect this terminating case by a membership check
            if pair not in self.merges:
                break  # nothing else can be merged anymore
            # Otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = self.merge(ids, pair, idx)
        return ids

    def save_encoder(self, path: str = "encoder.txt") -> None:
        """
        Save the encoder (vocab and merges) to a file.

        Args:
            path (str): The path to save the encoder to.
        """
        with open(path, "w") as file:
            # Save vocabulary
            file.write("### VOCAB ###\n")
            for idx, token in self.vocab.items():
                file.write(f"{idx} {token.hex()}\n")
            # Save merges
            file.write("### MERGES ###\n")
            for (p1, p2), idx in self.merges.items():
                file.write(f"{p1} {p2} {idx}\n")

    def load_encoder(self, path: str = "encoder.txt") -> None:
        """
        Load the encoder from a file.

        Args:
            path (str): The path to load the encoder from.
        """
        self.vocab = {}
        self.merges = {}
        with open(path, "r") as file:
            section = ""
            for line in file:
                if line.startswith("###"):
                    section = line.strip()
                    continue
                if not line.strip():
                    continue
                if section == "### VOCAB ###":
                    idx, token_hex = line.strip().split()
                    self.vocab[int(idx)] = bytes.fromhex(token_hex)
                elif section == "### MERGES ###":
                    p1, p2, idx = line.strip().split()
                    self.merges[(int(p1), int(p2))] = int(idx)

    def save_decoder(self, path: str = "decoder.txt") -> None:
        """
        Save the decoder to a file.

        Args:
            path (str): The path to save the decoder to.
        """
        with open(path, "w") as file:
            for idx, token in self.vocab.items():
                file.write(f"{idx} {token.hex()}\n")

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
---------------
Question: block_size
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
```
