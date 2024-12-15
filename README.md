**WIP**

## Vector Search Over Code

### Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main script: `python main.py`
4. :pizza:

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

### Example Output

Embeddings trained on my [language_model](https://github.com/tmickleydoyle/language_model) repo

```text
Question: What is the context length for the model?
File: /Users/thomasmickley-doyle/repos/language_model/config.py
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
---------------
Question: What is the learning rate used in the model?
File: /Users/thomasmickley-doyle/repos/language_model/config.py
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
---------------
Question: What is the dropout rate used in the model?
File: /Users/thomasmickley-doyle/repos/language_model/config.py
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
---------------
Question: How is the training data loaded into the model?
File: /Users/thomasmickley-doyle/repos/language_model/train.py
    # Save the model
    torch.save(model.state_dict(), 'model.pth')

    # load the model
    loaded_model = GPTLanguageModel(config, dataset.vocab_size)
    loaded_model.load_state_dict(torch.load('model.pth'))
    loaded_model = loaded_model.to(config.device)

    # Example question
    question = "What is Monstera?"
---------------
Question: What is the Byte Pair Encoding tokenizer?
File: /Users/thomasmickley-doyle/repos/language_model/bpe.py
"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

---------------
Question: GPTLanguageModel
File: /Users/thomasmickley-doyle/repos/language_model/model.py
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    """
    GPT language model with transformer blocks. Generates text autoregressively.
    """
    def __init__(self, config, vocab_size):
        super().__init__()
---------------
Question: How can I load the decoder from the tokenizer?
File: /Users/thomasmickley-doyle/repos/language_model/bpe.py
        """
        with open(path, "w") as file:
            for idx, token in self.vocab.items():
                file.write(f"{idx} {token.hex()}\n")

    def load_decoder(self, path: str = "decoder.txt") -> None:
        """
        Load the decoder from a file.

        Args:
---------------
```
