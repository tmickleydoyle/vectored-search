## Vector Search Over Code

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
├── utils.py              # Shared utility functions (if needed)
├── __init__.py           # Marks the folder as a package
└── requirements.txt      # Dependency list
```

### Example Output

```text
File: https://github.com/tmickleydoyle/language_model/blob/main/config.py
import torch

class Config:
  # Model hyperparameters
  batch_size = 64  # how many independent sequences will we process in parallel?
  block_size = 128   # maximum context length for predictions
  n_embd = 384    # embedding dimension
  n_head = 6      # number of attention heads
---------------
Question: What is the learning rate used in the model?
File: https://github.com/tmickleydoyle/language_model/blob/main/config.py
  vocab_size = 512  # Added this parameter for BPE tokenizer

  # Training hyperparameters
  max_iters = 2500
  eval_interval = 500
  learning_rate = 3e-4
  eval_iters = 200

  # System
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
---------------
Question: What is the dropout rate used in the model?
File: https://github.com/tmickleydoyle/language_model/blob/main/config.py
class Config:
  # Model hyperparameters
  batch_size = 64  # how many independent sequences will we process in parallel?
  block_size = 128   # maximum context length for predictions
  n_embd = 384    # embedding dimension
  n_head = 6      # number of attention heads
  n_layer = 6     # number of transformer blocks
  dropout = 0.2   # dropout rate
  vocab_size = 512  # Added this parameter for BPE tokenizer

---------------
Question: Where is the training data?
File: https://github.com/tmickleydoyle/language_model/blob/main/train.py
from data import TextDataset
from model import GPTLanguageModel

def train():
  # Initialize configuration
  config = Config()

  # Load and prepare data
  dataset = TextDataset(config)
  dataset.load_data('input.txt')
---------------
Question: What is the Byte Pair Encoding tokenizer?
File: https://github.com/tmickleydoyle/language_model/blob/main/bpe.py
"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

---------------
Question: GPTLanguageModel
File: https://github.com/tmickleydoyle/language_model/blob/main/model.py
      x = x + self.sa(self.ln1(x))
      x = x + self.ffwd(self.ln2(x))
      return x

class GPTLanguageModel(nn.Module):
  def __init__(self, config, vocab_size):
      super().__init__()
      self.config = config

      self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)
---------------
Question: How can I load the decoder from the tokenizer?
File: https://github.com/tmickleydoyle/language_model/blob/main/bpe.py
              file.write(f"{idx} {token.hex()}\n")

  def load_decoder(self, path: str = "decoder.txt") -> None:
      """
      Load the decoder from a file.

      Args:
          path (str): The path to load the decoder from.
      """
      self.vocab = {}
---------------
