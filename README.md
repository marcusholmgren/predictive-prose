# N-Gram Text Generation

This script generates text using an n-gram model built from a provided text file.

## Usage

```bash
uv run main.py [file_path] [options]
```

### Arguments

*   `file_path`: Path to the input text file. Required if not loading a model.

### Options

*   `--n-gram-size`: The size of the n-grams (default: 3).
*   `--num-words`: The number of words to generate (default: 15).
*   `--seed`: Random seed for reproducibility.
*   `--initial-words`: A sequence of words to start the generation from. The number of words must be `n-gram-size - 1`.
*   `--load-model`: Path to a saved model file to load.
*   `--save-model`: Path to save the generated model.

### Examples

**Generate 20 words from `data/jekyll.txt` using a 3-gram model:**

```bash
uv run main.py data/jekyll.txt --num-words 20
```

**Generate text with a 4-gram model and a specific starting sequence:**

```bash
uv run main.py data/jekyll.txt --n-gram-size 4 --initial-words "he was not"
```

**Save a model:**

```bash
uv run main.py data/jekyll.txt --save-model jekyll.model
```

**Load a model and generate text:**

```bash
uv run main.py --load-model jekyll.model --num-words 50
```
