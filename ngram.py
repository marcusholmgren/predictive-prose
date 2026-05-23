# -*- coding: utf-8 -*-
"""
Language model with n-gram successor map
"""
from collections import defaultdict, deque
import pickle
import random

# Modern PEP 695 Type Alias (Python 3.12+)
type NGramModel = dict[tuple[str, ...], list[str]]


def build_model(file_path: str, n_gram_size: int) -> NGramModel:
    """
    Builds an n-gram model from a text file.

    Args:
        file_path (str): The path to the input text file.
        n_gram_size (int): The size of the n-grams.

    Returns:
        NGramModel: A dictionary representing the n-gram model.

    Raises:
        ValueError: If n_gram_size is less than 1.
        FileNotFoundError: If the input file does not exist.
    """
    if n_gram_size < 1:
        raise ValueError("n-gram size must be at least 1.")

    successor_map: dict[tuple[str, ...], list[str]] = defaultdict(list)
    
    # Explicit encoding avoids cross-platform locale issues
    with open(file_path, 'r', encoding='utf-8') as reader:
        # Use deque for idiomatic and fast sliding window
        window: deque[str] = deque(maxlen=n_gram_size)
        
        for line in reader:
            for word in line.split():
                clean_word = word.strip('.;,-“’”:?—‘!()_').lower()
                if not clean_word:
                    continue
                window.append(clean_word)

                if len(window) == n_gram_size:
                    key = tuple(window)[:-1]
                    value = window[-1]
                    successor_map[key].append(value)

    # Convert back to standard dict for clean output / serialization
    return dict(successor_map)


def generate_text(model: NGramModel, initial_words: list[str], num_words_to_generate: int) -> str:
    """
    Generates text using the n-gram model.

    Args:
        model (NGramModel): The n-gram model.
        initial_words (list): A list of initial words.
        num_words_to_generate (int): The number of words to generate.

    Returns:
        str: The generated text.
    """
    if not model:
        return "Model is empty. Cannot generate text."

    current_words = list(initial_words)
    generated_text = list(initial_words)

    for _ in range(num_words_to_generate):
        key = tuple(current_words)
        if key in model:
            next_word = random.choice(model[key])
            generated_text.append(next_word)
            current_words.pop(0)
            current_words.append(next_word)
        else:
            generated_text.append(f"\n(Sequence { ' '.join(current_words) } not in model, stopping.)")
            break
            
    return " ".join(generated_text)


def load_model(model_path: str) -> NGramModel:
    """
    Load the n-gram model.

    Args:
        model_path (str): Name and path of model to load.

    Returns:
        NGramModel: A dictionary representing the n-gram model.
    """
    with open(model_path, 'rb') as f:
        return pickle.load(f)
