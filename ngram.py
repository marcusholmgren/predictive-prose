# -*- coding: utf-8 -*-
"""
Language model with n-gram successor map
"""
import random
from typing import Dict, List
import pickle

NGramModel = Dict[tuple, List[str]]

def build_model(file_path: str, n_gram_size: int) -> NGramModel:
    """
    Builds an n-gram model from a text file.

    Args:
        file_path (str): The path to the input text file.
        n_gram_size (int): The size of the n-grams.

    Returns:
        dict: A dictionary representing the n-gram model.
    """
    successor_map = {}
    try:
        with open(file_path, 'r') as reader:
            window = []
            for line in reader:
                for word in line.split():
                    clean_word = word.strip('.;,-“’”:?—‘!()_').lower()
                    if not clean_word:
                        continue
                    window.append(clean_word)

                    if len(window) == n_gram_size:
                        key = tuple(window[:-1])
                        value = window[-1]
                        if key in successor_map:
                            successor_map[key].append(value)
                        else:
                            successor_map[key] = [value]
                        window.pop(0)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    return successor_map

def generate_text(model: NGramModel, initial_words: List[str], num_words_to_generate: int) -> str:
    """
    Generates text using the n-gram model.

    Args:
        model (dict): The n-gram model.
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
        dict: A dictionary representing the n-gram model.
    """
    with open(model_path, 'rb') as f:
        return pickle.load(f)
