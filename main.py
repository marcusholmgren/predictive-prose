# -*- coding: utf-8 -*-
import random
import argparse
import pickle
import sys
from ngram import build_model, generate_text, load_model

def get_model(args, parser):
    """Gets the model, either by loading or building."""
    if args.load_model:
        print(f"Loading model from {args.load_model}...")
        try:
            return load_model(args.load_model)
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            print(f"Error loading model: {e}", file=sys.stderr)
            sys.exit(1)

    if args.file_path:
        print(f"Building model from {args.file_path}...")
        return build_model(args.file_path, args.n_gram_size)

    parser.error("Either file_path or --load-model must be provided.")

def save_model(model, file_path):
    """Saves the model to a file."""
    if not file_path:
        return
    print(f"Saving model to {file_path}...")
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
    except IOError as e:
        print(f"Error saving model: {e}", file=sys.stderr)

def get_initial_words(args, model):
    """Determines and validates the initial words for generation."""
    if not args.initial_words:
        return list(random.choice(list(model.keys())))

    if len(args.initial_words) == 1:
        initial_words = args.initial_words[0].split()
    else:
        initial_words = args.initial_words

    model_n_gram_size = len(next(iter(model.keys()))) + 1
    if len(initial_words) != model_n_gram_size - 1:
        print(f"Error: Number of initial words must be {model_n_gram_size - 1} for the loaded model.", file=sys.stderr)
        return None

    if tuple(initial_words) not in model:
        print(f"Error: The initial word sequence '{' '.join(initial_words)}' is not in the model.", file=sys.stderr)
        print("Try one of the following sequences (or run without initial words for a random start):", file=sys.stderr)
        for i, key in enumerate(model.keys()):
            if i > 5:
                break
            print(f"  --initial-words {' '.join(key)}", file=sys.stderr)
        return None

    return initial_words

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate text using an n-gram model.")
    parser.add_argument("file_path", nargs='?', default=None, help="Path to the input text file (required if not loading a model).")
    parser.add_argument("--n-gram-size", type=int, default=3, help="Size of the n-grams (default: 3).")
    parser.add_argument("--num-words", type=int, default=15, help="Number of words to generate (default: 15).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--initial-words", nargs='+', default=None, help="Initial words to start generation (default: random).")
    parser.add_argument("--load-model", help="Path to a saved model file to load.")
    parser.add_argument("--save-model", help="Path to save the generated model.")
    return parser, parser.parse_args()

def main(args, parser):
    """
    Main function to orchestrate n-gram model loading, text generation, and saving.
    """
    if args.seed is not None:
        random.seed(args.seed)

    model = get_model(args, parser)

    save_model(model, args.save_model)

    initial_words = get_initial_words(args, model)
    if not initial_words:
        sys.exit(1)

    generated_text = generate_text(model, initial_words, args.num_words)
    print(generated_text)

if __name__ == "__main__":
    parser, arguments = parse_arguments()
    main(arguments, parser)
