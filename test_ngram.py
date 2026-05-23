# -*- coding: utf-8 -*-
import pytest
from ngram import build_model, generate_text


def test_build_model_standard(tmp_path):
    """Test building a model with normal text file input."""
    file = tmp_path / "sample.txt"
    file.write_text("The quick brown fox jumps over the lazy dog. The quick brown fox.", encoding="utf-8")

    model = build_model(str(file), n_gram_size=3)
    assert model is not None
    # 3-gram model prefixes are size 2:
    # "the quick" -> ["brown"]
    # "quick brown" -> ["fox", "fox"]
    # "brown fox" -> ["jumps"]
    # ...
    assert model[("the", "quick")] == ["brown", "brown"]
    assert model[("quick", "brown")] == ["fox", "fox"]
    assert model[("brown", "fox")] == ["jumps"]


def test_build_model_invalid_ngram_size(tmp_path):
    """Test that build_model raises ValueError when n-gram size is < 1."""
    file = tmp_path / "sample.txt"
    file.write_text("Some text", encoding="utf-8")

    with pytest.raises(ValueError, match="n-gram size must be at least 1"):
        build_model(str(file), n_gram_size=0)

    with pytest.raises(ValueError, match="n-gram size must be at least 1"):
        build_model(str(file), n_gram_size=-2)


def test_build_model_file_not_found():
    """Test that build_model propagates FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        build_model("this_file_definitely_does_not_exist.txt", n_gram_size=3)


def test_build_model_empty_file(tmp_path):
    """Test that an empty file produces an empty model."""
    file = tmp_path / "empty.txt"
    file.write_text("", encoding="utf-8")

    model = build_model(str(file), n_gram_size=3)
    assert model == {}


def test_generate_text_standard():
    """Test text generation on a valid model."""
    model = {
        ("hello", "world"): ["how", "are"],
        ("world", "how"): ["are"],
        ("how", "are"): ["you"],
    }
    # Test generation from a specific seed
    res = generate_text(model, ["hello", "world"], num_words_to_generate=3)
    # The generation is probabilistic, but let's assert key characteristics
    assert res.startswith("hello world")
    words = res.split()
    assert len(words) == 10  # 2 starting words + 3 generated


def test_generate_text_empty_model():
    """Test that calling generate_text on empty model returns an error message."""
    res = generate_text({}, ["hello"], 5)
    assert "Model is empty" in res


def test_generate_text_missing_sequence():
    """Test stopping generation when sequence is not in model."""
    model = {
        ("hello", "world"): ["how"],
    }
    res = generate_text(model, ["hello", "world"], num_words_to_generate=5)
    assert "stopping" in res
    assert "Sequence" in res
