from __future__ import annotations

import pytest

from tokenizer_workshop.tokenizers import CharTokenizer


def test_char_tokenizer_train_builds_vocab() -> None:
    tokenizer = CharTokenizer()

    tokenizer.train("merhaba")

    assert tokenizer.vocab_size == len(set("merhaba"))


def test_char_tokenizer_encode_returns_integer_token_ids() -> None:
    tokenizer = CharTokenizer()
    tokenizer.train("merhaba")

    encoded = tokenizer.encode("merhaba")

    assert isinstance(encoded, list)
    assert all(isinstance(token_id, int) for token_id in encoded)
    assert len(encoded) == len("merhaba")


def test_char_tokenizer_decode_reconstructs_original_text() -> None:
    tokenizer = CharTokenizer()
    tokenizer.train("merhaba")

    encoded = tokenizer.encode("merhaba")
    decoded = tokenizer.decode(encoded)

    assert decoded == "merhaba"


def test_char_tokenizer_encode_decode_roundtrip() -> None:
    tokenizer = CharTokenizer()
    tokenizer.train("token")

    text = "token"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_char_tokenizer_train_with_empty_text_raises_error() -> None:
    tokenizer = CharTokenizer()

    with pytest.raises(ValueError, match="Training text cannot be empty"):
        tokenizer.train("")


def test_char_tokenizer_encode_before_training_raises_error() -> None:
    tokenizer = CharTokenizer()

    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.encode("merhaba")


def test_char_tokenizer_decode_before_training_raises_error() -> None:
    tokenizer = CharTokenizer()

    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.decode([0, 1, 2])


def test_char_tokenizer_encode_unknown_character_raises_error() -> None:
    tokenizer = CharTokenizer()
    tokenizer.train("abc")

    with pytest.raises(ValueError, match="Unknown character encountered"):
        tokenizer.encode("abcd")


def test_char_tokenizer_decode_unknown_token_id_raises_error() -> None:
    tokenizer = CharTokenizer()
    tokenizer.train("abc")

    with pytest.raises(ValueError, match="Unknown token id encountered"):
        tokenizer.decode([0, 1, 99])


def test_char_tokenizer_vocab_is_deterministic_for_same_input() -> None:
    tokenizer_a = CharTokenizer()
    tokenizer_b = CharTokenizer()

    tokenizer_a.train("merhaba")
    tokenizer_b.train("merhaba")

    assert tokenizer_a._stoi == tokenizer_b._stoi
    assert tokenizer_a._itos == tokenizer_b._itos