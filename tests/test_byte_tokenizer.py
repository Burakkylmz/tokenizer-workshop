from __future__ import annotations

import pytest

from tokenizer_workshop.api.services.tokenizer_factory import TokenizerFactory


def test_byte_tokenizer_has_fixed_vocab_size() -> None:
    tokenizer = TokenizerFactory.create("byte")

    assert tokenizer.vocab_size == 256


def test_byte_tokenizer_train_with_empty_text_raises_error() -> None:
    tokenizer = TokenizerFactory.create("byte")

    with pytest.raises(ValueError, match="Training text cannot be empty"):
        tokenizer.train("")


def test_byte_tokenizer_encode_before_training_raises_error() -> None:
    tokenizer = TokenizerFactory.create("byte")

    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.encode("merhaba")


def test_byte_tokenizer_decode_before_training_raises_error() -> None:
    tokenizer = TokenizerFactory.create("byte")

    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.decode([109, 101, 114, 104, 97, 98, 97])


def test_byte_tokenizer_encode_returns_integer_token_ids() -> None:
    tokenizer = TokenizerFactory.create("byte")
    tokenizer.train("merhaba")

    encoded = tokenizer.encode("merhaba")

    assert isinstance(encoded, list)
    assert all(isinstance(token_id, int) for token_id in encoded)
    assert encoded == [109, 101, 114, 104, 97, 98, 97]


def test_byte_tokenizer_decode_reconstructs_original_ascii_text() -> None:
    tokenizer = TokenizerFactory.create("byte")
    tokenizer.train("merhaba")

    encoded = tokenizer.encode("merhaba")
    decoded = tokenizer.decode(encoded)

    assert decoded == "merhaba"


def test_byte_tokenizer_roundtrip_with_turkish_characters() -> None:
    tokenizer = TokenizerFactory.create("byte")
    tokenizer.train("çğüşöıİ")

    text = "çğüşöıİ"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text


def test_byte_tokenizer_utf8_may_use_multiple_tokens_for_one_character() -> None:
    tokenizer = TokenizerFactory.create("byte")
    tokenizer.train("ğ")

    encoded = tokenizer.encode("ğ")

    assert len(encoded) > 1


def test_byte_tokenizer_decode_invalid_byte_value_raises_error() -> None:
    tokenizer = TokenizerFactory.create("byte")
    tokenizer.train("abc")

    with pytest.raises(ValueError, match="Invalid byte token encountered"):
        tokenizer.decode([65, 300, 66])


def test_byte_tokenizer_decode_invalid_utf8_sequence_raises_error() -> None:
    tokenizer = TokenizerFactory.create("byte")
    tokenizer.train("abc")

    # 0xC3 tek başına geçerli bir UTF-8 dizisi oluşturmaz.
    with pytest.raises(ValueError, match="valid UTF-8 byte sequence"):
        tokenizer.decode([195])


def test_byte_tokenizer_encode_is_deterministic_for_same_input() -> None:
    tokenizer = TokenizerFactory.create("byte")
    tokenizer.train("token")

    encoded_a = tokenizer.encode("token")
    encoded_b = tokenizer.encode("token")

    assert encoded_a == encoded_b