from __future__ import annotations

import pytest

from tokenizer_workshop.api.services.tokenizer_factory import TokenizerFactory


def test_byte_bpe_tokenizer_init_raises_error_for_invalid_num_merges() -> None:
    with pytest.raises(ValueError, match="num_merges must be at least 1"):
        TokenizerFactory.create("byte_bpe", num_merges=0)


def test_byte_bpe_tokenizer_train_with_empty_text_raises_error() -> None:
    tokenizer = TokenizerFactory.create("byte_bpe", num_merges=3)

    with pytest.raises(ValueError, match="Training text cannot be empty"):
        tokenizer.train("")


def test_byte_bpe_tokenizer_encode_before_training_raises_error() -> None:
    tokenizer = TokenizerFactory.create("byte_bpe", num_merges=3)

    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.encode("abababa")


def test_byte_bpe_tokenizer_decode_before_training_raises_error() -> None:
    tokenizer = TokenizerFactory.create("byte_bpe", num_merges=3)

    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.decode([0, 1, 2])


def test_byte_bpe_tokenizer_always_has_full_byte_base_vocabulary() -> None:
    tokenizer = TokenizerFactory.create("byte_bpe", num_merges=3)
    tokenizer.train("abababa")

    # Base vocabulary her zaman 256 byte'ı içerir, merged tokens eklenir.
    assert tokenizer.vocab_size >= 256


def test_byte_bpe_tokenizer_vocab_size_grows_with_num_merges() -> None:
    tokenizer = TokenizerFactory.create("byte_bpe", num_merges=5)
    tokenizer.train("abababa")

    # Training text'inde öğrenilen benzersiz merge'ler vocab'a eklenir.
    # 256 base + en az 1 merge öğrenilmiş olmalı.
    assert tokenizer.vocab_size > 256


def test_byte_bpe_tokenizer_encode_returns_integer_token_ids() -> None:
    tokenizer = TokenizerFactory.create("byte_bpe", num_merges=3)
    tokenizer.train("abababa")

    encoded = tokenizer.encode("abababa")

    assert isinstance(encoded, list)
    assert all(isinstance(token_id, int) for token_id in encoded)
    assert len(encoded) > 0


def test_byte_bpe_tokenizer_decode_reconstructs_original_text() -> None:
    tokenizer = TokenizerFactory.create("byte_bpe", num_merges=3)
    tokenizer.train("abababa")

    encoded = tokenizer.encode("abababa")
    decoded = tokenizer.decode(encoded)

    assert decoded == "abababa"


def test_byte_bpe_tokenizer_encode_decode_roundtrip() -> None:
    tokenizer = TokenizerFactory.create("byte_bpe", num_merges=5)
    tokenizer.train("tokenization")

    text = "tokenization"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_byte_bpe_tokenizer_roundtrip_with_turkish_characters() -> None:
    tokenizer = TokenizerFactory.create("byte_bpe", num_merges=5)
    tokenizer.train("çğüşöıİ")

    text = "çğüşöıİ"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text


def test_byte_bpe_tokenizer_can_encode_characters_unseen_during_training() -> None:
    # Byte-level BPE'nin en güçlü yanı: base vocabulary 256 byte'ı içerdiği için
    # training sırasında hiç görülmemiş karakterler bile encode edilebilir.
    tokenizer = TokenizerFactory.create("byte_bpe", num_merges=3)
    tokenizer.train("abababa")

    text = "xyz"  # training text'inde yok
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text


def test_byte_bpe_tokenizer_learns_merge_steps() -> None:
    tokenizer = TokenizerFactory.create("byte_bpe", num_merges=3)
    tokenizer.train("abababa")

    assert len(tokenizer.merge_steps) >= 1
    # "a" ve "b" byte'ları sırasıyla 97 ve 98 byte değerlerine sahiptir.
    # İlk merge bu pair olmalı: chr(97) + chr(98) -> "ab" (iki karakterli sembol).
    first_step = tokenizer.merge_steps[0]
    assert first_step.pair == (chr(97), chr(98))
    assert first_step.merged_token == chr(97) + chr(98)
    assert first_step.frequency == 3


def test_byte_bpe_tokenizer_can_reduce_token_count_for_repetitive_text() -> None:
    tokenizer = TokenizerFactory.create("byte_bpe", num_merges=3)
    text = "abababa"

    tokenizer.train(text)
    encoded = tokenizer.encode(text)

    # Merge'ler tekrarlayan byte'ları birleştirdiği için token sayısı
    # raw byte sayısından az olmalı.
    assert len(encoded) < len(text.encode("utf-8"))


def test_byte_bpe_tokenizer_decode_unknown_token_id_raises_error() -> None:
    tokenizer = TokenizerFactory.create("byte_bpe", num_merges=3)
    tokenizer.train("abababa")

    with pytest.raises(ValueError, match="Unknown token id encountered"):
        tokenizer.decode([99999])


def test_byte_bpe_tokenizer_is_deterministic_for_same_input() -> None:
    tokenizer_a = TokenizerFactory.create("byte_bpe", num_merges=3)
    tokenizer_b = TokenizerFactory.create("byte_bpe", num_merges=3)

    tokenizer_a.train("abababa")
    tokenizer_b.train("abababa")

    assert tokenizer_a.merge_steps == tokenizer_b.merge_steps
    assert tokenizer_a.encode("abababa") == tokenizer_b.encode("abababa")