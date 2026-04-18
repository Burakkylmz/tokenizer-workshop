from __future__ import annotations

import pytest

from tokenizer_workshop.tokenizers import SimpleBPETokenizer


def test_simple_bpe_tokenizer_init_raises_error_for_invalid_num_merges() -> None:
    with pytest.raises(ValueError, match="num_merges must be at least 1"):
        SimpleBPETokenizer(num_merges=0)


def test_simple_bpe_tokenizer_train_with_empty_text_raises_error() -> None:
    tokenizer = SimpleBPETokenizer(num_merges=3)

    with pytest.raises(ValueError, match="Training text cannot be empty"):
        tokenizer.train("")


def test_simple_bpe_tokenizer_encode_before_training_raises_error() -> None:
    tokenizer = SimpleBPETokenizer(num_merges=3)

    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.encode("abababa")


def test_simple_bpe_tokenizer_decode_before_training_raises_error() -> None:
    tokenizer = SimpleBPETokenizer(num_merges=3)

    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.decode([0, 1, 2])


def test_simple_bpe_tokenizer_train_builds_a_vocabulary() -> None:
    tokenizer = SimpleBPETokenizer(num_merges=3)

    tokenizer.train("abababa")

    assert tokenizer.vocab_size >= len(set("abababa"))


def test_simple_bpe_tokenizer_encode_returns_integer_token_ids() -> None:
    tokenizer = SimpleBPETokenizer(num_merges=3)
    tokenizer.train("abababa")

    encoded = tokenizer.encode("abababa")

    assert isinstance(encoded, list)
    assert all(isinstance(token_id, int) for token_id in encoded)
    assert len(encoded) > 0


def test_simple_bpe_tokenizer_decode_reconstructs_original_text() -> None:
    tokenizer = SimpleBPETokenizer(num_merges=3)
    tokenizer.train("abababa")

    encoded = tokenizer.encode("abababa")
    decoded = tokenizer.decode(encoded)

    assert decoded == "abababa"


def test_simple_bpe_tokenizer_encode_decode_roundtrip() -> None:
    tokenizer = SimpleBPETokenizer(num_merges=5)
    tokenizer.train("tokenization")

    text = "tokenization"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_simple_bpe_tokenizer_learns_merge_steps() -> None:
    tokenizer = SimpleBPETokenizer(num_merges=3)
    tokenizer.train("abababa")

    assert len(tokenizer.merge_steps) >= 1
    assert tokenizer.merge_steps[0].pair == ("a", "b")
    assert tokenizer.merge_steps[0].merged_token == "ab"
    assert tokenizer.merge_steps[0].frequency == 3


def test_simple_bpe_tokenizer_can_reduce_token_count_for_repetitive_text() -> None:
    tokenizer = SimpleBPETokenizer(num_merges=3)
    text = "abababa"

    tokenizer.train(text)
    encoded = tokenizer.encode(text)

    assert len(encoded) < len(text)


def test_simple_bpe_tokenizer_decode_unknown_token_id_raises_error() -> None:
    tokenizer = SimpleBPETokenizer(num_merges=3)
    tokenizer.train("abababa")

    with pytest.raises(ValueError, match="Unknown token id encountered"):
        tokenizer.decode([999])


def test_simple_bpe_tokenizer_is_deterministic_for_same_input() -> None:
    tokenizer_a = SimpleBPETokenizer(num_merges=3)
    tokenizer_b = SimpleBPETokenizer(num_merges=3)

    tokenizer_a.train("abababa")
    tokenizer_b.train("abababa")

    assert tokenizer_a.merge_steps == tokenizer_b.merge_steps
    assert tokenizer_a.encode("abababa") == tokenizer_b.encode("abababa")