from __future__ import annotations

import pytest

from tokenizer_workshop.evaluators import (
    TokenizationMetrics,
    evaluate_tokenizer,
    evaluate_tokenizers,
)

from tokenizer_workshop.api.services.tokenizer_factory import TokenizerFactory


def test_evaluate_tokenizer_returns_metrics_object() -> None:
    tokenizer = TokenizerFactory.create("char")

    result = evaluate_tokenizer(tokenizer=tokenizer, text="merhaba")

    assert isinstance(result, TokenizationMetrics)
    assert result.tokenizer_name == "char"


def test_evaluate_tokenizer_sets_basic_lengths_correctly() -> None:
    tokenizer = TokenizerFactory.create("char")
    text = "merhaba"

    result = evaluate_tokenizer(tokenizer=tokenizer, text=text)

    assert result.char_length == len(text)
    assert result.byte_length == len(text.encode("utf-8"))
    assert result.token_count == len(text)


def test_evaluate_tokenizer_roundtrip_is_true_for_char_tokenizer() -> None:
    tokenizer = TokenizerFactory.create("char")

    result = evaluate_tokenizer(tokenizer=tokenizer, text="merhaba")

    assert result.roundtrip_ok is True


def test_evaluate_tokenizer_roundtrip_is_true_for_byte_tokenizer() -> None:
    tokenizer = TokenizerFactory.create("byte")

    result = evaluate_tokenizer(tokenizer=tokenizer, text="merhaba")

    assert result.roundtrip_ok is True


def test_evaluate_tokenizer_roundtrip_is_true_for_simple_bpe_tokenizer() -> None:
    tokenizer = TokenizerFactory.create("simple_bpe", num_merges=3)

    result = evaluate_tokenizer(tokenizer=tokenizer, text="abababa")

    assert result.roundtrip_ok is True


def test_evaluate_tokenizer_uses_custom_train_text_when_provided() -> None:
    tokenizer = TokenizerFactory.create("char")

    result = evaluate_tokenizer(
        tokenizer=tokenizer,
        text="aba",
        train_text="ababa",
    )

    assert result.roundtrip_ok is True
    assert result.vocab_size == len(set("ababa"))


def test_evaluate_tokenizer_raises_error_for_empty_text() -> None:
    tokenizer = TokenizerFactory.create("char")

    with pytest.raises(ValueError, match="Evaluation text cannot be empty"):
        evaluate_tokenizer(tokenizer=tokenizer, text="")


def test_evaluate_tokenizers_returns_one_result_per_tokenizer() -> None:
    tokenizers = [
        TokenizerFactory.create("char"),
        TokenizerFactory.create("byte"),
        TokenizerFactory.create("simple_bpe", num_merges=3),
    ]

    results = evaluate_tokenizers(tokenizers=tokenizers, text="merhaba")

    assert len(results) == 3
    assert [result.tokenizer_name for result in results] == [
        "char",
        "byte",
        "simple_bpe",
    ]


def test_evaluate_tokenizers_raises_error_for_empty_tokenizer_list() -> None:
    with pytest.raises(ValueError, match="At least one tokenizer must be provided"):
        evaluate_tokenizers(tokenizers=[], text="merhaba")


def test_char_tokenizer_has_ratio_one_against_char_length_for_ascii_text() -> None:
    tokenizer = TokenizerFactory.create("char")
    text = "token"

    result = evaluate_tokenizer(tokenizer=tokenizer, text=text)

    assert result.compression_ratio_vs_chars == 1.0


def test_byte_tokenizer_has_ratio_one_against_byte_length_for_ascii_text() -> None:
    tokenizer = TokenizerFactory.create("byte")
    text = "token"

    result = evaluate_tokenizer(tokenizer=tokenizer, text=text)

    assert result.compression_ratio_vs_bytes == 1.0


def test_simple_bpe_can_reduce_token_count_on_repetitive_text() -> None:
    tokenizer = TokenizerFactory.create("simple_bpe", num_merges=3)
    text = "abababa"

    result = evaluate_tokenizer(tokenizer=tokenizer, text=text)

    assert result.token_count < len(text)
    assert result.compression_ratio_vs_chars < 1.0


def test_evaluate_tokenizer_roundtrip_is_true_for_byte_bpe_tokenizer() -> None:
    tokenizer = TokenizerFactory.create("byte_bpe", num_merges=3)

    result = evaluate_tokenizer(tokenizer=tokenizer, text="abababa")

    assert result.roundtrip_ok is True


def test_byte_bpe_can_reduce_token_count_on_repetitive_text() -> None:
    tokenizer = TokenizerFactory.create("byte_bpe", num_merges=3)
    text = "abababa"

    result = evaluate_tokenizer(tokenizer=tokenizer, text=text)

    # ByteBPE byte-level çalıştığı için byte_length ile karşılaştırmak daha anlamlı.
    assert result.token_count < result.byte_length
    assert result.compression_ratio_vs_bytes < 1.0