from __future__ import annotations

from dataclasses import dataclass

from tokenizer_workshop.tokenizers import BaseTokenizer


@dataclass(frozen=True)
class TokenizationMetrics:
    """
    Tek bir tokenizer'ın tek bir text sample üzerindeki özet metrics bilgisini tutar.

    Educational purpose:
    - tokenizer davranışını ölçülebilir hale getirir.
    - aynı input üzerinde farklı tokenizer'ları karşılaştırmayı kolaylaştırır.
    - "daha iyi" tokenization'ın neyi ölçtüğümüze bağlı olduğunu gösterir.
    """

    tokenizer_name: str
    vocab_size: int
    token_count: int
    char_length: int
    byte_length: int
    compression_ratio_vs_chars: float
    compression_ratio_vs_bytes: float
    roundtrip_ok: bool


def evaluate_tokenizer(
    tokenizer: BaseTokenizer,
    text: str,
    train_text: str | None = None,
) -> TokenizationMetrics:
    """
    Bir tokenizer'ı verilen text üzerinde train eder ve evaluate eder.

    Args:
        tokenizer: evaluate edilecek tokenizer instance'ı.
        text: encode/decode edilecek ve ölçülecek text.
        train_text: Opsiyonel training corpus. Verilmezse `text` kullanılır.

    Why train_text is optional:
    Küçük educational experiment'lerde aynı text üzerinde hem train hem evaluate etmek pratiktir.
    Daha sonra generalization konusunu tartışmak için bu ikisini ayırabiliriz.
    """
    if not text:
        raise ValueError("Evaluation text cannot be empty.")

    corpus = train_text if train_text is not None else text
    tokenizer.train(corpus)

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    char_length = len(text)
    byte_length = len(text.encode("utf-8"))
    token_count = len(encoded)

    return TokenizationMetrics(
        tokenizer_name=tokenizer.name,
        vocab_size=tokenizer.vocab_size,
        token_count=token_count,
        char_length=char_length,
        byte_length=byte_length,
        compression_ratio_vs_chars=token_count / char_length,
        compression_ratio_vs_bytes=token_count / byte_length,
        roundtrip_ok=(decoded == text),
    )


def evaluate_tokenizers(
    tokenizers: list[BaseTokenizer],
    text: str,
    train_text: str | None = None,
) -> list[TokenizationMetrics]:
    """
    Aynı text üzerinde birden fazla tokenizer'ı evaluate eder.

    Bu yaklaşım educational demo'larda side-by-side comparison yapmak için faydalıdır.
    """
    if not tokenizers:
        raise ValueError("At least one tokenizer must be provided.")

    return [
        evaluate_tokenizer(tokenizer=tokenizer, text=text, train_text=train_text)
        for tokenizer in tokenizers
    ]