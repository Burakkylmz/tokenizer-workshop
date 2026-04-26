from __future__ import annotations

import pytest

from tokenizer_workshop.api.services.tokenizer_factory import TokenizerFactory


# ---------------------------------------------------------
# INIT TESTS
# ---------------------------------------------------------

def test_ngram_tokenizer_init_invalid_n_raises_error() -> None:
    """
    n değeri 1'den küçük verilirse tokenizer oluşturulmamalıdır.

    Çünkü:
        n = 1  -> unigram
        n = 2  -> bigram
        n = 3  -> trigram

    n = 0 veya negatif değerler geçerli bir n-gram üretmez.
    """
    with pytest.raises(ValueError, match="n must be at least 1"):
        TokenizerFactory.create("ngram", n=0)


# ---------------------------------------------------------
# TRAIN TESTS
# ---------------------------------------------------------

def test_ngram_tokenizer_train_with_empty_text_raises_error() -> None:
    """
    Boş metinle train() çağrıldığında ValueError beklenir.

    Eğitim metni boş olursa vocabulary üretilemez.
    """
    tokenizer = TokenizerFactory.create("ngram", n=2)

    with pytest.raises(ValueError, match="Training text cannot be empty"):
        tokenizer.train("")


def test_ngram_tokenizer_builds_vocab() -> None:
    """
    train() sonrası tokenizer vocabulary oluşturmalıdır.

    Input:
        "the cat sat"

    n = 2 olduğu için bigram tokenları:
        ["the cat", "cat sat"]

    Beklenen vocab_size:
        2
    """
    tokenizer = TokenizerFactory.create("ngram", n=2)

    tokenizer.train("the cat sat")

    assert tokenizer.vocab_size == 2


# ---------------------------------------------------------
# ENCODE / DECODE TESTS
# ---------------------------------------------------------

def test_ngram_tokenizer_encode_before_training_raises_error() -> None:
    """
    train() çağrılmadan encode() çalışmamalıdır.

    Çünkü encode() işlemi için önce token -> id mapping'i oluşturulmalıdır.
    """
    tokenizer = TokenizerFactory.create("ngram", n=2)

    with pytest.raises(ValueError, match="not been trained"):
        tokenizer.encode("the cat")


def test_ngram_tokenizer_decode_before_training_raises_error() -> None:
    """
    train() çağrılmadan decode() çalışmamalıdır.

    Çünkü decode() işlemi için önce id -> token mapping'i oluşturulmalıdır.
    """
    tokenizer = TokenizerFactory.create("ngram", n=2)

    with pytest.raises(ValueError, match="not been trained"):
        tokenizer.decode([0])


def test_ngram_tokenizer_encode_returns_ids() -> None:
    """
    encode() metni integer token id listesine dönüştürmelidir.

    Input:
        "the cat sat"

    Bigram tokenları:
        ["the cat", "cat sat"]

    Bu nedenle encode çıktısı:
        - list olmalı
        - tüm elemanları int olmalı
        - uzunluğu 2 olmalı
    """
    tokenizer = TokenizerFactory.create("ngram", n=2)

    tokenizer.train("the cat sat")

    encoded = tokenizer.encode("the cat sat")

    assert isinstance(encoded, list)
    assert all(isinstance(i, int) for i in encoded)
    assert len(encoded) == 2


def test_ngram_tokenizer_decode_returns_string() -> None:
    """
    decode() token id listesini tekrar string çıktıya dönüştürmelidir.

    Not:
        N-gram decode işlemi overlap nedeniyle birebir reconstruction sağlamaz.
        Burada sadece okunabilir string çıktı üretildiği doğrulanır.
    """
    tokenizer = TokenizerFactory.create("ngram", n=2)

    tokenizer.train("the cat sat")

    encoded = tokenizer.encode("the cat sat")
    decoded = tokenizer.decode(encoded)

    assert isinstance(decoded, str)
    assert "the cat" in decoded


def test_ngram_tokenizer_roundtrip_behavior() -> None:
    """
    N-gram tokenizer için roundtrip davranışı bilinçli olarak farklıdır.

    Input:
        "the cat sat"

    Bigram tokenları:
        ["the cat", "cat sat"]

    Basit decode:
        "the cat cat sat"

    Bu yüzden decoded text'in orijinal metne birebir eşit olması beklenmez.
    """
    tokenizer = TokenizerFactory.create("ngram", n=2)

    text = "the cat sat"
    tokenizer.train(text)

    decoded = tokenizer.decode(tokenizer.encode(text))

    assert decoded == "the cat cat sat"


# ---------------------------------------------------------
# UNKNOWN TOKEN / ID TESTS
# ---------------------------------------------------------

def test_ngram_tokenizer_encode_unknown_token_raises_error() -> None:
    """
    Eğitim sırasında görülmeyen n-gram encode edilmeye çalışılırsa hata vermelidir.

    Train text:
        "the cat sat"

    Vocab:
        ["the cat", "cat sat"]

    Encode text:
        "unknown text here"

    Bu text'ten gelen n-gram'lar vocab içinde olmadığı için ValueError beklenir.
    """
    tokenizer = TokenizerFactory.create("ngram", n=2)

    tokenizer.train("the cat sat")

    with pytest.raises(ValueError, match="Unknown token"):
        tokenizer.encode("unknown text here")


def test_ngram_tokenizer_decode_unknown_id_raises_error() -> None:
    """
    Vocabulary içinde olmayan token id decode edilmeye çalışılırsa hata vermelidir.

    Bu test strict tokenizer contract'ını doğrular.
    """
    tokenizer = TokenizerFactory.create("ngram", n=2)

    tokenizer.train("the cat sat")

    with pytest.raises(ValueError, match="Unknown token id"):
        tokenizer.decode([999])


# ---------------------------------------------------------
# TOKENIZE TESTS
# ---------------------------------------------------------

def test_ngram_tokenizer_tokenize_returns_ngrams() -> None:
    """
    tokenize() metni string n-gram token listesine dönüştürmelidir.

    Input:
        "the cat sat"

    n = 2

    Beklenen:
        ["the cat", "cat sat"]
    """
    tokenizer = TokenizerFactory.create("ngram", n=2)

    tokens = tokenizer.tokenize("the cat sat")

    assert tokens == ["the cat", "cat sat"]


def test_ngram_tokenizer_tokenize_handles_short_input() -> None:
    """
    Input token sayısı n değerinden küçükse mevcut tokenlar döndürülür.

    Örnek:
        text = "hello"
        n = 3

    Normalde trigram üretmek için 3 kelime gerekir.
    Ancak input kısa olduğu için fallback olarak ["hello"] döner.
    """
    tokenizer = TokenizerFactory.create("ngram", n=3)

    tokens = tokenizer.tokenize("hello")

    assert tokens == ["hello"]


# ---------------------------------------------------------
# DETERMINISM TESTS
# ---------------------------------------------------------

def test_ngram_tokenizer_is_deterministic() -> None:
    """
    Aynı input ile eğitilen iki tokenizer aynı encode çıktısını üretmelidir.

    Bu test internal state'e doğrudan bakmaz.
    Davranış üzerinden determinism doğrular.
    """
    text = "the cat sat"

    t1 = TokenizerFactory.create("ngram", n=2)
    t2 = TokenizerFactory.create("ngram", n=2)

    t1.train(text)
    t2.train(text)

    assert t1.encode(text) == t2.encode(text)