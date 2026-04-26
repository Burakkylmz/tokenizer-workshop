from __future__ import annotations

import pytest

from tokenizer_workshop.api.services.tokenizer_factory import TokenizerFactory


# ---------------------------------------------------------
# Train / Vocabulary Tests
# ---------------------------------------------------------

def test_regex_tokenizer_train_builds_vocab() -> None:
    """
    train() çağrıldığında RegexTokenizer'ın vocabulary oluşturduğunu test eder.

    Input:
        "Hello world! Hello"

    Regex tokenları:
        ["Hello", "world", "!", "Hello"]

    Unique tokenlar:
        {"Hello", "world", "!"}

    Beklenen vocab_size:
        3
    """
    tokenizer = TokenizerFactory.create("regex")

    tokenizer.train("Hello world! Hello")

    assert tokenizer.vocab_size == 3


def test_regex_tokenizer_vocab_size_reflects_unique_regex_tokens() -> None:
    """
    vocab_size değerinin unique regex token sayısını yansıttığını test eder.

    Input:
        "one two two three!"

    Tokenlar:
        ["one", "two", "two", "three", "!"]

    Unique tokenlar:
        {"one", "two", "three", "!"}

    Beklenen vocab_size:
        4
    """
    tokenizer = TokenizerFactory.create("regex")

    tokenizer.train("one two two three!")

    assert tokenizer.vocab_size == 4


def test_regex_tokenizer_vocab_is_deterministic_for_same_input() -> None:
    """
    Aynı metinle eğitilen iki tokenizer'ın aynı vocabulary mapping'i
    üretmesini test eder.

    Bu önemlidir çünkü tokenizer deterministik olmalıdır.

    Yani aynı input:
        "elma armut kiraz elma"

    her çalıştırmada aynı token-id eşleşmesini üretmelidir.
    """
    tokenizer_a = TokenizerFactory.create("regex")
    tokenizer_b = TokenizerFactory.create("regex")

    text = "elma armut kiraz elma"

    tokenizer_a.train(text)
    tokenizer_b.train(text)

    assert tokenizer_a.token_to_id == tokenizer_b.token_to_id
    assert tokenizer_a.id_to_token == tokenizer_b.id_to_token


# ---------------------------------------------------------
# Tokenize Tests
# ---------------------------------------------------------

def test_regex_tokenizer_tokenize_splits_words_and_punctuation() -> None:
    """
    tokenize() metodunun kelimeleri ve noktalama işaretlerini ayrı tokenlara
    böldüğünü test eder.

    Input:
        "Hello world!"

    Beklenen çıktı:
        ["Hello", "world", "!"]
    """
    tokenizer = TokenizerFactory.create("regex")

    tokens = tokenizer.tokenize("Hello world!")

    assert tokens == ["Hello", "world", "!"]


def test_regex_tokenizer_punctuation_is_separate_token() -> None:
    """
    Virgül ve ünlem gibi noktalama işaretlerinin ayrı token olarak
    ayrıldığını test eder.

    Input:
        "Merhaba, dünya!"

    Beklenen çıktı:
        ["Merhaba", ",", "dünya", "!"]
    """
    tokenizer = TokenizerFactory.create("regex")
    text = "Merhaba, dünya!"

    tokenizer.train(text)

    tokens = tokenizer.tokenize(text)

    assert tokens == ["Merhaba", ",", "dünya", "!"]
    assert len(tokens) == 4


def test_regex_tokenizer_handles_turkish_characters() -> None:
    """
    RegexTokenizer'ın Türkçe karakterleri doğru tokenize ettiğini test eder.

    Python regex içinde \\w ifadesi Unicode-aware çalışır.
    Bu yüzden:
        "dünya"

    tek kelime tokenı olarak yakalanmalıdır.
    """
    tokenizer = TokenizerFactory.create("regex")
    text = "Merhaba dünya!"

    tokenizer.train(text)

    assert tokenizer.tokenize(text) == ["Merhaba", "dünya", "!"]
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_regex_tokenizer_tokenize_empty_string_returns_empty_list() -> None:
    """
    Boş string tokenize edildiğinde boş liste dönmelidir.

    Bu davranış API/report tarafında daha güvenli çalışmayı sağlar.
    """
    tokenizer = TokenizerFactory.create("regex")

    assert tokenizer.tokenize("") == []


def test_regex_tokenizer_tokenize_whitespace_only_returns_empty_list() -> None:
    """
    Sadece boşluklardan oluşan input tokenize edildiğinde boş liste dönmelidir.

    Örnek:
        "   " -> []
    """
    tokenizer = TokenizerFactory.create("regex")

    assert tokenizer.tokenize("   ") == []


def test_regex_tokenizer_supports_custom_pattern() -> None:
    """
    RegexTokenizer'ın custom regex pattern desteklediğini test eder.

    Burada pattern sadece İngilizce harfleri yakalar:

        r"[A-Za-z]+"

    Input:
        "Hello, world! 123"

    Beklenen:
        ["Hello", "world"]

    Çünkü:
        ","  punctuation olduğu için yakalanmaz
        "!"  punctuation olduğu için yakalanmaz
        "123" sayı olduğu için yakalanmaz
    """
    tokenizer = TokenizerFactory.create("regex", pattern=r"[A-Za-z]+")

    tokens = tokenizer.tokenize("Hello, world! 123")

    assert tokens == ["Hello", "world"]


# ---------------------------------------------------------
# Encode Tests
# ---------------------------------------------------------

def test_regex_tokenizer_encode_returns_integer_token_ids() -> None:
    """
    encode() metodunun string tokenları integer id listesine dönüştürdüğünü test eder.

    Input:
        "Hello world!"

    Tokenlar:
        ["Hello", "world", "!"]

    Beklenen:
        - çıktı list olmalı
        - listedeki tüm elemanlar int olmalı
        - token sayısı 3 olmalı
    """
    tokenizer = TokenizerFactory.create("regex")
    tokenizer.train("Hello world!")

    encoded = tokenizer.encode("Hello world!")

    assert isinstance(encoded, list)
    assert all(isinstance(token_id, int) for token_id in encoded)
    assert len(encoded) == 3


def test_regex_tokenizer_encode_before_training_raises_error() -> None:
    """
    train() çağrılmadan encode() çağrılırsa hata vermelidir.

    Çünkü encode() token_to_id vocabulary mapping'ine ihtiyaç duyar.
    """
    tokenizer = TokenizerFactory.create("regex")

    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.encode("Hello")


def test_regex_tokenizer_encode_unknown_token_raises_error() -> None:
    """
    Eğitim sırasında görülmeyen token encode edilmeye çalışılırsa hata vermelidir.

    Train text:
        "Hello world"

    Encode text:
        "unknown"

    "unknown" vocabulary içinde olmadığı için OOV hatası beklenir.
    """
    tokenizer = TokenizerFactory.create("regex")
    tokenizer.train("Hello world")

    with pytest.raises(ValueError, match="Unknown token"):
        tokenizer.encode("unknown")


# ---------------------------------------------------------
# Decode / Roundtrip Tests
# ---------------------------------------------------------

def test_regex_tokenizer_decode_reconstructs_readable_text() -> None:
    """
    decode() metodunun token id listesini tekrar okunabilir metne çevirdiğini test eder.

    Input:
        "Hello world!"

    encode:
        [id("Hello"), id("world"), id("!")]

    decode:
        "Hello world!"

    Not:
        RegexTokenizer whitespace bilgisini birebir saklamaz.
        Fakat noktalama işaretlerinden önceki gereksiz boşlukları temizlediği için
        bu örnekte birebir roundtrip sağlanır.
    """
    tokenizer = TokenizerFactory.create("regex")
    tokenizer.train("Hello world!")

    encoded = tokenizer.encode("Hello world!")
    decoded = tokenizer.decode(encoded)

    assert decoded == "Hello world!"


def test_regex_tokenizer_encode_decode_roundtrip() -> None:
    """
    encode -> decode akışının basit bir cümlede orijinal metni geri verdiğini test eder.

    Bu test tokenizer'ın minimum reconstruction davranışını doğrular.
    """
    tokenizer = TokenizerFactory.create("regex")
    text = "Hello world!"

    tokenizer.train(text)

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_regex_tokenizer_decode_before_training_raises_error() -> None:
    """
    train() çağrılmadan decode() çağrılırsa hata vermelidir.

    Çünkü decode() id_to_token mapping'ine ihtiyaç duyar.
    """
    tokenizer = TokenizerFactory.create("regex")

    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.decode([0, 1])


def test_regex_tokenizer_decode_unknown_token_id_raises_error() -> None:
    """
    Vocabulary içinde olmayan token id decode edilmeye çalışılırsa hata vermelidir.

    Örnek:
        decode([9999])

    9999 id_to_token içinde olmadığı için hata beklenir.
    """
    tokenizer = TokenizerFactory.create("regex")
    tokenizer.train("Hello world")

    with pytest.raises(ValueError, match="Unknown token id"):
        tokenizer.decode([9999])