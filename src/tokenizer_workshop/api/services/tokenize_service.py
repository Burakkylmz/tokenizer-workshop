"""
tokenize_service.py

Bu modül, API'nin tokenize endpoint'i için ana iş mantığını içerir.

Bu servis, verilen metni seçilen tokenizer ile tokenize eder ve standart bir response formatında sonuç döndürür.

Bu servis, controller/router katmanından ayrılarak:
    - Tokenizer oluşturma, interface doğrulama, optional training ve output normalization gibi business logic'i kapsar.
    - Böylece controller katmanı yalnızca HTTP davranışıyla ilgilenir, tokenize_service ise tokenize işleminin tüm detaylarını yönetir.
    - Bu ayrım, kodun daha modüler, test edilebilir ve genişletilebilir olmasını sağlar.

Tokenize akışı:
    1. tokenizer_name normalize edilir.
    2. TokenizerFactory üzerinden tokenizer instance oluşturulur.
    3. Tokenizer interface'i doğrulanır.
    4. Tokenizer train() destekliyorsa input text ile eğitilir.
    5. tokenize(text) çağrılır.
    6. Tokenizer çıktısı list[str] formatına normalize edilir.
    7. API response için standart payload döndürülür.

Bu servis, farklı tokenizer türlerini tek bir pipeline içinde çalıştırabilmek için gerekli adapter ve helper fonksiyonlarını içerir.
"""

from __future__ import annotations

from typing import Any

from tokenizer_workshop.api.services.exceptions import (
    TokenizationServiceError,
    UnsupportedTokenizerError,
)
from tokenizer_workshop.api.services.tokenizer_factory import TokenizerFactory
from tokenizer_workshop.api.services.utils import (
    normalize_tokens,
    validate_tokenizer_interface,
)


def train_tokenizer_if_supported(tokenizer: object, text: str) -> None:
    """
    Tokenizer destekliyorsa train() metodunu çalıştırır.

    Bazı tokenizer'lar stateless çalışır:
        - CharTokenizer
        - WordTokenizer
        - WhitespaceTokenizer

    Bazı tokenizer'lar ise tokenize işleminden önce eğitim gerektirir:
        - BPE tokenizer
        - Unigram tokenizer
        - Byte-Level BPE tokenizer

    Bu helper, iki tokenizer tipini aynı pipeline içinde çalıştırabilmek için
    adapter görevi görür.

    Args:
        tokenizer:
            Train edilme ihtimali olan tokenizer instance'ı.

        text:
            Training için kullanılacak ham metin.

    Raises:
        TokenizationServiceError:
            Tokenizer train() metoduna sahip olduğu halde eğitim sırasında hata oluşursa.
    """
    try:
        if hasattr(tokenizer, "train") and callable(getattr(tokenizer, "train")):
            tokenizer.train(text)
    except Exception as exc:
        raise TokenizationServiceError(
            "Tokenizer training failed before tokenization."
        ) from exc


def tokenize_text(text: str, tokenizer_name: str) -> dict[str, Any]:
    """
    Verilen metni seçilen tokenizer ile tokenize eder.

    Bu fonksiyon, tekli tokenization service akışının ana orkestrasyon noktasıdır.

    Pipeline:
        1. tokenizer_name normalize edilir.
        2. TokenizerFactory üzerinden tokenizer instance oluşturulur.
        3. Tokenizer interface'i doğrulanır.
        4. Tokenizer train() destekliyorsa input text ile eğitilir.
        5. tokenize(text) çağrılır.
        6. Tokenizer çıktısı list[str] formatına normalize edilir.
        7. API response için standart payload döndürülür.

    Bu servis neden gerekli?

    Controller / router katmanı yalnızca HTTP davranışıyla ilgilenmelidir.
    Tokenizer oluşturma, interface doğrulama, optional training ve output
    normalization gibi business logic bu service içinde tutulur.

    Args:
        text:
            Tokenize edilecek ham metin.

        tokenizer_name:
            Kullanılacak tokenizer adı.
            Örnek: "char", "word", "byte_level_bpe", "pretrained"

    Returns:
        Standart tokenize response payload'u:

        {
            "tokenizer_name": "word",
            "tokens": ["Hello", "world", "!"],
            "token_count": 3,
            "vocab_size": 3
        }

    Raises:
        UnsupportedTokenizerError:
            tokenizer_name registry/factory tarafından desteklenmiyorsa.

        TokenizationServiceError:
            Tokenizer oluşturma, eğitim, tokenize veya normalization aşamasında
            beklenmeyen hata oluşursa.
    """
    try:
        # TokenizerFactory, verilen tokenizer_name'e göre uygun tokenizer instance'ını döndürür.
        tokenizer = TokenizerFactory.create(tokenizer_name)

        # Tokenizer'ın tokenize(text) metodunu destekleyip desteklemediği doğrulanır
        validate_tokenizer_interface(tokenizer, tokenizer_name)

        # Tokenizer train() metodunu destekliyorsa, tokenize işlemi öncesinde eğitim yapılır.
        train_tokenizer_if_supported(tokenizer, text)

        # Tokenizer ile metin tokenize edilir.
        # Tokenizer çıktısı list[str] formatına normalize edilir.
        raw_tokens = tokenizer.tokenize(text)
        # Tokenizer'lar farklı token formatları döndürebilir (örneğin, bazıları byte dizisi döndürebilir).
        # normalize_tokens helper'ı, farklı formatlardaki token listelerini API response'u için uygun hale getirir.
        # Örneğin, byte tokenları string'e çevirir veya özel token objelerini string formatına dönüştürür.
        # Bu adım, API response'unun tutarlı ve anlaşılır olmasını sağlar, tokenizer'ların iç detaylarını gizler.
        normalized_tokens = normalize_tokens(raw_tokens)

        return {
            "tokenizer_name": TokenizerFactory.normalize_name(tokenizer_name),
            "tokens": normalized_tokens,
            "token_count": len(normalized_tokens),
            "vocab_size": len(set(normalized_tokens)),
        }

    except UnsupportedTokenizerError:
        raise

    except TokenizationServiceError:
        raise

    except Exception as exc:
        raise TokenizationServiceError(
            f"Tokenization failed for '{tokenizer_name}': {type(exc).__name__}: {exc}"
        ) from exc