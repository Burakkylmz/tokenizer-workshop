"""
utils.py

Service katmanında kullanılan yardımcı fonksiyonlar.

Bu modül, tokenizer pipeline'ında:
    - interface doğrulama
    - veri normalizasyonu
    - input temizleme

işlemlerini merkezi ve tutarlı şekilde yönetir.

Amaç:
    - Service katmanında defensive programming sağlamak
    - Tokenizer implementasyonları arasındaki tutarsızlıkları normalize etmek
    - Hataları erken yakalamak ve anlamlı exception üretmek
    - Kod tekrarını azaltmak
    - Tokenizer'ların beklenen kontratlara uyduğundan emin olmak
    - Tokenizer'ların çıktısını uniform bir formata dönüştürmek
    - Kullanıcıdan gelen tokenizer isimlerini normalize etmek ve tekrarları kaldırmak
"""

from __future__ import annotations

from typing import Any

from tokenizer_workshop.api.services.exceptions import TokenizationServiceError
from tokenizer_workshop.api.services.tokenizer_factory import TokenizerFactory


# ---------------------------------------------------------
# TOKENIZER INTERFACE VALIDATION
# ---------------------------------------------------------

def validate_tokenizer_interface(tokenizer: Any, tokenizer_name: str) -> None:
    """
    Verilen tokenizer nesnesinin beklenen interface'i sağlayıp sağlamadığını doğrular.

    Beklenen minimum kontrat:
        tokenizer.tokenize(text: str) -> Iterable[str]

    Bu kontrol neden önemli?
        - Farklı tokenizer implementasyonları (custom, pretrained, experimental)
          sisteme entegre edilebilir
        - Bu fonksiyon runtime'da interface uyumsuzluklarını yakalar
        - Silent failure yerine açık hata verir

    Args:
        tokenizer:
            Doğrulanacak tokenizer nesnesi.

        tokenizer_name:
            Hata mesajında kullanılacak tokenizer adı.

    Raises:
        TokenizationServiceError:
            tokenize metodu yoksa veya callable değilse.
    """
    # tokenize metodu var mı?
    # tokenize metodu callable mı? (Yani gerçekten bir fonksiyon mu?)
    # tokenize metodu yoksa veya callable değilse, bu tokenizer'ın beklenen interface'i sağlamadığı anlamına gelir.
    if not hasattr(tokenizer, "tokenize") or not callable(getattr(tokenizer, "tokenize")):
        raise TokenizationServiceError(
            f"Tokenizer '{tokenizer_name}' does not implement a callable 'tokenize' method."
        )


# ---------------------------------------------------------
# TOKEN NORMALIZATION
# ---------------------------------------------------------

def normalize_tokens(tokens: Any) -> list[str]:
    """
    Tokenizer çıktısını güvenli şekilde list[str] formatına normalize eder.

    Bu fonksiyon:
        - iterable olmayan çıktıları yakalar
        - None / invalid token değerlerini güvenli şekilde işler
        - tüm tokenları string'e çevirir

    Neden gerekli?
        - Bazı tokenizer'lar farklı tipte output döndürebilir (int, bytes, vs.)
        - Compare pipeline uniform veri bekler
        - UI ve raporlama string token ister

    Args:
        tokens:
            Tokenizer tarafından döndürülen ham çıktı.

    Returns:
        list[str]: Normalize edilmiş token listesi.

    Raises:
        TokenizationServiceError:
            Token output iterable değilse veya dönüştürülemezse.
    """
    if tokens is None:
        raise TokenizationServiceError(
            "Tokenizer returned None instead of an iterable token sequence."
        )

    try:
        # Tüm tokenları string'e çevirir. 
        # Eğer token zaten string ise, bu işlem etkisizdir.
        return [str(token) for token in list(tokens)]
    except Exception as exc:
        raise TokenizationServiceError(
            "Tokenizer output could not be normalized into a list of strings."
        ) from exc


# ---------------------------------------------------------
# TOKENIZER NAME NORMALIZATION
# ---------------------------------------------------------

def deduplicate_tokenizer_names(tokenizer_names: list[str]) -> list[str]:
    """
    Tokenizer isimlerini normalize eder ve tekrar edenleri kaldırır.

    Özellikler:
        ✔ Case-insensitive normalization
        ✔ Duplicate removal
        ✔ Order preservation

    Örnek:
        input:
            ["Char", "word", "char", "WORD"]

        output:
            ["char", "word"]

    Args:
        tokenizer_names:
            Kullanıcıdan gelen tokenizer isim listesi.

    Returns:
        list[str]: Normalize edilmiş ve tekrarları kaldırılmış tokenizer listesi.

    Raises:
        TokenizationServiceError:
            tokenizer_names boşsa veya geçersiz tip içeriyorsa.
    """
    if not tokenizer_names:
        raise TokenizationServiceError(
            "At least one tokenizer must be provided."
        )

    normalized_names: list[str] = []
    seen: set[str] = set()

    for name in tokenizer_names:
        if not isinstance(name, str):
            raise TokenizationServiceError(
                f"Invalid tokenizer name type: {type(name)}. Expected string."
            )

        normalized_name = TokenizerFactory.normalize_name(name)

        if not normalized_name:
            raise TokenizationServiceError(
                "Tokenizer name cannot be empty after normalization."
            )

        if normalized_name not in seen:
            seen.add(normalized_name)
            normalized_names.append(normalized_name)

    return normalized_names