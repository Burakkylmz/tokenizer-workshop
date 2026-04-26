"""
tokenizer_factory.py

Tokenizer üretiminden sorumlu merkezi factory sınıfı.

Bu modülün amacı:
- API veya servis katmanından string olarak gelen tokenizer adını almak
- ilgili tokenizer class'ını registry üzerinden bulmak
- her istek için yeni bir tokenizer instance'ı üretmek

Bu factory artık manuel tokenizer mapping tutmaz.

Eski yaklaşım:
    {
        "char": CharTokenizer(),
        "word": WordTokenizer(),
        "regex": RegexTokenizer(),
    }

Yeni yaklaşım:
    tokenizer module
        ↓
    @register_tokenizer(...)
        ↓
    TokenizerRegistry
        ↓
    TokenizerFactory

Bu sayede yeni tokenizer eklemek için bu dosyaya import veya mapping eklemek gerekmez.
"""

from __future__ import annotations

import logging
from typing import Any

from tokenizer_workshop.api.services.exceptions import UnsupportedTokenizerError
from tokenizer_workshop.tokenizers.discovery import auto_import_tokenizers
from tokenizer_workshop.tokenizers.registry import TokenizerRegistry

logger = logging.getLogger(__name__)


class TokenizerFactory:
    """
    String tokenizer adlarını gerçek tokenizer instance'larına dönüştüren factory.

    Bu sınıf API/service katmanı ile tokenizer implementasyonları arasında
    merkezi bir oluşturma noktası sağlar.

    Neden factory kullanıyoruz?

    Çünkü uygulamanın farklı yerlerinde doğrudan şunu yapmak istemiyoruz:

        CharTokenizer()
        WordTokenizer()
        RegexTokenizer()

    Bunun yerine uygulama şunu yapar:

        TokenizerFactory.create("word")

    Böylece:
        - tokenizer oluşturma mantığı tek yerde kalır
        - desteklenmeyen tokenizer adları kontrollü şekilde yakalanır
        - yeni tokenizer eklemek kolaylaşır
        - UI/API tarafı concrete class'lara bağımlı olmaz

    Registry/discovery mimarisiyle birlikte bu factory artık plug-in benzeri
    bir yapıyı destekler.
    """

    @staticmethod
    def _ensure_registry_loaded() -> None:
        """
        TokenizerRegistry'nin dolu olmasını garanti eder.

        Önemli nokta:
            @register_tokenizer(...) decorator'ı, ilgili tokenizer modülü import
            edilmeden çalışmaz.

        Örnek:
            @register_tokenizer("regex")
            class RegexTokenizer(...):
                ...

        Bu class'ın registry'ye eklenmesi için regex_tokenizer.py dosyasının
        import edilmesi gerekir.

        auto_import_tokenizers() fonksiyonu:
            - tokenizers package'i altındaki tokenizer modüllerini tarar
            - tokenizer dosyalarını import eder
            - import sırasında decorator'ların çalışmasını sağlar
            - TokenizerRegistry'nin dolmasını garanti eder

        Bu method factory içindeki tüm public methodlardan önce çağrılır.
        Böylece registry'nin boş kalması engellenir.
        """
        auto_import_tokenizers()

    @classmethod
    def get_registry(cls) -> dict[str, Any]:
        """
        Kayıtlı tüm tokenizer class'larından yeni instance'lar üretir.

        Returns:
            dict[str, Any]:
                Key:
                    tokenizer adı

                Value:
                    yeni oluşturulmuş tokenizer instance'ı

        Örnek çıktı:
            {
                "char": CharTokenizer(),
                "word": WordTokenizer(),
                "regex": RegexTokenizer(),
                "byte_bpe": ByteBPETokenizer(),
            }

        Neden class yerine instance döndürüyoruz?
            API/service katmanı tokenizer'ı doğrudan kullanmak ister.
            Bu yüzden burada registry'deki class referanslarından yeni nesneler
            oluşturulur.

        Neden her çağrıda yeni instance?
            Bazı tokenizer'lar train() sonrası state tutar:
                - vocabulary
                - merges
                - token_to_id
                - id_to_token

            Aynı instance'ın farklı isteklerde paylaşılması state leakage
            oluşturabilir. Bu nedenle her çağrıda fresh instance üretmek daha güvenlidir.
        """
        cls._ensure_registry_loaded()

        return {
            name: tokenizer_cls()
            for name, tokenizer_cls in TokenizerRegistry.get_all().items()
        }

    @classmethod
    def get_supported_tokenizers(cls) -> list[str]:
        """
        Sistemde desteklenen tokenizer adlarını alfabetik olarak döndürür.

        Bu method genellikle /api/tokenizers endpoint'i tarafından kullanılır.

        UI akışı:
            1. Frontend /api/tokenizers endpoint'ini çağırır.
            2. Bu method desteklenen tokenizer adlarını döndürür.
            3. UI checkbox/dropdown listesini dinamik oluşturur.

        Registry/discovery sayesinde yeni tokenizer eklendiğinde:
            - factory değişmez
            - endpoint değişmez
            - UI otomatik olarak yeni tokenizer'ı görebilir

        Returns:
            list[str]:
                Alfabetik sıralı tokenizer adları.

        Örnek:
            ["byte", "byte_bpe", "char", "regex", "regex_bpe", "word"]
        """
        cls._ensure_registry_loaded()
        return sorted(TokenizerRegistry.get_all().keys())

    @staticmethod
    def normalize_name(name: str) -> str:
        """
        Tokenizer adını standart forma dönüştürür.

        Normalizasyon kuralları:
            - Değer string olmalıdır.
            - Başındaki ve sonundaki boşluklar silinir.
            - Küçük harfe çevrilir.
            - Boş string kabul edilmez.

        Neden gerekli?
            Kullanıcı veya frontend farklı formatlarda tokenizer adı gönderebilir.

        Örnek:
            " Regex_BPE " -> "regex_bpe"
            "WORD"        -> "word"
            " char "      -> "char"

        Args:
            name:
                Normalize edilecek tokenizer adı.

        Returns:
            str:
                Normalize edilmiş tokenizer adı.

        Raises:
            TypeError:
                name string değilse.

            ValueError:
                name boş veya sadece whitespace ise.
        """
        if not isinstance(name, str):
            raise TypeError("Tokenizer name must be a string.")

        normalized = name.strip().lower()

        if not normalized:
            raise ValueError("Tokenizer name cannot be empty.")

        return normalized

    @classmethod
    def create(cls, tokenizer_name: str, **kwargs) -> Any:
        """
        Verilen tokenizer adına göre yeni tokenizer instance'ı üretir.

        Örnek:
            TokenizerFactory.create("regex")
            TokenizerFactory.create("regex", pattern=r"[A-Za-z]+")
            TokenizerFactory.create("byte_bpe", num_merges=3)
        """
        cls._ensure_registry_loaded()

        key = cls.normalize_name(tokenizer_name)
        registry = TokenizerRegistry.get_all()

        if key not in registry:
            supported = sorted(registry.keys())

            logger.warning(
                "Unsupported tokenizer requested: %s. Supported tokenizers: %s",
                tokenizer_name,
                supported,
            )

            raise UnsupportedTokenizerError(
                tokenizer_name=tokenizer_name,
                supported_tokenizers=supported,
            )

        tokenizer_cls = registry[key]

        logger.debug(
            "Creating tokenizer instance: %s with kwargs=%s",
            key,
            kwargs,
        )

        return tokenizer_cls(**kwargs)

    @classmethod
    def create_many(cls, tokenizer_names: list[str]) -> dict[str, Any]:
        """
        Birden fazla tokenizer instance'ını tek seferde üretir.

        Bu method özellikle compare/report akışlarında kullanılır.

        Örnek input:
            ["char", "word", "regex"]

        Örnek output:
            {
                "char": CharTokenizer(),
                "word": WordTokenizer(),
                "regex": RegexTokenizer(),
            }

        Davranış:
            - tokenizer isimleri normalize edilir
            - duplicate isimler kaldırılır
            - input sırası korunur
            - her tokenizer için fresh instance oluşturulur

        Args:
            tokenizer_names:
                Oluşturulacak tokenizer adları.

        Returns:
            dict[str, Any]:
                Normalize edilmiş tokenizer adı -> tokenizer instance mapping'i.
        """
        normalized_names = cls.normalize_many(tokenizer_names)

        return {
            name: cls.create(name)
            for name in normalized_names
        }

    @classmethod
    def normalize_many(cls, tokenizer_names: list[str]) -> list[str]:
        """
        Tokenizer isim listesini normalize eder ve duplicate değerleri kaldırır.

        Neden gerekli?
            Frontend veya kullanıcı aynı tokenizer'ı birden fazla gönderebilir.

        Örnek input:
            ["word", " Word ", "REGEX", "regex", "char"]

        Örnek output:
            ["word", "regex", "char"]

        Özellikler:
            - Sıra korunur.
            - Duplicate değerler kaldırılır.
            - Her değer normalize_name() ile validate edilir.

        Args:
            tokenizer_names:
                Normalize edilecek tokenizer isim listesi.

        Returns:
            list[str]:
                Normalize edilmiş ve tekrarları kaldırılmış tokenizer adları.

        Raises:
            ValueError:
                Liste boşsa.

            TypeError:
                Listenin içindeki herhangi bir değer string değilse.
        """
        if not tokenizer_names:
            raise ValueError("At least one tokenizer name must be provided.")

        normalized_names: list[str] = []
        seen: set[str] = set()

        for tokenizer_name in tokenizer_names:
            normalized = cls.normalize_name(tokenizer_name)

            if normalized in seen:
                continue

            seen.add(normalized)
            normalized_names.append(normalized)

        return normalized_names