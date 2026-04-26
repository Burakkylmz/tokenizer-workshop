from typing import Type, Dict
import logging

from tokenizer_workshop.tokenizers.base import BaseTokenizer

logger = logging.getLogger(__name__)


class TokenizerRegistry:
    """
    TokenizerRegistry

    Bu sınıf, sistemde bulunan tüm tokenizer implementasyonlarını merkezi bir
    registry (kayıt havuzu) içinde tutar.

    Neden bu yapı kullanılır?

    Geleneksel yaklaşım:
        TokenizerFactory içinde manuel mapping:
            {
                "char": CharTokenizer(),
                "word": WordTokenizer(),
                ...
            }

    Bu yaklaşımın problemi:
        - Yeni tokenizer eklemek için factory değiştirmek gerekir
        - Open/Closed Principle ihlal edilir
        - Kod genişledikçe bakım zorlaşır

    Bu registry yapısı sayesinde:
        ✔ Yeni tokenizer eklemek için sadece class yazmak yeterlidir
        ✔ Factory değiştirilmez (plug-in mimarisi)
        ✔ UI (/api/tokenizers) otomatik güncellenir
        ✔ Sistem extensible hale gelir

    Mimari akış:
        Tokenizer class (@register_tokenizer)
                ↓
        TokenizerRegistry (kayıt)
                ↓
        TokenizerFactory (okuma / instance üretme)
                ↓
        API + UI

    Bu yapı bir "plugin system" davranışı kazandırır.
    """

    # Global tokenizer registry
    #
    # Key   : tokenizer adı (örn: "regex_bpe")
    # Value : tokenizer class referansı (örn: RegexBPETokenizer)
    #
    # Örnek:
    # {
    #     "char": CharTokenizer,
    #     "word": WordTokenizer,
    #     "regex_bpe": RegexBPETokenizer
    # }
    _registry: Dict[str, Type[BaseTokenizer]] = {}

    @classmethod
    def register(cls, name: str, tokenizer_cls: Type[BaseTokenizer]) -> None:
        """
        Yeni bir tokenizer'ı registry'e ekler.

        Bu method genellikle doğrudan çağrılmaz.
        Bunun yerine @register_tokenizer decorator'ı üzerinden otomatik tetiklenir.

        Args:
            name:
                Tokenizer'ın sistemde kullanılacak adı.
                (örn: "char", "word", "regex_bpe")

                Not:
                    - Case-insensitive normalize edilir
                    - UI ve API bu ismi kullanır

            tokenizer_cls:
                Tokenizer class referansı (instance değil!)

        Raises:
            TypeError:
                Eğer verilen class BaseTokenizer'dan türemiyorsa.

                Bu kontrol çok kritiktir çünkü:
                    - encode / decode / train kontratının garanti edilmesini sağlar
                    - runtime hataları önler

            ValueError:
                Aynı isimle daha önce tokenizer kayıtlıysa.

                Bu durum:
                    - yanlış import
                    - duplicate registration
                    - plugin çakışması

                gibi hatalara işaret eder.

        Örnek:
            register("regex_bpe", RegexBPETokenizer)
        """
        key = name.strip().lower()

        # TYPE SAFETY
        #
        # Sadece BaseTokenizer'dan türeyen class'ların kayıt edilmesine izin verilir.
        #
        # Bu sayede:
        # - encode()
        # - decode()
        # - train()
        #
        # metodlarının varlığı garanti edilir.
        if not issubclass(tokenizer_cls, BaseTokenizer):
            raise TypeError(
                f"{tokenizer_cls.__name__} must inherit from BaseTokenizer"
            )

        # DUPLICATE KONTROLÜ
        #
        # Aynı isimle ikinci kez kayıt yapılmasını engeller.
        #
        # Neden önemli?
        # - Silent override buglarını önler
        # - Plugin çakışmalarını yakalar
        if key in cls._registry:
            raise ValueError(f"Tokenizer already registered: {key}")

        cls._registry[key] = tokenizer_cls

        # Debug log (prod'da INFO yapılabilir)
        logger.debug(f"Tokenizer registered: {key}")

    @classmethod
    def get_all(cls) -> Dict[str, Type[BaseTokenizer]]:
        """
        Registry'deki tüm tokenizer class'larını döndürür.

        Returns:
            Dict[str, Type[BaseTokenizer]]

        Örnek çıktı:
            {
                "char": CharTokenizer,
                "word": WordTokenizer,
                "regex_bpe": RegexBPETokenizer
            }

        Not:
            dict(...) ile kopya döndürülür.
            Bu sayede dışarıdan registry mutasyonu engellenir.
        """
        return dict(cls._registry)

    @classmethod
    def create(cls, name: str) -> BaseTokenizer:
        """
        Verilen isimden bir tokenizer instance üretir.

        Bu method aslında TokenizerFactory'nin yaptığı işi doğrudan sağlar.

        Args:
            name:
                Oluşturulmak istenen tokenizer adı.

        Returns:
            BaseTokenizer instance

        Raises:
            ValueError:
                Eğer verilen isim registry'de yoksa.

        Örnek:
            tokenizer = TokenizerRegistry.create("regex_bpe")

        İç akış:
            1. isim normalize edilir
            2. registry'de kontrol edilir
            3. ilgili class instantiate edilir

        Not:
            Her çağrıda yeni instance üretilir.
            (stateless veya request-based kullanım için uygundur)
        """
        key = name.strip().lower()

        if key not in cls._registry:
            raise ValueError(f"Unsupported tokenizer: {name}")

        return cls._registry[key]()

    @classmethod
    def clear(cls) -> None:
        """
        Registry içeriğini tamamen temizler.

        Bu method production'da kullanılmaz.
        Sadece test senaryoları için eklenmiştir.

        Kullanım senaryoları:
            - unit test isolation
            - test suite reset
            - plugin reload

        Örnek:
            TokenizerRegistry.clear()
        """
        cls._registry.clear()

def register_tokenizer(name: str):
    """
    Tokenizer class'larını otomatik olarak registry'e ekleyen decorator.

    Bu decorator sayesinde tokenizer eklemek için merkezi bir yere
    dokunmaya gerek kalmaz.

    Kullanım:
        @register_tokenizer("regex_bpe")
        class RegexBPETokenizer(BaseTokenizer):
            ...

    Çalışma mantığı:
        Class tanımlandığında otomatik olarak:
            TokenizerRegistry.register(...) çağrılır

    Avantaj:
        - Boilerplate azaltır
        - Plugin sistemini aktif hale getirir
        - Kod okunabilirliğini artırır
    """
    def decorator(tokenizer_cls: Type[BaseTokenizer]):
        TokenizerRegistry.register(name, tokenizer_cls)
        return tokenizer_cls

    return decorator