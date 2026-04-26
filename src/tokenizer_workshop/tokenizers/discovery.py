from __future__ import annotations

import importlib
import logging
import pkgutil
from types import ModuleType
from typing import Set

from tokenizer_workshop import tokenizers

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Imported Module Cache
# ---------------------------------------------------------
# Bu set, auto-discovery sırasında daha önce başarıyla import edilmiş
# tokenizer modüllerini takip eder.
#
# Neden gerekli?
#   auto_import_tokenizers() fonksiyonu uygulama boyunca birden fazla kez
#   çağrılabilir. Örneğin:
#       - /api/tokenizers endpoint'i çağrıldığında
#       - /api/compare çalıştığında
#       - /api/report üretildiğinde
#
# Eğer her çağrıda tüm tokenizer modülleri yeniden import edilmeye çalışılırsa:
#   - gereksiz performans maliyeti oluşur
#   - decorator tabanlı registration tekrar tetiklenebilir
#   - duplicate registration hataları oluşabilir
#
# Bu nedenle import edilen modülleri burada cache'liyoruz.
_IMPORTED_MODULES: Set[str] = set()


def auto_import_tokenizers() -> None:
    """
    tokenizers package'i altındaki tokenizer modüllerini otomatik olarak keşfeder
    ve import eder.

    Bu fonksiyon, dinamik tokenizer registration sisteminin en kritik parçasıdır.

    Temel problem:
        @register_tokenizer(...) decorator'ı ancak ilgili tokenizer modülü
        Python tarafından import edildiğinde çalışır.

        Örneğin:
            @register_tokenizer("regex")
            class RegexTokenizer(BaseTokenizer):
                ...

        Bu class'ın registry'ye eklenebilmesi için regex_tokenizer.py dosyasının
        import edilmiş olması gerekir.

    Bu fonksiyon ne yapar?
        1. tokenizer_workshop.tokenizers package path'ini okur.
        2. Bu klasör altındaki Python modüllerini tarar.
        3. registry.py, discovery.py, base.py gibi altyapı dosyalarını atlar.
        4. Gerçek tokenizer modüllerini import eder.
        5. Import sırasında decorator'lar çalışır.
        6. TokenizerRegistry otomatik olarak dolar.

    Sonuç:
        Yeni tokenizer eklemek için artık TokenizerFactory değiştirilmez.

        Sadece:
            - yeni tokenizer dosyası oluşturulur
            - class üzerine @register_tokenizer("tokenizer_name") eklenir

    Örnek:
        src/tokenizer_workshop/tokenizers/regex_tokenizer.py

        @register_tokenizer("regex")
        class RegexTokenizer(BaseTokenizer):
            ...

        auto_import_tokenizers() çağrıldığında bu modül import edilir ve
        RegexTokenizer otomatik olarak registry'ye kaydedilir.

    Idempotency:
        Bu fonksiyon tekrar tekrar çağrılabilir.
        Daha önce import edilen modüller _IMPORTED_MODULES içinde tutulduğu için
        aynı modül ikinci kez işlenmez.

    Hata yönetimi:
        Bir tokenizer modülü import edilirken hata oluşursa hata loglanır ve
        RuntimeError olarak yeniden yükseltilir. Böylece uygulama sessizce
        eksik tokenizer listesiyle çalışmaz.
    """

    # Package adı:
    # Normalde "tokenizer_workshop.tokenizers" değerini alır.
    #
    # Bu değer dinamik import path'i oluşturmak için kullanılır.
    #
    # Örnek:
    #   package_name = "tokenizer_workshop.tokenizers"
    #   module_name = "regex_tokenizer"
    #   full_module_name = "tokenizer_workshop.tokenizers.regex_tokenizer"
    package_name = tokenizers.__name__

    # Package path:
    # pkgutil.iter_modules() bu path üzerinden klasör içindeki modülleri tarar.
    #
    # tokenizers.__path__ sadece package'larda bulunur.
    # Bu nedenle tokenizers klasörünün __init__.py içeren gerçek bir package
    # olması gerekir.
    package_path = tokenizers.__path__

    logger.debug("Starting tokenizer auto-discovery...")

    # pkgutil.iter_modules(package_path), tokenizers klasörü altındaki
    # import edilebilir modülleri listeler.
    #
    # module_info içinde:
    #   - module_info.name      → modül adı
    #   - module_info.ispkg     → package olup olmadığı
    #
    # Burada sadece name alanını kullanıyoruz.
    for module_info in pkgutil.iter_modules(package_path):
        module_name = module_info.name

        # ---------------------------------------------------------
        # Skip Core Infrastructure Modules
        # ---------------------------------------------------------
        # Bu dosyalar tokenizer implementasyonu değildir.
        #
        # registry.py:
        #   TokenizerRegistry ve register_tokenizer decorator'ını içerir.
        #
        # discovery.py:
        #   Bu auto-discovery fonksiyonunun bulunduğu dosyadır.
        #
        # base.py:
        #   BaseTokenizer abstract class'ını içerir.
        #
        # __init__:
        #   Package initializer dosyasıdır.
        #
        # Bu modülleri import etmek gereksizdir ve bazı durumlarda
        # circular import riskini artırabilir.
        if module_name in {"registry", "discovery", "base", "__init__"}:
            continue

        # Full import path oluşturulur.
        #
        # Örnek:
        #   module_name      = "regex_tokenizer"
        #   full_module_name = "tokenizer_workshop.tokenizers.regex_tokenizer"
        #
        # importlib.import_module() bu tam path'i bekler.
        full_module_name = f"{package_name}.{module_name}"

        # ---------------------------------------------------------
        # Idempotency Guard
        # ---------------------------------------------------------
        # Aynı modül daha önce import edildiyse tekrar işlemiyoruz.
        #
        # Bunun faydaları:
        #   - duplicate registration riskini azaltır
        #   - gereksiz import maliyetini önler
        #   - endpoint tekrar çağrılarında stabil davranış sağlar
        if full_module_name in _IMPORTED_MODULES:
            continue

        try:
            # ---------------------------------------------------------
            # Dynamic Module Import
            # ---------------------------------------------------------
            # Bu satır ilgili tokenizer modülünü runtime'da import eder.
            #
            # Import sırasında modülün top-level kodu çalışır.
            # Yani tokenizer class'ının üzerindeki decorator tetiklenir:
            #
            #   @register_tokenizer("regex")
            #   class RegexTokenizer(BaseTokenizer):
            #       ...
            #
            # Bu decorator da TokenizerRegistry.register(...) çağırır.
            module: ModuleType = importlib.import_module(full_module_name)

            # Import başarılı olduysa modül cache'e eklenir.
            #
            # Not:
            #   Cache'e sadece başarılı import sonrası ekliyoruz.
            #   Böylece hatalı importlar düzeltilip tekrar denenebilir.
            _IMPORTED_MODULES.add(full_module_name)

            logger.debug(
                "Tokenizer module loaded successfully: %s",
                full_module_name,
            )

        except Exception as exc:
            # ---------------------------------------------------------
            # Import Failure Handling
            # ---------------------------------------------------------
            # Bir tokenizer modülü import edilirken hata oluşabilir:
            #
            #   - syntax error
            #   - yanlış import path
            #   - BaseTokenizer uyumsuzluğu
            #   - duplicate registration
            #   - missing dependency
            #
            # Bu durumda hatayı sadece yutmak doğru değildir.
            # Çünkü registry eksik dolar ve UI/API yanlış tokenizer listesi döner.
            #
            # logger.exception traceback bilgisini de loglar.
            logger.exception(
                "Failed to import tokenizer module: %s",
                full_module_name,
            )

            # RuntimeError ile daha anlamlı bir üst seviye hata fırlatıyoruz.
            # Orijinal exception da `from exc` ile korunur.
            raise RuntimeError(
                f"Tokenizer auto-discovery failed for module: {full_module_name}"
            ) from exc

    logger.debug("Tokenizer auto-discovery completed.")