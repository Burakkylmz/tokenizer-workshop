"""
compare_service.py

Çoklu tokenizer karşılaştırma ve detaylı evaluation işlemlerinin service katmanı.

Bu modül iki farklı karşılaştırma akışı sağlar:

1. compare_tokenizers()
   Daha sade, hızlı ve temel compare çıktısı üretir.

2. evaluate_tokenizers()
   Metrik, latency ve pairwise comparison içeren zengin evaluation çıktısı üretir.

Bu servis, controller katmanından ayrılarak:
    - Tokenizer oluşturma, interface doğrulama, optional training, tokenize işlemi, metrik hesaplama ve pairwise comparison gibi business logic'i kapsar.
    - Böylece controller katmanı yalnızca HTTP davranışıyla ilgilenir, compare_service ise karşılaştırma işleminin tüm detaylarını yönetir.
    - Bu ayrım, kodun daha modüler, test edilebilir ve genişletilebilir olmasını sağlar.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

from tokenizer_workshop.api.services.exceptions import (
    TokenizationServiceError,
    UnsupportedTokenizerError,
)
from tokenizer_workshop.api.services.metrics_service import (
    build_pairwise_comparisons,
    calculate_metrics,
)
from tokenizer_workshop.api.services.tokenize_service import (
    tokenize_text,
    train_tokenizer_if_supported,
)
from tokenizer_workshop.api.services.tokenizer_factory import TokenizerFactory
from tokenizer_workshop.api.services.utils import (
    deduplicate_tokenizer_names,
    normalize_tokens,
    validate_tokenizer_interface,
)


def compare_tokenizers(text: str, tokenizer_names: list[str]) -> dict[str, Any]:
    """
    Aynı ham metni birden fazla tokenizer ile çalıştırarak temel karşılaştırma çıktısı üretir.

    Bu fonksiyon lightweight compare akışıdır. Amaç, her tokenizer'ın aynı input
    üzerinde hangi token listesini ürettiğini görmek ve temel sayısal bilgileri
    döndürmektir.

    Pipeline:
        1. tokenizer_names boş mu kontrol edilir.
        2. Tokenizer isimleri normalize edilir.
        3. Tekrar eden tokenizer isimleri sıralama korunarak temizlenir.
        4. Her tokenizer için tokenize_text() çalıştırılır.
        5. Sonuçlar standart response payload'u olarak döndürülür.

    Bu fonksiyon özellikle:
        - UI compare ekranı
        - hızlı token preview
        - temel API response
        - report generation öncesi sade compare payload'u

    için uygundur.

    Args:
        text:
            Karşılaştırılacak kaynak metin.

        tokenizer_names:
            Çalıştırılacak tokenizer adları.

    Returns:
        Temel tokenizer karşılaştırma payload'u.

        {
            "text": "...",
            "total_tokenizers": 3,
            "results": [
                {
                    "tokenizer_name": "char",
                    "tokens": [...],
                    "token_count": 10,
                    "vocab_size": 8
                }
            ]
        }

    Raises:
        TokenizationServiceError:
            tokenizer_names boşsa veya compare sırasında beklenmeyen hata oluşursa.

        UnsupportedTokenizerError:
            Desteklenmeyen tokenizer adı verilirse.
    """
    if not tokenizer_names:
        raise TokenizationServiceError("At least one tokenizer must be selected.")

    try:
        # Tokenizer isimleri normalize edilir ve tekrar edenler temizlenir.
        normalized_names = deduplicate_tokenizer_names(tokenizer_names)

        results = [
            tokenize_text(text=text, tokenizer_name=name) 
            for name in normalized_names
        ] # Tokenizer'lar tek tek çalıştırılır ve sonuçları evaluations listesinde toplanır.
        # Her tokenizer için tokenize_text fonksiyonu çağrılır, bu fonksiyon tokenizer'ı oluşturur, 
        # gerekirse train eder, metni tokenize eder ve normalize eder.

        return {
            "text": text,
            "total_tokenizers": len(results),
            "results": results,
        }

    except UnsupportedTokenizerError:
        raise

    except TokenizationServiceError:
        raise

    except Exception as exc:
        raise TokenizationServiceError(
            f"An unexpected error occurred while comparing tokenizers: {type(exc).__name__}: {exc}"
        ) from exc


def evaluate_tokenizers(text: str, tokenizer_names: list[str]) -> dict[str, Any]:
    """
    Aynı ham metni birden fazla tokenizer ile çalıştırarak detaylı evaluation çıktısı üretir.

    Bu fonksiyon compare_tokenizers() fonksiyonundan daha zengindir.
    Sadece token listesini döndürmez; aynı zamanda her tokenizer için metrik,
    latency ve pairwise comparison bilgisi üretir.

    Pipeline:
        1. tokenizer_names boş mu kontrol edilir.
        2. Tokenizer isimleri normalize edilip duplicate değerler temizlenir.
        3. Her tokenizer factory üzerinden oluşturulur.
        4. Tokenizer interface'i doğrulanır.
        5. train() destekliyorsa tokenizer input text ile eğitilir.
        6. tokenize() süresi perf_counter() ile ölçülür.
        7. Tokenizer çıktısı list[str] formatına normalize edilir.
        8. calculate_metrics() ile detaylı metrikler hesaplanır.
        9. build_pairwise_comparisons() ile tokenizer çiftleri karşılaştırılır.
        10. Zenginleştirilmiş evaluation payload'u döndürülür.

    Bu fonksiyon özellikle:
        - detaylı API evaluation endpoint'i
        - TXT / MD / PDF rapor üretimi
        - tokenizer benchmark analizi
        - latency ve compression karşılaştırması
        - pairwise token overlap analizi
    için kullanılır.

    Args:
        text:
            Karşılaştırma yapılacak kaynak metin.

        tokenizer_names:
            Çalıştırılacak tokenizer adları.

    Returns:
        Zengin tokenizer evaluation payload'u.

        {
            "source_text": "...",
            "evaluations": [
                {
                    "tokenizer_name": "word",
                    "tokens": [...],
                    "metrics": {...}
                }
            ],
            "pairwise_comparisons": [...]
         }

    Raises:
        TokenizationServiceError:
            tokenizer_names boşsa veya evaluation sırasında beklenmeyen hata oluşursa.

        UnsupportedTokenizerError:
            Desteklenmeyen tokenizer adı verilirse.
    """
    if not tokenizer_names:
        raise TokenizationServiceError("At least one tokenizer must be selected.")

    try:
        normalized_names = deduplicate_tokenizer_names(tokenizer_names)

        evaluations: list[dict[str, Any]] = []

        for name in normalized_names:
            tokenizer = TokenizerFactory.create(name)

            # Tokenizer'ın tokenize(text) metodunu destekleyip desteklemediği doğrulanır.
            validate_tokenizer_interface(tokenizer, name)

            # Tokenizer train() metodunu destekliyorsa, tokenize işlemi öncesinde eğitim yapılır.
            train_tokenizer_if_supported(tokenizer, text)

            start = perf_counter() # Tokenizer ile metin tokenize edilirken geçen süre ölçülür.
            raw_tokens = tokenizer.tokenize(text) # Tokenizer'lar farklı token formatları döndürebilir (örneğin, bazıları byte dizisi döndürebilir).
            end = perf_counter() # Tokenizer'ın tokenize() metodunun çalışması tamamlandığında süre ölçümü sona erer.

            normalized_tokens = normalize_tokens(raw_tokens) # normalize_tokens helper'ı, farklı formatlardaki token listelerini API response'u için uygun hale getirir.
            # Örneğin, byte tokenları string'e çevirir veya özel token objelerini string formatına dönüştürür. Bu adım, API response'unun tutarlı ve anlaşılır olmasını sağlar, tokenizer'ların iç detaylarını gizler.

            latency_seconds = end - start # Tokenizer'ın tokenize() metodunun çalışması sırasında geçen süre, latency_seconds değişkenine saniye cinsinden atanır.  

            metrics = calculate_metrics(
                tokens=normalized_tokens,
                latency_seconds=latency_seconds,
                source_text=text,
            )

            evaluations.append(
                {
                    "tokenizer_name": TokenizerFactory.normalize_name(name),
                    "tokens": normalized_tokens,
                    "metrics": metrics,
                }
            )

        # Tüm tokenizer değerlendirmeleri tamamlandıktan sonra, build_pairwise_comparisons fonksiyonu çağrılarak 
        # tokenizer çiftleri arasında karşılaştırmalar yapılır ve sonuçlar pairwise_comparisons değişkenine atanır.
        # Bu karşılaştırmalar, her bir tokenizer çiftinin token listeleri arasındaki benzerlik ve farkları analiz eder, 
        # böylece hangi tokenizer'ların benzer çıktılar ürettiği veya birbirlerinden ne kadar farklı oldukları hakkında bilgi sağlar.
        # Sonuç olarak, evaluations listesinde her tokenizer için detaylı metrikler ve token listeleri bulunurken,
        # pairwise_comparisons listesinde ise tokenizer çiftleri arasındaki karşılaştırma sonuçları yer alır.
        pairwise_comparisons = build_pairwise_comparisons(evaluations)

        return {
            "source_text": text,
            "evaluations": evaluations,
            "pairwise_comparisons": pairwise_comparisons,
        }

    except UnsupportedTokenizerError:
        raise

    except TokenizationServiceError:
        raise

    except Exception as exc:
        raise TokenizationServiceError(
            f"An unexpected error occurred while generating the evaluation result: {type(exc).__name__}: {exc}"
        ) from exc  