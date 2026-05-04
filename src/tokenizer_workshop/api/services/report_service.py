"""
report_service.py

Tokenizer karşılaştırma sonuçlarından rapor üretme servisleri.

Bu modül, API katmanında rapor üretimiyle ilgili tüm logic'i içerir.

Sorumlulukları:
    - Compare sonuçlarını istenen formatta raporlara dönüştürmek
    - Rapor formatı için uygun builder fonksiyonunu seçmek
    - API response için standart rapor payload'u oluşturmak
    - Rapor üretimi sırasında oluşabilecek hataları yönetmek

Rapor formatları:
    - "txt": Düz metin raporu
    - "md": Markdown formatında rapor
    - "pdf": PDF formatında rapor (ileride eklenebilir)

Rapor builder'lar:
    - Her format için ayrı builder fonksiyonları bulunur.
    - Builder fonksiyonları, compare sonuçlarını alır ve formatına uygun rapor çıktısı üretir.
    - Builder'lar, raporun içeriği ve düzeniyle ilgili tüm logici kapsar.
    - Bu sayede rapor üretimi modüler ve genişletilebilir olur.

Api response formatı:
    {
        "report": "...",       # Rapor içeriği (string)
        "format": "md",        # Rapor formatı (örneğin "txt", "md", "pdf")
        "filename": "tokenizer_report.md"  # İndirilebilir dosya adı
    }

Hata yönetimi:
    - Desteklenmeyen rapor formatı verilirse UnsupportedReportFormatError fırlatılır.
    - Desteklenmeyen tokenizer adı verilirse UnsupportedTokenizerError fırlatılır.
    - Compare veya rapor üretimi sırasında beklenmeyen hata oluşursa TokenizationServiceError fırlatılır.

Not:
    Bu servis, API katmanında rapor üretimiyle ilgili tüm logic'i içerir.
    Compare sonuçlarını istenen formatta raporlara dönüştürmekten sorumludur.
    Rapor formatı için uygun builder fonksiyonunu seçer ve API response için standart rapor payload'u oluşturur.
    Rapor üretimi sırasında oluşabilecek hataları yönetir.
"""

from typing import Any

from tokenizer_workshop.api.services.compare_service import run_compare
from tokenizer_workshop.api.reports.factory import get_report_builder


def generate_report_service(
    text: str,
    tokenizer_names: list[str],
    fmt: str,
) -> dict[str, Any]:
    """
    Tokenizer karşılaştırma sonucundan istenen formatta rapor üretir.

    Bu servis, report generation flow'unun ana orkestrasyon noktasıdır.

    Sorumlulukları:
        1. Verilen text ve tokenizer listesiyle compare pipeline'ını çalıştırmak
        2. İstenen rapor formatına uygun builder fonksiyonunu seçmek
        3. Compare sonucunu okunabilir rapor çıktısına dönüştürmek
        4. API layer'a standart response payload'u döndürmek

    Akış:
        text + tokenizer_names
            ↓
        run_compare(...)
            ↓
        compare_result
            ↓
        get_report_builder(fmt)
            ↓
        builder(compare_result)
            ↓
        report response

    Args:
        text:
            Raporlanacak ham metin.

        tokenizer_names:
            Karşılaştırmada kullanılacak tokenizer adları.

        fmt:
            Üretilecek rapor formatı.
            Örnek: "txt", "md", "pdf"

    Returns:
        API response içinde kullanılabilecek standart report payload'u.

        {
            "report": "...",
            "format": "md",
            "filename": "tokenizer_report.md"
        }

    Raises:
        UnsupportedReportFormatError:
            fmt desteklenmeyen bir rapor formatıysa.

        UnsupportedTokenizerError:
            tokenizer_names içinde desteklenmeyen tokenizer varsa.

        TokenizationServiceError:
            Compare veya report generation sırasında beklenmeyen hata oluşursa.
    """
    compare_result = run_compare(text, tokenizer_names)

    builder, filename = get_report_builder(fmt)
    report = builder(compare_result)

    return {
        "report": report,
        "format": fmt,
        "filename": filename,
    }