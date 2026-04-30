"""
helpers.py - Rapor oluşturma sürecinde kullanılan yardımcı fonksiyonları içerir.

Bu modül, raporlarda sıkça kullanılan işlemleri tek bir yerde toplayarak, rapor oluşturma sürecini daha düzenli ve sürdürülebilir hale getirir.

İçerdiği fonksiyonlar:
    - wide_hr: Geniş bir yatay çizgi oluşturur.
    - utc_now_iso: Geçerli UTC zamanını ISO 8601 formatında döndürür.
    - safe_str: Verilen değeri string'e dönüştürür, None ise fallback değeri döndürür.
    - truncate_list: Verilen listeyi belirtilen maksimum öğe sayısına göre kısaltır.
    - format_number: Verilen sayıyı belirtilen ondalık basamak sayısına göre formatlar.
    - hr: Raporlarda kullanılacak yatay çizgi oluşturur.
    - normalize_text: Reconstructed text comparison için metni normalize eder.
    - is_reconstruction_match: Original text ile reconstructed text değerlerini normalize ederek karşılaştırır.
    - get_metrics: Item'dan metrikleri alır.
    - get_metric: Belirli bir metrik değerini alır.
    - latency_microseconds: Latency değerini mikro saniye cinsinden döner.
    - format_top_tokens: Top tokenleri formatlar.
    - format_reconstruction: Reconstruction metriklerini formatlar.
    - append_section_title: Rapor bölümü başlığı ekler.
    - extract_compare_payload: Compare tokenizers servisinden gelen sonucu güvenli bir şekilde ayrıştırır.
    - interpret_similarity_level: Similarity level değerini yorumlar.
    - format_pairwise_comparisons: Pairwise comparison sonuçlarını yorumlar ve formatlar.

Bu yardımcı fonksiyonlar, rapor oluşturma sürecinde tekrar eden kodları azaltarak, kodun daha temiz, anlaşılır ve bakımı kolay olmasını sağlar.
"""

from datetime import datetime, timezone
from typing import Any

import re

REPORT_WIDTH = 88 # Rapor tabloları için genişlik sınırı
REPORT_TABLE_WIDTH = 120 # Rapor tablolarında geniş veri setleri için daha geniş bir sınır (örneğin, top tokens listesi)


def wide_hr(char: str = "=") -> str:
    """
    Geniş bir yatay çizgi (horizontal rule) oluşturur.
    Bu, raporlarda bölümleri görsel olarak ayırmak için kullanılır.

    Args:
        char:
            Yatay çizgi karakteri. Varsayılan olarak "=" kullanılır.

    Returns:        
        Geniş bir yatay çizgi string'i.
    """
    return char * REPORT_TABLE_WIDTH


def utc_now_iso() -> str:
    """
    UTC zaman diliminde geçerli olan şu anki zamanı ISO 8601 formatında döndürür.
    Bu, raporlarda zaman damgası olarak kullanılabilir.

    Örnek çıktı: "2024-06-01T12:34:56.789012+00:00"

    Bu fonksiyon, raporlarda zaman damgası olarak kullanılmak üzere standart bir format sağlar.

    ISO 8601 formatı, tarih ve saat bilgisini uluslararası standartlara uygun şekilde ifade eder ve 
    zaman dilimi bilgisini de içerir. Bu sayede raporlarda tutarlı ve anlaşılır zaman bilgisi sunar.

    Returns:
        Geçerli UTC zamanını ISO 8601 formatında string olarak döndürür.
    """
    return datetime.now(timezone.utc).isoformat()


def safe_str(value: Any, fallback: str = "-") -> str:
    """
    Verilen değeri string'e dönüştürür. 
    Eğer değer None ise, belirtilen fallback değerini döndürür.

    Bu fonksiyon, raporlarda eksik veya None olan değerler için tutarlı bir gösterim sağlar.

    Örnek kullanımlar:
        safe_str("hello") -> "hello"
        safe_str(None) -> "-"
        safe_str(123) -> "123"

    Args:
        value: String'e dönüştürülecek değer.
        fallback: Değer None ise döndürülecek yedek string. Varsayılan olarak "-" kullanılır.

    Returns:
        String'e dönüştürülmüş değer veya fallback değeri.
    """
    if value is None:
        return fallback
    return str(value)


def truncate_list(items: list[Any], max_items: int = 20) -> str:
    """
    Verilen listeyi belirtilen maksimum öğe sayısına göre kısaltır.
    Eğer liste boşsa "[]" döndürür.
    Eğer liste maksimum öğe sayısından az veya eşitse, listeyi string olarak döndürür.
    Aksi takdirde, görünür öğeleri ve kalan öğe sayısını gösterir.

    Bu fonksiyon, raporlarda uzun listelerin görsel karmaşasını azaltmak için kullanılır.

    Örnek kullanımlar:
        truncate_list([1, 2, 3, 4, 5], max_items=3) -> "[1, 2, 3] ... (+2 more)"
        truncate_list([], max_items=3) -> "[]"
        truncate_list([1, 2], max_items=3) -> "[1, 2]"

    Args:
        items: Kısaltılacak liste.
        max_items: Görüntülenecek maksimum öğe sayısı. Varsayılan olarak 20 kullanılır.

    Returns:
        Kısaltılmış listeyi temsil eden string.
    """
    # Eğer liste boşsa, "[]" döndürülür.
    if not items:
        return "[]"

    # Eğer öğe sayısı maksimumdan az veya eşitse, tüm listeyi string olarak döndürür.
    if len(items) <= max_items:
        return str(items)

    visible = items[:max_items] # Görüntülenecek öğeler
    remaining = len(items) - max_items # Kalan öğe sayısı

    return f"{visible} ... (+{remaining} more)"


def format_number(value: Any, digits: int = 2) -> str:
    """
    Verilen sayıyı belirtilen ondalık basamak sayısına göre formatlar.
    Eğer değer bir sayı değilse, "-" döndürür.

    Bu fonksiyon, raporlarda sayısal değerlerin tutarlı ve okunabilir bir formatta sunulmasını sağlar.

    Örnek kullanımlar:
        format_number(123.456) -> "123.46"
        format_number(123.456, digits=1) -> "123.5"
        format_number("abc") -> "-"
        format_number(None) -> "-"
        format_number(123) -> "123.00"
        format_number(123, digits=0) -> "123"

    Args:
        value: Formatlanacak sayı.
        digits: Ondalık basamak sayısı. Varsayılan olarak 2 kullanılır.

    Returns:
        Formatlanmış sayı string'i veya "-" değeri.
    """
    # Eğer değer int veya float ise, belirtilen ondalık basamak sayısına göre formatlanır.
    # Aksi takdirde, "-" döndürülür.
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return "-"


def hr(char: str = "=") -> str:
    """
    Raporlarda kullanılacak yatay çizgi (horizontal rule) oluşturur.

    Bu fonksiyon, raporlarda bölümleri görsel olarak ayırmak için kullanılır.

    Args:
        char: Yatay çizgi karakteri. Varsayılan olarak "=" kullanılır.

    Returns:
        Yatay çizgi string'i.
    """
    return char * REPORT_WIDTH


def normalize_text(text: str) -> str:
    """
    Reconstructed text comparison için metni normalize eder.

    Normalizasyon adımları:
    - Başındaki ve sonundaki boşlukları kaldırır
    - Birden fazla boşluğu tek bir boşluğa indirger
    - Noktalama işaretlerinden önceki boşlukları kaldırır
    - Tokenizer'ların farklı join stratejilerinden kaynaklanan küçük formatlama farklılıklarını ortadan kaldırır
    - Bu normalizasyon, tokenizer'ların farklı join stratejilerinden kaynaklanan küçük formatlama farklılıklarını ortadan kaldırarak, 
    reconstructed text'in orijinal metinle daha doğru bir şekilde karşılaştırılmasını sağlar.
    
    Normalizasyon sayesinde, tokenizer'ların ürettiği reconstructed text'in orijinal metinle eşleşip eşleşmediği daha güvenilir bir şekilde değerlendirilebilir. 
    Özellikle, bazı tokenizer'lar token'ları birleştirirken ekstra boşluklar ekleyebilir veya noktalama işaretlerinden önce gereksiz boşluklar bırakabilir. 
    Bu tür küçük farklılıklar, doğrudan string karşılaştırmalarında false-negative sonuçlara yol açabilir. 
    normalize_text fonksiyonu bu tür durumları azaltarak, tokenizer'ların gerçek performansını daha doğru bir şekilde yansıtmayı amaçlar.

    Amaç:
    - fazla boşlukları tek boşluğa indirmek,
    - noktalama işaretlerinden önceki gereksiz boşlukları kaldırmak,
    - tokenizer join farklılıklarından kaynaklanan false-negative durumları azaltmak.

    Args:
        text: Normalizasyon uygulanacak metin.
        
    Returns:
        Normalize edilmiş metin.
    """

    text = text.strip() # Başındaki ve sonundaki boşlukları kaldırır
    text = re.sub(r"\s+", " ", text) # Birden fazla boşluğu tek bir boşluğa indirger
    text = re.sub(r"\s+([.,!?;:])", r"\1", text) # Noktalama işaretlerinden önceki boşlukları kaldırır

    return text


def is_reconstruction_match(original_text: str, reconstructed_text: str) -> bool:
    """
    Original text ile reconstructed text değerlerini normalize ederek karşılaştırır.

    Bu fonksiyon, tokenizer'ların ürettiği reconstructed text'in orijinal metinle eşleşip eşleşmediğini değerlendirmek için kullanılır.
    Normalizasyon adımları, tokenizer'ların farklı join stratejilerinden kaynaklanan küçük formatlama farklılıklarını ortadan kaldırarak, reconstructed text'in orijinal metinle daha doğru bir şekilde karşılaştırılmasını sağlar.
    Bu sayede, tokenizer'ların gerçek performansını daha doğru bir şekilde yansıtmayı amaçlar.

    Args:
        original_text: Orijinal metin.
        reconstructed_text: Yeniden oluşturulmuş metin.

    Returns:
        Metinlerin eşleşip eşleşmediğini belirten boolean değer.
    """
    # Normalizasyon sayesinde, tokenizer'ların ürettiği reconstructed text'in orijinal metinle eşleşip eşleşmediği daha güvenilir bir şekilde değerlendirilebilir.
    # Özellikle, bazı tokenizer'lar token'ları birleştirirken ekstra boşluklar ekleyebilir veya noktalama işaretlerinden önce gereksiz boşluklar bırakabilir.
    # Bu tür küçük farklılıklar, doğrudan string karşılaştırmalarında false-negative sonuçlara yol açabilir.
    # normalize_text fonksiyonu bu tür durumları azaltarak, tokenizer'ların gerçek performansını daha doğru bir şekilde yansıtmayı amaçlar.
    # Amaç:
    # - fazla boşlukları tek boşluğa indirmek,
    # - noktalama işaretlerinden önceki gereksiz boşlukları kaldırmak,
    # - tokenizer join farklılıklarından kaynaklanan false-negative durumları azaltmak.
    return normalize_text(original_text) == normalize_text(reconstructed_text)


def get_metrics(item: dict[str, Any]) -> dict[str, Any]:
    """
    Item'dan metrikleri alır.

    Bu fonksiyon, raporlarda her tokenizer için hesaplanan metriklerin bulunduğu sözlüğü güvenli bir şekilde almak için kullanılır.
    Eğer item içinde "metrics" anahtarı yoksa veya değeri bir sözlük değilse, boş bir sözlük döndürür. 
    Bu sayede raporlarda metriklere erişim sırasında oluşabilecek hataların önüne geçilir ve eksik metrikler için tutarlı bir gösterim sağlanır.

    Args:
        item: Metriklerin bulunduğu sözlük.

    Returns:
        Metriklerin bulunduğu sözlük. Eğer metrikler yoksa boş bir sözlük döner.
    """
    # item sözlüğünden "metrics" anahtarıyla metrikler alınır. 
    # Eğer "metrics" anahtarı yoksa veya değeri bir sözlük değilse, boş bir sözlük döndürülür.
    metrics = item.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def get_metric(item: dict[str, Any], key: str, fallback: Any = "-") -> Any:
    """
    Belirli bir metrik değerini alır.

    Bu fonksiyon, raporlarda her tokenizer için hesaplanan metriklerin bulunduğu sözlükten belirli bir metrik değerini güvenli bir şekilde almak için kullanılır.
    Eğer istenen metrik anahtarı mevcut değilse veya metrikler sözlüğü geçerli değilse, belirtilen fallback değeri döndürülür.
    Bu sayede raporlarda metriklere erişim sırasında oluşabilecek hataların önüne geçilir ve eksik metrikler için tutarlı bir gösterim sağlanır.

    Args:
        item: Metriklerin bulunduğu sözlük.
        key: Alınacak metrik anahtarı.
        fallback: Metrik bulunamazsa dönecek varsayılan değer.

    Returns:
        Metrik değeri veya varsayılan değer.
    """
    # item sözlüğünden metrikler alınır. Eğer metrikler geçerli değilse, boş bir sözlük kullanılır.
    # Daha sonra, metrikler sözlüğünden istenen anahtara karşılık gelen değer alınır. 
    # Eğer anahtar mevcut değilse veya metrikler sözlüğü geçerli değilse, fallback değeri döndürülür.
    metrics = get_metrics(item)
    return metrics.get(key, fallback)


def latency_microseconds(metrics: dict[str, Any]) -> str:
    """
    Latency değerini mikro saniye cinsinden döner.

    Bu fonksiyon, raporlarda her tokenizer için hesaplanan latency değerini mikro saniye cinsinden döndürmek için kullanılır.
    Eğer latency değeri geçerli değilse, "-" döndürür.

    Args:
        metrics: Metriklerin bulunduğu sözlük.

    Returns:
        Latency değeri mikro saniye cinsinden veya "-" ifadesi.
    """
    # "latency_seconds" anahtarıyla latency değeri alınır. 
    # Eğer değer geçerli bir sayı ise, mikro saniye cinsine çevrilir ve string olarak döndürülür.
    # Aksi takdirde, "-" döndürülür.

    latency = metrics.get("latency_seconds") # Latency değeri saniye cinsinden alınır.

    if isinstance(latency, (int, float)):
        return str(int(latency * 1_000_000)) # Saniyeyi mikro saniyeye çevirir

    return "-"


def format_top_tokens(top_tokens: Any) -> list[str]:
    """
    Top tokenleri formatlar.

    Bu fonksiyon, raporlarda her tokenizer için en sık kullanılan tokenleri formatlamak için kullanılır.
    Eğer top_tokens geçerli bir liste değilse, boş bir liste döndürülür.

    Args:
        top_tokens: Top tokenlerin bulunduğu liste.

    Returns:
        Formatlanmış top tokenler listesi.
    """
    # Eğer top_tokens geçerli bir liste değilse, boş bir liste döndürülür.
    if not isinstance(top_tokens, list):
        return []

    formatted: list[str] = [] # Formatlanmış top tokenler listesi oluşturulur.

    for item in top_tokens[:5]:
        # Eğer item bir sözlükse, "token" ve "count" anahtarlarıyla token değeri ve sayısı alınır ve formatlanmış bir string oluşturulur.
        if isinstance(item, dict):
            token = item.get("token") # Token değeri alınır
            count = item.get("count") # Tokenin kaç kez geçtiği sayısı alınır
            formatted.append(f"  {token} → {count}") # Formatlanmış string oluşturulur ve listeye eklenir
        else:
            formatted.append(f"  {item}")

    return formatted


def format_reconstruction(
    metrics: dict[str, Any],
    original_text: str | None = None,
) -> list[str]:
    """
    Reconstruction metriklerini formatlar.

    Bu fonksiyon, raporlarda her tokenizer için hesaplanan reconstruction metriklerini formatlamak için kullanılır.
    Eğer reconstruction metrikleri geçerli değilse, boş bir liste döndürülür.

    Args:
        metrics: Metriklerin bulunduğu sözlük.
        original_text: Orijinal metin.

    Returns:
        Formatlanmış reconstruction metrikleri listesi.
    """
    # Reconstruction match ve reconstructed text değerleri alınır. 
    # Eğer her ikisi de geçerli değilse, boş bir liste döndürülür.
    
    reconstruction_match = metrics.get("reconstruction_match") # Reconstruction match değeri alınır
    reconstructed_text = metrics.get("reconstructed_text") # Reconstructed text değeri alınır

    # Eğer her ikisi de geçerli değilse, boş bir liste döndürülür.
    if reconstruction_match is None and reconstructed_text is None:
        return []

    # Reconstruction match değeri ve reconstructed text değeri üzerinden karşılaştırma yapılır.
    # original_text ve reconstructed_text normalize edilerek karşılaştırılır.
    # Bu normalizasyon, tokenizer'ların farklı join stratejilerinden kaynaklanan küçük formatlama farklılıklarını ortadan kaldırarak, 
    # reconstructed text'in orijinal metinle daha doğru bir şekilde karşılaştırılmasını sağlar. 
    # normalize_text fonksiyonu bu tür durumları azaltarak, tokenizer'ların gerçek performansını daha doğru bir şekilde yansıtmayı amaçlar. 
    # Amaç:
    # - fazla boşlukları tek boşluğa indirmek,
    # - noktalama işaretlerinden önceki gereksiz boşlukları kaldırmak,
    # - tokenizer join farklılıklarından kaynaklanan false-negative durumları azaltmak.
    if original_text is not None and reconstructed_text is not None:
        reconstruction_match = is_reconstruction_match(
            original_text=original_text,
            reconstructed_text=reconstructed_text,
        )

    # Reconstruction match değeri üzerinden bir ikon belirlenir. 
    # Eğer match sağlanıyorsa "✔", sağlanmıyorsa "✘" kullanılır.
    icon = "✔" if reconstruction_match else "✘"

    # Reconstruction match sonucu ve reconstructed text bilgisi formatlanarak bir liste halinde döndürülür.
    lines = [
        f"  Match: {icon} {reconstruction_match}",
    ]

    # Eğer reconstructed text geçerli ise, reconstructed text bilgisi de formatlanarak listeye eklenir.
    if reconstructed_text:
        lines.append(f'  Output: "{reconstructed_text}"')

    # Eğer reconstruction match sağlanmıyorsa ve reconstructed text geçerli ise, 
    # bu durumun tokenizer'ların farklı join stratejilerinden kaynaklanan küçük formatlama farklılıklarından kaynaklanabileceği bilgisi de listeye eklenir.
    if reconstruction_match is False and reconstructed_text:
        lines.append(
            "  Note: This tokenizer output cannot be directly reconstructed into the original text "
            "(likely due to encoded or non-text token representations)."
        )

    return lines


def append_section_title(lines: list[str], title: str) -> None:
    """
    Rapor bölümü başlığı ekler.

    Bu fonksiyon, raporlarda bölümleri görsel olarak ayırmak ve başlıklandırmak için kullanılır.
    Bölüm başlığı, yatay çizgi ile ayrılarak vurgulanır.
    """
    lines.append(title)
    lines.append(hr("-"))


def extract_compare_payload(
    compare_result: dict[str, Any],
) -> tuple[str, int, list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Compare tokenizers servisinden gelen sonucu güvenli bir şekilde ayrıştırır.

    Bu fonksiyon, compare_tokenizers() servisinden gelen sonucu güvenli bir şekilde ayrıştırmak için kullanılır.

    Compare tokenizers servisi, iki veya daha fazla tokenizer'ın aynı metin üzerinde karşılaştırıldığı bir pipeline'ın sonucunu döndürür. 
    Bu sonuç, karşılaştırılan metin, toplam tokenizer sayısı, her tokenizer için hesaplanan metrikler ve tokenizer çiftleri arasındaki benzerlik analizlerini içerebilir.
    Bu fonksiyon, bu sonucu güvenli bir şekilde ayrıştırarak, raporlarda kullanılacak formatta veriler elde etmeyi amaçlar. 
    Eğer beklenen veri yapısı sağlanmazsa, fonksiyon varsayılan değerler döndürerek raporlarda hataların önüne geçer.   
    
    Args:
        compare_result: Compare tokenizers servisinden gelen sonuç sözlüğü.
    
    Returns:
        Tuple[str, int, list[dict[str, Any]], list[dict[str, Any]]]: Ayrıştırılmış metin, toplam tokenizer sayısı, her tokenizer için hesaplanan metrikler ve tokenizer çiftleri arasındaki benzerlik analizleri.
    """
    # Compare tokenizers servisinden gelen sonuç sözlüğünden metin, toplam tokenizer sayısı, her tokenizer için hesaplanan metrikler ve tokenizer çiftleri arasındaki benzerlik analizleri güvenli bir şekilde ayrıştırılır.
    # Eğer beklenen veri yapısı sağlanmazsa, fonksiyon varsayılan değerler döndürerek raporlarda hataların önüne geçer.

    text = safe_str(
        compare_result.get("text")
        or compare_result.get("source_text")
        or "",
        "",
    )

    results = (
        compare_result.get("results")
        or compare_result.get("evaluations")
        or []
    )

    if not isinstance(results, list):
        results = []

    total = int(
        compare_result.get("total_tokenizers")
        or len(results)
        or 0
    )

    pairwise = compare_result.get("pairwise_comparisons") or []

    if not isinstance(pairwise, list):
        pairwise = []

    return text, total, results, pairwise


def interpret_similarity_level(similarity_level: str) -> str:
    """
    Pairwise tokenizer similarity seviyesini kısa, okunabilir bir yoruma dönüştürür.

    Bu fonksiyon, tokenizer çiftleri arasındaki similarity seviyesini yorumlamak için kullanılır.

    Similarity seviyeleri, tokenizer çiftlerinin token setleri arasındaki benzerliği ifade eder ve 
    "Completely Different", "Highly Similar" gibi kategorilere ayrılabilir.

    Bu fonksiyon, bu kategorilere karşılık gelen kısa ve okunabilir yorumlar sağlayarak kullanıcıların sonuçları daha hızlı yorumlamasını sağlar.
    Bu alan özellikle rapor üretiminde faydalıdır. 
    Kullanıcı yalnızca "Highly Similar" gibi bir kategori görmek yerine, tokenizer çiftlerinin ne kadar benzer olduğunu daha hızlı yorumlayabilir.

    Args:
        similarity_level: Tokenizer çiftlerinin similarity seviyesini ifade eden kategori.

    Returns:
        Similarity seviyesinin kısa, okunabilir yorumu.
    """
    # Similarity seviyeleri için yorumlar sözlüğü oluşturulur.
    interpretations = {
        "Completely Different": (
            "These tokenizers produce almost entirely different token sets. "
            "Their segmentation strategies are strongly different."
        ),
        "Highly Different": (
            "These tokenizers share very few tokens. "
            "They likely segment the text at different granularity levels."
        ),
        "Moderately Similar": (
            "These tokenizers share some tokenization behavior, "
            "but still differ in several segmentation decisions."
        ),
        "Highly Similar": (
            "These tokenizers produce highly similar token sets. "
            "One may be usable as a close alternative to the other."
        ),
        "Nearly Identical": (
            "These tokenizers produce almost identical token sets. "
            "Their practical difference may be minimal for this input."
        ),
    }

    return interpretations.get(
        similarity_level,
        "Similarity level could not be interpreted.",
    )


def format_pairwise_interpretation(pairwise_comparisons: Any) -> list[str]:
    """
    Pairwise comparison sonuçlarını report'a yazılabilir interpretation satırlarına dönüştürür.

    Bu fonksiyon, raporlarda tokenizer çiftleri arasındaki similarity seviyesini yorumlamak için kullanılır.
    
    Similarity seviyeleri, tokenizer çiftlerinin token setleri arasındaki benzerliği ifade eder ve 
    "Completely Different", "Highly Similar" gibi kategorilere ayrılabilir.
    
    Bu fonksiyon, bu kategorilere karşılık gelen kısa ve okunabilir yorumlar sağlayarak kullanıcıların sonuçları daha hızlı yorumlamasını sağlar.
    Bu alan özellikle rapor üretiminde faydalıdır. 
    Kullanıcı yalnızca "Highly Similar" gibi bir kategori görmek yerine, tokenizer çiftlerinin ne kadar benzer olduğunu daha hızlı yorumlayabilir.
    
    Args:
        pairwise_comparisons: Tokenizer çiftleri arasındaki similarity seviyelerini içeren veri yapısı.

    Returns:
        Pairwise comparison sonuçlarının yorumlanmış ve formatlanmış hali.
    """
    # Eğer pairwise_comparisons geçerli bir liste değilse veya boşsa, "No pairwise comparison data available." ifadesi içeren bir liste döndürülür.
    if not isinstance(pairwise_comparisons, list) or not pairwise_comparisons:
        return ["No pairwise comparison data available."]

    lines: list[str] = [] # Pairwise comparison sonuçlarının yorumlanmış ve formatlanmış hali için boş bir liste oluşturulur.

    # Her bir pairwise comparison sonucu için yorumlar oluşturulur ve lines listesine eklenir.
    for item in pairwise_comparisons:
        # Eğer item bir sözlük değilse, bu item atlanır ve bir sonraki item'a geçilir.
        if not isinstance(item, dict):
            continue

        left = safe_str(item.get("left_name")) # Sol tokenizer adı güvenli bir şekilde alınır.
        right = safe_str(item.get("right_name")) # Sağ tokenizer adı güvenli bir şekilde alınır.
        overlap = format_number(item.get("overlap_ratio")) # Overlap ratio değeri güvenli bir şekilde formatlanır.
        similarity_level = safe_str(item.get("similarity_level")) # Similarity level değeri güvenli bir şekilde alınır.
        similarity_level_interpretation = interpret_similarity_level(similarity_level) # Similarity level yorumlanır.

        lines.append(f"{left} ↔ {right}") # Tokenizer çiftinin adı formatlanarak lines listesine eklenir.
        lines.append(f"Overlap Ratio    : {overlap}") # Overlap ratio değeri formatlanarak lines listesine eklenir.
        lines.append(f"Similarity Level : {similarity_level}") # Similarity level değeri formatlanarak lines listesine eklenir.
        lines.append(f"Interpretation   : {similarity_level_interpretation}") # Similarity level yorumlanmış hali lines listesine eklenir.
        lines.append("") # Boş bir satır eklenir, böylece her pairwise comparison sonucu arasında görsel bir ayrım sağlanır.

    return lines