"""
metrics_service.py

Tokenizer çıktıları üzerinden metrik ve pairwise comparison üreten servisler.

Bu modül, tokenizer çıktılarını yalnızca token listesi olarak bırakmaz;
onları ölçülebilir, karşılaştırılabilir ve raporlanabilir metriklere dönüştürür.

Temel sorumluluklar:
    - Tek tokenizer çıktısı için istatistiksel metrik hesaplamak
    - Token frekanslarını çıkarmak
    - Token uzunluğu dağılımını üretmek
    - Unknown token oranını hesaplamak
    - Latency ve token başına maliyet metriklerini üretmek
    - Tokenizer çiftleri arasında overlap analizi yapmak
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any


def calculate_metrics(
    tokens: list[str],
    latency_seconds: float,
    source_text: str,
) -> dict[str, Any]:
    """
    Tek bir tokenizer çıktısı üzerinden detaylı metrikler hesaplar.

    Bu fonksiyon, tokenizer'ın aynı input metni nasıl temsil ettiğini sayısal
    olarak analiz eder. Çıktı, API response, Markdown/TXT/PDF raporları ve UI
    tabloları tarafından doğrudan kullanılabilir.

    Hesaplanan metrik grupları:
        1. Sayım metrikleri:
            - token_count
            - unique_token_count
            - unique_ratio

        2. Uzunluk metrikleri:
            - average_token_length
            - min_token_length
            - max_token_length
            - avg_chars_per_token

        3. Unknown token metrikleri:
            - unknown_count
            - unknown_rate

        4. Performans metrikleri:
            - latency_seconds
            - latency_per_token

        5. Verimlilik metrikleri:
            - efficiency_score
            - compression_ratio

        6. Analiz çıktıları:
            - top_tokens
            - token_length_distribution
            - reconstructed_text
            - reconstruction_match

    Not:
        reconstructed_text burada basit şekilde " ".join(tokens) ile üretilir.
        Bu her tokenizer için gerçek decode davranışını temsil etmez.
        Özellikle BPE, byte-level BPE, SentencePiece veya pretrained tokenizer'larda
        gerçek reconstruction için tokenizer.decode(...) kullanılmalıdır.

    Args:
        tokens:
            Tokenizer tarafından üretilmiş normalize token listesi.

        latency_seconds:
            tokenize() işleminin saniye cinsinden ölçülen süresi.

        source_text:
            Tokenize edilen orijinal metin.

    Returns:
        Detaylı tokenizer metriklerini içeren dict.
    """
    token_count = len(tokens)
    unique_token_count = len(set(tokens))

    # unique_ratio, token_count sıfırsa 0.0 olarak atanır, 
    # aksi halde unique_token_count / token_count olarak hesaplanır.
    unique_ratio = (unique_token_count / token_count) if token_count > 0 else 0.0

    token_lengths = [len(token) for token in tokens]

    # average_token_length, min_token_length, max_token_length token_lengths listesine göre hesaplanır.
    average_token_length = (
        sum(token_lengths) / token_count if token_count > 0 else 0.0
    )
    
    min_token_length = min(token_lengths) if token_lengths else 0
    max_token_length = max(token_lengths) if token_lengths else 0

    # avg_chars_per_token, unknown_count, unknown_rate, efficiency_score, compression_ratio, latency_per_token gibi diğer metrikler 
    # token_count sıfır durumuna göre güvenli şekilde hesaplanır.
    avg_chars_per_token = (
        len(source_text) / token_count if token_count > 0 else 0.0
    )

    unknown_count = sum(1 for token in tokens if token in {"[UNK]", "<unk>", "UNK", "<UNK>", "[unk]"})
    unknown_rate = (unknown_count / token_count) if token_count > 0 else 0.0

    # Unknown token oranı yüksekse verimlilik skorunu cezalandırıyoruz.
    # Böylece sadece kısa token dizisi üretmek değil, anlamlı temsil üretmek de skora yansır.
    efficiency_score = (
        avg_chars_per_token * (1 - unknown_rate)
        if token_count > 0
        else 0.0
    )

    compression_ratio = (
        len(source_text) / token_count if token_count > 0 else 0.0
    )

    latency_per_token = (
        latency_seconds / token_count if token_count > 0 else 0.0
    )

    top_tokens = Counter(tokens).most_common(5)

    token_length_distribution: dict[str, int] = {}
    for length in token_lengths:
        key = str(length)
        # token_length_distribution sözlüğünde, her token uzunluğu için kaç token olduğunu sayar.
        token_length_distribution[key] = token_length_distribution.get(key, 0) + 1

    # reconstruction_match metrikleri, tokenizer'ın ürettiği token listesinin orijinal metni ne kadar iyi temsil ettiğini analiz eder.
    reconstructed_text = " ".join(tokens) if tokens else ""
    
    # reconstructed_text ile source_text'in tam olarak eşleşip eşleşmediğini boolean olarak gösterir.
    reconstruction_match = (
        reconstructed_text == source_text
        if reconstructed_text
        else False
    )

    return {
        "token_count": token_count,
        "unique_token_count": unique_token_count,
        "unique_ratio": unique_ratio,
        "average_token_length": average_token_length,
        "min_token_length": min_token_length,
        "max_token_length": max_token_length,
        "avg_chars_per_token": avg_chars_per_token,
        "unknown_count": unknown_count,
        "unknown_rate": unknown_rate,
        "latency_seconds": latency_seconds,
        "latency_per_token": latency_per_token,
        "efficiency_score": efficiency_score,
        "compression_ratio": compression_ratio,
        "top_tokens": [
            {"token": token, "count": count}
            for token, count in top_tokens
        ],
        "token_length_distribution": token_length_distribution,
        "reconstructed_text": reconstructed_text,
        "reconstruction_match": reconstruction_match,
    }


def classify_similarity(overlap_ratio: float) -> str:
    """
    Pairwise overlap oranını insan tarafından okunabilir similarity seviyesine çevirir.

    Bu fonksiyon, tokenizer çiftleri arasındaki token overlap oranını yorumlamak için kullanılır.
    Overlap ratio, iki tokenizer'ın token setleri arasındaki benzerliği sayısal olarak ifade eder.
    Bu sayısal oranı, "Completely Different", "Highly Similar" gibi kategorilere çevirerek 
    kullanıcıların sonuçları daha hızlı yorumlamasını sağlar.

    Bu alan özellikle rapor üretiminde faydalıdır. Kullanıcı yalnızca 0.37 gibi
    sayısal bir oran görmek yerine, tokenizer çiftlerinin ne kadar benzer olduğunu
    daha hızlı yorumlayabilir.

    Args:
        overlap_ratio:
            İki tokenizer arasındaki Jaccard-style token overlap oranı.

    Returns:
        Benzerlik seviyesi.
    """
    if overlap_ratio == 0:
        return "Completely Different"

    if overlap_ratio < 0.25:
        return "Highly Different"

    if overlap_ratio < 0.60:
        return "Moderately Similar"

    if overlap_ratio < 0.85:
        return "Highly Similar"

    return "Nearly Identical"


def build_pairwise_comparisons(
    evaluations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Tokenizer sonuçları arasında pairwise token overlap analizi üretir.

    Bu fonksiyon, compare_tokenizers() fonksiyonunun bir parçası olarak çalışır ve 
    her tokenizer çiftini karşılaştırarak benzerlik ve farkları analiz eder.

    Bu fonksiyon her tokenizer çiftini karşılaştırır ve şu bilgileri çıkarır:
        - Ortak tokenlar
        - Sadece sol tokenizer'da bulunan tokenlar
        - Sadece sağ tokenizer'da bulunan tokenlar
        - Overlap ratio
        - Semantic similarity level

    Overlap ratio:
        common_tokens / union_tokens

    Bu Jaccard-style oran, iki tokenizer'ın token setleri açısından ne kadar
    benzer davrandığını gösterir.

    Args:
        evaluations:
            evaluate_tokenizers() çıktısındaki evaluations listesi.

    Returns:
        Tokenizer çiftleri için pairwise comparison listesi.
    """
    pairwise_results: list[dict[str, Any]] = []

    for left, right in combinations(evaluations, 2):
        left_tokens = set(left["tokens"])
        right_tokens = set(right["tokens"])

        common_tokens = sorted(left_tokens & right_tokens)
        unique_to_left = sorted(left_tokens - right_tokens)
        unique_to_right = sorted(right_tokens - left_tokens)

        union_count = len(left_tokens | right_tokens)
        overlap_ratio = len(common_tokens) / union_count if union_count > 0 else 0.0

        pairwise_results.append(
            {
                "left_name": left["tokenizer_name"],
                "right_name": right["tokenizer_name"],
                "common_tokens": common_tokens,
                "unique_to_left": unique_to_left,
                "unique_to_right": unique_to_right,
                "overlap_ratio": overlap_ratio,
                "similarity_level": classify_similarity(overlap_ratio),
            }
        )

    return pairwise_results