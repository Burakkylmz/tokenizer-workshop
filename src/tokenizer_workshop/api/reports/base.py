"""
Ortak rapor yardımcıları.

`markdown_report.py` ve `text_report.py` arasında paylaşılan domain mantığı:
- güvenli sayısal dönüşüm
- metrik erişimi
- "en iyi" seçim heuristikleri
- tokenizer kalite skoru
- statik açıklama metinleri (kategori bazlı kullanım rehberi, trade-off'lar)

Bu modül **format-agnostik** olmalıdır. Markdown veya plain-text üretmez,
sadece veri ve string döndürür. Format katmanı (markdown/text builder)
buradan gelen değerleri kendi gösterim kuralına göre sarar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .helpers import get_metrics, safe_str


# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

# Skoru sayısal olarak garanti elenecek hâle getirmek için kullanılan sentinel.
# `tokenizer_quality_score` collapse / geçersiz girdi durumunda bu değeri döner.
DISQUALIFIED_SCORE: float = -1000.0

# Bir tokenizer'ın "valid" sayılması için minimum token sayısı.
# Bu eşiğin altındaki sonuçlar collapse olmuş kabul edilir.
MIN_VALID_TOKEN_COUNT: int = 3

# `select_best_tokenizer` içinde "balanced" seçim yaparken hedeflenen
# token sayısı. Top 3 aday içinden bu hedefe en yakın olan seçilir.
BALANCED_TOKEN_TARGET: int = 10

# Top kaç aday arasından balanced seçim yapılacağı.
BALANCED_TOP_K: int = 3


@dataclass(frozen=True)
class QualityWeights:
    """
    `tokenizer_quality_score` içinde kullanılan ağırlıklar ve eşikler.

    Skor mantığını ayarlamak için tek değiştirme noktası.
    """

    # Çekirdek sinyal ağırlıkları
    efficiency_weight: float = 3.0
    efficiency_cap: float = 5.0
    token_count_weight: float = 1.5
    token_count_cap: int = 20
    latency_penalty: float = 20.0
    unknown_penalty: float = 25.0

    # Reconstruction ödül / cezası
    reconstruction_bonus: float = 30.0
    non_reconstruction_penalty: float = 20.0
    short_token_no_reconstruct_penalty: float = 10.0
    short_token_avg_chars_threshold: float = 4.0

    # Yapısal cezalar
    very_low_token_penalty: float = 20.0   # token_count < 3
    low_token_penalty: float = 10.0        # token_count < 5
    high_token_penalty: float = 10.0       # token_count > high_token_threshold
    high_token_threshold: int = 30

    # Ortalama karakter cezası
    over_long_token_penalty: float = 15.0
    max_avg_chars: float = 15.0


DEFAULT_QUALITY_WEIGHTS = QualityWeights()


# Okunabilirlik / debugging için tercih sırası.
# `best_readable_tokenizer` bu listeyi sırayla tarayıp ilk eşleşeni döner.
READABLE_TOKENIZER_PREFERENCE: tuple[str, ...] = (
    "word",
    "regex",
    "punctuation",
    "white_space",
)


# Tokenizer adı -> kullanım önerisi.
# Hem markdown hem text raporda aynı metinleri kullanmak için tek noktada
# tutuluyor. Yeni tokenizer eklendiğinde sadece burası güncellenmeli.
TOKENIZER_GUIDANCE: dict[str, str] = {
    "char": "best for debugging and maximum granularity.",
    "byte": "best when ultra-fast byte-level tokenization is required.",
    "word": "best when readability and simple word-level segmentation matter.",
    "regex": "best for custom tokenization patterns and domain-specific text.",
    "bpe": "balanced option between compression and flexibility.",
    "simple_bpe": "educational BPE baseline for understanding merge-based compression.",
    "byte_bpe": "best for byte-level coverage with BPE-style compression.",
    "byte_level_bpe": "best for robust byte-level coverage and unseen text handling.",
    "regex_bpe": "best for pattern-aware BPE tokenization.",
    "ngram": "best for capturing local context and multi-word expressions.",
    "wordpiece": "best for fixed-vocabulary subword tokenization.",
    "unigram": "best for probabilistic subword tokenization.",
    "sentencepiece": "best for language-agnostic subword tokenization.",
    "white_space": "best for simple baseline tokenization and debugging.",
    "punctuation": "best for separating words and punctuation into distinct tokens.",
    "subword": "best for fixed-size subword tokenization.",
    "morpheme": "best for linguistically motivated subword analysis.",
    "pretrained": "best for leveraging existing production tokenizer models.",
}


# Statik trade-off açıklamaları. Format katmanı bunları bullet'a sarar.
TRADEOFF_LINES: tuple[str, ...] = (
    "Character-level tokenization provides high granularity but usually increases sequence length.",
    "Word-level tokenization is compact but language-dependent.",
    "Subword/BPE tokenization balances flexibility and compression.",
    "Byte-level tokenization ensures full coverage of any input.",
    "Regex-based tokenization allows for custom patterns and domain-specific text handling.",
    (
        "N-gram tokenization captures local context and multi-word expressions, which can be "
        "beneficial for certain languages and tasks, but may increase token count compared to "
        "word-level tokenization."
    ),
    (
        "WordPiece tokenization is effective for subword tokenization with a fixed vocabulary, "
        "commonly used in transformer models, but may require careful handling of unknown tokens "
        "and vocabulary management."
    ),
    (
        "Unigram tokenization is effective for probabilistic subword tokenization with a fixed "
        "vocabulary, balancing flexibility and compression."
    ),
    (
        "SentencePiece tokenization is effective for flexible subword tokenization with a fixed "
        "vocabulary, balancing flexibility and compression."
    ),
    (
        "Whitespace tokenization is a simple baseline that can be useful for debugging and "
        "educational purposes, but it may not provide the best performance or compression for "
        "most real-world applications."
    ),
    (
        "Punctuation tokenization is effective for separating words and punctuation into distinct "
        "tokens, which can improve readability and downstream processing. However, it may "
        "increase token count and latency compared to simpler tokenization strategies."
    ),
    "Subword tokenization is effective for fixed-size subword tokenization, balancing flexibility and compression.",
    (
        "Morpheme tokenization is effective for linguistically motivated subword tokenization, "
        "capturing meaningful units of language. However, it may require language-specific "
        "resources and may not always align with the needs of downstream models."
    ),
    "Byte-level BPE tokenization is effective for handling complex or unseen text, ensuring full coverage of any input.",
    (
        "Pretrained tokenization is effective for leveraging existing tokenization models, "
        "providing strong performance on many tasks. However, it may require additional "
        "dependencies and may not be customizable for specific domains or languages."
    ),
)


TRADEOFF_CLOSING = (
    "Ultimately, the best tokenizer choice depends on whether the target priority is "
    "speed, compression, interpretability, semantic fidelity, or robustness."
)


# ---------------------------------------------------------------------------
# Düşük seviye yardımcılar
# ---------------------------------------------------------------------------

def safe_float(value: Any, fallback: float = 0.0) -> float:
    """
    Verilen değeri güvenli bir şekilde float'a dönüştürür.
    Dönüştürülemezse `fallback` değerini döner.

    Skor / sıralama fonksiyonlarında `None`, eksik anahtar veya beklenmeyen
    string değerlere karşı koruma sağlar.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def metric(item: Mapping[str, Any], key: str, fallback: Any = 0) -> Any:
    """
    Item sözlüğünden bir metriği güvenli şekilde okur.

    Önce `metrics` alt sözlüğüne, orada yoksa top-level alanlara bakar.
    Pipeline farklı versiyonlarda metrikleri farklı seviyelerde tutabildiği
    için bu çift kademeli erişim kasıtlıdır.
    """
    metrics_dict = get_metrics(item)
    if key in metrics_dict:
        return metrics_dict[key]
    return item.get(key, fallback)


# ---------------------------------------------------------------------------
# Skor & seçim
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ScoreInputs:
    """Skor hesaplama için ham (sayısal) sinyaller."""
    token_count: float
    avg_chars: float
    efficiency: float
    unknown_rate: float
    latency: float
    reconstruct: bool


def _extract_score_inputs(item: Mapping[str, Any]) -> _ScoreInputs:
    """Item'dan skor sinyallerini tek seferde çıkarır."""
    metrics = get_metrics(item)
    return _ScoreInputs(
        token_count=safe_float(metric(item, "token_count")),
        avg_chars=safe_float(metrics.get("avg_chars_per_token")),
        efficiency=safe_float(metrics.get("efficiency_score")),
        unknown_rate=safe_float(metrics.get("unknown_rate")),
        latency=safe_float(metrics.get("latency_seconds")),
        reconstruct=bool(metrics.get("reconstruction_match")),
    )


def _core_signal_score(inp: _ScoreInputs, w: QualityWeights) -> float:
    """Çekirdek sinyaller: efficiency, sequence length, latency, unknown."""
    score = 0.0
    score += min(inp.efficiency, w.efficiency_cap) * w.efficiency_weight
    score += min(inp.token_count, w.token_count_cap) * w.token_count_weight
    score -= inp.latency * w.latency_penalty
    score -= inp.unknown_rate * w.unknown_penalty
    return score


def _reconstruction_score(inp: _ScoreInputs, w: QualityWeights) -> float:
    """Reconstruction ödül / cezası."""
    score = 0.0
    if inp.reconstruct:
        score += w.reconstruction_bonus
    else:
        score -= w.non_reconstruction_penalty
        # Hem reconstruct edemiyor hem de avg_chars çok düşükse extra ceza:
        # bu kombinasyon "çöp segmentation" sinyalidir.
        if inp.avg_chars < w.short_token_no_reconstruct_penalty:
            score -= w.short_token_no_reconstruct_penalty
    return score


def _structural_penalty(inp: _ScoreInputs, w: QualityWeights) -> float:
    """Token sayısı ve avg_chars üzerinden yapısal cezalar."""
    penalty = 0.0
    if inp.token_count < 3:
        penalty += w.very_low_token_penalty
    elif inp.token_count < 5:
        penalty += w.low_token_penalty

    if inp.token_count > w.high_token_threshold:
        penalty += w.high_token_penalty

    if inp.avg_chars > w.max_avg_chars:
        penalty += w.over_long_token_penalty

    return -penalty


def tokenizer_quality_score(
    item: Mapping[str, Any],
    weights: QualityWeights = DEFAULT_QUALITY_WEIGHTS,
) -> float:
    """
    Tokenizer için bileşik kalite skoru hesaplar.

    Skor üç parçadan oluşur:
        1. Çekirdek sinyaller (efficiency, length, latency, unknown)
        2. Reconstruction ödül / cezası
        3. Yapısal cezalar (collapse, over-segmentation, çok uzun token)

    Token sayısı `MIN_VALID_TOKEN_COUNT`'tan azsa tokenizer collapse olmuş
    kabul edilir ve `DISQUALIFIED_SCORE` döner.
    """
    inp = _extract_score_inputs(item)

    # Collapse: tokenizer girdiyi anlamlı şekilde parçalayamamış.
    if inp.token_count < MIN_VALID_TOKEN_COUNT:
        return DISQUALIFIED_SCORE

    return (
        _core_signal_score(inp, weights)
        + _reconstruction_score(inp, weights)
        + _structural_penalty(inp, weights)
    )


# ---------------------------------------------------------------------------
# Listeden seçim heuristikleri
# ---------------------------------------------------------------------------

def best_by_metric(
    results: list[dict[str, Any]],
    metric_key: str,
    *,
    reverse: bool = True,
) -> dict[str, Any] | None:
    """
    Verilen metriğe göre en iyi sonucu döner.

    `reverse=True`  -> en yüksek değer (efficiency, compression, vs.)
    `reverse=False` -> en düşük değer (latency, token_count, vs.)
    Liste boşsa `None`.
    """
    if not results:
        return None

    return max(
        results,
        key=lambda item: (
            safe_float(metric(item, metric_key))
            if reverse
            else -safe_float(metric(item, metric_key))
        ),
    )


def compression_gain_percent(item: Mapping[str, Any], source_text: str) -> float:
    """
    Sıkıştırma kazancını yüzde olarak hesaplar:

        (1 - token_count / source_length) * 100

    Boş kaynak metin veya geçersiz token sayısı durumunda 0.0 döner.
    """
    source_length = len(source_text)
    if source_length == 0:
        return 0.0

    token_count = safe_float(metric(item, "token_count"))
    if token_count <= 0:
        return 0.0

    return (1 - token_count / source_length) * 100


def _is_valid_result(item: Mapping[str, Any]) -> bool:
    """Collapse olmamış (kullanılabilir) sonuçları filtreler."""
    return safe_float(metric(item, "token_count")) >= MIN_VALID_TOKEN_COUNT


def select_best_tokenizer(
    results: list[dict[str, Any]],
    *,
    target_token_count: int = BALANCED_TOKEN_TARGET,
    top_k: int = BALANCED_TOP_K,
) -> dict[str, Any] | None:
    """
    Genel dengeye göre önerilecek tokenizer'ı seçer.

    Önce `tokenizer_quality_score`'a göre sıralanır, ardından top-K aday
    içinden `target_token_count`'a en yakın olan döner. Bu, tek başına
    skor maksimizasyonunun yarattığı "en yüksek skor en uzun çıktı"
    tarzı yan etkileri dengeler.

    Geçersiz (collapse olmuş) sonuçlar elenmeden önce filtrelenir.
    Hiç geçerli sonuç kalmazsa `None`.
    """
    valid = [item for item in results if _is_valid_result(item)]
    if not valid:
        return None

    scored = sorted(valid, key=tokenizer_quality_score, reverse=True)
    top_candidates = scored[:top_k]

    return min(
        top_candidates,
        key=lambda item: abs(
            safe_float(metric(item, "token_count")) - target_token_count
        ),
    )


def best_readable_tokenizer(
    results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Okunabilirlik / debugging için en uygun tokenizer'ı seçer.

    `READABLE_TOKENIZER_PREFERENCE` sırasına göre ilk bulduğunu döner.
    Hiçbiri eşleşmezse `None`.
    """
    by_name = {
        safe_str(item.get("tokenizer_name")).lower(): item
        for item in results
    }

    for name in READABLE_TOKENIZER_PREFERENCE:
        match = by_name.get(name)
        if match is not None:
            return match
    return None


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Winners:
    """
    Bir sonuç listesi için tüm "en iyi" hesaplamalarının bir kerede
    yapılmış hâli.

    Her rapor bölümü bu nesneyi alır ve içinden ilgili alanları kullanır.
    Aksi takdirde her bölüm aynı `best_by_metric` çağrılarını tekrar tekrar
    yapardı.

    `best_balance` collapse olmamış sonuçlar üzerinden seçilir; diğer
    alanlar ham metrik tabanlı olduğu için tüm sonuçları görür.
    """
    best_balance: dict[str, Any] | None
    best_efficiency: dict[str, Any] | None
    best_compression: dict[str, Any] | None
    fastest: dict[str, Any] | None
    lowest_token: dict[str, Any] | None
    highest_token: dict[str, Any] | None
    highest_unique: dict[str, Any] | None
    lowest_unknown: dict[str, Any] | None
    most_readable: dict[str, Any] | None


def compute_winners(results: list[dict[str, Any]]) -> Winners:
    """
    Tüm rapor bölümlerinin ihtiyaç duyduğu "en iyi" tokenizer seçimlerini
    tek seferde hesaplar.
 
    `best_efficiency`, `best_compression` ve `lowest_token` collapse olmuş
    (`token_count < MIN_VALID_TOKEN_COUNT`) sonuçlardan korunur. Bu metrikler
    collapse durumunda suni olarak şişer (örn. 1 token tüm metni "compress"
    ediyor gibi görünür) ve kullanıcıya yanıltıcı sinyal verir.
 
    Latency, unique count ve unknown rate gibi diğer metrikler collapse'tan
    etkilenmediği için tüm sonuçlar üzerinden hesaplanır.
    """
    valid_results = [item for item in results if _is_valid_result(item)]

    return Winners(
        best_balance=select_best_tokenizer(results),
        best_efficiency=best_by_metric(valid_results, "efficiency_score", reverse=True),
        best_compression=best_by_metric(valid_results, "compression_ratio", reverse=True),
        fastest=best_by_metric(results, "latency_seconds", reverse=False),
        lowest_token=best_by_metric(valid_results, "token_count", reverse=False),
        highest_token=best_by_metric(results, "token_count", reverse=True),
        highest_unique=best_by_metric(results, "unique_token_count", reverse=True),
        lowest_unknown=best_by_metric(results, "unknown_rate", reverse=False),
        most_readable=best_readable_tokenizer(results),
    )