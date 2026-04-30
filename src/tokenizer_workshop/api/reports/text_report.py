"""
Plain-text rapor oluşturucu.

Tokenizer karşılaştırma sonuçlarını terminal ve düz metin gösterimi için
optimize edilmiş bir rapora dönüştürür. Domain mantığı `_report_common.py`'da
paylaşılır; bu modül yalnızca plain-text formatlamadan sorumludur.
"""

from __future__ import annotations

from typing import Any

from .base import (
    TOKENIZER_GUIDANCE,
    TRADEOFF_CLOSING,
    TRADEOFF_LINES,
    Winners,
    compression_gain_percent,
    compute_winners,
    metric as _metric,
    safe_float as _safe_float,
    tokenizer_quality_score,
)
from .helpers import (
    append_section_title,
    extract_compare_payload,
    format_number,
    format_reconstruction,
    format_top_tokens,
    get_metrics,
    hr,
    latency_microseconds,
    safe_str,
    truncate_list,
    utc_now_iso,
    wide_hr,
)


REPORT_TITLE_WIDTH = 120


# ---------------------------------------------------------------------------
# Plain-text spesifik küçük yardımcılar
# ---------------------------------------------------------------------------

def _name(item: dict[str, Any] | None) -> str:
    """Tokenizer adını güvenli şekilde okur."""
    if item is None:
        return ""
    return safe_str(item.get("tokenizer_name"))


def _similarity_level(overlap_ratio: Any) -> str:
    """
    Overlap oranını niteliksel bir seviyeye dönüştürür.

    Bu fonksiyon plain-text'e özgü kalır; markdown raporu pipeline tarafından
    hesaplanan `similarity_level` alanını doğrudan kullanır.
    """
    ratio = _safe_float(overlap_ratio)

    if ratio == 0:
        return "Completely Different"
    if ratio < 0.25:
        return "Highly Different"
    if ratio < 0.60:
        return "Moderately Similar"
    return "Highly Similar"


_PAIRWISE_OBSERVATIONS = {
    "Completely Different": (
        "No shared tokens were found, indicating completely different tokenization strategies."
    ),
    "Highly Different": (
        "Minimal overlap exists; tokenization strategies differ significantly."
    ),
    "Moderately Similar": (
        "Moderate overlap suggests partial similarity in segmentation."
    ),
    "Highly Similar": (
        "High overlap indicates similar tokenization behavior."
    ),
}


def _pairwise_observation(overlap_ratio: Any) -> str:
    """Overlap seviyesine göre kısa gözlem cümlesi döner."""
    return _PAIRWISE_OBSERVATIONS[_similarity_level(overlap_ratio)]


# ---------------------------------------------------------------------------
# Bölümler
# ---------------------------------------------------------------------------

def _append_header(lines: list[str], total: int) -> None:
    """Rapor başlığını ve genel meta bilgileri ekler."""
    lines.extend(
        [
            wide_hr("="),
            "TOKENIZER EVALUATION REPORT".center(REPORT_TITLE_WIDTH),
            wide_hr("="),
            f"Generated At (UTC): {utc_now_iso()}",
            f"Total Tokenizers   : {total}",
            "",
        ]
    )


def _append_source_text(lines: list[str], text: str) -> None:
    """Kaynak metni rapora ekler."""
    append_section_title(lines, "SOURCE TEXT")
    lines.append(text if text else "No source text provided.")
    lines.append("")


def _append_overview(lines: list[str], total: int) -> None:
    """Raporun genel bakış bölümünü ekler."""
    lines.extend(
        [
            "OVERVIEW",
            wide_hr("-"),
            (
                f"This report evaluates {total} tokenizer(s) on the same input text. "
                "It compares token count, vocabulary diversity, segmentation granularity, latency, "
                "compression behavior, reconstruction quality, and pairwise token overlap."
            ),
            "",
        ]
    )


def _append_executive_summary(
    lines: list[str],
    results: list[dict[str, Any]],
    winners: Winners,
) -> None:
    """Yürütücü özet bölümünü ekler."""
    lines.extend(["EXECUTIVE SUMMARY", wide_hr("-")])

    if not results:
        lines.extend(["No executive summary available.", ""])
        return

    best_overall = max(
        results,
        key=tokenizer_quality_score,
        default=None,
    )

    bullets: list[tuple[dict[str, Any] | None, str]] = [
        (best_overall, "Best overall tokenizer    "),
        (winners.fastest, "Fastest tokenizer         "),
        (winners.lowest_token, "Shortest sequence         "),
        (winners.highest_token, "Most granular tokenizer   "),
    ]

    for item, label in bullets:
        if item:
            lines.append(f"• {label}: {_name(item)}")

    lines.extend(
        [
            "",
            (
                "Tokenizer selection should depend on the target use case: speed, compression, "
                "readability, interpretability, or robustness across diverse input types."
            ),
            "",
        ]
    )


def _append_summary_table(
    lines: list[str],
    results: list[dict[str, Any]],
    source_text: str,
) -> None:
    """Sabit genişlikli özet tabloyu ekler."""
    lines.extend(
        [
            "SUMMARY TABLE",
            wide_hr("-"),
            (
                "Tokenizer | Tokens | Unique | Uniq Ratio | Avg Len | Min | Max | "
                "Chars/Token | Unknown | Latency µs | Eff. Score | Comp. | Gain %"
            ),
            (
                "----------+--------+--------+------------+---------+-----+-----+"
                "-------------+---------+------------+------------+-------+--------"
            ),
        ]
    )

    if not results:
        lines.extend(["No summary data available.", ""])
        return

    for item in results:
        metrics = get_metrics(item)
        gain = compression_gain_percent(item, source_text)

        lines.append(
            f"{_name(item):<9} | "
            f"{_metric(item, 'token_count'):<6} | "
            f"{_metric(item, 'unique_token_count', item.get('vocab_size', 0)):<6} | "
            f"{format_number(metrics.get('unique_ratio'), 2):<10} | "
            f"{format_number(metrics.get('average_token_length'), 2):<7} | "
            f"{metrics.get('min_token_length', '-'):<3} | "
            f"{metrics.get('max_token_length', '-'):<3} | "
            f"{format_number(metrics.get('avg_chars_per_token'), 2):<11} | "
            f"{format_number(metrics.get('unknown_rate'), 2):<7} | "
            f"{latency_microseconds(metrics):<10} | "
            f"{format_number(metrics.get('efficiency_score'), 2):<10} | "
            f"{format_number(metrics.get('compression_ratio'), 2):<5} | "
            f"{format_number(gain, 2)}"
        )

    lines.append("")


def _append_key_insights(lines: list[str], winners: Winners) -> None:
    """Anahtar bulguları rapora ekler."""
    lines.extend(["KEY INSIGHTS", wide_hr("-")])

    if winners.best_efficiency is None:
        lines.extend(["No insights available.", ""])
        return

    insights: list[tuple[dict[str, Any] | None, str, str]] = [
        (
            winners.lowest_token, "Lowest token count        ",
            f"({_metric(winners.lowest_token or {}, 'token_count')})",
        ),
        (
            winners.highest_token, "Highest token count       ",
            f"({_metric(winners.highest_token or {}, 'token_count')})",
        ),
        (
            winners.best_efficiency, "Best efficiency score     ",
            f"({format_number(_metric(winners.best_efficiency or {}, 'efficiency_score'), 2)})",
        ),
        (
            winners.highest_unique, "Highest unique token count",
            f"({_metric(winners.highest_unique or {}, 'unique_token_count')})",
        ),
        (
            winners.fastest, "Fastest tokenizer         ",
            f"({latency_microseconds(get_metrics(winners.fastest or {}))} µs)",
        ),
        (
            winners.best_compression, "Best compression ratio    ",
            f"({format_number(_metric(winners.best_compression or {}, 'compression_ratio'), 2)})",
        ),
    ]

    for item, label, suffix in insights:
        if item:
            lines.append(f"• {label}: {_name(item)} {suffix}")

    lines.append("")


def _append_interpretation(lines: list[str], winners: Winners) -> None:
    """Bulguların kısa yorumunu ekler."""
    lines.extend(["INTERPRETATION", wide_hr("-")])

    if winners.best_efficiency is None:
        lines.extend(["No interpretation available.", ""])
        return

    if winners.lowest_token:
        lines.append(
            f"The '{_name(winners.lowest_token)}' tokenizer produces the most compact "
            f"segmentation with {_metric(winners.lowest_token, 'token_count')} tokens."
        )

    if winners.highest_token:
        lines.append(
            f"The '{_name(winners.highest_token)}' tokenizer produces the most granular "
            f"segmentation with {_metric(winners.highest_token, 'token_count')} tokens."
        )

    if winners.best_efficiency:
        eff = format_number(_metric(winners.best_efficiency, "efficiency_score"), 2)
        lines.append(
            f"The '{_name(winners.best_efficiency)}' tokenizer achieves the strongest "
            f"efficiency score ({eff}), which indicates better compression behavior per token."
        )

    if winners.fastest:
        lat = latency_microseconds(get_metrics(winners.fastest))
        lines.append(
            f"The fastest tokenizer is '{_name(winners.fastest)}' with {lat} µs latency."
        )

    lines.extend(
        [
            (
                "Overall, tokenizer choice directly affects sequence length, processing cost, "
                "semantic granularity, compression behavior, and downstream model efficiency."
            ),
            "",
        ]
    )


def _append_recommendation(
    lines: list[str],
    results: list[dict[str, Any]],
    winners: Winners,
) -> None:
    """
    Bölümlü öneri:
        1. Genel kazananlardan kategori öneriler
        2. Mevcut tokenizer'lar için statik kullanım rehberi
        3. Trade-off açıklamaları
    """
    lines.extend(["RECOMMENDATION", wide_hr("-")])

    if not results:
        lines.extend(["No recommendation available.", ""])
        return

    lines.extend(["When to use each tokenizer:", ""])

    added: set[str] = set()

    def add(name: str, text: str) -> None:
        key = name.lower()
        if key and key not in added:
            lines.append(f"• {name:<14}: {text}")
            added.add(key)

    if winners.best_efficiency:
        add(_name(winners.best_efficiency),
            "best when compression and token efficiency matter.")
    if winners.fastest:
        add(_name(winners.fastest),
            "best when low-latency tokenization matters.")
    if winners.lowest_token:
        add(_name(winners.lowest_token),
            "best when minimizing total token count matters.")

    for item in results:
        name = _name(item)
        guidance = TOKENIZER_GUIDANCE.get(name.lower())
        if guidance:
            add(name, guidance)

    lines.extend(["", "Trade-offs:"])
    lines.extend(f"• {line}" for line in TRADEOFF_LINES)
    lines.extend(["", TRADEOFF_CLOSING, ""])


def _append_tokenizer_details(
    lines: list[str],
    results: list[dict[str, Any]],
    source_text: str,
) -> None:
    """Her tokenizer için detaylı bölüm ekler."""
    lines.extend(["TOKENIZER DETAILS", wide_hr("-"), ""])

    if not results:
        lines.extend(["No tokenizer details available.", ""])
        return

    for index, item in enumerate(results, start=1):
        tokenizer_name = _name(item)
        metrics = get_metrics(item)
        tokens = item.get("tokens", [])

        lines.extend(
            [
                f"[{index}] {tokenizer_name}",
                f"Token Count          : {_metric(item, 'token_count')}",
                f"Vocab Size           : {_metric(item, 'unique_token_count', item.get('vocab_size', 0))}",
                "Token Preview:",
                f"  {truncate_list(tokens)}",
                f"Unique Token Count   : {_metric(item, 'unique_token_count', '-')}",
                f"Unique Ratio         : {format_number(metrics.get('unique_ratio'), 2)}",
                f"Average Token Length : {format_number(metrics.get('average_token_length'), 2)}",
                f"Min Token Length     : {metrics.get('min_token_length', '-')}",
                f"Max Token Length     : {metrics.get('max_token_length', '-')}",
                f"Avg Chars / Token    : {format_number(metrics.get('avg_chars_per_token'), 2)}",
                f"Unknown Count        : {metrics.get('unknown_count', '-')}",
                f"Unknown Rate         : {format_number(metrics.get('unknown_rate'), 2)}",
                (
                    f"Latency              : {format_number(metrics.get('latency_seconds'), 6)}s "
                    f"({latency_microseconds(metrics)} µs)"
                ),
                f"Latency / Token      : {format_number(metrics.get('latency_per_token'), 6)}",
                f"Efficiency Score     : {format_number(metrics.get('efficiency_score'), 2)}",
                f"Compression Ratio    : {format_number(metrics.get('compression_ratio'), 2)}",
                "",
                "Top Tokens:",
            ]
        )

        top_tokens = format_top_tokens(metrics.get("top_tokens"))
        if top_tokens:
            lines.extend(top_tokens)
        else:
            lines.append("  No top token data available.")

        lines.extend(["", "Token Length Distribution:"])

        distribution = metrics.get("token_length_distribution")
        if isinstance(distribution, dict) and distribution:
            for length, count in sorted(
                distribution.items(),
                key=lambda pair: _safe_float(pair[0]),
            ):
                lines.append(f"  Length {length}: {count} tokens")
        else:
            lines.append("  No token length distribution available.")

        lines.extend(["", "Reconstruction:"])

        reconstruction = format_reconstruction(metrics, original_text=source_text)
        if reconstruction:
            lines.extend(reconstruction)
        else:
            lines.append("  No reconstruction details available.")

        lines.extend(["", hr("-")])


def _append_winner_explanation(
    lines: list[str],
    results: list[dict[str, Any]],
) -> None:
    """En yüksek skorlu tokenizer'ın neden kazandığını açıklar."""
    lines.extend(["WHY THIS TOKENIZER WON", wide_hr("-")])

    if not results:
        lines.extend(["No winner explanation available.", ""])
        return

    winner = max(results, key=tokenizer_quality_score)
    metrics = get_metrics(winner)

    name = _name(winner)
    score = tokenizer_quality_score(winner)
    token_count = _metric(winner, "token_count")
    efficiency = _safe_float(metrics.get("efficiency_score"))
    unknown_rate = _safe_float(metrics.get("unknown_rate"))
    latency = latency_microseconds(metrics)
    reconstruct = bool(metrics.get("reconstruction_match"))
    avg_chars = _safe_float(metrics.get("avg_chars_per_token"))

    lines.extend(
        [
            f"Winner: {name}",
            f"Composite Score: {format_number(score, 2)}",
            "",
            "Why it ranked first:",
        ]
    )

    if reconstruct:
        lines.append("• It can reconstruct the original text, which strongly improves usability.")
    else:
        lines.append("• It cannot reconstruct the original text, which limits production usability.")

    if unknown_rate == 0:
        lines.append("• It produced no unknown tokens.")
    else:
        lines.append(f"• It has an unknown-token rate of {format_number(unknown_rate, 2)}.")

    lines.append(f"• It produced {token_count} tokens, which affects sequence length and cost.")
    lines.append(f"• Its efficiency score is {format_number(efficiency, 2)}.")
    lines.append(f"• Its latency is {latency} µs.")

    if avg_chars > 15:
        lines.append("• Warning: average characters per token is very high, which may indicate over-compression.")

    if token_count <= 2:
        lines.append("• Warning: token count is extremely low, which indicates a collapsed tokenization result.")

    lines.extend(
        [
            "",
            (
                "In short, this tokenizer won because it achieved the strongest balance "
                "between scoring signals such as reconstruction, token count, efficiency, "
                "latency, unknown-token rate, and structural penalties."
            ),
            "",
        ]
    )


def _append_ranking(lines: list[str], results: list[dict[str, Any]]) -> None:
    """
    Bileşik kalite skoruna göre genel sıralama bölümünü ekler.

    Markdown raporuyla tutarlılık için aynı `tokenizer_quality_score`
    kullanılır.
    """
    lines.extend(["OVERALL RANKING", wide_hr("-")])

    if not results:
        lines.extend(["No ranking available.", ""])
        return

    ranking = sorted(results, key=tokenizer_quality_score, reverse=True)

    lines.extend(
        [
            (
                "Ranking is based on a composite quality score including efficiency, "
                "latency, unknown rate, reconstruction quality, and structural penalties."
            ),
            "",
        ]
    )

    for index, item in enumerate(ranking, start=1):
        metrics = get_metrics(item)
        score = format_number(tokenizer_quality_score(item), 2)
        eff = format_number(metrics.get("efficiency_score"), 2)
        lat = latency_microseconds(metrics)

        lines.append(
            f"{index}. {_name(item)} (score={score}, eff={eff}, latency={lat} µs)"
        )

    lines.append("")


def _append_pairwise_comparisons(
    lines: list[str],
    pairwise: list[dict[str, Any]],
) -> None:
    """Pairwise tokenizer comparison sonuçlarını ekler."""
    lines.extend(["PAIRWISE COMPARISONS", wide_hr("-")])

    if not pairwise:
        lines.extend(["No pairwise comparison data available.", ""])
        return

    for item in pairwise:
        left_name = safe_str(item.get("left_name"))
        right_name = safe_str(item.get("right_name"))
        ratio = item.get("overlap_ratio")

        lines.extend(
            [
                f"{left_name} ↔ {right_name}",
                f"Overlap Ratio             : {format_number(ratio, 2)}",
                f"Semantic Difference Level : {_similarity_level(ratio)}",
                "Observation:",
                f"  {_pairwise_observation(ratio)}",
                "",
                f"Common Tokens             : {truncate_list(item.get('common_tokens', []))}",
                f"Only In {left_name:<16}: {truncate_list(item.get('unique_to_left', []))}",
                f"Only In {right_name:<16}: {truncate_list(item.get('unique_to_right', []))}",
                hr("-"),
            ]
        )

    lines.append("")


def _append_categorical_recommendation(
    lines: list[str],
    winners: Winners,
) -> None:
    """
    Kategori bazlı öneri.

    Tek bir tokenizer'ı mutlak en iyi ilan etmek yerine kullanım senaryosuna
    göre kategori bazlı öneri üretir. `Winners` üzerinden çalışır, böylece
    aynı seçimler diğer raporlarla tutarlı kalır.
    """
    lines.extend(["CATEGORICAL RECOMMENDATION", wide_hr("-")])

    if winners.best_balance is None:
        lines.extend(["No tokenizer recommendation available.", ""])
        return

    lines.extend(
        [
            (
                "Tokenizer choice depends on the target use case. "
                "For this input, the recommended options are:"
            ),
            "",
        ]
    )

    balanced = winners.best_balance
    balanced_metrics = get_metrics(balanced)

    rows: list[str] = [
        (
            f"• Best balanced choice          : {_name(balanced)} "
            f"(efficiency={format_number(balanced_metrics.get('efficiency_score'), 2)}, "
            f"latency={latency_microseconds(balanced_metrics)} µs)"
        ),
    ]

    if winners.fastest:
        rows.append(
            f"• Best for speed                : {_name(winners.fastest)} "
            f"(latency={latency_microseconds(get_metrics(winners.fastest))} µs)"
        )

    if winners.best_compression:
        rows.append(
            f"• Best for compression          : {_name(winners.best_compression)} "
            f"(compression={format_number(_metric(winners.best_compression, 'compression_ratio'), 2)})"
        )

    if winners.lowest_unknown:
        rows.append(
            f"• Best for low unknown-token risk: {_name(winners.lowest_unknown)} "
            f"(unknown_rate={format_number(_metric(winners.lowest_unknown, 'unknown_rate'), 2)})"
        )

    if winners.most_readable:
        rows.append(
            f"• Best for readability/debugging: {_name(winners.most_readable)} "
            f"(tokens={_metric(winners.most_readable, 'token_count')})"
        )

    lines.extend(rows)

    # Balanced choice için ek metrik dump.
    lines.extend(
        [
            "",
            f"Balanced choice metrics: {_name(balanced)}",
            f"  Efficiency Score : {format_number(balanced_metrics.get('efficiency_score'), 2)}",
            f"  Compression Ratio: {format_number(balanced_metrics.get('compression_ratio'), 2)}",
            f"  Unknown Rate     : {format_number(balanced_metrics.get('unknown_rate'), 2)}",
            f"  Latency          : {latency_microseconds(balanced_metrics)} µs",
            "",
            (
                "Use the balanced recommendation as the default. If the application has a "
                "strict priority such as speed, compression, readability, or robustness, "
                "prefer the category-specific recommendation above instead."
            ),
            "",
        ]
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_text_report(compare_result: dict[str, Any]) -> str:
    """
    Tokenizer karşılaştırma sonuçlarından production-ready bir plain-text
    raporu üretir.

    Rapor şu bölümleri içerir: header, source text, overview, executive
    summary, summary table, key insights, interpretation, recommendation,
    why-this-won, tokenizer details, ranking, pairwise comparisons,
    categorical recommendation.
    """
    text, total, results, pairwise = extract_compare_payload(compare_result)
    winners = compute_winners(results)

    lines: list[str] = []

    _append_header(lines, total)
    _append_source_text(lines, text)
    _append_overview(lines, total)
    _append_executive_summary(lines, results, winners)
    _append_summary_table(lines, results, text)
    _append_key_insights(lines, winners)
    _append_interpretation(lines, winners)
    _append_recommendation(lines, results, winners)
    _append_winner_explanation(lines, results)
    _append_tokenizer_details(lines, results, text)
    _append_ranking(lines, results)
    _append_pairwise_comparisons(lines, pairwise)
    _append_categorical_recommendation(lines, winners)

    lines.extend(
        [
            "END OF REPORT".center(REPORT_TITLE_WIDTH),
            wide_hr("="),
        ]
    )

    return "\n".join(lines)