"""
Markdown rapor oluşturucu.

Tokenizer karşılaştırma sonuçlarını Markdown gösterimi için optimize edilmiş
bir rapora dönüştürür. Domain mantığı `base.py`'da paylaşılır; bu modül
yalnızca markdown formatlamadan sorumludur.

Bölümler text/PDF raporlarıyla tutarlı bir çekirdek paylaşır; ek olarak
markdown'a özgü daha zengin "Categorical Recommendation" bölümü içerir.
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
    extract_compare_payload,
    format_number,
    format_reconstruction,
    format_top_tokens,
    get_metrics,
    interpret_similarity_level,
    latency_microseconds,
    safe_str,
    truncate_list,
    utc_now_iso,
)


# ---------------------------------------------------------------------------
# Düşük seviye yardımcılar
# ---------------------------------------------------------------------------

def _name(item: dict[str, Any] | None) -> str:
    """Tokenizer adını güvenli şekilde okur."""
    if item is None:
        return ""
    return safe_str(item.get("tokenizer_name"))


def _bullet(label: str, value: Any) -> str:
    """Tutarlı `- **Label:** value` bullet üretir."""
    return f"- **{label}:** {value}"


# ---------------------------------------------------------------------------
# Bölümler
# ---------------------------------------------------------------------------

def _append_header(lines: list[str], total: int) -> None:
    """Rapor başlığı ve genel meta bilgileri."""
    lines.extend(
        [
            "# Tokenizer Evaluation Report",
            "",
            f"**Generated At (UTC):** {utc_now_iso()}",
            f"**Total Tokenizers:** {total}",
            "",
        ]
    )


def _append_source_text(lines: list[str], text: str) -> None:
    """Kaynak metin bölümü."""
    lines.extend(["## Source Text", ""])

    if text:
        lines.extend(["```text", text, "```", ""])
    else:
        lines.extend(["_No source text provided._", ""])

    # Çok kısa girdiler için uyarı.
    if text and len(text) < 100:
        lines.extend(
            [
                f"> ⚠ **Note:** Source text is short ({len(text)} characters). "
                "Findings on small inputs may not generalize.",
                "",
            ]
        )


def _append_overview(lines: list[str], total: int) -> None:
    """Raporun genel bakış bölümü."""
    lines.extend(
        [
            "## Overview",
            "",
            (
                f"This report evaluates **{total} tokenizer(s)** on the same input text. "
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
    """Yürütücü özet bölümü."""
    lines.extend(["## Executive Summary", ""])

    if not results:
        lines.extend(["_No executive summary available._", ""])
        return

    best_overall = max(results, key=tokenizer_quality_score, default=None)

    rows: list[tuple[dict[str, Any] | None, str]] = [
        (best_overall, "Best overall tokenizer"),
        (winners.fastest, "Fastest tokenizer"),
        (winners.lowest_token, "Shortest sequence tokenizer"),
        (winners.highest_token, "Most granular tokenizer"),
    ]

    for item, label in rows:
        if item:
            lines.append(_bullet(label, _name(item)))

    lines.extend(
        [
            "",
            (
                "This suggests that tokenizer selection should depend on the target use case: "
                "speed, compression, interpretability, or robustness across diverse input types."
            ),
            "",
        ]
    )


def _append_summary_table(
    lines: list[str],
    results: list[dict[str, Any]],
    source_text: str,
) -> None:
    """Özet tablo."""
    lines.extend(
        [
            "## Summary Table",
            "",
            (
                "| Tokenizer | Tokens | Unique | Uniq Ratio | Avg Len | Chars/Token | "
                "Latency µs | Latency/Token | Eff. Score | Comp. | Gain % |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    if not results:
        lines.extend(["", "_No summary data available._", ""])
        return

    for item in results:
        metrics = get_metrics(item)
        gain = compression_gain_percent(item, source_text)

        lines.append(
            f"| {_name(item)} "
            f"| {_metric(item, 'token_count')} "
            f"| {_metric(item, 'unique_token_count', item.get('vocab_size', 0))} "
            f"| {format_number(metrics.get('unique_ratio'), 2)} "
            f"| {format_number(metrics.get('average_token_length'), 2)} "
            f"| {format_number(metrics.get('avg_chars_per_token'), 2)} "
            f"| {latency_microseconds(metrics)} "
            f"| {format_number(metrics.get('latency_per_token'), 6)} "
            f"| {format_number(metrics.get('efficiency_score'), 2)} "
            f"| {format_number(metrics.get('compression_ratio'), 2)} "
            f"| {format_number(gain, 2)} |"
        )

    lines.append("")


def _append_key_insights(lines: list[str], winners: Winners) -> None:
    """Anahtar metrik bulguları."""
    lines.extend(["## Key Insights", ""])

    if winners.best_efficiency is None:
        lines.extend(["_No insights available._", ""])
        return

    insights: list[tuple[dict[str, Any] | None, str, str]] = [
        (
            winners.lowest_token, "Lowest token count",
            f"({_metric(winners.lowest_token or {}, 'token_count')})",
        ),
        (
            winners.highest_token, "Highest token count",
            f"({_metric(winners.highest_token or {}, 'token_count')})",
        ),
        (
            winners.best_efficiency, "Best efficiency score",
            f"({format_number(_metric(winners.best_efficiency or {}, 'efficiency_score'), 2)})",
        ),
        (
            winners.highest_unique, "Highest unique token count",
            f"({_metric(winners.highest_unique or {}, 'unique_token_count')})",
        ),
        (
            winners.fastest, "Fastest tokenizer",
            f"({latency_microseconds(get_metrics(winners.fastest or {}))} µs)",
        ),
        (
            winners.best_compression, "Best compression ratio",
            f"({format_number(_metric(winners.best_compression or {}, 'compression_ratio'), 2)})",
        ),
    ]

    for item, label, suffix in insights:
        if item:
            lines.append(f"- **{label}:** {_name(item)} {suffix}")

    lines.append("")


def _append_interpretation(lines: list[str], winners: Winners) -> None:
    """Bulguların kısa yorumu."""
    lines.extend(["## Interpretation", ""])

    if winners.best_efficiency is None:
        lines.extend(["_No interpretation available._", ""])
        return

    if winners.lowest_token:
        lines.extend(
            [
                f"The **{_name(winners.lowest_token)}** tokenizer produces the most compact "
                f"segmentation with **{_metric(winners.lowest_token, 'token_count')} tokens**.",
                "",
            ]
        )

    if winners.highest_token:
        lines.extend(
            [
                f"The **{_name(winners.highest_token)}** tokenizer produces the most granular "
                f"segmentation with **{_metric(winners.highest_token, 'token_count')} tokens**.",
                "",
            ]
        )

    if winners.best_efficiency:
        eff = format_number(_metric(winners.best_efficiency, "efficiency_score"), 2)
        lines.extend(
            [
                f"The **{_name(winners.best_efficiency)}** tokenizer achieves the strongest "
                f"efficiency score (**{eff}**), which indicates better compression behavior per token.",
                "",
            ]
        )

    if winners.fastest:
        lat = latency_microseconds(get_metrics(winners.fastest))
        lines.extend(
            [
                f"The fastest tokenizer is **{_name(winners.fastest)}** with **{lat} µs** latency.",
                "",
            ]
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
        1. Genel kazananlardan kategori önerileri (en iyi efficiency, fastest, lowest token)
        2. `TOKENIZER_GUIDANCE`'tan statik kullanım rehberi
        3. Trade-off açıklamaları
    """
    lines.extend(["## Recommendation", ""])

    if not results:
        lines.extend(["No tokenizer results are available for recommendation.", ""])
        return

    lines.extend(["### When to use each tokenizer", ""])

    added: set[str] = set()

    def add(name: str, text: str) -> None:
        key = name.lower()
        if key and key not in added:
            lines.append(f"- **{name}** → {text}")
            added.add(key)

    if winners.best_efficiency:
        add(
            _name(winners.best_efficiency),
            "best when compression and token efficiency matter.",
        )
    if winners.fastest:
        add(_name(winners.fastest), "best when low-latency tokenization matters.")
    if winners.lowest_token:
        add(
            _name(winners.lowest_token),
            "best when minimizing total token count matters.",
        )

    for item in results:
        name = _name(item)
        guidance = TOKENIZER_GUIDANCE.get(name.lower())
        if guidance:
            add(name, guidance)

    lines.extend(["", "### Trade-offs", ""])
    for line in TRADEOFF_LINES:
        lines.append(f"- {line}")

    lines.extend(["", TRADEOFF_CLOSING, ""])


def _append_tokenizer_details(
    lines: list[str],
    results: list[dict[str, Any]],
    source_text: str,
) -> None:
    """Her tokenizer için detaylı bölüm."""
    lines.extend(["## Tokenizer Details", ""])

    if not results:
        lines.extend(["_No tokenizer details available._", ""])
        return

    for index, item in enumerate(results, start=1):
        tokenizer_name = _name(item)
        tokens = item.get("tokens", [])
        metrics = get_metrics(item)

        lines.extend(
            [
                f"### {index}. {tokenizer_name}",
                "",
                _bullet("Token Count", _metric(item, "token_count")),
                _bullet(
                    "Vocab Size",
                    _metric(item, "unique_token_count", item.get("vocab_size", 0)),
                ),
                _bullet("Token Preview", f"`{truncate_list(tokens)}`"),
                _bullet("Unique Token Count", _metric(item, "unique_token_count", "-")),
                _bullet("Unique Ratio", format_number(metrics.get("unique_ratio"), 2)),
                _bullet(
                    "Average Token Length",
                    format_number(metrics.get("average_token_length"), 2),
                ),
                _bullet(
                    "Min / Max Token Length",
                    f"{metrics.get('min_token_length', '-')} / "
                    f"{metrics.get('max_token_length', '-')}",
                ),
                _bullet(
                    "Avg Chars / Token",
                    format_number(metrics.get("avg_chars_per_token"), 2),
                ),
                _bullet(
                    "Unknown Tokens",
                    f"{metrics.get('unknown_count', '-')} "
                    f"({format_number(metrics.get('unknown_rate'), 2)})",
                ),
                _bullet(
                    "Latency",
                    f"{latency_microseconds(metrics)} µs "
                    f"({format_number(metrics.get('latency_seconds'), 6)} s)",
                ),
                _bullet(
                    "Latency / Token",
                    format_number(metrics.get("latency_per_token"), 6),
                ),
                _bullet(
                    "Efficiency Score",
                    format_number(metrics.get("efficiency_score"), 2),
                ),
                _bullet(
                    "Compression Ratio",
                    format_number(metrics.get("compression_ratio"), 2),
                ),
                "",
                "**Top Tokens:**",
            ]
        )

        top_tokens = format_top_tokens(metrics.get("top_tokens"))
        if top_tokens:
            for token_line in top_tokens:
                lines.append(f"- `{token_line.strip()}`")
        else:
            lines.append("- No top token data available.")

        lines.extend(["", "**Token Length Distribution:**"])

        distribution = metrics.get("token_length_distribution")
        if isinstance(distribution, dict) and distribution:
            for length, count in sorted(
                distribution.items(),
                key=lambda pair: _safe_float(pair[0]),
            ):
                lines.append(f"- Length `{length}`: `{count}` tokens")
        else:
            lines.append("- No token length distribution available.")

        lines.extend(["", "**Reconstruction:**"])

        reconstruction = format_reconstruction(metrics, original_text=source_text)
        if reconstruction:
            for reconstruction_line in reconstruction:
                lines.append(f"- {reconstruction_line.strip()}")
        else:
            lines.append("- No reconstruction details available.")

        lines.append("")


def _append_ranking(lines: list[str], results: list[dict[str, Any]]) -> None:
    """
    Bileşik kalite skoruna göre genel sıralama.

    `tokenizer_quality_score` text/PDF raporlarıyla aynı mantığı kullanır.
    """
    lines.extend(["## Overall Ranking", ""])

    if not results:
        lines.extend(["_No ranking available._", ""])
        return

    lines.extend(
        [
            (
                "Ranking is based on a composite quality score including "
                "**efficiency**, **latency**, **unknown rate**, reconstruction quality, "
                "and structural penalties."
            ),
            "",
        ]
    )

    ranking = sorted(results, key=tokenizer_quality_score, reverse=True)

    for index, item in enumerate(ranking, start=1):
        metrics = get_metrics(item)
        score = format_number(tokenizer_quality_score(item), 2)
        eff = format_number(metrics.get("efficiency_score"), 2)
        lat = latency_microseconds(metrics)

        lines.append(
            f"{index}. **{_name(item)}** "
            f"(score={score}, eff={eff}, latency={lat} µs)"
        )

    lines.append("")


def _append_pairwise_comparisons(
    lines: list[str],
    pairwise: list[dict[str, Any]],
) -> None:
    """
    Pairwise tokenizer karşılaştırmaları.

    `similarity_level` pipeline tarafından hesaplanır; bu builder yalnızca
    onu gösterir ve yorumlar.
    """
    lines.extend(["## Pairwise Comparisons", ""])

    if not pairwise:
        lines.extend(["No pairwise comparison data available.", ""])
        return

    for item in pairwise:
        left_name = safe_str(item.get("left_name"))
        right_name = safe_str(item.get("right_name"))
        ratio = item.get("overlap_ratio")
        similarity_level = safe_str(
            item.get("similarity_level"),
            fallback="Unknown",
        )

        lines.extend(
            [
                f"### {left_name} ↔ {right_name}",
                "",
                _bullet("Overlap Ratio", format_number(ratio, 2)),
                _bullet("Similarity Level", similarity_level),
                _bullet("Interpretation", interpret_similarity_level(similarity_level)),
                _bullet(
                    "Common Tokens",
                    f"`{truncate_list(item.get('common_tokens', []))}`",
                ),
                _bullet(
                    f"Only In {left_name}",
                    f"`{truncate_list(item.get('unique_to_left', []))}`",
                ),
                _bullet(
                    f"Only In {right_name}",
                    f"`{truncate_list(item.get('unique_to_right', []))}`",
                ),
                "",
            ]
        )


def _append_categorical_recommendation(
    lines: list[str],
    winners: Winners,
) -> None:
    """
    Kategori bazlı öneri.

    Tek bir tokenizer'ı mutlak en iyi ilan etmek yerine kullanım senaryosuna
    göre kategori bazlı öneri üretir. "Best balanced choice" bölümü, eski
    `Final Recommendation` bölümünün metrik dump'ını da içerir.
    """
    lines.extend(["## Categorical Recommendation", ""])

    if winners.best_balance is None:
        lines.extend(["No tokenizer recommendation available.", ""])
        return

    lines.extend(
        [
            "Tokenizer choice depends on the target use case. "
            "For this input, the recommended options are:",
            "",
        ]
    )

    # Best balanced choice — eski Final Recommendation buraya gömüldü.
    balanced = winners.best_balance
    metrics = get_metrics(balanced)
    lines.extend(
        [
            f"- **Best balanced choice:** `{_name(balanced)}` "
            f"(efficiency={format_number(metrics.get('efficiency_score'), 2)}, "
            f"latency={latency_microseconds(metrics)} µs)",
        ]
    )

    if winners.fastest:
        fast_metrics = get_metrics(winners.fastest)
        lines.append(
            f"- **Best for speed:** `{_name(winners.fastest)}` "
            f"(latency={latency_microseconds(fast_metrics)} µs)"
        )

    if winners.best_compression:
        lines.append(
            f"- **Best for compression:** `{_name(winners.best_compression)}` "
            f"(compression={format_number(_metric(winners.best_compression, 'compression_ratio'), 2)})"
        )

    if winners.lowest_unknown:
        lines.append(
            f"- **Best for low unknown-token risk:** `{_name(winners.lowest_unknown)}` "
            f"(unknown_rate={format_number(_metric(winners.lowest_unknown, 'unknown_rate'), 2)})"
        )

    if winners.most_readable:
        lines.append(
            f"- **Best for readability/debugging:** `{_name(winners.most_readable)}` "
            f"(tokens={_metric(winners.most_readable, 'token_count')})"
        )

    # Balanced choice için ek metrik dump (eski Final Recommendation içeriği)
    lines.extend(
        [
            "",
            f"### Balanced choice metrics: `{_name(balanced)}`",
            "",
            _bullet("Efficiency Score", format_number(metrics.get("efficiency_score"), 2)),
            _bullet(
                "Compression Ratio",
                format_number(metrics.get("compression_ratio"), 2),
            ),
            _bullet("Unknown Rate", format_number(metrics.get("unknown_rate"), 2)),
            _bullet("Latency", f"{latency_microseconds(metrics)} µs"),
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

def build_markdown_report(compare_result: dict[str, Any]) -> str:
    """
    Tokenizer karşılaştırma sonuçlarından production-ready bir Markdown
    raporu üretir.

    Bölümler text/PDF raporlarıyla tutarlı bir çekirdek paylaşır:
    header, source text, overview, executive summary, summary table,
    key insights, interpretation, recommendation, tokenizer details,
    overall ranking, pairwise comparisons, categorical recommendation.
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
    _append_tokenizer_details(lines, results, text)
    _append_ranking(lines, results)
    _append_pairwise_comparisons(lines, pairwise)
    _append_categorical_recommendation(lines, winners)

    return "\n".join(lines)