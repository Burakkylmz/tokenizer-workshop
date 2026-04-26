from __future__ import annotations

from typing import Any

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


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _metric(item: dict[str, Any], key: str, fallback: Any = 0) -> Any:
    metrics = get_metrics(item)
    return metrics.get(key, item.get(key, fallback))


def _best_by_metric(
    results: list[dict[str, Any]],
    metric_key: str,
    *,
    reverse: bool = True,
) -> dict[str, Any] | None:
    if not results:
        return None

    return sorted(
        results,
        key=lambda item: _safe_float(_metric(item, metric_key)),
        reverse=reverse,
    )[0]


def _compression_gain_percent(item: dict[str, Any], source_text: str) -> float:
    source_length = len(source_text)

    if source_length == 0:
        return 0.0

    token_count = _safe_float(_metric(item, "token_count"))
    return (1 - token_count / source_length) * 100


def _similarity_level(overlap_ratio: Any) -> str:
    ratio = _safe_float(overlap_ratio)

    if ratio == 0:
        return "Completely Different"

    if ratio < 0.25:
        return "Highly Different"

    if ratio < 0.60:
        return "Moderately Similar"

    return "Highly Similar"


def _pairwise_observation(overlap_ratio: Any) -> str:
    level = _similarity_level(overlap_ratio)

    observations = {
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

    return observations[level]


def _append_header(lines: list[str], total: int) -> None:
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
    append_section_title(lines, "SOURCE TEXT")
    lines.append(text if text else "No source text provided.")
    lines.append("")


def _append_overview(lines: list[str], total: int) -> None:
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
) -> None:
    lines.extend(["EXECUTIVE SUMMARY", wide_hr("-")])

    if not results:
        lines.extend(["No executive summary available.", ""])
        return

    best_overall = _best_by_metric(results, "efficiency_score", reverse=True)
    fastest = _best_by_metric(results, "latency_seconds", reverse=False)
    shortest = _best_by_metric(results, "token_count", reverse=False)
    most_granular = _best_by_metric(results, "token_count", reverse=True)

    if best_overall:
        lines.append(f"• Best overall tokenizer    : {safe_str(best_overall.get('tokenizer_name'))}")

    if fastest:
        lines.append(f"• Fastest tokenizer         : {safe_str(fastest.get('tokenizer_name'))}")

    if shortest:
        lines.append(f"• Shortest sequence         : {safe_str(shortest.get('tokenizer_name'))}")

    if most_granular:
        lines.append(f"• Most granular tokenizer   : {safe_str(most_granular.get('tokenizer_name'))}")

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
        gain = _compression_gain_percent(item, source_text)

        lines.append(
            f"{safe_str(item.get('tokenizer_name')):<9} | "
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


def _append_key_insights(lines: list[str], results: list[dict[str, Any]]) -> None:
    lines.extend(["KEY INSIGHTS", wide_hr("-")])

    if not results:
        lines.extend(["No insights available.", ""])
        return

    lowest_token = _best_by_metric(results, "token_count", reverse=False)
    highest_token = _best_by_metric(results, "token_count", reverse=True)
    best_efficiency = _best_by_metric(results, "efficiency_score", reverse=True)
    highest_unique = _best_by_metric(results, "unique_token_count", reverse=True)
    fastest = _best_by_metric(results, "latency_seconds", reverse=False)
    best_compression = _best_by_metric(results, "compression_ratio", reverse=True)

    if lowest_token:
        lines.append(
            f"• Lowest token count        : {safe_str(lowest_token.get('tokenizer_name'))} "
            f"({_metric(lowest_token, 'token_count')})"
        )

    if highest_token:
        lines.append(
            f"• Highest token count       : {safe_str(highest_token.get('tokenizer_name'))} "
            f"({_metric(highest_token, 'token_count')})"
        )

    if best_efficiency:
        lines.append(
            f"• Best efficiency score     : {safe_str(best_efficiency.get('tokenizer_name'))} "
            f"({format_number(_metric(best_efficiency, 'efficiency_score'), 2)})"
        )

    if highest_unique:
        lines.append(
            f"• Highest unique token count: {safe_str(highest_unique.get('tokenizer_name'))} "
            f"({_metric(highest_unique, 'unique_token_count')})"
        )

    if fastest:
        lines.append(
            f"• Fastest tokenizer         : {safe_str(fastest.get('tokenizer_name'))} "
            f"({latency_microseconds(get_metrics(fastest))} µs)"
        )

    if best_compression:
        lines.append(
            f"• Best compression ratio    : {safe_str(best_compression.get('tokenizer_name'))} "
            f"({format_number(_metric(best_compression, 'compression_ratio'), 2)})"
        )

    lines.append("")


def _append_interpretation(lines: list[str], results: list[dict[str, Any]]) -> None:
    lines.extend(["INTERPRETATION", wide_hr("-")])

    if not results:
        lines.extend(["No interpretation available.", ""])
        return

    lowest_token = _best_by_metric(results, "token_count", reverse=False)
    highest_token = _best_by_metric(results, "token_count", reverse=True)
    best_efficiency = _best_by_metric(results, "efficiency_score", reverse=True)
    fastest = _best_by_metric(results, "latency_seconds", reverse=False)

    if lowest_token:
        lines.append(
            f"The '{safe_str(lowest_token.get('tokenizer_name'))}' tokenizer produces the most compact "
            f"segmentation with {_metric(lowest_token, 'token_count')} tokens."
        )

    if highest_token:
        lines.append(
            f"The '{safe_str(highest_token.get('tokenizer_name'))}' tokenizer produces the most granular "
            f"segmentation with {_metric(highest_token, 'token_count')} tokens."
        )

    if best_efficiency:
        lines.append(
            f"The '{safe_str(best_efficiency.get('tokenizer_name'))}' tokenizer achieves the strongest "
            f"efficiency score ({format_number(_metric(best_efficiency, 'efficiency_score'), 2)}), "
            "which indicates better compression behavior per token."
        )

    if fastest:
        lines.append(
            f"The fastest tokenizer is '{safe_str(fastest.get('tokenizer_name'))}' "
            f"with {latency_microseconds(get_metrics(fastest))} µs latency."
        )

    lines.extend(
        [
            (
                "Overall, tokenizer choice directly affects sequence length, processing cost, semantic granularity, "
                "compression behavior, and downstream model efficiency."
            ),
            "",
        ]
    )


def _append_recommendation(lines: list[str], results: list[dict[str, Any]]) -> None:
    lines.extend(["RECOMMENDATION", wide_hr("-")])

    if not results:
        lines.extend(["No recommendation available.", ""])
        return

    best_efficiency = _best_by_metric(results, "efficiency_score", reverse=True)
    fastest = _best_by_metric(results, "latency_seconds", reverse=False)
    lowest_token = _best_by_metric(results, "token_count", reverse=False)

    lines.extend(["### When to use each tokenizer", ""])

    if best_efficiency:
        lines.append(
            f"- **{safe_str(best_efficiency.get('tokenizer_name'))}** → best when compression and token efficiency matter."
        )

    if fastest:
        lines.append(
            f"- **{safe_str(fastest.get('tokenizer_name'))}** → best when low-latency tokenization matters."
        )

    if lowest_token:
        lines.append(
            f"- **{safe_str(lowest_token.get('tokenizer_name'))}** → best when minimizing total token count matters."
        )

    for item in results:
        name = safe_str(item.get("tokenizer_name")).lower()

        if name == "word":
            lines.append("• word      : best when compression and readability matter")

        elif name == "byte":
            lines.append("• byte      : best when ultra-fast tokenization is required")

        elif name == "char":
            lines.append("• char      : best for debugging and maximum granularity")

        elif name == "regex":
            lines.append("• regex     : best for custom tokenization patterns and domain-specific text")

        elif name == "byte_bpe":
            lines.append("• byte_bpe  : best for handling complex or unseen text")

        elif name == "bpe":
            lines.append("• bpe       : balanced option between compression and flexibility")

        elif name == "regex_bpe":
            lines.append("• regex_bpe : best for custom tokenization patterns and domain-specific text")

        elif name == "ngram":
            lines.append("• ngram      : best for capturing local context and multi-word expressions")

    lines.extend(
        [
            "",
            "Trade-offs:",
            "• Character-level tokenization provides high granularity but usually increases sequence length.",
            "• Word-level tokenization is compact but language-dependent.",
            "• Subword/BPE tokenization balances flexibility and compression.",
            "• Byte-level tokenization ensures full coverage of any input.",
            "• Regex-based tokenization allows for custom patterns and domain-specific text handling.",
            "• N-gram tokenization captures local context and multi-word expressions, which can be beneficial for certain languages and tasks, but may increase token count compared to word-level tokenization.",
            "",
        ]
    )


def _append_tokenizer_details(
    lines: list[str],
    results: list[dict[str, Any]],
    source_text: str,
) -> None:
    lines.extend(["TOKENIZER DETAILS", wide_hr("-"), ""])

    if not results:
        lines.extend(["No tokenizer details available.", ""])
        return

    for index, item in enumerate(results, start=1):
        tokenizer_name = safe_str(item.get("tokenizer_name"))
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


def _append_ranking(lines: list[str], results: list[dict[str, Any]]) -> None:
    lines.extend(["OVERALL RANKING", wide_hr("-")])

    if not results:
        lines.extend(["No ranking available.", ""])
        return

    ranking = sorted(
        results,
        key=lambda item: (
            _safe_float(_metric(item, "efficiency_score")),
            -_safe_float(_metric(item, "latency_seconds")),
        ),
        reverse=True,
    )

    lines.extend(
        [
            "Ranking is based primarily on efficiency score, with latency used as a secondary tie-breaker.",
            "",
        ]
    )

    for index, item in enumerate(ranking, start=1):
        metrics = get_metrics(item)

        lines.append(
            f"{index}. {safe_str(item.get('tokenizer_name'))} "
            f"(eff={format_number(metrics.get('efficiency_score'), 2)}, "
            f"latency={latency_microseconds(metrics)} µs)"
        )

    lines.append("")


def _append_pairwise_comparisons(
    lines: list[str],
    pairwise: list[dict[str, Any]],
) -> None:
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


def build_text_report(compare_result: dict[str, Any]) -> str:
    """
    Build a production-ready plain-text tokenizer evaluation report.

    The report includes:
    - source text,
    - executive summary,
    - summary table,
    - key insights,
    - interpretation,
    - recommendation,
    - tokenizer-level details,
    - ranking,
    - pairwise comparisons.

    Args:
        compare_result:
            Raw comparison result produced by the tokenizer evaluation pipeline.

    Returns:
        Plain-text report as a string.
    """

    text, total, results, pairwise = extract_compare_payload(compare_result)

    lines: list[str] = []

    _append_header(lines, total)
    _append_source_text(lines, text)
    _append_overview(lines, total)
    _append_executive_summary(lines, results)
    _append_summary_table(lines, results, text)
    _append_key_insights(lines, results)
    _append_interpretation(lines, results)
    _append_recommendation(lines, results)
    _append_tokenizer_details(lines, results, text)
    _append_ranking(lines, results)
    _append_pairwise_comparisons(lines, pairwise)

    lines.extend(
        [
            "END OF REPORT".center(REPORT_TITLE_WIDTH),
            wide_hr("="),
        ]
    )

    return "\n".join(lines)