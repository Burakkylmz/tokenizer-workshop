from __future__ import annotations

from typing import Any

from .helpers import (
    extract_compare_payload,
    format_number,
    format_reconstruction,
    format_top_tokens,
    get_metrics,
    latency_microseconds,
    safe_str,
    truncate_list,
    utc_now_iso,
    is_reconstruction_match,
    normalize_text,
    wide_hr,
)


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


def _pairwise_similarity_level(overlap_ratio: Any) -> str:
    ratio = _safe_float(overlap_ratio)

    if ratio == 0:
        return "Completely Different"

    if ratio < 0.25:
        return "Highly Different"

    if ratio < 0.60:
        return "Moderately Similar"

    return "Highly Similar"


def _pairwise_observation(overlap_ratio: Any) -> str:
    level = _pairwise_similarity_level(overlap_ratio)

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
            "# Tokenizer Evaluation Report",
            "",
            f"**Generated At (UTC):** {utc_now_iso()}",
            f"**Total Tokenizers:** {total}",
            "",
        ]
    )


def _append_source_text(lines: list[str], text: str) -> None:
    lines.extend(["## Source Text", ""])

    if text:
        lines.extend(["```text", text, "```", ""])
    else:
        lines.extend(["_No source text provided._", ""])


def _append_overview(lines: list[str], total: int) -> None:
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
) -> None:
    lines.extend(["## Executive Summary", ""])

    if not results:
        lines.extend(["_No executive summary available._", ""])
        return

    best_efficiency = _best_by_metric(results, "efficiency_score", reverse=True)
    fastest = _best_by_metric(results, "latency_seconds", reverse=False)
    lowest_token = _best_by_metric(results, "token_count", reverse=False)
    highest_token = _best_by_metric(results, "token_count", reverse=True)

    if best_efficiency:
        lines.append(
            f"- **Best overall tokenizer:** {safe_str(best_efficiency.get('tokenizer_name'))}"
        )

    if fastest:
        lines.append(
            f"- **Fastest tokenizer:** {safe_str(fastest.get('tokenizer_name'))}"
        )

    if lowest_token:
        lines.append(
            f"- **Shortest sequence tokenizer:** {safe_str(lowest_token.get('tokenizer_name'))}"
        )

    if highest_token:
        lines.append(
            f"- **Most granular tokenizer:** {safe_str(highest_token.get('tokenizer_name'))}"
        )

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
        gain = _compression_gain_percent(item, source_text)

        lines.append(
            f"| {safe_str(item.get('tokenizer_name'))} "
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


def _append_key_insights(lines: list[str], results: list[dict[str, Any]]) -> None:
    lines.extend(["## Key Insights", ""])

    if not results:
        lines.extend(["_No insights available._", ""])
        return

    lowest_token = _best_by_metric(results, "token_count", reverse=False)
    highest_token = _best_by_metric(results, "token_count", reverse=True)
    best_efficiency = _best_by_metric(results, "efficiency_score", reverse=True)
    highest_unique = _best_by_metric(results, "unique_token_count", reverse=True)
    fastest = _best_by_metric(results, "latency_seconds", reverse=False)
    best_compression = _best_by_metric(results, "compression_ratio", reverse=True)

    if lowest_token:
        lines.append(
            f"- **Lowest token count:** {safe_str(lowest_token.get('tokenizer_name'))} "
            f"({_metric(lowest_token, 'token_count')})"
        )

    if highest_token:
        lines.append(
            f"- **Highest token count:** {safe_str(highest_token.get('tokenizer_name'))} "
            f"({_metric(highest_token, 'token_count')})"
        )

    if best_efficiency:
        lines.append(
            f"- **Best efficiency score:** {safe_str(best_efficiency.get('tokenizer_name'))} "
            f"({format_number(_metric(best_efficiency, 'efficiency_score'), 2)})"
        )

    if highest_unique:
        lines.append(
            f"- **Highest unique token count:** {safe_str(highest_unique.get('tokenizer_name'))} "
            f"({_metric(highest_unique, 'unique_token_count')})"
        )

    if fastest:
        lines.append(
            f"- **Fastest tokenizer:** {safe_str(fastest.get('tokenizer_name'))} "
            f"({latency_microseconds(get_metrics(fastest))} µs)"
        )

    if best_compression:
        lines.append(
            f"- **Best compression ratio:** {safe_str(best_compression.get('tokenizer_name'))} "
            f"({format_number(_metric(best_compression, 'compression_ratio'), 2)})"
        )

    lines.append("")


def _append_interpretation(lines: list[str], results: list[dict[str, Any]]) -> None:
    lines.extend(["## Interpretation", ""])

    if not results:
        lines.extend(["_No interpretation available._", ""])
        return

    lowest_token = _best_by_metric(results, "token_count", reverse=False)
    highest_token = _best_by_metric(results, "token_count", reverse=True)
    best_efficiency = _best_by_metric(results, "efficiency_score", reverse=True)
    fastest = _best_by_metric(results, "latency_seconds", reverse=False)

    if lowest_token:
        lines.extend(
            [
                (
                    f"The **{safe_str(lowest_token.get('tokenizer_name'))}** tokenizer produces the most compact "
                    f"segmentation with **{_metric(lowest_token, 'token_count')} tokens**."
                ),
                "",
            ]
        )

    if highest_token:
        lines.extend(
            [
                (
                    f"The **{safe_str(highest_token.get('tokenizer_name'))}** tokenizer produces the most granular "
                    f"segmentation with **{_metric(highest_token, 'token_count')} tokens**."
                ),
                "",
            ]
        )

    if best_efficiency:
        lines.extend(
            [
                (
                    f"The **{safe_str(best_efficiency.get('tokenizer_name'))}** tokenizer achieves the strongest "
                    f"efficiency score (**{format_number(_metric(best_efficiency, 'efficiency_score'), 2)}**), "
                    "which indicates better compression behavior per token."
                ),
                "",
            ]
        )

    if fastest:
        lines.extend(
            [
                (
                    f"The fastest tokenizer is **{safe_str(fastest.get('tokenizer_name'))}** "
                    f"with **{latency_microseconds(get_metrics(fastest))} µs** latency."
                ),
                "",
            ]
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
    lines.extend(["## Recommendation", ""])

    if not results:
        lines.append("No tokenizer results are available for recommendation.")
        lines.append("")
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

        elif name == "wordpiece":
            lines.append("• wordpiece  : best for subword tokenization with a fixed vocabulary")

        elif name == "unigram":
            lines.append("• unigram    : best for probabilistic subword tokenization with a fixed vocabulary")

        elif name == "sentencepiece":
            lines.append("• sentencepiece : best for flexible subword tokenization with a fixed vocabulary")

        elif name == "white_space":
            lines.append(
                "• white_space : best for simple baseline tokenization and debugging"
            )

        elif name == "punctuation":
            lines.append(
                "• punctuation : best for separating words and punctuation into distinct tokens"
            )

        elif name == "subword":
            lines.append("• subword    : best for fixed-size subword tokenization")

        elif name == "morpheme":
            lines.append("• morpheme    : best for linguistically motivated subword tokenization")

        elif name == "byte_level_bpe":
            lines.append("• byte_level_bpe  : best for handling complex or unseen text")   

        elif name == "pretrained":
            lines.append("• pretrained : best for leveraging existing tokenization models") 

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
            "• WordPiece tokenization is effective for subword tokenization with a fixed vocabulary, commonly used in transformer models, but may require careful handling of unknown tokens and vocabulary management.",
            "• Unigram tokenization is effective for probabilistic subword tokenization with a fixed vocabulary, balancing flexibility and compression.",
            "• SentencePiece tokenization is effective for flexible subword tokenization with a fixed vocabulary, balancing flexibility and compression.",
            "• Whitespace tokenization is a simple baseline that can be useful for debugging and educational purposes, but it may not provide the best performance or compression for most real-world applications.",
            "• Punctuation tokenization is effective for separating words and punctuation into distinct tokens, which can improve readability and downstream processing. However, it may increase token count and latency compared to simpler tokenization strategies.",
            "• Subword tokenization is effective for fixed-size subword tokenization, balancing flexibility and compression.",
            "• Morpheme tokenization is effective for linguistically motivated subword tokenization, capturing meaningful units of language. However, it may require language-specific resources and may not always align with the needs of downstream models.",
            "• Byte-level BPE tokenization is effective for handling complex or unseen text, ensuring full coverage of any input.", 
            "• Pretrained tokenization is effective for leveraging existing tokenization models, providing strong performance on many tasks. However, it may require additional dependencies and may not be customizable for specific domains or languages.",
            "",
            "Ultimately, the best tokenizer choice depends on the specific requirements of the application, including the desired balance between compression, speed, interpretability, and robustness to diverse input types.",
        ]
    )


def _append_tokenizer_details(
    lines: list[str],
    results: list[dict[str, Any]],
    source_text: str,
) -> None:
    lines.extend(["## Tokenizer Details", ""])

    if not results:
        lines.extend(["_No tokenizer details available._", ""])
        return

    for index, item in enumerate(results, start=1):
        tokenizer_name = safe_str(item.get("tokenizer_name"))
        tokens = item.get("tokens", [])
        metrics = get_metrics(item)

        lines.extend(
            [
                f"### {index}. {tokenizer_name}",
                "",
                f"- **Token Count:** {_metric(item, 'token_count')}",
                f"- **Vocab Size:** {_metric(item, 'unique_token_count', item.get('vocab_size', 0))}",
                f"- **Token Preview:** `{truncate_list(tokens)}`",
                f"- **Unique Token Count:** {_metric(item, 'unique_token_count', '-')}",
                f"- **Unique Ratio:** {format_number(metrics.get('unique_ratio'), 2)}",
                f"- **Average Token Length:** {format_number(metrics.get('average_token_length'), 2)}",
                (
                    f"- **Min / Max Token Length:** "
                    f"{metrics.get('min_token_length', '-')} / {metrics.get('max_token_length', '-')}"
                ),
                f"- **Avg Chars / Token:** {format_number(metrics.get('avg_chars_per_token'), 2)}",
                (
                    f"- **Unknown Tokens:** {metrics.get('unknown_count', '-')} "
                    f"({format_number(metrics.get('unknown_rate'), 2)})"
                ),
                (
                    f"- **Latency:** {latency_microseconds(metrics)} µs "
                    f"({format_number(metrics.get('latency_seconds'), 6)} s)"
                ),
                f"- **Latency / Token:** {format_number(metrics.get('latency_per_token'), 6)}",
                f"- **Efficiency Score:** {format_number(metrics.get('efficiency_score'), 2)}",
                f"- **Compression Ratio:** {format_number(metrics.get('compression_ratio'), 2)}",
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
    lines.extend(["## Overall Ranking", ""])

    if not results:
        lines.extend(["_No ranking available._", ""])
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
            "Ranking is based primarily on **efficiency score**, with **latency** used as a secondary tie-breaker.",
            "",
        ]
    )

    for index, item in enumerate(ranking, start=1):
        metrics = get_metrics(item)

        lines.append(
            f"{index}. **{safe_str(item.get('tokenizer_name'))}** "
            f"(eff={format_number(metrics.get('efficiency_score'), 2)}, "
            f"latency={latency_microseconds(metrics)} µs)"
        )

    lines.append("")


def _append_pairwise_comparisons(
    lines: list[str],
    pairwise: list[dict[str, Any]],
) -> None:
    lines.extend(["## Pairwise Comparisons", ""])

    if not pairwise:
        lines.extend(["No pairwise comparison data available.", ""])
        return

    for item in pairwise:
        left_name = safe_str(item.get("left_name"))
        right_name = safe_str(item.get("right_name"))
        ratio = item.get("overlap_ratio")

        lines.extend(
            [
                f"### {left_name} ↔ {right_name}",
                "",
                f"- **Overlap Ratio:** {format_number(ratio, 2)}",
                f"- **Semantic Difference Level:** {_pairwise_similarity_level(ratio)}",
                f"- **Common Tokens:** `{truncate_list(item.get('common_tokens', []))}`",
                f"- **Only In {left_name}:** `{truncate_list(item.get('unique_to_left', []))}`",
                f"- **Only In {right_name}:** `{truncate_list(item.get('unique_to_right', []))}`",
                "- **Observation:**",
                f"  - {_pairwise_observation(ratio)}",
                "",
            ]
        )


def build_markdown_report(compare_result: dict[str, Any]) -> str:
    """
    Build a production-ready Markdown report from tokenizer comparison results.

    The report includes:
    - source text,
    - executive summary,
    - summary metrics,
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
        Markdown report as a string.
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

    return "\n".join(lines)