"""
PDF rapor oluşturucu.

Tokenizer karşılaştırma sonuçlarını ReportLab tabanlı bir PDF raporuna
dönüştürür. Domain mantığı `base.py`'da paylaşılır; bu modül yalnızca
PDF formatlamadan sorumludur.

Bölümler `text_report.py` ile aynı sırayı takip eder:
    header, source text, executive summary, summary table, key insights,
    interpretation, recommendation, why-this-won, tokenizer details,
    overall ranking, pairwise comparisons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
from xml.sax.saxutils import escape as xml_escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

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
    get_metrics,
    latency_microseconds,
    safe_str,
    utc_now_iso,
)


# ---------------------------------------------------------------------------
# Stil ve sabitler
# ---------------------------------------------------------------------------

# Tutarlı spacing scale (pt). Rastgele Spacer değerleri yerine bunları kullan.
SPACE_XS = 4
SPACE_SM = 8
SPACE_MD = 14
SPACE_LG = 22

# Standard 14 PostScript fontlarının Unicode kapsama desteği
# platform/font-renderer'lara göre değişir (örn. `µ` bazı poppler
# kurulumlarında boşluk olarak render olur). PDF için ASCII-güvenli
# "us" kullanıyoruz; text raporu kendi unicode gösterimini sürdürür.
LATENCY_UNIT = "us"

# Aynı font-encoding sebebiyle özel sembolleri ASCII fallback ile
# kullanıyoruz. Tek noktada toplu değiştirmek istersen burayı düzenle.
SYM_WARN = "(!)"
SYM_PAIR = "<->"

# Renk paleti — tek noktada toplandı.
COLOR_HEADER_BG = colors.HexColor("#1F3B73")
COLOR_HEADER_TEXT = colors.white
COLOR_ROW_ALT = colors.HexColor("#EEF2F7")
COLOR_ROW_BASE = colors.whitesmoke
COLOR_GRID = colors.HexColor("#B8C2CC")
COLOR_MUTED = colors.HexColor("#5A6470")
COLOR_WARNING = colors.HexColor("#A33A3A")

# Pairwise observation metinleri text_report.py ile birebir aynı.
_PAIRWISE_OBSERVATIONS: dict[str, str] = {
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


def _build_styles() -> dict[str, ParagraphStyle]:
    """
    Standart stylesheet üzerine raporun ihtiyaç duyduğu stilleri ekler.

    `getSampleStyleSheet`'in default'larını override etmek yerine yeni
    stiller türetiyoruz — böylece aynı isimle tekrar çağrıldığında
    "style already defined" hatası almayız.
    """
    base = getSampleStyleSheet()

    custom = {
        "Title": ParagraphStyle(
            "ReportTitle",
            parent=base["Heading1"],
            fontSize=20,
            leading=24,
            spaceAfter=4,
            textColor=COLOR_HEADER_BG,
        ),
        "Subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontSize=9,
            leading=12,
            textColor=COLOR_MUTED,
        ),
        "H2": ParagraphStyle(
            "ReportH2",
            parent=base["Heading2"],
            fontSize=14,
            leading=18,
            spaceBefore=8,
            spaceAfter=6,
            textColor=COLOR_HEADER_BG,
        ),
        "H3": ParagraphStyle(
            "ReportH3",
            parent=base["Heading3"],
            fontSize=11,
            leading=14,
            spaceBefore=6,
            spaceAfter=3,
        ),
        "Body": ParagraphStyle(
            "ReportBody",
            parent=base["Normal"],
            fontSize=9.5,
            leading=13,
        ),
        "Bullet": ParagraphStyle(
            "ReportBullet",
            parent=base["Normal"],
            fontSize=9.5,
            leading=13,
            leftIndent=12,
            bulletIndent=2,
        ),
        "Mono": ParagraphStyle(
            "ReportMono",
            parent=base["Normal"],
            fontName="Courier",
            fontSize=8.5,
            leading=11,
        ),
        "Warning": ParagraphStyle(
            "ReportWarning",
            parent=base["Normal"],
            fontSize=9,
            leading=12,
            textColor=COLOR_WARNING,
            leftIndent=12,
        ),
    }
    return custom


# ---------------------------------------------------------------------------
# Düşük seviye yardımcılar
# ---------------------------------------------------------------------------

def _name(item: dict[str, Any] | None) -> str:
    """Tokenizer adını güvenli şekilde okur."""
    if item is None:
        return ""
    return safe_str(item.get("tokenizer_name"))


def _esc(value: Any) -> str:
    """
    ReportLab Paragraph içine konacak her metin XML-escape edilmelidir.
    Aksi takdirde `<`, `&` gibi karakterler parse hatasına yol açar.
    """
    return xml_escape(str(value))


def _bullet(text: str, style: ParagraphStyle) -> Paragraph:
    """Tek satırlık bullet paragrafı üretir."""
    return Paragraph(f"• {text}", style)


def _similarity_level(overlap_ratio: Any) -> str:
    """Overlap oranını niteliksel seviyeye çevirir."""
    ratio = _safe_float(overlap_ratio)
    if ratio == 0:
        return "Completely Different"
    if ratio < 0.25:
        return "Highly Different"
    if ratio < 0.60:
        return "Moderately Similar"
    return "Highly Similar"


def _pairwise_observation(overlap_ratio: Any) -> str:
    """Overlap seviyesine göre kısa gözlem cümlesi döner."""
    return _PAIRWISE_OBSERVATIONS[_similarity_level(overlap_ratio)]


# ---------------------------------------------------------------------------
# Bölüm builder'ları — her biri Flowable listesi döner
# ---------------------------------------------------------------------------

def _build_header(
    styles: dict[str, ParagraphStyle],
    total: int,
) -> list[Any]:
    """Rapor başlığı ve meta bilgileri."""
    return [
        Paragraph("Tokenizer Evaluation Report", styles["Title"]),
        Paragraph(
            f"Generated at (UTC): {_esc(utc_now_iso())}  |  "
            f"Total tokenizers: {total}",
            styles["Subtitle"],
        ),
        Spacer(1, SPACE_LG),
    ]


def _build_source_text(
    styles: dict[str, ParagraphStyle],
    text: str,
) -> list[Any]:
    """Kaynak metin bölümü."""
    body = text if text else "No source text provided."
    elements = [
        Paragraph("Source Text", styles["H2"]),
        Paragraph(_esc(body), styles["Mono"]),
    ]

    # Çok kısa girdiler için uyarı — istatistiksel anlamlılık düşük.
    if text and len(text) < 100:
        elements.append(Spacer(1, SPACE_XS))
        elements.append(
            Paragraph(
                f"{SYM_WARN} Note: Source text is short ({len(text)} characters). "
                "Findings on small inputs may not generalize.",
                styles["Warning"],
            )
        )

    elements.append(Spacer(1, SPACE_MD))
    return elements


def _build_executive_summary(
    styles: dict[str, ParagraphStyle],
    results: list[dict[str, Any]],
    winners: Winners,
) -> list[Any]:
    """
    Yürütücü özet.

    "Best overall" artık `tokenizer_quality_score`'tan gelir — eski koddaki
    "best efficiency = best overall" yanılgısı kapatıldı.
    """
    elements: list[Any] = [Paragraph("Executive Summary", styles["H2"])]

    if not results:
        elements.append(Paragraph("No executive summary available.", styles["Body"]))
        elements.append(Spacer(1, SPACE_MD))
        return elements

    best_overall = max(results, key=tokenizer_quality_score, default=None)

    rows: list[tuple[dict[str, Any] | None, str]] = [
        (best_overall, "Best overall tokenizer"),
        (winners.fastest, "Fastest tokenizer"),
        (winners.lowest_token, "Shortest sequence"),
        (winners.highest_token, "Most granular tokenizer"),
    ]

    for item, label in rows:
        if item is None:
            continue
        elements.append(
            _bullet(f"<b>{_esc(label)}:</b> {_esc(_name(item))}", styles["Bullet"])
        )

    elements.append(Spacer(1, SPACE_SM))
    elements.append(
        Paragraph(
            "Tokenizer selection should depend on the target use case: speed, "
            "compression, readability, interpretability, or robustness.",
            styles["Body"],
        )
    )
    elements.append(Spacer(1, SPACE_MD))
    return elements


def _build_summary_table(
    styles: dict[str, ParagraphStyle],
    results: list[dict[str, Any]],
    source_text: str,
    page_width: float,
) -> list[Any]:
    """
    Özet tablo. Sütun genişlikleri sayfa genişliğine göre hesaplanır,
    böylece uzun isimler taşmaz.
    """
    elements: list[Any] = [Paragraph("Summary Table", styles["H2"])]

    if not results:
        elements.append(Paragraph("No summary data available.", styles["Body"]))
        elements.append(Spacer(1, SPACE_MD))
        return elements

    headers = [
        "Tokenizer", "Tokens", "Unique", "Uniq Ratio",
        "Chars/Tok", "Unknown", f"Latency {LATENCY_UNIT}", "Eff.", "Comp.", "Gain %",
    ]
    table_data: list[list[Any]] = [headers]

    for item in results:
        metrics = get_metrics(item)
        gain = compression_gain_percent(item, source_text)
        table_data.append(
            [
                _name(item),
                _metric(item, "token_count"),
                _metric(item, "unique_token_count", item.get("vocab_size", 0)),
                format_number(metrics.get("unique_ratio"), 2),
                format_number(metrics.get("avg_chars_per_token"), 2),
                format_number(metrics.get("unknown_rate"), 2),
                latency_microseconds(metrics),
                format_number(metrics.get("efficiency_score"), 2),
                format_number(metrics.get("compression_ratio"), 2),
                format_number(gain, 2),
            ]
        )

    # İlk sütun (tokenizer adı) geniş, kalanı eşit dağıt.
    first_col = page_width * 0.18
    rest = (page_width - first_col) / (len(headers) - 1)
    col_widths = [first_col] + [rest] * (len(headers) - 1)

    table = Table(table_data, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), COLOR_HEADER_BG),
                ("TEXTCOLOR", (0, 0), (-1, 0), COLOR_HEADER_TEXT),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [COLOR_ROW_BASE, COLOR_ROW_ALT],
                ),
                ("GRID", (0, 0), (-1, -1), 0.4, COLOR_GRID),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    elements.append(table)
    elements.append(Spacer(1, SPACE_MD))
    return elements


def _build_key_insights(
    styles: dict[str, ParagraphStyle],
    winners: Winners,
) -> list[Any]:
    """Anahtar metrik bulguları."""
    elements: list[Any] = [Paragraph("Key Insights", styles["H2"])]

    if winners.best_efficiency is None:
        elements.append(Paragraph("No insights available.", styles["Body"]))
        elements.append(Spacer(1, SPACE_MD))
        return elements

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
            f"({latency_microseconds(get_metrics(winners.fastest or {}))} {LATENCY_UNIT})",
        ),
        (
            winners.best_compression, "Best compression ratio",
            f"({format_number(_metric(winners.best_compression or {}, 'compression_ratio'), 2)})",
        ),
    ]

    for item, label, suffix in insights:
        if item is None:
            continue
        elements.append(
            _bullet(
                f"<b>{_esc(label)}:</b> {_esc(_name(item))} {_esc(suffix)}",
                styles["Bullet"],
            )
        )

    elements.append(Spacer(1, SPACE_MD))
    return elements


def _build_interpretation(
    styles: dict[str, ParagraphStyle],
    winners: Winners,
) -> list[Any]:
    """Bulguların kısa yorumu."""
    elements: list[Any] = [Paragraph("Interpretation", styles["H2"])]

    if winners.best_efficiency is None:
        elements.append(Paragraph("No interpretation available.", styles["Body"]))
        elements.append(Spacer(1, SPACE_MD))
        return elements

    if winners.lowest_token:
        elements.append(
            Paragraph(
                f"The <b>{_esc(_name(winners.lowest_token))}</b> tokenizer produces the "
                f"most compact segmentation with <b>{_esc(_metric(winners.lowest_token, 'token_count'))}</b> tokens.",
                styles["Body"],
            )
        )

    if winners.highest_token:
        elements.append(
            Paragraph(
                f"The <b>{_esc(_name(winners.highest_token))}</b> tokenizer produces the "
                f"most granular segmentation with <b>{_esc(_metric(winners.highest_token, 'token_count'))}</b> tokens.",
                styles["Body"],
            )
        )

    if winners.best_efficiency:
        eff = format_number(_metric(winners.best_efficiency, "efficiency_score"), 2)
        elements.append(
            Paragraph(
                f"The <b>{_esc(_name(winners.best_efficiency))}</b> tokenizer achieves the "
                f"strongest efficiency score (<b>{_esc(eff)}</b>), indicating better "
                "compression behavior per token.",
                styles["Body"],
            )
        )

    if winners.fastest:
        lat = latency_microseconds(get_metrics(winners.fastest))
        elements.append(
            Paragraph(
                f"The fastest tokenizer is <b>{_esc(_name(winners.fastest))}</b> "
                f"with <b>{_esc(lat)} {LATENCY_UNIT}</b> latency.",
                styles["Body"],
            )
        )

    elements.append(Spacer(1, SPACE_XS))
    elements.append(
        Paragraph(
            "Overall, tokenizer choice directly affects sequence length, processing cost, "
            "semantic granularity, compression behavior, and downstream model efficiency.",
            styles["Body"],
        )
    )
    elements.append(Spacer(1, SPACE_MD))
    return elements


def _build_recommendation(
    styles: dict[str, ParagraphStyle],
    results: list[dict[str, Any]],
    winners: Winners,
) -> list[Any]:
    """
    Bölümlü öneri:
        1. Genel kazananlardan kategori önerileri
        2. `TOKENIZER_GUIDANCE`'tan statik kullanım rehberi
        3. Trade-off açıklamaları
    """
    elements: list[Any] = [Paragraph("Recommendation", styles["H2"])]

    if not results:
        elements.append(Paragraph("No recommendation available.", styles["Body"]))
        elements.append(Spacer(1, SPACE_MD))
        return elements

    elements.append(Paragraph("<b>When to use each tokenizer:</b>", styles["Body"]))
    elements.append(Spacer(1, SPACE_XS))

    added: set[str] = set()

    def add(name: str, text: str) -> None:
        key = name.lower()
        if key and key not in added:
            elements.append(
                _bullet(f"<b>{_esc(name)}:</b> {_esc(text)}", styles["Bullet"])
            )
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

    elements.append(Spacer(1, SPACE_SM))
    elements.append(Paragraph("<b>Trade-offs:</b>", styles["Body"]))
    elements.append(Spacer(1, SPACE_XS))

    for line in TRADEOFF_LINES:
        elements.append(_bullet(_esc(line), styles["Bullet"]))

    elements.append(Spacer(1, SPACE_XS))
    elements.append(Paragraph(_esc(TRADEOFF_CLOSING), styles["Body"]))
    elements.append(Spacer(1, SPACE_MD))
    return elements


def _build_winner_explanation(
    styles: dict[str, ParagraphStyle],
    results: list[dict[str, Any]],
) -> list[Any]:
    """En yüksek skorlu tokenizer'ın neden kazandığını açıklar."""
    elements: list[Any] = [Paragraph("Why This Tokenizer Won", styles["H2"])]

    if not results:
        elements.append(Paragraph("No winner explanation available.", styles["Body"]))
        elements.append(Spacer(1, SPACE_MD))
        return elements

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

    elements.append(
        Paragraph(
            f"<b>Winner:</b> {_esc(name)}  |  "
            f"<b>Composite score:</b> {_esc(format_number(score, 2))}",
            styles["Body"],
        )
    )
    elements.append(Spacer(1, SPACE_XS))
    elements.append(Paragraph("<b>Why it ranked first:</b>", styles["Body"]))

    reasons: list[str] = []
    if reconstruct:
        reasons.append("It can reconstruct the original text, which strongly improves usability.")
    else:
        reasons.append("It cannot reconstruct the original text, which limits production usability.")

    if unknown_rate == 0:
        reasons.append("It produced no unknown tokens.")
    else:
        reasons.append(f"It has an unknown-token rate of {format_number(unknown_rate, 2)}.")

    reasons.append(f"It produced {token_count} tokens, which affects sequence length and cost.")
    reasons.append(f"Its efficiency score is {format_number(efficiency, 2)}.")
    reasons.append(f"Its latency is {latency} {LATENCY_UNIT}.")

    for reason in reasons:
        elements.append(_bullet(_esc(reason), styles["Bullet"]))

    if avg_chars > 15:
        elements.append(
            Paragraph(
                f"{SYM_WARN} Warning: average characters per token is very high, which may indicate "
                "over-compression.",
                styles["Warning"],
            )
        )
    if _safe_float(token_count) <= 2:
        elements.append(
            Paragraph(
                f"{SYM_WARN} Warning: token count is extremely low, which indicates a collapsed "
                "tokenization result.",
                styles["Warning"],
            )
        )

    elements.append(Spacer(1, SPACE_XS))
    elements.append(
        Paragraph(
            "In short, this tokenizer won because it achieved the strongest balance "
            "between scoring signals such as reconstruction, token count, efficiency, "
            "latency, unknown-token rate, and structural penalties.",
            styles["Body"],
        )
    )
    elements.append(Spacer(1, SPACE_MD))
    return elements


def _build_tokenizer_details(
    styles: dict[str, ParagraphStyle],
    results: list[dict[str, Any]],
) -> list[Any]:
    """Her tokenizer için detaylı blok."""
    elements: list[Any] = [
        PageBreak(),
        Paragraph("Tokenizer Details", styles["H2"]),
        Spacer(1, SPACE_SM),
    ]

    if not results:
        elements.append(Paragraph("No tokenizer details available.", styles["Body"]))
        return elements

    for index, item in enumerate(results, start=1):
        metrics = get_metrics(item)
        name = _name(item)

        block: list[Any] = [
            Paragraph(f"{index}. {_esc(name)}", styles["H3"]),
        ]

        detail_pairs = [
            ("Token Count", _metric(item, "token_count")),
            ("Unique Token Count", _metric(item, "unique_token_count", "-")),
            ("Unique Ratio", format_number(metrics.get("unique_ratio"), 2)),
            ("Average Token Length", format_number(metrics.get("average_token_length"), 2)),
            ("Avg Chars / Token", format_number(metrics.get("avg_chars_per_token"), 2)),
            ("Unknown Rate", format_number(metrics.get("unknown_rate"), 2)),
            ("Latency", f"{latency_microseconds(metrics)} {LATENCY_UNIT}"),
            ("Efficiency Score", format_number(metrics.get("efficiency_score"), 2)),
            ("Compression Ratio", format_number(metrics.get("compression_ratio"), 2)),
            ("Reconstruction", "yes" if metrics.get("reconstruction_match") else "no"),
        ]

        for label, value in detail_pairs:
            block.append(
                Paragraph(
                    f"<b>{_esc(label)}:</b> {_esc(value)}",
                    styles["Body"],
                )
            )

        block.append(Spacer(1, SPACE_SM))
        # Aynı tokenizer'ın bilgileri sayfa başında bölünmesin.
        elements.append(KeepTogether(block))

    elements.append(Spacer(1, SPACE_MD))
    return elements


def _build_overall_ranking(
    styles: dict[str, ParagraphStyle],
    results: list[dict[str, Any]],
) -> list[Any]:
    """Bileşik skora göre genel sıralama."""
    elements: list[Any] = [
        PageBreak(),
        Paragraph("Overall Ranking", styles["H2"]),
    ]

    if not results:
        elements.append(Paragraph("No ranking available.", styles["Body"]))
        elements.append(Spacer(1, SPACE_MD))
        return elements

    elements.append(
        Paragraph(
            "Ranking is based on a composite quality score including efficiency, "
            "latency, unknown rate, reconstruction quality, and structural penalties.",
            styles["Body"],
        )
    )
    elements.append(Spacer(1, SPACE_SM))

    ranking = sorted(results, key=tokenizer_quality_score, reverse=True)

    table_data: list[list[Any]] = [["#", "Tokenizer", "Score", "Eff.", f"Latency {LATENCY_UNIT}"]]
    for index, item in enumerate(ranking, start=1):
        metrics = get_metrics(item)
        table_data.append(
            [
                index,
                _name(item),
                format_number(tokenizer_quality_score(item), 2),
                format_number(metrics.get("efficiency_score"), 2),
                latency_microseconds(metrics),
            ]
        )

    table = Table(
        table_data,
        colWidths=[20 * mm, 50 * mm, 30 * mm, 25 * mm, 30 * mm],
        hAlign="LEFT",
        repeatRows=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), COLOR_HEADER_BG),
                ("TEXTCOLOR", (0, 0), (-1, 0), COLOR_HEADER_TEXT),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("ALIGN", (2, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [COLOR_ROW_BASE, COLOR_ROW_ALT],
                ),
                ("GRID", (0, 0), (-1, -1), 0.4, COLOR_GRID),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    elements.append(table)
    elements.append(Spacer(1, SPACE_MD))
    return elements


def _build_pairwise_comparisons(
    styles: dict[str, ParagraphStyle],
    pairwise: list[dict[str, Any]],
) -> list[Any]:
    """Pairwise tokenizer karşılaştırmaları."""
    elements: list[Any] = [
        PageBreak(),
        Paragraph("Pairwise Comparisons", styles["H2"]),
    ]

    if not pairwise:
        elements.append(Paragraph("No pairwise comparison data available.", styles["Body"]))
        elements.append(Spacer(1, SPACE_MD))
        return elements

    for item in pairwise:
        left_name = safe_str(item.get("left_name"))
        right_name = safe_str(item.get("right_name"))
        ratio = item.get("overlap_ratio")
        level = _similarity_level(ratio)

        block: list[Any] = [
            Paragraph(
                f"{_esc(left_name)} {SYM_PAIR} {_esc(right_name)}",
                styles["H3"],
            ),
            Paragraph(
                f"<b>Overlap ratio:</b> {_esc(format_number(ratio, 2))}  |  "
                f"<b>Similarity:</b> {_esc(level)}",
                styles["Body"],
            ),
            Paragraph(
                f"<i>{_esc(_pairwise_observation(ratio))}</i>",
                styles["Body"],
            ),
            Spacer(1, SPACE_SM),
        ]
        elements.append(KeepTogether(block))

    return elements


def _build_categorical_recommendation(
    styles: dict[str, ParagraphStyle],
    winners: Winners,
) -> list[Any]:
    """
    Kategori bazlı öneri.

    Tek bir tokenizer'ı mutlak en iyi ilan etmek yerine kullanım
    senaryosuna göre kategori bazlı öneri üretir. `Winners` üzerinden
    çalışır; aynı seçimler text/markdown raporlarıyla tutarlı kalır.
    """
    elements: list[Any] = [
        Paragraph("Categorical Recommendation", styles["H2"]),
    ]

    if winners.best_balance is None:
        elements.append(
            Paragraph("No tokenizer recommendation available.", styles["Body"])
        )
        elements.append(Spacer(1, SPACE_MD))
        return elements

    elements.append(
        Paragraph(
            "Tokenizer choice depends on the target use case. "
            "For this input, the recommended options are:",
            styles["Body"],
        )
    )
    elements.append(Spacer(1, SPACE_XS))

    balanced = winners.best_balance
    balanced_metrics = get_metrics(balanced)

    rows: list[tuple[str, str]] = [
        (
            "Best balanced choice",
            f"{_name(balanced)} "
            f"(efficiency={format_number(balanced_metrics.get('efficiency_score'), 2)}, "
            f"latency={latency_microseconds(balanced_metrics)} {LATENCY_UNIT})",
        ),
    ]

    if winners.fastest:
        rows.append(
            (
                "Best for speed",
                f"{_name(winners.fastest)} "
                f"(latency={latency_microseconds(get_metrics(winners.fastest))} {LATENCY_UNIT})",
            )
        )

    if winners.best_compression:
        rows.append(
            (
                "Best for compression",
                f"{_name(winners.best_compression)} "
                f"(compression={format_number(_metric(winners.best_compression, 'compression_ratio'), 2)})",
            )
        )

    if winners.lowest_unknown:
        rows.append(
            (
                "Best for low unknown-token risk",
                f"{_name(winners.lowest_unknown)} "
                f"(unknown_rate={format_number(_metric(winners.lowest_unknown, 'unknown_rate'), 2)})",
            )
        )

    if winners.most_readable:
        rows.append(
            (
                "Best for readability/debugging",
                f"{_name(winners.most_readable)} "
                f"(tokens={_metric(winners.most_readable, 'token_count')})",
            )
        )

    for label, value in rows:
        elements.append(
            _bullet(f"<b>{_esc(label)}:</b> {_esc(value)}", styles["Bullet"])
        )

    # Balanced choice için ek metrik dump.
    elements.extend(
        [
            Spacer(1, SPACE_SM),
            Paragraph(
                f"<b>Balanced choice metrics: {_esc(_name(balanced))}</b>",
                styles["Body"],
            ),
            Spacer(1, SPACE_XS),
        ]
    )

    detail_pairs = [
        ("Efficiency Score", format_number(balanced_metrics.get("efficiency_score"), 2)),
        ("Compression Ratio", format_number(balanced_metrics.get("compression_ratio"), 2)),
        ("Unknown Rate", format_number(balanced_metrics.get("unknown_rate"), 2)),
        ("Latency", f"{latency_microseconds(balanced_metrics)} {LATENCY_UNIT}"),
    ]
    for label, value in detail_pairs:
        elements.append(
            _bullet(f"<b>{_esc(label)}:</b> {_esc(value)}", styles["Bullet"])
        )

    elements.append(Spacer(1, SPACE_SM))
    elements.append(
        Paragraph(
            "Use the balanced recommendation as the default. If the application has a "
            "strict priority such as speed, compression, readability, or robustness, "
            "prefer the category-specific recommendation above instead.",
            styles["Body"],
        )
    )
    elements.append(Spacer(1, SPACE_MD))
    return elements


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_pdf_report(
    compare_result: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """
    Tokenizer karşılaştırma sonuçlarından production-ready bir PDF
    raporu üretir.

    Bölümler `text_report.py` ile tutarlıdır: header, source text,
    executive summary, summary table, key insights, interpretation,
    recommendation, why-this-won, tokenizer details, overall ranking,
    pairwise comparisons.
    """
    text, total, results, pairwise = extract_compare_payload(compare_result)
    winners = compute_winners(results)

    output_path = Path(output_path)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="Tokenizer Evaluation Report",
    )

    page_width = A4[0] - doc.leftMargin - doc.rightMargin
    styles = _build_styles()

    sections: Iterable[list[Any]] = (
        _build_header(styles, total),
        _build_source_text(styles, text),
        _build_executive_summary(styles, results, winners),
        _build_summary_table(styles, results, text, page_width),
        _build_key_insights(styles, winners),
        _build_interpretation(styles, winners),
        _build_recommendation(styles, results, winners),
        _build_winner_explanation(styles, results),
        _build_tokenizer_details(styles, results),
        _build_overall_ranking(styles, results),
        _build_pairwise_comparisons(styles, pairwise),
        _build_categorical_recommendation(styles, winners),
    )

    flowables: list[Any] = []
    for section in sections:
        flowables.extend(section)

    doc.build(flowables)
    return output_path