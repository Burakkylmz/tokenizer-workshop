"""
tokenize.py

Bu modül, tokenization ve tokenizer karşılaştırma işlemleri ile ilgili HTTP endpoint'lerini içerir.

Route katmanının sorumluluğu:
- HTTP request almak
- request model validation sonrası service katmanını çağırmak
- uygun response modelini döndürmek
- service exception'larını HTTP seviyesine map etmek

Bu dosyada business logic tutulmaz.
Asıl uygulama davranışı service katmanında yaşar.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from pathlib import Path
import tempfile

from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from tokenizer_workshop.api.reports import (
    build_markdown_report,
    build_pdf_report,
    get_report_builder,
)

from tokenizer_workshop.api.reports import get_report_builder
from tokenizer_workshop.api.schemas import (
    CompareRequest,
    CompareResponse,
    ReportRequest,
    ReportResponse,
    TokenizeRequest,
    TokenizeResponse,
    TokenizerComparisonResult,
)
from tokenizer_workshop.api.services import (
    TokenizationServiceError,
    TokenizerFactory,
    UnsupportedTokenizerError,
    compare_tokenizers,
    evaluate_tokenizers,
    tokenize_text,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["tokenization"],
)


# ---------------------------------------------------------
# Report Endpoint
# ---------------------------------------------------------
@router.post(
    "/report",
    response_model=ReportResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate downloadable tokenizer evaluation report",
    description=(
        "Seçilen tokenizer'ları detaylı evaluation akışıyla çalıştırır ve "
        "metrik/pairwise comparison içeren indirilebilir rapor üretir."
    ),
    response_description="Metrik içeren tokenizer evaluation raporu",
)
def generate_report(request: ReportRequest) -> ReportResponse:
    """
    Detaylı evaluation sonucundan indirilebilir rapor üretir.

    Not:
        Rapor çıktısı her zaman evaluate_tokenizers(...) üzerinden üretilir.
        Böylece raporda metrikler ve pairwise comparison sonuçları bulunur.
    """
    try:
        analysis_result = evaluate_tokenizers(
            text=request.text,
            tokenizer_names=request.tokenizer_names,
        )

        report_source = {
            "text": analysis_result.get("source_text", request.text),
            "total_tokenizers": len(analysis_result.get("evaluations", [])),
            "results": [
                {
                    "tokenizer_name": item.get("tokenizer_name"),
                    "tokens": item.get("tokens", []),
                    "token_count": item.get("metrics", {}).get("token_count", 0),
                    "vocab_size": item.get("metrics", {}).get("unique_token_count", 0),
                    "metrics": item.get("metrics", {}),
                }
                for item in analysis_result.get("evaluations", [])
            ],
            "pairwise_comparisons": analysis_result.get("pairwise_comparisons", []),
        }

        builder, filename = get_report_builder(request.format)
        report = builder(report_source)

        return ReportResponse(
            report=report,
            format=request.format,
            filename=filename,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except UnsupportedTokenizerError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except TokenizationServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected error occurred while generating report.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while generating the report.",
        ) from exc


# ---------------------------------------------------------
# Tokenize Endpoint
# ---------------------------------------------------------
@router.post(
    "/tokenize",
    response_model=TokenizeResponse,
    status_code=status.HTTP_200_OK,
    summary="Tokenize text with a selected tokenizer",
    description=(
        "İstemciden gelen metni seçilen tokenizer ile işler ve "
        "tokenization sonucunu standartlaştırılmış bir response modeli ile döndürür."
    ),
    response_description="Başarılı tokenization sonucu",
)
def tokenize(request: TokenizeRequest) -> TokenizeResponse:
    """
    İstemciden gelen metni seçilen tokenizer ile tokenize eder.
    """
    try:
        result = tokenize_text(
            text=request.text,
            tokenizer_name=request.tokenizer_name,
        )
        return TokenizeResponse(**result)

    except UnsupportedTokenizerError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except TokenizationServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected error occurred while processing tokenize request.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the tokenization request.",
        ) from exc


# ---------------------------------------------------------
# Compare Endpoint
# ---------------------------------------------------------
@router.post(
    "/compare",
    response_model=CompareResponse,
    status_code=status.HTTP_200_OK,
    summary="Compare multiple tokenizers on the same text",
    description=(
        "Aynı ham metni bir veya birden fazla tokenizer ile işler ve "
        "sonuçları karşılaştırmalı biçimde döndürür."
    ),
    response_description="Çoklu tokenizer karşılaştırma sonucu",
)
def compare(request: CompareRequest) -> CompareResponse:
    """
    Aynı metni seçilen bir veya birden fazla tokenizer ile işler ve
    toplu karşılaştırma sonucunu döndürür.
    """
    try:
        result = compare_tokenizers(
            text=request.text,
            tokenizer_names=request.tokenizer_names,
        )
        return CompareResponse(**result)

    except UnsupportedTokenizerError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except TokenizationServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected error occurred while processing compare request.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the compare request.",
        ) from exc


# ---------------------------------------------------------
# Evaluate Endpoint
# ---------------------------------------------------------
@router.post(
    "/evaluate",
    response_model=TokenizerComparisonResult,
    status_code=status.HTTP_200_OK,
    summary="Run advanced tokenizer evaluation",
    description=(
        "Aynı metni birden fazla tokenizer ile çalıştırır, detaylı metrikler "
        "ve pairwise comparison sonuçları üretir."
    ),
    response_description="Detaylı tokenizer evaluation sonucu",
)
def evaluate(request: CompareRequest) -> TokenizerComparisonResult:
    """
    Aynı metni birden fazla tokenizer ile değerlendirir ve
    detaylı analiz sonucunu döndürür.
    """
    try:
        result = evaluate_tokenizers(
            text=request.text,
            tokenizer_names=request.tokenizer_names,
        )
        return TokenizerComparisonResult(**result)

    except UnsupportedTokenizerError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except TokenizationServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected error occurred while processing evaluation request.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the evaluation request.",
        ) from exc


# ---------------------------------------------------------
# Get Available Tokenizers Endpoint
# ---------------------------------------------------------
@router.get(
    "/tokenizers",
    status_code=status.HTTP_200_OK,
    summary="List available tokenizers",
    description=(
        "Sistem tarafından desteklenen tokenizer türlerini döndürür. "
        "Bu endpoint, özellikle frontend uygulamaların dinamik olarak "
        "tokenizer seçim listelerini oluşturması için kullanılır."
    ),
    response_description="Desteklenen tokenizer'ların listesi ve metadata bilgisi",
)
def get_tokenizers() -> dict[str, object]:
    """
    Desteklenen tokenizer türlerini ve ilgili metadata bilgilerini döndürür.
    """
    try:
        tokenizers = TokenizerFactory.get_supported_tokenizers()

        return {
            "available_tokenizers": tokenizers,
            "count": len(tokenizers),
            "notes": "Tokenizer isimleri case-insensitive olarak kullanılabilir.",
        }

    except Exception as exc:
        logger.exception("Failed to fetch supported tokenizers.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch tokenizer list.",
        ) from exc


@router.post(
    "/report/pdf",
    status_code=status.HTTP_200_OK,
    summary="Generate PDF tokenizer report",
)
def generate_pdf_report_endpoint(request: ReportRequest) -> FileResponse:
    try:
        analysis_result = evaluate_tokenizers(
            text=request.text,
            tokenizer_names=request.tokenizer_names,
        )

        report_source = {
            "text": analysis_result.get("source_text", request.text),
            "total_tokenizers": len(analysis_result.get("evaluations", [])),
            "results": [
                {
                    "tokenizer_name": item.get("tokenizer_name"),
                    "tokens": item.get("tokens", []),
                    "token_count": item.get("metrics", {}).get("token_count", 0),
                    "vocab_size": item.get("metrics", {}).get("unique_token_count", 0),
                    "metrics": item.get("metrics", {}),
                }
                for item in analysis_result.get("evaluations", [])
            ],
            "pairwise_comparisons": analysis_result.get("pairwise_comparisons", []),
        }

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as output:
            pdf_path = Path(output.name)
        try:
            build_pdf_report(report_source, pdf_path)
        except Exception:
            pdf_path.unlink(missing_ok=True)
            raise

        return FileResponse(
            path=pdf_path,
            filename="tokenizer_report.pdf",
            media_type="application/pdf",
            background=BackgroundTask(pdf_path.unlink, missing_ok=True),
        )

    except Exception as exc:
        logger.exception("PDF generation failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF generation failed: {str(exc)}",
        ) from exc
