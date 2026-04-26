"""
tokenizer_factory.py

Tokenizer üretiminden sorumlu merkezi factory sınıfı.
"""

from __future__ import annotations

from typing import Any

from tokenizer_workshop.tokenizers.registry import TokenizerRegistry
from tokenizer_workshop.tokenizers.discovery import auto_import_tokenizers

from tokenizer_workshop.api.services.exceptions import UnsupportedTokenizerError
from tokenizer_workshop.tokenizers.byte_bpe_tokenizer import ByteBPETokenizer
from tokenizer_workshop.tokenizers.byte_tokenizer import ByteTokenizer
from tokenizer_workshop.tokenizers.char_tokenizer import CharTokenizer
from tokenizer_workshop.tokenizers.simple_bpe_tokenizer import SimpleBPETokenizer
from tokenizer_workshop.tokenizers.word_tokenizer import WordTokenizer
from tokenizer_workshop.tokenizers.regex_tokenizer import RegexTokenizer
from tokenizer_workshop.tokenizers.regex_bpe_tokenizer import RegexBPETokenizer



class TokenizerFactory:
    """
    String olarak gelen tokenizer adını uygun tokenizer nesnesine dönüştürür.
    """

    @staticmethod
    def get_registry() -> dict[str, Any]:
        """
        Sistemin desteklediği tokenizer kayıtlarını döndürür.
        """
        return {
            "char": CharTokenizer(),
            "byte": ByteTokenizer(),
            "bpe": SimpleBPETokenizer(),
            "byte_bpe": ByteBPETokenizer(),
            "word": WordTokenizer(),
            "regex": RegexTokenizer(),
            "regex_bpe": RegexBPETokenizer(),
        }

    @staticmethod
    def get_supported_tokenizers() -> list[str]:
        """
        Desteklenen tokenizer adlarını döndürür.
        """
        return list(TokenizerFactory.get_registry().keys())

    @staticmethod
    def normalize_name(name: str) -> str:
        """
        Tokenizer adını standart forma dönüştürür.
        """
        return name.strip().lower()

    @staticmethod
    def create(name: str) -> Any:
        """
        Verilen tokenizer adına karşılık gelen tokenizer nesnesini döndürür.
        """
        normalized_name = TokenizerFactory.normalize_name(name)
        registry = TokenizerFactory.get_registry()

        tokenizer = registry.get(normalized_name)
        if tokenizer is None:
            supported_tokenizers = ", ".join(TokenizerFactory.get_supported_tokenizers())
            raise UnsupportedTokenizerError(
                f"Unsupported tokenizer: '{name}'. "
                f"Supported tokenizers: {supported_tokenizers}"
            )

        return tokenizer