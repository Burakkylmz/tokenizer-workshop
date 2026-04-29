from __future__ import annotations

"""
cli/runner.py

CLI comparison execution katmanını içerir.

Bu modülün sorumluluğu:
- seçili tokenizer'ları train etmek
- comparison pipeline'ını çalıştırmak
- sonucu terminale yazdırmak
- raporu belirtilen path'e kaydetmek

Not:
    Kullanıcı input'u, menü akışı veya tokenizer seçimi burada değildir.
    Bunlar CLIController sorumluluğundadır.
"""

from collections.abc import Mapping

from tokenizer_workshop.cli.menu import print_header
from tokenizer_workshop.comparisons.compare_manager import CompareManager
from tokenizer_workshop.tokenizers.base import BaseTokenizer


DEFAULT_REPORT_PATH = "report.md"


class CLIComparisonRunner:
    """
    CLI üzerinden tokenizer comparison pipeline'ını çalıştıran runner.

    Bu sınıf execution sorumluluğunu üstlenir.
    Flow yönetmez, input okumaz, menü göstermez.
    """

    def __init__(
        self,
        manager: CompareManager,
        train_text: str,
        report_path: str = DEFAULT_REPORT_PATH,
    ) -> None:
        self.manager = manager
        self.train_text = train_text
        self.report_path = report_path

    def run(
        self,
        text: str,
        tokenizers: Mapping[str, BaseTokenizer],
    ) -> None:
        """
        Seçili tokenizer'larla comparison pipeline'ını çalıştırır.

        Args:
            text:
                Tokenizer'ların karşılaştırılacağı kaynak metin.

            tokenizers:
                Tokenizer adı -> tokenizer instance mapping'i.
        """

        if not self._is_valid_text(text):
            print("\nComparison text cannot be empty.")
            return

        if not tokenizers:
            print("\nAt least one tokenizer must be selected.")
            return

        print_header("RUNNING TOKENIZER COMPARISON")

        self._train_tokenizers(tokenizers)
        result = self._compare_tokenizers(text, tokenizers)
        self._print_and_save_result(result)

    def _is_valid_text(self, text: str) -> bool:
        """
        Comparison text'in çalıştırılabilir olup olmadığını kontrol eder.
        """

        return bool(text.strip())

    def _train_tokenizers(
        self,
        tokenizers: Mapping[str, BaseTokenizer],
    ) -> None:
        """
        Train destekleyen tokenizer'ları eğitim metni ile hazırlar.
        """

        print("Training supported tokenizers...")

        self.manager.train_tokenizers(
            tokenizers=dict(tokenizers),
            train_text=self.train_text,
        )

    def _compare_tokenizers(
        self,
        text: str,
        tokenizers: Mapping[str, BaseTokenizer],
    ):
        """
        Seçili tokenizer'ları aynı text üzerinde karşılaştırır.
        """

        print("Comparing tokenizers...")

        return self.manager.compare_multiple(
            text=text,
            tokenizers=dict(tokenizers),
        )

    def _print_and_save_result(self, result) -> None:
        """
        Comparison sonucunu terminale yazdırır ve rapor olarak kaydeder.
        """

        print("\nComparison result:\n")

        self.manager.print_comparison_result(
            result,
            save_path=self.report_path,
        )

        print(f"\nReport saved to: {self.report_path}")