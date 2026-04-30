from __future__ import annotations

"""
cli/controller.py

Tokenizer Workshop CLI akışını yöneten controller modülü.

Bu dosyanın sorumluluğu:
- ana CLI döngüsünü çalıştırmak
- kullanıcıdan text seçimini almak
- tokenizer seçimini almak
- seçilen akışı runner'a devretmek

Not:
    Bu dosyada comparison business logic bulunmaz.
    Comparison execution sorumluluğu CLIComparisonRunner içindedir.
"""

from typing import Any

from tokenizer_workshop.api.services.tokenizer_factory import TokenizerFactory
from tokenizer_workshop.cli.input import (
    parse_tokenizer_selection,
    read_custom_text,
    read_menu_choice,
)
from tokenizer_workshop.cli.menu import (
    clear_screen,
    display_main_menu,
    display_tokenizer_menu,
    pause,
)
from tokenizer_workshop.cli.runner import CLIComparisonRunner
from tokenizer_workshop.tokenizers.base import BaseTokenizer


class CLIController:
    """
    Tokenizer Workshop CLI kullanıcı akışını yöneten controller.

    Sorumluluklar:
        - ana menüyü göstermek
        - kullanıcı seçimini okumak
        - default/custom text akışını yönetmek
        - tokenizer seçimini almak
        - comparison çalıştırma işini runner'a devretmek

    Bu sınıf tokenization veya comparison logic içermez.
    """

    def __init__(
        self,
        runner: CLIComparisonRunner,
        tokenizer_config: dict[str, dict[str, Any]],
        default_compare_text: str,
    ) -> None:
        self.runner = runner
        self.tokenizer_config = tokenizer_config
        self.default_compare_text = default_compare_text
        self.all_tokenizers = self._build_tokenizers()

    def run(self) -> None:
        """
        Ana CLI loop'unu başlatır.
        """

        while True:
            clear_screen()
            display_main_menu()

            choice = read_menu_choice()

            if choice == "1":
                self._run_default_text_flow()

            elif choice == "2":
                self._run_custom_text_flow()

            elif choice == "3":
                print("\nExiting Tokenizer Workshop CLI.")
                break

            else:
                print("\nInvalid option. Please select 1, 2, or 3.")
                pause()

    def _run_default_text_flow(self) -> None:
        """
        Default comparison text ile karşılaştırma akışını başlatır.
        """

        print("\nDefault comparison text selected.\n")

        self._run_compare_flow(
            text=self.default_compare_text,
        )

        pause()

    def _run_custom_text_flow(self) -> None:
        """
        Kullanıcıdan custom text alır ve karşılaştırma akışını başlatır.
        """

        comparison_text = read_custom_text(
            default_text=self.default_compare_text,
        )

        print("\nCustom comparison text selected.\n")

        self._run_compare_flow(
            text=comparison_text,
        )

        pause()

    def _run_compare_flow(self, text: str) -> None:
        """
        Text belirlendikten sonra tokenizer seçimini alır
        ve comparison execution işini runner'a devreder.
        """

        selected_tokenizers = self._select_tokenizers()

        self.runner.run(
            text=text,
            tokenizers=selected_tokenizers,
        )

    def _select_tokenizers(self) -> dict[str, BaseTokenizer]:
        """
        Kullanıcıdan tokenizer seçimi alır ve seçilen tokenizer nesnelerini döndürür.
        """

        tokenizer_names = list(self.all_tokenizers.keys())

        display_tokenizer_menu(self.tokenizer_config)

        raw_selection = input("\nSelect tokenizer numbers: ")

        selected_names = parse_tokenizer_selection(
            raw_input=raw_selection,
            tokenizer_names=tokenizer_names,
        )

        return {
            name: self.all_tokenizers[name]
            for name in selected_names
        }

    def _build_tokenizers(self) -> dict[str, BaseTokenizer]:
        """
        Config üzerinden tokenizer instance'larını üretir.

        Yeni tokenizer eklemek için:
            1. Tokenizer registry/factory tarafında tokenizer'ı tanıt
            2. compare.py içindeki TOKENIZER_CONFIG'e config ekle
        """

        return {
            name: TokenizerFactory.create(name, **config)
            for name, config in self.tokenizer_config.items()
        }