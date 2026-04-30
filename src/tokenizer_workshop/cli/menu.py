from __future__ import annotations

"""
cli/menu.py

CLI ekran çıktılarından sorumlu yardımcı fonksiyonları içerir.

Bu modül:
- terminal ekranını temizler
- standart başlık formatı basar
- ana menüyü gösterir
- tokenizer seçim menüsünü gösterir
- tokenizer config bilgisini okunabilir hale getirir

Not:
    Bu dosyada business logic bulunmaz.
    Sadece CLI presentation/output sorumluluğu vardır.
"""

import os
from collections.abc import Mapping
from typing import Any


SEPARATOR_WIDTH = 70
SEPARATOR_CHAR = "="


def clear_screen() -> None:
    """
    Terminal ekranını işletim sistemine uygun komutla temizler.

    Windows:
        cls

    macOS / Linux:
        clear
    """

    os.system("cls" if os.name == "nt" else "clear")


def print_separator(
    char: str = SEPARATOR_CHAR,
    width: int = SEPARATOR_WIDTH,
) -> None:
    """
    Standart ayırıcı çizgi basar.
    """

    print(char * width)


def print_header(title: str) -> None:
    """
    Standart CLI başlığı basar.
    """

    print_separator()
    print(title)
    print_separator()


def pause() -> None:
    """
    Kullanıcının terminal çıktısını okuyabilmesi için ekranı bekletir.
    """

    input("\nPress ENTER to continue...")


def display_main_menu() -> None:
    """
    Ana CLI menüsünü gösterir.
    """

    print_header("TOKENIZER WORKSHOP CLI")
    print("1. Use default comparison text")
    print("2. Enter custom comparison text")
    print("3. Exit")
    print_separator()


def display_tokenizer_menu(
    tokenizer_configs: Mapping[str, Mapping[str, Any]],
) -> None:
    """
    Kullanılabilir tokenizer'ları numaralı şekilde gösterir.

    Args:
        tokenizer_configs:
            Tokenizer adlarını ve ilgili constructor/config değerlerini
            içeren mapping.

    Örnek:
        {
            "word": {},
            "byte_bpe": {"num_merges": 10},
            "ngram": {"n": 2},
        }
    """

    print_header("SELECT TOKENIZERS")

    for index, (name, config) in enumerate(tokenizer_configs.items(), start=1):
        readable_config = format_config(config)
        print(f"{index:>2}. {name:<18} config: {readable_config}")

    print_separator()
    print("Selection examples:")
    print("  1,2,3")
    print("  4,7,10")
    print("  all")
    print_separator()


def format_config(config: Mapping[str, Any]) -> str:
    """
    Tokenizer config bilgisini okunabilir string formatına dönüştürür.

    Args:
        config:
            Tokenizer'a ait constructor/config değerleri.

    Returns:
        Config boşsa "-".
        Doluysa "key=value" çiftlerinden oluşan okunabilir metin.
    """

    if not config:
        return "-"

    return ", ".join(
        f"{key}={value!r}"
        for key, value in config.items()
    )