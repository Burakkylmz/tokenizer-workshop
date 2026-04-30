from __future__ import annotations

"""
cli/input.py

CLI kullanıcı input işlemlerinden sorumlu yardımcı fonksiyonları içerir.

Bu modül:
- ana menü seçimini okur
- custom comparison text alır
- tokenizer seçim input'unu parse eder

Not:
    Bu dosyada comparison logic bulunmaz.
    Sadece input okuma ve input parsing sorumluluğu vardır.
"""

from collections.abc import Sequence


ALL_SELECTION_KEYWORD = "all"


def read_menu_choice() -> str:
    """
    Ana menüden kullanıcı seçimini okur.

    Returns:
        Kullanıcının normalize edilmiş menü seçimi.
    """

    return input("\nSelect option: ").strip()


def read_custom_text(default_text: str) -> str:
    """
    Kullanıcıdan comparison text alır.

    Boş input girilirse default text döner.

    Args:
        default_text:
            Kullanıcı boş input girerse kullanılacak fallback metin.

    Returns:
        Kullanıcının girdiği text veya default text.
    """

    print("\nEnter text to compare.")
    print("Leave empty to use default text.")

    text = input("\n> ").strip()

    if not text:
        print("\nNo text entered. Default text will be used.")
        return default_text

    return text


def parse_tokenizer_selection(
    raw_input: str,
    tokenizer_names: Sequence[str],
) -> list[str]:
    """
    Kullanıcının tokenizer seçim input'unu tokenizer isimlerine dönüştürür.

    Desteklenen input formatları:
        all
        1
        1,2,3
        2, 5, 8

    Geçersiz, boş veya sonuç üretmeyen input durumunda tüm tokenizer'lar döner.

    Args:
        raw_input:
            Kullanıcının terminalden girdiği ham seçim metni.

        tokenizer_names:
            Menüde gösterilen tokenizer isimleri.
            Index hesaplaması bu sıraya göre yapılır.

    Returns:
        Seçilen tokenizer isimleri.
        Fallback olarak tüm tokenizer isimleri döner.
    """

    names = list(tokenizer_names)

    if not names:
        return []

    normalized_input = raw_input.strip().lower()

    if not normalized_input:
        return names

    if normalized_input == ALL_SELECTION_KEYWORD:
        return names

    selected_names = _parse_numeric_selection(
        raw_input=normalized_input,
        tokenizer_names=names,
    )

    return selected_names or names


def _parse_numeric_selection(
    raw_input: str,
    tokenizer_names: Sequence[str],
) -> list[str]:
    """
    Virgülle ayrılmış numerik tokenizer seçimlerini parse eder.

    Örnek:
        raw_input = "1,3"
        tokenizer_names = ["word", "char", "byte"]

        output = ["word", "byte"]
    """

    selected_names: list[str] = []

    try:
        selected_indices = _parse_indices(raw_input)
    except ValueError:
        return []

    for index in selected_indices:
        if 0 <= index < len(tokenizer_names):
            selected_names.append(tokenizer_names[index])

    return _deduplicate_preserving_order(selected_names)


def _parse_indices(raw_input: str) -> list[int]:
    """
    Kullanıcı input'unu zero-based index listesine dönüştürür.

    Örnek:
        "1,2,3" -> [0, 1, 2]
    """

    return [
        int(value.strip()) - 1
        for value in raw_input.split(",")
        if value.strip()
    ]


def _deduplicate_preserving_order(items: Sequence[str]) -> list[str]:
    """
    Tekrarlı seçimleri sıralamayı bozmadan temizler.

    Örnek:
        ["word", "word", "char"] -> ["word", "char"]
    """

    seen: set[str] = set()
    unique_items: list[str] = []

    for item in items:
        if item not in seen:
            seen.add(item)
            unique_items.append(item)

    return unique_items