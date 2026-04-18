from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    """Projede yer alan tüm tokenizer'lar için ortak arayüzü tanımlar."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def train(self, text: str) -> None:
        """Ham metinden tokenizer durumunu / bilgisini oluşturur."""
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Metni token id listesine dönüştürür."""
        raise NotImplementedError

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """Token id listesini tekrar metne dönüştürür."""
        raise NotImplementedError

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Mevcut vocabulary boyutunu döndürür."""
        raise NotImplementedError