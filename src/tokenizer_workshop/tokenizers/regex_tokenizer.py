from __future__ import annotations

import re

from tokenizer_workshop.tokenizers.base import BaseTokenizer


class RegexTokenizer(BaseTokenizer):
    """
    RegexTokenizer

    Bu tokenizer, metni regex (regular expression) kullanarak tokenize eder.

    Temel yaklaşım:
        - Kelimeleri (words)
        - Sayıları
        - Noktalama işaretlerini

        ayrı tokenlar olarak ayırır.

    Default regex pattern:
        r"\\w+|[^\\w\\s]"

    Bu pattern ne yapar?

        \\w+       → kelimeleri yakalar (harf, sayı, underscore)
        [^\\w\\s]  → kelime olmayan ve boşluk olmayan karakterleri yakalar (punctuation)

    Örnek:
        Input:
            "Hello world!"

        Output:
            ["Hello", "world", "!"]

    Not:
        Bu tokenizer:
            - whitespace bilgisini saklamaz
            - reconstruction (decode) sırasında boşlukları yeniden oluşturur
            - punctuation öncesindeki boşlukları temizler

    Mimari özellik:
        - BaseTokenizer kontratına uyar
        - registry decorator ile otomatik sisteme dahil edilir
        - strict davranır (train zorunlu)
    """

    def __init__(self, pattern: str | None = None) -> None:
        """
        RegexTokenizer instance oluşturur.

        Args:
            pattern:
                Custom regex pattern.

                Eğer verilmezse default pattern kullanılır:
                    r"\\w+|[^\\w\\s]"

        İç state:
            token_to_id:
                string token → integer id mapping

            id_to_token:
                integer id → string token mapping

            _is_trained:
                tokenizer'ın train edilip edilmediğini gösterir
        """
        super().__init__(name="regex")

        # Kullanıcı custom pattern verebilir, yoksa default kullanılır
        self.pattern = pattern or r"\w+|[^\w\s]"

        # Regex compile edilir (performans için önemli)
        self._compiled_pattern = re.compile(self.pattern)

        # Vocabulary mapping'leri
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

        # Train state flag
        self._is_trained = False

    def train(self, text: str) -> None:
        """
        Verilen metin üzerinden tokenizer vocabulary oluşturur.

        Adımlar:
            1. Metin tokenize edilir
            2. Unique tokenlar çıkarılır
            3. Deterministik olması için sıralanır
            4. token_to_id mapping oluşturulur
            5. id_to_token mapping oluşturulur

        Args:
            text:
                Eğitim metni

        Raises:
            ValueError:
                Eğer text boş veya sadece whitespace ise
        """
        if not text or not text.strip():
            raise ValueError("Training text cannot be empty.")

        # Metni regex ile tokenize et
        tokens = self._tokenize_to_strings(text)

        # Unique tokenları al ve sırala (deterministic mapping için)
        unique_tokens = sorted(set(tokens))

        # Token → id mapping
        self.token_to_id = {
            token: index
            for index, token in enumerate(unique_tokens)
        }

        # Id → token mapping (reverse mapping)
        self.id_to_token = {
            index: token
            for token, index in self.token_to_id.items()
        }

        # Train flag set edilir
        self._is_trained = True

    def encode(self, text: str) -> list[int]:
        """
        Metni integer token id listesine dönüştürür.

        Akış:
            1. Train edilmiş mi kontrol edilir
            2. Metin tokenize edilir
            3. Her token id'ye çevrilir

        Args:
            text:
                Encode edilecek metin

        Returns:
            list[int]:
                Token id listesi

        Raises:
            ValueError:
                - Eğer train edilmemişse
                - Eğer vocabulary dışında token varsa (OOV)
        """
        if not self._is_trained:
            raise ValueError("Tokenizer has not been trained yet.")

        tokens = self._tokenize_to_strings(text)

        token_ids: list[int] = []

        for token in tokens:
            if token not in self.token_to_id:
                raise ValueError(
                    f"Unknown token encountered during encoding: {token}"
                )

            token_ids.append(self.token_to_id[token])

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """
        Token id listesini tekrar metne dönüştürür.

        Akış:
            1. Train edilmiş mi kontrol edilir
            2. Her id token'a çevrilir
            3. Tokenlar birleştirilir

        Args:
            token_ids:
                Decode edilecek id listesi

        Returns:
            str:
                Oluşturulan metin

        Raises:
            ValueError:
                - Eğer train edilmemişse
                - Eğer bilinmeyen token id varsa
        """
        if not self._is_trained:
            raise ValueError("Tokenizer has not been trained yet.")

        tokens: list[str] = []

        for token_id in token_ids:
            if token_id not in self.id_to_token:
                raise ValueError(
                    f"Unknown token id encountered during decoding: {token_id}"
                )

            tokens.append(self.id_to_token[token_id])

        return self._join_tokens(tokens)

    def tokenize(self, text: str) -> list[str]:
        """
        Metni string token listesine dönüştürür.

        Not:
            Bu method train gerektirmez.
            Sadece regex segmentation yapar.

        Bu method özellikle:
            - CompareManager
            - Reporting
            - UI

        gibi katmanlarda kullanılır.

        Args:
            text:
                Tokenize edilecek metin

        Returns:
            list[str]:
                Token listesi
        """
        return self._tokenize_to_strings(text)

    @property
    def vocab_size(self) -> int:
        """
        Vocabulary boyutunu döndürür.

        Returns:
            int:
                Unique token sayısı
        """
        return len(self.token_to_id)

    # ---------------------------------------------------------
    # INTERNAL HELPERS
    # ---------------------------------------------------------

    def _tokenize_to_strings(self, text: str) -> list[str]:
        """
        Regex pattern kullanarak metni tokenlara ayırır.

        Bu method:
            - tokenize()
            - encode()

        tarafından ortak kullanılır.

        Args:
            text:
                Tokenize edilecek metin

        Returns:
            list[str]:
                Regex ile ayrılmış token listesi
        """
        if not text or not text.strip():
            return []

        return self._compiled_pattern.findall(text)

    @staticmethod
    def _join_tokens(tokens: list[str]) -> str:
        """
        Token listesini tekrar okunabilir metne dönüştürür.

        Adımlar:
            1. Tokenlar boşluk ile birleştirilir
            2. Noktalama işaretlerinden önceki boşluklar kaldırılır
            3. Açılış parantezlerinden sonraki boşluklar temizlenir

        Args:
            tokens:
                Birleştirilecek token listesi

        Returns:
            str:
                Oluşturulan metin
        """
        if not tokens:
            return ""

        text = " ".join(tokens)

        # "word !" -> "word!"
        text = re.sub(r"\s+([.,!?;:%\)\]\}])", r"\1", text)

        # "( word" -> "(word"
        text = re.sub(r"([\(\[\{])\s+", r"\1", text)

        return text