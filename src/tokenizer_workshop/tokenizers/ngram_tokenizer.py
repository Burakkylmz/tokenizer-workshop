from __future__ import annotations

from typing import List

from tokenizer_workshop.tokenizers.base import BaseTokenizer
from tokenizer_workshop.tokenizers.registry import register_tokenizer


@register_tokenizer("ngram")
class NgramTokenizer(BaseTokenizer):
    """
    NgramTokenizer

    N-gram tabanlı tokenizer implementasyonudur.

    Bu tokenizer, metni önce whitespace bazlı kelimelere ayırır.
    Daha sonra ardışık n adet kelimeyi birleştirerek token üretir.

    N-gram nedir?
        N-gram, ardışık n birimden oluşan token grubudur.

        n = 1 → unigram
            "the cat sat" -> ["the", "cat", "sat"]

        n = 2 → bigram
            "the cat sat" -> ["the cat", "cat sat"]

        n = 3 → trigram
            "the cat sat on" -> ["the cat sat", "cat sat on"]

    Bu tokenizer özellikle:
        - bağlam bazlı token analizi
        - kelime grubu örüntülerini inceleme
        - tokenizer karşılaştırmalarında context-aware baseline oluşturma
        - BPE / word / regex tokenizer davranışlarını kıyaslama

    için kullanışlıdır.

    Not:
        Bu implementasyon intentionally simple tutulmuştur.
        Punctuation normalization, lowercasing veya gelişmiş preprocessing yapmaz.
        Amaç, n-gram mantığını sade ve anlaşılır şekilde projeye eklemektir.
    """

    def __init__(self, n: int = 2) -> None:
        """
        NgramTokenizer instance oluşturur.

        Args:
            n:
                Kaç kelimelik gruplar oluşturulacağını belirler.

                Örnek:
                    n=1 -> unigram
                    n=2 -> bigram
                    n=3 -> trigram

        Raises:
            ValueError:
                n değeri 1'den küçükse fırlatılır.

        Internal state:
            _token_to_id:
                N-gram string tokenlarını integer id değerlerine map eder.

            _id_to_token:
                Integer id değerlerini tekrar n-gram string tokenlarına map eder.

            _trained:
                Tokenizer'ın train edilip edilmediğini takip eder.
        """
        super().__init__(name="ngram")

        # n en az 1 olmalıdır.
        # n=0 veya negatif değerler anlamlı bir n-gram üretmez.
        if n < 1:
            raise ValueError("n must be at least 1")

        self.n = n

        # Token -> id mapping.
        #
        # Örnek:
        # {
        #     "the cat": 0,
        #     "cat sat": 1
        # }
        self._token_to_id: dict[str, int] = {}

        # Id -> token reverse mapping.
        #
        # Decode işlemi sırasında id listesini tekrar string tokenlara çevirmek
        # için kullanılır.
        self._id_to_token: dict[int, str] = {}

        # train() çağrılmadan encode/decode yapılmasını engellemek için flag.
        self._trained = False

    def train(self, text: str) -> None:
        """
        Verilen text üzerinden n-gram vocabulary oluşturur.

        İş akışı:
            1. Input text boş mu kontrol edilir.
            2. Metin whitespace bazlı kelimelere ayrılır.
            3. Kelimelerden n-gram listesi oluşturulur.
            4. Tekrar eden n-gram'lar kaldırılır.
            5. Deterministik sırayla token_to_id mapping oluşturulur.
            6. Reverse mapping olarak id_to_token oluşturulur.
            7. Tokenizer trained state'e geçirilir.

        Args:
            text:
                Vocabulary oluşturmak için kullanılacak eğitim metni.

        Raises:
            ValueError:
                text boş veya sadece whitespace ise fırlatılır.
        """
        if not text or not text.strip():
            raise ValueError("Training text cannot be empty")

        # Basit whitespace tokenization.
        #
        # Örnek:
        # "the cat sat" -> ["the", "cat", "sat"]
        words = text.split()

        # Kelime listesinden n-gram tokenları oluşturulur.
        #
        # n=2 için:
        # ["the", "cat", "sat"] -> ["the cat", "cat sat"]
        ngrams = self._build_ngrams(words)

        # dict.fromkeys(...) insertion order korur.
        # Böylece tekrar eden n-gram'lar temizlenir ama ilk görülme sırası korunur.
        #
        # Bu deterministic davranış sağlar:
        # aynı input -> aynı vocabulary mapping
        unique_tokens = list(dict.fromkeys(ngrams))

        # N-gram tokenlarına integer id atanır.
        self._token_to_id = {
            token: index
            for index, token in enumerate(unique_tokens)
        }

        # Decode için reverse mapping oluşturulur.
        self._id_to_token = {
            index: token
            for token, index in self._token_to_id.items()
        }

        self._trained = True

    def encode(self, text: str) -> List[int]:
        """
        Verilen text'i n-gram token id listesine dönüştürür.

        Args:
            text:
                Encode edilecek metin.

        Returns:
            List[int]:
                N-gram token id listesi.

        Raises:
            ValueError:
                - Tokenizer train edilmemişse
                - Encode edilen text içinde vocabulary'de olmayan n-gram varsa

        Örnek:
            train text:
                "the cat sat"

            vocabulary:
                {
                    "the cat": 0,
                    "cat sat": 1
                }

            encode("the cat sat"):
                [0, 1]
        """
        if not self._trained:
            raise ValueError("Tokenizer has not been trained yet")

        words = text.split()
        ngrams = self._build_ngrams(words)

        token_ids: list[int] = []

        for token in ngrams:
            # Strict tokenizer davranışı:
            # Eğitim sırasında görülmeyen n-gram encode edilmez.
            #
            # Bu, diğer tokenizer'larla aynı kontratı korur.
            if token not in self._token_to_id:
                raise ValueError(f"Unknown token: {token}")

            token_ids.append(self._token_to_id[token])

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Token id listesini tekrar okunabilir n-gram string'lerine dönüştürür.

        Args:
            token_ids:
                Decode edilecek integer token id listesi.

        Returns:
            str:
                N-gram string'lerinin birleştirilmiş hali.

        Raises:
            ValueError:
                - Tokenizer train edilmemişse
                - Bilinmeyen token id verilirse

        Önemli not:
            N-gram tokenization overlap içerir.

            Örnek:
                input:
                    "the cat sat"

                bigram tokenları:
                    ["the cat", "cat sat"]

                Basit decode:
                    "the cat cat sat"

            Bu yüzden decode çıktısı her zaman orijinal text ile birebir aynı
            olmak zorunda değildir. Bu davranış bilinçli olarak sade tutulmuştur.
        """
        if not self._trained:
            raise ValueError("Tokenizer has not been trained yet")

        tokens: list[str] = []

        for token_id in token_ids:
            if token_id not in self._id_to_token:
                raise ValueError("Unknown token id encountered")

            tokens.append(self._id_to_token[token_id])

        # Basit reconstruction:
        # Overlap çözümü yapmadan n-gram string'lerini boşlukla birleştirir.
        #
        # Örnek:
        # ["the cat", "cat sat"] -> "the cat cat sat"
        return " ".join(tokens)

    def tokenize(self, text: str) -> list[str]:
        """
        Metni string n-gram token listesine dönüştürür.

        Bu method CompareManager, API ve reporting katmanları için kullanışlıdır.
        encode() gibi integer id döndürmez; doğrudan okunabilir token listesi verir.

        Not:
            Bu method train gerektirmez.
            Sadece segmentation yapar.

        Args:
            text:
                Tokenize edilecek metin.

        Returns:
            list[str]:
                N-gram string token listesi.
        """
        if not text or not text.strip():
            return []

        words = text.split()
        return self._build_ngrams(words)

    @property
    def vocab_size(self) -> int:
        """
        Vocabulary boyutunu döndürür.

        Returns:
            int:
                Eğitim sonrası öğrenilen unique n-gram token sayısı.
        """
        return len(self._token_to_id)

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------

    def _build_ngrams(self, tokens: list[str]) -> list[str]:
        """
        Kelime listesinden n-gram tokenları üretir.

        Args:
            tokens:
                Whitespace ile ayrılmış kelime listesi.

        Returns:
            list[str]:
                N-gram token listesi.

        Davranış:
            Eğer token sayısı n değerinden küçükse, mevcut tokenlar döndürülür.

            Örnek:
                tokens = ["hello"]
                n = 2

                output:
                    ["hello"]

            Bu fallback sayesinde kısa inputlarda boş sonuç üretmek yerine
            anlamlı bir token listesi korunur.
        """
        if len(tokens) < self.n:
            return tokens

        # Sliding window mantığı:
        #
        # tokens = ["the", "cat", "sat"]
        # n = 2
        #
        # i=0 -> tokens[0:2] -> ["the", "cat"] -> "the cat"
        # i=1 -> tokens[1:3] -> ["cat", "sat"] -> "cat sat"
        return [
            " ".join(tokens[index : index + self.n])
            for index in range(len(tokens) - self.n + 1)
        ]