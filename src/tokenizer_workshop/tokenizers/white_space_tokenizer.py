from __future__ import annotations

from tokenizer_workshop.tokenizers.base import BaseTokenizer
from tokenizer_workshop.tokenizers.registry import register_tokenizer


@register_tokenizer("white_space")
class WhitespaceTokenizer(BaseTokenizer):
    """
    Whitespace tabanlı tokenizer.

    Bu tokenizer, metni Python'un built-in str.split() davranışını kullanarak
    whitespace karakterlerine göre böler.

    WhitespaceTokenizer en basit tokenizer türlerinden biridir ve genellikle
    diğer tokenizer'larla karşılaştırma yapabilmek için baseline olarak kullanılır.

    - Bu tokenizer metni yalnızca boşluk karakterlerine göre böler.
    - En basit ve en hızlı tokenization yaklaşımıdır.

    Örnek:
        Input:
            "hello world tokenizer"

        Output:
            ["hello", "world", "tokenizer"]

    Önemli davranış:
        Noktalama işaretleri ayrı token yapılmaz.

        Input:
            "hello, world!"

        Output:
            ["hello,", "world!"]

    Kullanım amacı:
        - Baseline tokenizer sağlamak
        - Word/Regex/Punctuation/Subword tokenizer'larla karşılaştırma yapmak
        - Basit token-count analizleri yapmak
        - Debug ve eğitim amaçlı örnekler üretmek

    Sınırlamalar:
        - Noktalama işaretlerini ayırmaz
        - Subword tokenization yapmaz
        - Morphological analysis yapmaz
        - Whitespace türlerini normalize ederek ele alır
        - Orijinal whitespace biçimini birebir korumaz
        - Unicode normalization içermez

    Bu tokenizer:
        - deterministiktir
        - stateless çalışır
        - training gerektirmez
            - ancak train() metodu opsiyoneldir ve token-id mapping oluşturmak için kullanılabilir
    """

    def __init__(self) -> None:
        """
        WhitespaceTokenizer instance'ını başlatır.

        Bu tokenizer tokenize() için state'e ihtiyaç duymaz.
        Ancak encode/decode akışı için token -> id ve id -> token
        mapping tabloları tutulur.

        _trained flag'i:
            encode/decode çağrılarının eğitimden önce çalışmasını engeller.
        """
        super().__init__(name="whitespace")

        # WordTokenizer gibi bir vocab tutmak istersek kullanılabilir
        # Token string'lerini integer id değerlerine map eder.
        # Örnek: {"hello": 0, "world": 1}
        self._token_to_id: dict[str, int] = {}

        # Integer id değerlerini tekrar token string'lerine map eder.
        # Örnek: {0: "hello", 1: "world"}
        self._id_to_token: dict[int, str] = {}

        # train() çağrılıp çağrılmadığını takip eder.
        # encode/decode için vocabulary hazır olmalıdır.
        self._trained = False

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------

    def train(self, text: str) -> None:
        """
        Eğitim metninden vocabulary oluşturur.

        WhitespaceTokenizer için training algoritmik olarak karmaşık değildir.
        Buradaki amaç:
            - metni whitespace tokenlarına ayırmak
            - unique tokenları bulmak
            - token -> id mapping oluşturmak
            - id -> token mapping oluşturmak

        Örnek:
            Input:
                "hello world hello"

            Tokenlar:
                ["hello", "world", "hello"]

            Unique tokenlar:
                ["hello", "world"]

            Mapping:
                {"hello": 0, "world": 1}

        Neden dict.fromkeys kullanıyoruz?
            Python dict insertion order koruduğu için tokenların ilk görülme
            sırasını kaybetmeden duplicate değerleri temizler.

        Raises:
            ValueError:
                Eğitim metni boşsa veya sadece whitespace içeriyorsa.
        """
        # Boş veya sadece whitespace içeren input eğitim için anlamlı değildir.
        # Çünkü vocabulary oluşturmak için en az bir gerçek token gerekir.
        if not text or not text.strip():
            raise ValueError("Training text cannot be empty")

        # Metin whitespace kurallarına göre tokenlara ayrılır.
        tokens = self.tokenize(text)

        # Duplicate tokenları temizlerken ilk görülme sırasını korur.
        # Örnek: ["hello", "world", "hello"] -> ["hello", "world"]
        unique_tokens = list(dict.fromkeys(tokens))

        # Her unique token için deterministik integer id oluşturulur.
        # İlk görülen token 0, sonraki 1, ... şeklinde ilerler.
        self._token_to_id = {
            token: idx for idx, token in enumerate(unique_tokens)
        }

        # Decode işlemi için ters mapping oluşturulur.
        self._id_to_token = {
            idx: token for token, idx in self._token_to_id.items()
        }

        # Vocabulary hazır olduğu için tokenizer artık encode/decode yapabilir.
        self._trained = True

    # ---------------------------------------------------------
    # TOKENIZE
    # ---------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        """
        Metni whitespace karakterlerine göre tokenlara böler.

        Python str.split() default olarak:
            - boşluk
            - tab
            - newline
            - ardışık whitespace karakterleri

        üzerinde çalışır.

        Örnek:
            "hello   world" -> ["hello", "world"]
            " hello world " -> ["hello", "world"]
            "hello\\nworld" -> ["hello", "world"]

        Returns:
            list[str]:
                Whitespace'e göre ayrılmış token listesi.
                Boş veya whitespace-only input için [] döner.
        """
        # Boş string veya sadece whitespace içeren input için token yoktur.
        if not text or not text.strip():
            return []

        # split() argümansız kullanıldığında tüm whitespace türlerini ayırıcı kabul eder.
        # split() default olarak:
        # - ardışık whitespace'leri tek kabul eder
        # - leading/trailing whitespace'i ignore eder
        return text.split()

    # ---------------------------------------------------------
    # ENCODE
    # ---------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """
        Metni integer token id listesine dönüştürür.

        İşleyiş:
            1. Metin tokenize edilir.
            2. Her token vocabulary içinde aranır.
            3. Token karşılık gelen integer id değerine çevrilir.

        Örnek:
            Vocabulary:
                {"hello": 0, "world": 1}

            Input:
                "hello world"

            Output:
                [0, 1]

        Raises:
            ValueError:
                Tokenizer henüz train edilmemişse.
            ValueError:
                Input içinde vocabulary'de olmayan token varsa.
        """
        # encode için token -> id mapping gerekir.
        # Bu mapping train() sırasında oluşturulur.
        if not self._trained:
            raise ValueError("Tokenizer has not been trained yet")

        tokens = self.tokenize(text)

        ids: list[int] = []

        for token in tokens:
            # WhitespaceTokenizer strict vocabulary davranışı kullanır.
            # Eğitimde görülmeyen token encode edilmeye çalışılırsa açık hata verir.
            if token not in self._token_to_id:
                raise ValueError(f"Unknown token: {token}")

            # Token'ın integer id karşılığı output listesine eklenir.
            ids.append(self._token_to_id[token])

        return ids

    # ---------------------------------------------------------
    # DECODE
    # ---------------------------------------------------------

    def decode(self, token_ids: list[int]) -> str:
        """
        Integer token id listesini tekrar string'e dönüştürür.

        İşleyiş:
            1. Her id vocabulary içinde aranır.
            2. Id karşılık gelen token string'ine çevrilir.
            3. Tokenlar tek boşluk ile birleştirilir.

        Önemli:
            Decode işlemi orijinal whitespace formatını birebir korumaz.

            Örnek:
                Original:
                    "hello   world"

                Decode:
                    "hello world"

        Raises:
            ValueError:
                Tokenizer henüz train edilmemişse.
            ValueError:
                Bilinmeyen token id verilirse.
        """
        # decode için id -> token mapping gerekir.
        # Bu mapping train() sırasında oluşturulur.
        if not self._trained:
            raise ValueError("Tokenizer has not been trained yet")

        tokens: list[str] = []

        for tid in token_ids:
            # Vocabulary içinde olmayan id decode edilemez.
            # Bu strict davranış hatalı model çıktılarının sessizce kabul edilmesini engeller.
            if tid not in self._id_to_token:
                raise ValueError("Unknown token id")

            tokens.append(self._id_to_token[tid])

        # WhitespaceTokenizer tokenları tek boşluk ile birleştirir.
        # Orijinal multiple-space/tab/newline bilgisi korunmaz.
        return " ".join(tokens)

    # ---------------------------------------------------------
    # VOCAB
    # ---------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """
        Mevcut vocabulary boyutunu döndürür.

        Eğitim öncesi:
            0

        Eğitim sonrası:
            train() sırasında görülen unique whitespace token sayısı.
        """
        return len(self._token_to_id)