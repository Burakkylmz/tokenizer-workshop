from __future__ import annotations

import re

from tokenizer_workshop.tokenizers.base import BaseTokenizer
from tokenizer_workshop.tokenizers.registry import register_tokenizer


@register_tokenizer("punctuation")
class PunctuationTokenizer(BaseTokenizer):
    """
    Noktalama duyarlı tokenizer.

    Bu tokenizer, metni kelime tokenları ve noktalama tokenları olarak ayırır.

    Temel fark:
        WhitespaceTokenizer noktalama işaretlerini kelimeye yapışık bırakır.

    Örnek:
            "Hello, world!"
            -> ["Hello,", "world!"]

        PunctuationTokenizer ise noktalama işaretlerini ayrı token olarak üretir.

        Örnek:
            "Hello, world!"
            -> ["hello", ",", "world", "!"]

    ---------------------------------------------------------
    📌 FARKI NEDİR?
    ---------------------------------------------------------

    WhitespaceTokenizer:
        ["hello,", "world!"]

    PunctuationTokenizer:
        ["hello", ",", "world", "!"]

    Kullanım amacı:
        - Noktalama işaretlerinin token sayısına etkisini göstermek
        - WhitespaceTokenizer ile RegexTokenizer arasında basit bir ara seviye sağlamak
        - Baseline ama noktalama-aware bir tokenizer sunmak
        - Compare/report çıktılarında daha açıklayıcı token breakdown üretmek

   Tasarım kararları:
        - Regex tabanlı çalışır.
        - Metni lowercase normalize eder.
        - Whitespace karakterlerini token olarak saklamaz.
        - Noktalama işaretlerini ayrı token olarak üretir.
        - tokenize() için eğitim gerekmez.
        - encode/decode için train() ile vocabulary oluşturulmalıdır.

    Sınırlamalar:
    - NLP seviyesinde tokenization değildir (örneğin contractions handling yok)
    - Unicode punctuation sınırlı kapsanır
    - Subword segmentation yapmaz.
    - Morfolojik analiz yapmaz.
    - Decode sırasında orijinal spacing birebir korunmaz.
        - Punctuation spacing basit join nedeniyle normalize edilir.

    Decode davranışı:
        Tokenlar " ".join(...) ile birleştirilir.

    tokenize() → train gerekmez  
    encode/decode → train gerekir

    Bu yüzden:
            ["hello", ",", "world", "!"]
            -> "hello , world !"

        Bu bilinçli olarak basit tutulmuştur.
    """

    def __init__(self) -> None:
        """
        PunctuationTokenizer instance'ını başlatır.

        Bu tokenizer tokenize() için eğitim gerektirmez.
        Ancak encode/decode için token -> id ve id -> token mapping tabloları gerekir.

        Internal state:
            _token_to_id:
                Token string'lerini integer id değerlerine map eder.

            _id_to_token:
                Integer id değerlerini tekrar token string'lerine map eder.

            _trained:
                Vocabulary'nin hazır olup olmadığını takip eder.

            _pattern:
                Kelimeleri ve noktalama işaretlerini yakalayan regex pattern.
        """
        super().__init__(name="punctuation")

        # WordTokenizer gibi bir vocab tutmak istersek kullanılabilir
        # Token string'lerini integer id değerlerine map eder.
        # Örnek: {"hello": 0, ",": 1, "world": 2, "!": 3}
        self._token_to_id: dict[str, int] = {}
        # Integer id değerlerini tekrar token string'lerine map eder.
        # Örnek: {0: "hello", 1: ",", 2: "world", 3: "!"}
        self._id_to_token: dict[int, str] = {}

        # train() çağrılıp çağrılmadığını takip eder.
        # encode/decode için vocabulary hazır olmalıdır.
        self._trained = False

        # Regex açıklaması:
        #   \w+
        #       Harf, rakam ve underscore gibi word-character dizilerini yakalar.
        #       Python regex motorunda Unicode-aware çalıştığı için Türkçe karakterleri
        #       de çoğu durumda kelime parçası olarak yakalar.
        #   [^\w\s]
        #       Word-character olmayan ve whitespace olmayan tek karakterleri yakalar.
        #       Bu genellikle noktalama/sembol karakterlerini temsil eder.
        # Örnek:
        #   "Merhaba, dünya!"
        #   -> ["merhaba", ",", "dünya", "!"]
        self._pattern = re.compile(r"\w+|[^\w\s]")

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------

    def train(self, text: str) -> None:
        """
        Eğitim metninden vocabulary oluşturur.

        Bu tokenizer için train() kompleks bir model öğrenmez.
        Sadece encode/decode için gerekli mapping tablolarını oluşturur.

        İşleyiş:
            1. Input doğrulanır.
            2. Metin kelime + noktalama tokenlarına ayrılır.
            3. Duplicate tokenlar temizlenir.
            4. Tokenların ilk görülme sırası korunur.
            5. token -> id mapping oluşturulur.
            6. id -> token mapping oluşturulur.
            7. Tokenizer trained state'e geçirilir.

        Bu tokenizer için training:
            - tokenize et
            - unique tokenları al
            - mapping oluştur

        Örnek:
            Input:
                "Hello, world! Hello"

            Tokenlar:
                ["hello", ",", "world", "!", "hello"]

            Unique tokenlar:
                ["hello", ",", "world", "!"]

            Mapping:
                {
                    "hello": 0,
                    ",": 1,
                    "world": 2,
                    "!": 3
                }

        Neden dict.fromkeys?
            Python dict insertion-order koruduğu için duplicate tokenları silerken
            tokenların ilk görülme sırasını korur.

        Raises:
            ValueError:
                Eğitim metni boşsa veya yalnızca whitespace içeriyorsa.
        """
        # Boş veya sadece whitespace içeren input eğitim için anlamlı değildir.
        # Çünkü vocabulary oluşturmak için en az bir gerçek token gerekir.
        if not text or not text.strip():
            raise ValueError("Training text cannot be empty")

        # Metin whitespace kurallarına göre tokenlara ayrılır.
        tokens = self.tokenize(text)

        # Duplicate tokenları temizlerken ilk görülme sırasını korur.
        # Bu, vocabulary id atamasını deterministik hale getirir.
        # Örnek: ["hello", ",", "world", "!", "hello"] -> ["hello", ",", "world", "!"]
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

        # Tokenizer trained state'e geçirilir.
        # Bu, encode/decode işlemlerinin yapılabilmesi için gereklidir.
        self._trained = True

    # ---------------------------------------------------------
    # TOKENIZE
    # ---------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        """
        Metni kelime ve noktalama tokenlarına böler.

        Davranış:
            - Boş input için [] döner.
            - Whitespace-only input için [] döner.
            - Metni lowercase normalize eder.
            - Kelime dizilerini tek token yapar.
            - Noktalama işaretlerini ayrı token yapar.
            - Whitespace karakterlerini token olarak saklamaz.

        Örnek:
            Input:
                "Hello, world!"

            Output:
                ["hello", ",", "world", "!"]

        Returns:
            list[str]:
                Kelime ve noktalama tokenlarından oluşan liste.
        """
        # Boş string veya sadece whitespace içeren input için token yoktur.
        if not text or not text.strip():
            return []

        # Lowercase normalization:
        # Aynı kelimenin farklı case varyasyonlarının farklı token id almasını önler.
        # Örnek:
        #   "Hello" ve "hello" aynı token olarak değerlendirilir.
        text = text.lower()

        # Regex pattern'ine göre metni tokenize eder.
        # Kelime dizilerini (\w+) ve noktalama işaretlerini ([^\w\s]) ayrı token olarak yakalar.
        # Örnek:
        #   "Hello, world!"
        #   -> ["hello", ",", "world", "!"]
        return self._pattern.findall(text)

    # ---------------------------------------------------------
    # ENCODE
    # ---------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """
        Metni integer token id listesine dönüştürür.

        İşleyiş:
            1. Tokenizer'ın train edilmiş olduğu doğrulanır.
            2. Input tokenize edilir.
            3. Her token vocabulary içinde aranır.
            4. Tokenlar integer id değerlerine çevrilir.

        Örnek:
            Vocabulary:
                {
                    "hello": 0,
                    ",": 1,
                    "world": 2,
                    "!": 3
                }

            Input:
                "Hello, world!"

            Output:
                [0, 1, 2, 3]

        Strict vocabulary davranışı:
            Eğitim sırasında görülmeyen token encode edilmeye çalışılırsa ValueError fırlatılır.

        Raises:
            ValueError:
                Tokenizer train edilmemişse.
            ValueError:
                Input içinde vocabulary dışında kalan token varsa.
        """
        # Encode işlemi için tokenizer'ın train edilmiş olması gerekir.
        # Çünkü token -> id mapping tabloları train() sırasında oluşturulur.
        if not self._trained:
            raise ValueError("Tokenizer has not been trained yet")

        # Input tokenize edilir.
        # Örnek:
        #   "Hello, world!"
        #   -> ["hello", ",", "world", "!"]
        tokens = self.tokenize(text)

        # Her token için vocabulary kontrolü yapılır ve integer id'lere çevrilir.
        ids: list[int] = []

        for token in tokens:
            # OOV kontrolü:
            # Bu tokenizer bilinmeyen tokenları otomatik [UNK] ile karşılamaz.
            # Strict davranarak veri/vocabulary uyumsuzluğunu açık şekilde gösterir.
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
        Token id listesini tekrar string'e dönüştürür.

        İşleyiş:
            1. Tokenizer'ın train edilmiş olduğu doğrulanır.
            2. Her id vocabulary içinde aranır.
            3. Id karşılık gelen token string'ine çevrilir.
            4. Tokenlar tek boşluk ile birleştirilir.

        Örnek:
            Input ids:
                [0, 1, 2, 3]

            Tokenlar:
                ["hello", ",", "world", "!"]

            Decode:
                "hello , world !"

        Önemli:
            Decode işlemi orijinal spacing'i birebir korumaz.
            Noktalama öncesi/sonrası boşluklar basit join nedeniyle normalize edilir.

        Not:
            Basit join kullanır → spacing tam NLP gibi değildir.

        Örnek:
            ["hello", ",", "world", "!"]
            → "hello , world !"

        Raises:
            ValueError:
                Tokenizer train edilmemişse.
            ValueError:
                Bilinmeyen token id verilirse.
        """
        # Decode işlemi için tokenizer'ın train edilmiş olması gerekir.
        # Çünkü id -> token mapping tabloları train() sırasında oluşturulur.
        if not self._trained:
            raise ValueError("Tokenizer has not been trained yet")

        # Her id için vocabulary kontrolü yapılır ve token string'lerine çevrilir.
        tokens: list[str] = []

        for tid in token_ids:
            # Model çıktısından gelen id vocabulary içinde yoksa
            # bunu sessizce ignore etmek yerine açık hata veririz.
            if tid not in self._id_to_token:
                raise ValueError(f"Unknown token id: {tid}")

            # Token id'sinin karşılığı olan token string'i output listesine eklenir.
            tokens.append(self._id_to_token[tid])

        # Basit join (space-aware değil)
        # Bu tokenizer'ın decode davranışı bilinçli olarak basit tutulmuştur.
        # Orijinal spacing'i birebir korumaz, ancak tokenları tek boşluk ile birleştirerek normalize edilmiş bir string döndürür.
        return " ".join(tokens)

    # ---------------------------------------------------------
    # VOCAB
    # ---------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """
        Vocabulary'deki token sayısını döner.
    
        Bu, tokenizer'ın train edilip edilmediği ve kaç unique token öğrendiği hakkında bilgi verir.

        Eğer tokenizer train edilmemişse vocab_size 0 döner.
        Eğer tokenizer train edilmişse vocab_size, eğitim metnindeki unique token sayısına eşit olur.   
    
        Örnek:
            Eğer train() sırasında "hello, world!" metni kullanıldıysa,
            vocabulary ["hello", ",", "world", "!"] tokenlarını içerir ve vocab_size 4 olur.
        """
        return len(self._token_to_id)