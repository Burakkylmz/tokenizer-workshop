from __future__ import annotations

import re

from tokenizer_workshop.tokenizers.base import BaseTokenizer
from tokenizer_workshop.tokenizers.registry import register_tokenizer


@register_tokenizer("subword")
class SubwordTokenizer(BaseTokenizer):
    """
    Basit subword tabanlı tokenizer.

    Bu tokenizer, kelimeleri sabit uzunluklu alt parçalara böler.
    Production-level bir BPE, WordPiece veya SentencePiece değildir.
    Ama subword tokenization kavramını basit ve okunabilir biçimde göstermek için güçlü bir baseline sağlar.

    Temel Fikir:

    Word-level tokenizer kelimeyi tek parça olarak ele alır:

        "tokenization" -> ["tokenization"]

    SubwordTokenizer ise kelimeyi daha küçük parçalara böler:

        subword_size = 3

        "tokenization" -> ["tok", "eni", "zat", "ion"]

    Bu sayede:
        - kelime altı parçalama mantığı gösterilir
        - token sayısı değişimi analiz edilebilir
        - WordTokenizer / WhitespaceTokenizer / BPE benzeri tokenizerlarla karşılaştırma yapılabilir

    Nasıl çalışır?
    1. Metin lowercase normalize edilir.
    2. Regex ile kelime ve noktalama tokenlarına ayrılır.
    3. Kelime tokenları sabit uzunluklu subword parçalarına bölünür.
    4. Noktalama tokenları ayrı token olarak korunur.
    5. Whitespace token olarak saklanmaz.

    Örnek:
        text = "tokenization"
        subword_size = 3

        tokenize(text)
        -> ["tok", "eni", "zat", "ion"]

    Örnek:
        Input:
            "Hello, world!"

        subword_size:
            3

        Output:
            ["hel", "lo", ",", "wor", "ld", "!"]    

    Tasarım Kararları:
        - tokenize() eğitim gerektirmez.
        - encode/decode için train() gerekir.
        - Vocabulary train() sırasında oluşturulur.
        - Mapping deterministiktir.
        - Duplicate tokenlar temizlenirken ilk görülme sırası korunur.
        - Decode tokenları doğrudan birleştirir.
        - Orijinal whitespace bilgisi korunmaz.

    Amaç:
        Bu tokenizer production-level bir WordPiece/BPE değildir.
        Daha çok subword tokenization fikrini basit, okunabilir ve
        karşılaştırılabilir şekilde göstermek için tasarlanmıştır.

    Kullanım alanı:
        - WordTokenizer ile Subword tokenization farkını göstermek
        - Token sayısı / parça uzunluğu analizleri yapmak
        - Compare endpoint içinde basit subword baseline sağlamak

    Tasarım notları:
        - tokenize() eğitim gerektirmez
        - encode/decode için train() gerekir
        - lowercase normalization uygulanır
        - punctuation ayrı token olarak korunur
        - whitespace token olarak saklanmaz

    Sınırlamalar:
        - Öğrenilmiş merge veya probability yoktur
        - Olasılıksal segmentation yoktur.
        - Morfolojik analiz yapmaz.
        - Gerçek BPE / WordPiece / SentencePiece alternatifi değildir
        - Decode spacing bilgisini birebir korumaz (Decode sonucu whitespace açısından lossless değildir)
    """

    def __init__(self, subword_size: int = 3) -> None:
        """
        SubwordTokenizer instance'ını başlatır.

        Parameters:
            subword_size:
                Kelime tokenlarının kaç karakterlik parçalara bölüneceğini belirler.

                Örnek:
                    subword_size = 3
                    "tokenizer" -> ["tok", "eni", "zer"]

                    subword_size = 4
                    "tokenizer" -> ["toke", "nize", "r"]

        Raises:
            ValueError:
                subword_size 1'den küçükse.

        Neden validation var?
            subword_size = 0 olursa range step mantığı bozulur.
            Negatif değerler de anlamlı bir tokenization davranışı üretmez.
        """
        super().__init__(name="subword")

        # subword_size'ın geçerli bir değer olduğundan emin olmak için validation yapılır.
        # subword_size 1'den küçükse ValueError fırlatılır.
        if subword_size < 1:
            raise ValueError("subword_size must be at least 1")

        # subword_size instance değişkeni olarak saklanır.
        self.subword_size = subword_size

        # Token string -> integer id mapping tabloları  
        # Örnek: {"tok": 0, "eni": 1, "zat": 2}
        # encode() bu mapping'i kullanır.
        self._token_to_id: dict[str, int] = {}

        # Integer id -> token string mapping.
        # Örnek:  {0: "tok", 1: "eni", 2: "zat"}
        # decode() bu mapping'i kullanır.
        self._id_to_token: dict[int, str] = {}

        # encode/decode için train() çağrılıp çağrılmadığını takip eder.
        self._trained = False

        # Regex açıklaması:
        #   \w+
        #       Harf, rakam ve underscore gibi word-character dizilerini yakalar.
        #       Python regex motorunda Unicode-aware çalıştığı için Türkçe karakterleri
        #       de kelime tokenı olarak yakalayabilir.
        #   [^\w\s]
        #       Word-character olmayan ve whitespace olmayan tek karakterleri yakalar.
        #       Bu genellikle noktalama veya sembol karakterleridir.
        # Örnek:
        #   "Merhaba, dünya!"
        #   -> ["merhaba", ",", "dünya", "!"]
        self._pattern = re.compile(r"\w+|[^\w\s]")

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------

    def train(self, text: str) -> None:
        """
        Eğitim metninden subword vocabulary oluşturur.

        Bu tokenizer öğrenilmiş merge veya probability üretmez.
        Buradaki training aşaması yalnızca encode/decode için gerekli vocabulary mapping'lerini oluşturur.

        İşleyiş:
            1. Input doğrulanır.
            2. Metin tokenize() ile subword tokenlara ayrılır.
            3. Duplicate tokenlar temizlenir.
            4. İlk görülme sırası korunur.
            5. token -> id mapping oluşturulur.
            6. id -> token mapping oluşturulur.
            7. Tokenizer trained state'e alınır.

        Örnek:
            text = "token tokenization"
            subword_size = 3

            tokenize(text):
                ["tok", "en", "tok", "eni", "zat", "ion"]

            unique tokens:
                ["tok", "en", "eni", "zat", "ion"]

            mapping:
                {
                    "tok": 0,
                    "en": 1,
                    "eni": 2,
                    "zat": 3,
                    "ion": 4
                }

        Neden dict.fromkeys?
            Python dict insertion order korur.
            Böylece duplicate tokenlar silinirken ilk görülme sırası korunur.
            Bu da deterministic vocabulary id ataması sağlar.

        Raises:
            ValueError:
                Eğitim metni boşsa veya sadece whitespace içeriyorsa.
        """
        # Boş veya sadece whitespace içeren input eğitim için anlamlı değildir.
        # Çünkü vocabulary oluşturmak için en az bir gerçek token gerekir.
        if not text or not text.strip():
            raise ValueError("Training text cannot be empty")

        # Metin tokenize() ile subword tokenlara ayrılır.
        # Örnek: "token tokenization" -> ["tok", "en", "tok", "eni", "zat", "ion"]
        # tokenize() eğitim gerektirmez, bu yüzden doğrudan çağrılabilir.
        tokens = self.tokenize(text)

        # Duplicate tokenları temizlerken ilk görülme sırasını koruruz.
        # Bu sayede aynı input her zaman aynı token-id mapping üretir.
        unique_tokens = list(dict.fromkeys(tokens))

        # Her unique token'a integer id atanır.
        # İlk görülen token 0, sonraki 1, ... şeklinde ilerler.
        self._token_to_id = {
            token: idx for idx, token in enumerate(unique_tokens)
        }

        # Decode işlemi için ters mapping oluşturulur.
        # Bu sayede token id'lerden tekrar string'e dönüştürülebilir.
        self._id_to_token = {
            idx: token for token, idx in self._token_to_id.items()
        }

        # Tokenizer trained state'e alınır.
        # Bu sayede encode/decode işlemlerinde train edilip edilmediği kontrol edilebilir.
        self._trained = True

    # ---------------------------------------------------------
    # TOKENIZE
    # ---------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        """
        Metni sabit uzunluklu subword tokenlara böler.

        Davranış:
            - boş input için [] döner
            - whitespace-only input için [] döner
            - metni lowercase normalize eder
            - kelimeleri subword_size uzunluğunda parçalara böler
            - punctuation tokenlarını ayrı tutar
            - whitespace karakterlerini token olarak saklamaz

        Örnek:
            Input:
                "Hello, world!"

            subword_size = 3

            Output:
                ["hel", "lo", ",", "wor", "ld", "!"]

        Neden punctuation ayrı korunuyor?
            Noktalama işaretleri kelime içeriği değildir.
            Bu yüzden kelime parçalama algoritmasına dahil edilmeden
            ayrı token olarak bırakılır.

        Returns:
            list[str]:
                Subword ve punctuation tokenlarından oluşan liste.
        """
        # Boş string veya sadece whitespace içeren input için token yoktur.
        if not text or not text.strip():
            return []

        # Lowercase normalization:
        # "Hello" ve "hello" farklı vocabulary entry'leri üretmesin diye uygulanır.
        normalized_text = text.lower()

        # Önce metin kelime ve noktalama seviyesinde ayrılır.
        # \w+ regex ile kelime tokenları yakalanır, 
        # [^\w\s] regex ile punctuation tokenları yakalanır.
        # Örnek:
        #   "hello, world!"
        #   -> ["hello", ",", "world", "!"]
        raw_tokens = self._pattern.findall(normalized_text)

        # Kelime tokenları subword parçalarına bölünürken, punctuation tokenları ayrı tutulur.
        output_tokens: list[str] = []

        for token in raw_tokens:
            # Kelime tokenları subword parçalarına bölünür.
            # Punctuation tokenları doğrudan output listesine eklenir.
            # Örnek:
            #   "tokenizer" -> ["tok", "eni", "zer"]
            if self._is_word_token(token):
                output_tokens.extend(self._split_word_into_subwords(token))
            # Noktalama/sembol tokenları olduğu gibi korunur.
            else:
                output_tokens.append(token)

        return output_tokens

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
                    "tok": 0,
                    "eni": 1,
                    "zat": 2,
                    "ion": 3
                }

            Input:
                "tokenization"

            tokenize(input):
                ["tok", "eni", "zat", "ion"]

            encode(input):
                [0, 1, 2, 3]

        Strict vocabulary davranışı:
            Eğitim sırasında görülmeyen subword parçası encode edilmeye çalışılırsa
            ValueError fırlatılır.

        Raises:
            ValueError:
                Tokenizer henüz train edilmemişse.
            ValueError:
                Vocabulary içinde olmayan token görülürse.
        """
        # encode için token -> id mapping gerekir.
        # Bu mapping train() sırasında oluşturulur.
        if not self._trained:
            raise ValueError("Tokenizer has not been trained yet")

        # Input tokenize edilir.
        tokens = self.tokenize(text)

        # Her token vocabulary içinde aranır ve karşılık gelen integer id'ye çevrilir.
        ids: list[int] = []

        for token in tokens:
            # OOV kontrolü:
            # Bu tokenizer [UNK] token kullanmaz.
            # Vocabulary dışı tokenları açık hata ile bildirir.
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
            4. Tokenlar doğrudan birleştirilir.

        Örnek:
            Tokenlar:
                ["tok", "eni", "zat", "ion"]

            Decode:
                "tokenization"

        Noktalama örneği:
            Tokenlar:
                ["hel", "lo", ",", "wor", "ld", "!"]

            Decode:
                "hello,world!"

        Önemli:
            Decode işlemi whitespace bilgisini korumaz.
            Çünkü tokenize() whitespace karakterlerini saklamaz.
            Bu nedenle decode sonucu orijinal metinden farklı olabilir.

       Raises:
            ValueError:
                Tokenizer henüz train edilmemişse.
            ValueError:
                Bilinmeyen token id verilirse.
        """
        # encode/decode için train() çağrılıp çağrılmadığı kontrol edilir.
        if not self._trained:
            raise ValueError("Tokenizer has not been trained yet")

        # Her token id vocabulary içinde aranır ve karşılık gelen token string'ine çevrilir.
        tokens: list[str] = []

        for token_id in token_ids:
            # OOV kontrolü:
            # Bu tokenizer [UNK] token kullanmaz.
            # Vocabulary dışı token id'leri açık hata ile bildirir.
            if token_id not in self._id_to_token:
                raise ValueError(f"Unknown token id: {token_id}")
            
            # Token id'sinin karşılığı olan token string'i output listesine eklenir.
            tokens.append(self._id_to_token[token_id])

        # Subword tokenlar kelime parçaları olduğu için doğrudan join edilir.
        # Whitespace bilgisi saklanmadığından kelimeler arası boşluk geri üretilemez.
        return "".join(tokens)

    # ---------------------------------------------------------
    # INTERNAL HELPERS
    # ---------------------------------------------------------

    def _split_word_into_subwords(self, word: str) -> list[str]:
        """
        Tek bir kelimeyi sabit uzunluklu subword parçalarına böler.

        Örnek:
            word = "tokenizer"
            subword_size = 3

            output:
                ["tok", "eni", "zer"]

        Edge case:
            Kelime uzunluğu subword_size'a tam bölünmezse son parça daha kısa olur.

            Örnek:
                word = "hello"
                subword_size = 3

                output:
                    ["hel", "lo"]
        """
        return [
            word[index:index + self.subword_size] # Kelimenin index'ten başlayarak subword_size uzunluğunda parçası
            for index in range(0, len(word), self.subword_size)
        ] # index'i 0'dan kelime uzunluğuna kadar subword_size adımlarla iterasyon yapar.

    def _is_word_token(self, token: str) -> bool:
        """
        Token'ın kelime tokenı olup olmadığını kontrol eder.

        Kelime tokenı:
            - harf
            - rakam
            - underscore
        içerebilir.

        Noktalama tokenları bu kontrolden geçmez.

        Örnek:
            "hello" -> True
            "abc123" -> True
            "," -> False
            "!" -> False
        """
        # \w regex'i kelime karakterlerini yakalar (harf, rakam, underscore).
        # Bu tokenizer'ın tokenize() regex'i ile uyumlu olması için aynı tanım kullanılır.  
        return bool(re.fullmatch(r"\w+", token))

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
        # Vocabulary token -> id mapping tablosundaki entry sayısı vocab_size'ı verir.
        return len(self._token_to_id)