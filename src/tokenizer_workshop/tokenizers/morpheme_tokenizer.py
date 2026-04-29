from __future__ import annotations

import re

from tokenizer_workshop.tokenizers.base import BaseTokenizer
from tokenizer_workshop.tokenizers.registry import register_tokenizer


@register_tokenizer("morpheme")
class MorphemeTokenizer(BaseTokenizer):
    """
    Basitleştirilmiş morfem tabanlı tokenizer.

    Bu tokenizer, kelimeleri basit suffix kurallarına göre kök ve ek benzeri
    parçalara ayıran eğitim amaçlı bir tokenizer'dır.

    Temel amaç:
        Bu sınıf, gerçek bir NLP morfolojik analiz sistemi kurmak için değil;
        morpheme-level tokenization yaklaşımının token sayısı, vocabulary yapısı
        ve segmentation davranışı üzerindeki etkisini göstermek için tasarlanmıştır.

    Çalışma prensibi:
        1. Girdi metni lowercase normalize edilir.
        2. Metin kelime ve noktalama tokenlarına ayrılır.
        3. Kelime tokenları suffix listesine göre analiz edilir.
        4. En uzun suffix önce denenir.
        5. Uygun suffix bulunursa kelime stem + suffix parçalarına ayrılır.
        6. Noktalama işaretleri bağımsız token olarak korunur.

    Örnek:
        suffixes = ["lar", "ler", "dır", "dir", "ing", "ed", "s"]

        Input:
            "Books are running."

        Output:
            ["book", "s", "are", "runn", "ing", "."]

    Türkçe örnek:
        Input:
            "Çocuklar okulda."

        Output:
            ["çocuk", "lar", "okulda", "."]

    Tasarım kararı:
        Bu tokenizer train() çağrısında istatistiksel bir model öğrenmez.
        Training aşaması yalnızca encode/decode işlemleri için gerekli
        token-id vocabulary mapping tablolarını üretir.

    Kullanım amacı:
        - WordTokenizer ile morpheme-level yaklaşımı karşılaştırmak
        - Ek ayrıştırmanın token sayısına etkisini göstermek
        - Dilbilimsel tokenization kavramına giriş yapmak
        - Compare/report çıktılarında morfem benzeri parçalama sağlamak

    Sınırlamalar:
        - Gerçek kök bulma yapmaz.
        - Türkçeye özgü ses olaylarını modellemez.
        - Lemmatization veya stemming algoritması değildir.
        - Sadece tanımlı suffix listesine göre parçalama yapar.
        - Bir kelimede yalnızca greedy suffix decomposition uygular.
        - Whitespace bilgisini saklamadığı için decode işlemi lossless değildir.
        - Bilinmeyen tokenlar için [UNK] mekanizması kullanmaz.

    Kullanım alanı:
        - WordTokenizer ile morpheme-like tokenization karşılaştırması yapmak
        - Ek ayrıştırmanın token sayısına etkisini göstermek
        - Compare/report çıktılarında daha zengin tokenizer davranışı sunmak
        - Dilbilimsel tokenization kavramına giriş seviyesi bir örnek sağlamak
    """

    # Suffix listesi, tokenizer'ın kelimeleri nasıl parçaladığına dair temel bir kural setidir.
    # Bu listede yer alan suffixler, tokenize() sırasında kelimelerden ayrıştırılmaya çalışılır.
    DEFAULT_SUFFIXES = [
        # Türkçe çoğul ekleri
        "lar",
        "ler",

        # Türkçe bulunma / ayrılma / yönelme benzeri sık ekler
        "da",
        "de",
        "ta",
        "te",
        "dan",
        "den",
        "tan",
        "ten",
        "a",
        "e",

        # Türkçe iyelik / kişi / durum benzeri basit örnekler
        "ım",
        "im",
        "um",
        "üm",
        "ın",
        "in",
        "un",
        "ün",
        "ımız",
        "imiz",
        "umuz",
        "ümüz",

        # Türkçe bildirme / fiilimsi benzeri basit örnekler
        "dır",
        "dir",
        "dur",
        "dür",
        "tır",
        "tir",
        "tur",
        "tür",
        "mak",
        "mek",

        # İngilizce yaygın suffix örnekleri
        "ing",
        "ed",
        "er",
        "est",
        "ly",
        "ness",
        "ment",
        "tion",
        "s",
    ]

    def __init__(
        self,
        suffixes: list[str] | None = None,
        min_stem_length: int = 2,
    ) -> None:
        """
        MorphemeTokenizer instance'ını yapılandırır.

        Args:
            suffixes:
                Kelimelerden ayrıştırılacak suffix listesidir.
                None verilirse DEFAULT_SUFFIXES kullanılır.

                Liste normalize edilir:
                    - lowercase dönüşümü yapılır
                    - boş değerler temizlenir
                    - duplicate suffixler kaldırılır
                    - uzun suffixler önce denenecek şekilde sıralanır

            min_stem_length:
                Bir suffix ayrıldıktan sonra geriye kalan stem'in sahip olması
                gereken minimum karakter uzunluğudur.

                Bu kontrol, aşırı agresif ve anlamsız parçalamaları engeller.

                Örnek:
                    word = "as"
                    suffix = "s"

                    Eğer min_stem_length = 2 ise:
                        stem = "a" çok kısa olduğu için split yapılmaz.

        Raises:
            ValueError:
                min_stem_length değeri 1'den küçükse.

        Design Note:
            Suffixlerin uzunluktan kısaya sıralanması bilinçli bir karardır.
            Bu sayede "ness" gibi daha anlamlı uzun ekler, "s" gibi kısa eklerden
            önce değerlendirilir.
        """
        super().__init__(name="morpheme")

        # min_stem_length kontrolü, kelimenin aşırı parçalanmasını engellemek için önemlidir.
        # Örnek:
        #   word = "as"
        #   suffixes = ["s"]
        #   Eğer min_stem_length = 2 ise:
        #       "as" -> "a" + "s" şeklinde parçalanmaz, 
        #       çünkü "a" tek karakterdir ve anlamsız bir parçalama olurdu.
        # Bu kontrol, tokenizer'ın daha makul ve anlamlı parçalamalar yapmasını sağlar.
        if min_stem_length < 1:
            raise ValueError("min_stem_length must be at least 1")

        # Suffix listesi normalize edilir:
        # - Lowercase yapılır: "ING" -> "ing"   
        # - Boş stringler atılır: ["ing", "", "ed"] -> ["ing", "ed"]
        # - Duplicate suffixler temizlenir: ["ing", "ed", "ing"] -> ["ing", "ed"]
        # - Uzun suffixler önce denenecek şekilde sıralanır: ["s", "ness"] -> ["ness", "s"] 
        # Bu normalizasyon adımları, tokenize() sırasında tutarlı ve beklenen davranışın sağlanması için önemlidir. 
        self.min_stem_length = min_stem_length # Parçalama için minimum stem uzunluğu

        # Suffix listesi normalize edilir.
        # Lowercase yapılır, boş stringler atılır, tekrar eden suffixler temizlenir.
        # Uzun suffixler önce denenecek şekilde sıralanır.
        # Normalizasyonun amacı, tokenize() sırasında suffix eşleşmelerinin tutarlı ve beklenen şekilde çalışmasını sağlamaktır.
        # Örnek:
        #   suffixes = ["ING", "", "ed", "ing"]
        #   normalized_suffixes = ["ing", "ed"]
        #   Uzun suffixlerin önce denenmesi, "ness" gibi daha anlamlı eklerin "s" gibi kısa eklerden önce değerlendirilmesini sağlar.
        raw_suffixes = suffixes if suffixes is not None else self.DEFAULT_SUFFIXES

        normalized_suffixes = [
            suffix.lower() # Lowercase normalizasyonu, tokenize() sırasında suffix eşleşmelerinin case-insensitive olmasını sağlar.
            for suffix in raw_suffixes # Boş stringler atılır, çünkü boş suffixler anlamsızdır ve tokenize() sırasında gereksiz karmaşıklık yaratır.
            if suffix and suffix.strip() # Boş stringleri atmak için kontrol, çünkü boş veya sadece whitespace içeren suffixler tokenize() sırasında anlamsızdır ve gereksiz karmaşıklık yaratır.
        ]

        # Uzun suffixlerin önce denenmesi önemlidir.
        # Örnek:
        #   suffixes = ["s", "ness"]
        #   word = "kindness"
        #   Eğer "s" önce denenirse:
        #       "kindness" -> "kindnes" + "s" gibi zayıf parçalama olurdu.
        # Eğer "s" önce denenirse yanlış parçalama oluşabilir.
        # Bu yüzden greedy longest-suffix-first yaklaşımı kullanılır.
        #   Uzun suffixlerin önce denenmesiyle:
        #       "kindness" -> "kind" + "ness" gibi daha anlamlı bir parçalama olur. 
        self.suffixes = sorted(
            dict.fromkeys(normalized_suffixes), # Duplicate suffixleri temizlerken ilk görülme sırasını korur.
            key=len, # Uzun suffixler önce denenecek şekilde sıralama yapılır.  
            reverse=True, # Uzun suffixler önce denenecek şekilde sıralama yapılır.
        )

        # Tokenizer'ın encode/decode işlemleri için ihtiyaç duyduğu token-id mapping tabloları.
        # Bu tokenizer tokenize() için state'e ihtiyaç duymaz, ancak encode/decode akışı için token -> id ve id -> token mapping tabloları tutulur.
        # _trained flag'i, encode/decode çağrılarının eğitimden önce çalışmasını engeller.  
        # encode/decode işlemleri için vocabulary hazır olmalıdır, bu yüzden train() çağrılmadan önce bu işlemlerin yapılması engellenir.   
        # encode/decode işlemleri için token-id mapping tabloları tutulur, çünkü bu tokenizer'ın encode() ve decode() metodları, train() sırasında oluşturulan vocabulary'ye dayanır.
        # token -> id mapping tablosu, tokenize() tarafından üretilen tokenların integer id'lere çevrilmesi için kullanılır.
        # id -> token mapping tablosu, integer id'lerin tekrar token string'lerine çevrilmesi için kullanılır.
        self._token_to_id: dict[str, int] = {} # Token string'lerini integer id değerlerine map eder. Örnek: {"çocuk": 0, "lar": 1}
        self._id_to_token: dict[int, str] = {} # Integer id değerlerini token string'lerine map eder. Örnek: {0: "çocuk", 1: "lar"}

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
        Eğitim metninden deterministic vocabulary mapping oluşturur.

        Bu tokenizer istatistiksel bir model öğrenmez.
        train() metodunun görevi, verilen metindeki morpheme-like tokenları çıkarıp
        encode/decode işlemlerinde kullanılacak token-id tablolarını hazırlamaktır.

        Pipeline:
            1. Eğitim metninin boş olmadığı doğrulanır.
            2. Metin tokenize edilir.
            3. Duplicate tokenlar temizlenir.
            4. İlk görülme sırası korunur.
            5. token -> id mapping oluşturulur.
            6. id -> token mapping oluşturulur.
            7. Tokenizer trained state'e alınır.

        Determinism:
            dict.fromkeys(tokens) kullanılması bilinçli bir tercihtir.
            Python dictionary insertion order koruduğu için aynı input metni
            her çalıştırmada aynı token-id mapping sonucunu üretir.

        Args:
            text:
                Vocabulary oluşturmak için kullanılacak eğitim metni.

        Örnek:
            Input:
                "Çocuklar çocuklar okulda."

            Tokenize:
                ["çocuk", "lar", "çocuk", "lar", "okul", "da", "."]

            Unique:
                ["çocuk", "lar", "okul", "da", "."]

        Neden dict.fromkeys?
            Python dict insertion order korur.
            Böylece duplicate tokenlar silinirken ilk görülme sırası korunur.
            Bu da deterministic token-id mapping sağlar.

        Raises:
            ValueError:
                Eğitim metni boşsa veya sadece whitespace içeriyorsa.
        """
        # Boş veya sadece whitespace içeren input eğitim için anlamlı değildir.
        # Çünkü vocabulary oluşturmak için en az bir gerçek token gerekir.
        if not text or not text.strip():
            raise ValueError("Training text cannot be empty")

        # Metin tokenize edilir.
        # tokenize() metodu, verilen metni morpheme-like tokenlara böler.
        tokens = self.tokenize(text)

        # Duplicate tokenları temizlerken ilk görülme sırasını korur.
        # Örnek: ["çocuk", "lar", "çocuk", "lar", "okul", "da", "."] -> ["çocuk", "lar", "okul", "da", "."]
        # dict.fromkeys(tokens) kullanılarak tokenların ilk görülme sırası korunur ve duplicate tokenlar silinir.
        unique_tokens = list(dict.fromkeys(tokens))

        # Her unique token için deterministik integer id oluşturulur.
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
        Metni morpheme-like token listesine dönüştürür.

        Bu metod eğitim gerektirmez.
        Verilen metni önce kelime/noktalama düzeyinde parçalar, ardından kelime
        tokenlarını suffix kurallarına göre daha küçük morfem benzeri parçalara ayırır.

        Davranış:
            - Boş veya whitespace-only input için [] döner.
            - Metin lowercase normalize edilir.
            - Kelimeler greedy suffix decomposition ile parçalanır. 
            - Noktalama işaretleri ayrı token olarak korunur.
            - Whitespace karakterleri token olarak saklanmaz.
        
        Örnek:
            Input:
                "Çocuklar okulda."

            Output:
                ["çocuk", "lar", "okul", "da", "."]

        Args:
            text:
                Tokenize edilecek ham metin.

        Returns:
            list[str]:
                Stem, suffix ve punctuation tokenlarından oluşan liste.
        """
        # Boş string veya sadece whitespace içeren input için token yoktur.
        if not text or not text.strip():
            return []

        # Metin lowercase normalize edilir.
        # Normalizasyon, tokenize() sırasında suffix eşleşmelerinin case-insensitive olmasını sağlar.
        # Örnek:
        #   Input: "Çocuklar okulda."
        #   Output: "çocuklar okulda."
        normalized_text = text.lower()

        # Metin kelime ve noktalama tokenlarına ayrılır.
        # regex pattern'ı, kelime karakterlerinden oluşan dizileri (\w+) ve
        # kelime karakteri olmayan tek karakterleri ([^\w\s]) yakalar.
        # Bu sayede "Çocuklar okulda." -> ["çocuklar", "okulda", "."] gibi bir token listesi elde edilir.
        raw_tokens = self._pattern.findall(normalized_text)

        output_tokens: list[str] = []

        for token in raw_tokens:
            # Kelime tokenları suffix kurallarına göre parçalanır, noktalama işaretleri olduğu gibi korunur.
            # _is_word_token() metodu, token'ın kelime tokenı mı yoksa noktalama/sembol tokenı mı olduğunu belirler.
            if self._is_word_token(token):
                output_tokens.extend(self._split_word_into_morphemes(token))
            else:
                output_tokens.append(token)

        return output_tokens

    # ---------------------------------------------------------
    # ENCODE
    # ---------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """
        Metni integer token id listesine dönüştürür.

        encode() işlemi strict vocabulary yaklaşımı kullanır.
        Yani yalnızca train() sırasında vocabulary'ye eklenmiş tokenlar encode edilebilir.

        İşleyiş:
            1. Tokenizer'ın train edilmiş olduğu doğrulanır.
            2. Input tokenize edilir.
            3. Her token vocabulary içinde aranır.
            4. Tokenlar integer id değerlerine çevrilir.

        Strict vocabulary davranışı:
            Bu tokenizer [UNK] token kullanmaz.
            Eğitim sırasında görülmeyen token encode edilmeye çalışılırsa
            ValueError fırlatılır.

        Args:
            text:
                Encode edilecek metin.

        Returns:
            list[int]:
                Token id listesi.

        Raises:
            ValueError:
                Tokenizer henüz train edilmemişse.
            ValueError:
                Vocabulary içinde bulunmayan bir token görülürse.
        """
        # Encode işlemi için tokenizer'ın train edilmiş olması gerekir.
        # Çünkü token -> id mapping tabloları train() sırasında oluşturulur.
        if not self._trained:
            raise ValueError("Tokenizer has not been trained yet")

        # Input tokenize edilir.
        # tokenize() metodu, verilen metni morpheme-like tokenlara böler.
        # Örnek:
        #   Input: "Çocuklar okulda."
        #   tokenize() -> ["çocuk", "lar", "okul", "da", "."]
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
        Token id listesini string çıktıya dönüştürür.

        Decode işlemi tokenları doğrudan birleştirir.
        Bu tokenizer whitespace bilgisini saklamadığı için orijinal metin birebir
        geri üretilemez.

        Örnek:
            Tokenlar:
                ["çocuk", "lar", "okul", "da", "."]

            Decode:
                "çocuklarokulda."

        Bu davranış bilinçlidir.
        MorphemeTokenizer'ın temel amacı lossless reconstruction değil,
        morpheme-like segmentation davranışını göstermektir.

        Args:
            token_ids:
                Decode edilecek token id listesi.

        Returns:
            str:
                Tokenların birleştirilmesiyle oluşturulan metin.

        Raises:
            ValueError:
                Tokenizer henüz train edilmemişse.
            ValueError:
                Bilinmeyen token id verilirse.
        """
        # Decode işlemi için tokenizer'ın train edilmiş olması gerekir.
        # Çünkü id -> token mapping tabloları train() sırasında oluşturulur.
        if not self._trained:
            raise ValueError("Tokenizer has not been trained yet")

        tokens: list[str] = [] # Decode işlemi için token id'lerden tekrar token string'lerine çevrilir.

        for token_id in token_ids:
            # Vocabulary içinde olmayan token id'ler decode edilemez.
            # Bu strict davranış hatalı model çıktılarının sessizce kabul edilmesini engeller.
            if token_id not in self._id_to_token:
                raise ValueError("Unknown token id")

            # Token id'nin karşılığı olan token string'i output listesine eklenir.  
            tokens.append(self._id_to_token[token_id])

        # Tokenlar doğrudan birleştirilir.
        # MorphemeTokenizer whitespace bilgisini saklamadığı için tokenlar tek bir string olarak birleştirilir. 
        # Örnek:
        #   tokens = ["çocuk", "lar", "okul", "da", "."]
        #   decode() -> "çocuklarokulda."
        return "".join(tokens)

    # ---------------------------------------------------------
    # INTERNAL HELPERS
    # ---------------------------------------------------------

    def _split_word_into_morphemes(self, word: str) -> list[str]:
        """
        Tek bir kelimeyi stem + suffix parçalarına ayırır.

        Bu metod, tokenizer'ın temel morpheme-like segmentation mantığını içerir.
        Amaç, kelimenin sonundan başlayarak tanımlı suffix listesiyle eşleşen
        parçaları ayırmak ve kelimeyi daha küçük anlam-benzeri birimlere bölmektir.

        Önemli:
            Bu metod gerçek morfolojik analiz yapmaz.
            Yani kelimenin sözlük kökünü, ek türünü, çekim bilgisini veya dilbilgisel
            görevini tespit etmez. Sadece suffix listesine dayalı rule-based bir parçalama uygular.

        Algoritma:
            1. Kelime current değişkenine alınır.
            2. current kelimesi için en uygun suffix aranır.
            3. Suffix listesi uzunluktan kısaya sıralı olduğu için önce en uzun suffix eşleşmeleri denenir.
            4. Eğer current ilgili suffix ile bitiyorsa suffix ayrılır.
            5. Suffix ayrıldıktan sonra kalan stem'in min_stem_length şartını sağlayıp sağlamadığı kontrol edilir.
            6. Stem çok kısa kalıyorsa parçalama durdurulur.
            7. Geçerli suffix suffix_parts listesine eklenir.
            8. current artık kalan stem olacak şekilde güncellenir.
            9. Yeni current üzerinde aynı işlem tekrar denenir.
            10. Artık eşleşen suffix kalmadığında döngü sonlanır.
            11. Suffixler sondan başa doğru bulunduğu için sonuçta reverse edilir.
            12. Hiç suffix bulunamazsa kelime tek parça olarak döndürülür.

        Neden greedy yaklaşım?
            Kelimenin sonunda birden fazla suffix adayı olabilir.

        Örnek:
                suffixes = ["s", "ness"]
                word = "kindness"

            Eğer kısa suffix olan "s" önce ayrılırsa:
                ["kindnes", "s"]

            Bu zayıf ve anlamsız bir parçalama üretir.

            Bu yüzden suffix listesi uzunluktan kısaya sıralanır ve önce
            "ness" gibi daha spesifik suffixler denenir.

        Çoklu suffix ayrıştırma:
            Bu metod yalnızca tek suffix ayırmakla kalmaz.
            Her başarılı ayrıştırmadan sonra kalan stem üzerinde yeniden suffix
            aradığı için çoklu suffix parçalama da yapabilir.

            Örnek:
                word = "evlerimizde"

                Olası suffixler:
                    ["de", "imiz", "ler"]

                İşleyiş:
                    evlerimizde -> evlerimiz + de
                    evlerimiz   -> evler + imiz
                    evler       -> ev + ler

                Output:
                    ["ev", "ler", "imiz", "de"]

        min_stem_length neden önemli?
            Çok kısa stem'ler çoğu zaman anlamlı kök değildir.

            Örnek:
                word = "as"
                suffix = "s"

            Eğer kontrol yapılmazsa:
                ["a", "s"]

            Bu çoğu durumda istenmeyen bir parçalamadır.
            min_stem_length bu tür agresif split işlemlerini engeller.

        Args:
            word:
                Parçalanacak normalize edilmiş kelime tokenı.

        Returns:
            list[str]:
                Eğer suffix bulunursa stem + suffix parçalarından oluşan liste.
                Eğer suffix bulunmazsa kelimenin kendisini içeren tek elemanlı liste.

        Example:
            >>> tokenizer._split_word_into_morphemes("çocuklar")
            ["çocuk", "lar"]

            >>> tokenizer._split_word_into_morphemes("evlerimizde")
            ["ev", "ler", "imiz", "de"]
        """
        # current değişkeni, kelimenin parçalanma sürecinde güncellenen halidir.
        current = word
        suffix_parts: list[str] = [] # Bulunan suffix parçalarını saklar. 
        # En son current'ın kalan stem'i ile birleştirilir.

        while True:
            # current kelimesi için en uygun suffix aranır.
            # Suffix listesi uzunluktan kısaya sıralandığı için bu metod longest-suffix-first davranışı gösterir.
            # Örnek:
            #   suffixes = ["ness", "s"]
            #   word = "kindness"
            #   Eşleşme: "ness"
            matched_suffix = self._find_matching_suffix(current)

            # Eğer current kelimesi için uygun bir suffix bulunamazsa parçalama işlemi sonlanır.
            # matched_suffix None dönerse while döngüsünden çıkılır.
            if matched_suffix is None:
                break
            
            stem = current[: -len(matched_suffix)] # current kelimesinden matched_suffix uzunluğu kadar karakter çıkarılarak kalan stem elde edilir.    

            # Stem çok kısa kalıyorsa parçalama durdurulur.
            # min_stem_length kontrolü, aşırı agresif parçalamaları engeller.
            # Örnek:
            #   word = "as"
            #   suffix = "s"
            #   stem = "a" tek karakterdir ve anlamsız bir parçalama olurdu.
            if len(stem) < self.min_stem_length:
                break
            
            # Geçerli suffix suffix_parts listesine eklenir.
            suffix_parts.append(matched_suffix)
            current = stem # current artık kalan stem olacak şekilde güncellenir ve yeni current üzerinde aynı işlem tekrar denenir.    

        # Hiç suffix bulunamazsa kelimenin kendisini içeren tek elemanlı liste döndürülür.
        # Örnek:
        #   word = "okul"
        #   suffixes = ["lar", "ler", "da", "de", ...]
        #   Hiç suffix bulunamaz, bu yüzden ["okul"] döner.
        if not suffix_parts:
            return [word]

        # Suffixler sondan başa doğru bulunduğu için sonuçta reverse edilir.
        # Örnek:
        #   word = "evlerimizde"
        #   suffix_parts = ["de", "imiz", "ler"] sırayla bulunur, ancak doğru sıralama ["ev", "ler", "imiz", "de"] olmalıdır.
        #   Bu yüzden suffix_parts reverse edilir ve kalan stem current'ın başına eklenir.
        return [current] + list(reversed(suffix_parts))


    def _find_matching_suffix(self, word: str) -> str | None:
        """
        Verilen kelime için en uygun suffix eşleşmesini bulur.

        Bu metod, kelimenin tanımlı suffix listesinde yer alan hangi ek ile
        bittiğini kontrol eder. İlk eşleşen suffix döndürülür.

        Suffix listesi __init__ aşamasında uzunluktan kısaya sıralandığı için
        bu metod pratikte longest-suffix-first davranışı gösterir.

        Neden longest-suffix-first?
            Bazı suffixler başka suffixlerin son karakterlerini de içerebilir.

            Örnek:
                suffixes = ["ness", "s"]
                word = "kindness"

            Eğer "s" önce değerlendirilirse:
                matched_suffix = "s"

            Bu durumda kelime:
                "kindnes" + "s"

            şeklinde zayıf bir parçalamaya gider.

            Ancak suffix listesi uzunluktan kısaya sıralı olduğu için:
                matched_suffix = "ness"

            olur ve daha anlamlı bir segmentasyon elde edilir:
                "kind" + "ness"

        Davranış:
            - Eşleşen ilk suffix döndürülür.
            - Hiç suffix eşleşmezse None döndürülür.
            - Sadece kelimenin sonunda bulunan suffixler dikkate alınır.
            - Kelime ortasındaki veya başındaki parçalar analiz edilmez.

        Args:
            word:
                Suffix eşleşmesi aranacak kelime.

        Returns:
            str | None:
                Eşleşen suffix varsa suffix string'i.
                Eşleşme yoksa None.
        """
        # Suffix listesi uzunluktan kısaya sıralı olduğu için bu döngü pratikte longest-suffix-first davranışı gösterir.
        # Örnek:
        #   suffixes = ["ness", "s"]
        #   word = "kindness"
        #   İlk iterasyonda "ness" kontrol edilir ve eşleşme sağlanırsa "ness" döndürülür.
        #   Eğer "ness" eşleşmezse sonraki iterasyonda "s" kontrol edilir.
        #   Bu sayede daha uzun ve anlamlı suffixler önce değerlendirilir.
        for suffix in self.suffixes:
            # Kelimenin tanımlı suffix ile bitip bitmediği kontrol edilir.
            #  Eğer word.endswith(suffix) True dönerse, bu suffix kelimenin sonunda bulunur ve eşleşme sağlanır.
            if word.endswith(suffix):
                return suffix

        return None

    def _is_word_token(self, token: str) -> bool:
        """
        Token'ın kelime tokenı olup olmadığını kontrol eder.

        Bu metod, tokenize() sırasında elde edilen ham tokenların kelime mi yoksa
        noktalama/sembol mü olduğunu ayırt etmek için kullanılır.

        Word token kabul edilen karakterler:
            - harfler
            - rakamlar
            - underscore karakteri

        Noktalama ve sembol tokenları word token kabul edilmez.
        Böylece ".", ",", "!", "?", ":" gibi karakterler suffix parçalama
        sürecine sokulmadan doğrudan output token listesine eklenir.

        Örnek:
            "çocuklar" -> True
            "running"  -> True
            "token_1"  -> True
            "2026"     -> True
            "."        -> False
            ","        -> False
            "!"        -> False

        Neden ayrı kontrol gerekiyor?
            Morpheme-like parçalama yalnızca kelimeler için anlamlıdır.
            Noktalama işaretlerine suffix analizi uygulamak gereksizdir ve
            hatalı token çıktıları üretebilir.

         Args:
            token:
                Kelime olup olmadığı kontrol edilecek ham token.

        Returns:
            bool:
                Token kelime formatındaysa True, aksi halde False.
        """
        # \w regex'i, Unicode-aware olduğu için Türkçe karakterler de dahil olmak üzere harf, rakam ve underscore içeren tokenları yakalar. 
        # [^\w\s] regex'i ise kelime karakteri olmayan ve whitespace olmayan tek karakterleri yakalar, bu da genellikle noktalama veya sembol karakterleridir.
        # Bu sayede tokenize() sırasında "çocuklar" gibi kelime tokenları suffix analizi için _split_word_into_morphemes() metoduna gönderilirken, "." gibi noktalama işaretleri doğrudan output token listesine eklenir.
        # Örnek:
        #   "Çocuklar okulda." -> tokenize() -> ["çocuklar", "okulda", "."]
        #   "running!" -> tokenize() -> ["running", "!"]
        #   "token_1" -> tokenize() -> ["token_1"]
        #   "2026" -> tokenize() -> ["2026"]
        #   "!" -> tokenize() -> ["!"]
        #   "." -> tokenize() -> ["."]
        # Bu regex tabanlı kontrol, tokenize() sırasında kelime tokenları ile noktalama/sembol tokenlarını ayırmak için basit ve etkili bir yöntem sağlar.
        return bool(re.fullmatch(r"\w+", token))

    # ---------------------------------------------------------
    # VOCAB
    # ---------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """
        Tokenizer vocabulary'sindeki unique token sayısını döndürür.

        vocab_size, train() aşamasında oluşturulan token -> id mapping tablosunun
        kaç farklı token içerdiğini gösterir.

        Eğitim öncesi davranış:
            Tokenizer henüz train edilmemişse _token_to_id boş olur.
            Bu durumda vocab_size 0 döner.

        Eğitim sonrası davranış:
            train() çağrısı sonrası vocab_size, eğitim metninde görülen unique
            morpheme-like token sayısına eşittir.

        Örnek:
            train("Çocuklar okulda.")

            tokenize sonucu:
                ["çocuk", "lar", "okul", "da", "."]

            unique vocabulary:
                {
                    "çocuk": 0,
                    "lar": 1,
                    "okul": 2,
                    "da": 3,
                    ".": 4,
                }

            vocab_size: 5

        Neden property?
            vocab_size dışarıdan okunabilir bir metadata bilgisidir.
            Ancak doğrudan _token_to_id gibi internal state'e erişilmesini
            istemeyiz. Bu yüzden kontrollü ve read-only bir property olarak sunulur.

        Returns:
            int:
                Vocabulary içinde bulunan unique token sayısı.
        """
        return len(self._token_to_id)