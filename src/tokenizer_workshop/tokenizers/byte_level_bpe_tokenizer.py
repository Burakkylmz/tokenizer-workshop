from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from tokenizer_workshop.tokenizers.base import BaseTokenizer
from tokenizer_workshop.tokenizers.registry import register_tokenizer


@dataclass(frozen=True)
class ByteLevelBPEMerge:
    """
    Byte-Level BPE eğitiminde öğrenilen tek bir merge kuralını temsil eder.

    BPE algoritmasında her eğitim adımında en sık görülen ardışık token çifti
    seçilir ve bu çift yeni bir token id ile temsil edilir.

    Attributes:
        pair:
            Merge edilen iki token id'si.
            Örneğin (97, 98), byte seviyesinde "a" ve "b" tokenlarının
            yan yana gelmesini temsil eder.

        merged_token_id:
            Pair merge edildiğinde üretilecek yeni token id'si.

            Base byte vocabulary 0-255 aralığını kullandığı için merge ile oluşturulan token id'leri 256'dan başlar.

        frequency:
            Bu pair'in merge edildiği eğitim adımında token dizisi içinde kaç kez
            görüldüğünü belirtir.

            Bu bilgi encode için zorunlu değildir; ancak debug, analiz ve raporlama
            açısından değerlidir.
    """

    pair: tuple[int, int] # Merge edilen iki token id'si.
    merged_token_id: int # Bu pair birleştirildiğinde üretilecek yeni token id'si.  
    frequency: int # Eğitim anında bu pair'in token dizisi içinde kaç kez geçtiğini gösterir.


@register_tokenizer("byte_level_bpe")
class ByteLevelBPETokenizer(BaseTokenizer):
    """
    Eğitim amaçlı Byte-Level Byte Pair Encoding tokenizer implementasyonu.

    Byte-Level BPE, metni karakter ya da kelime seviyesinde değil, UTF-8 byte seviyesinde işler. 
    Bu nedenle tüm Unicode metinler önce byte dizisine çevrilir ve tokenizer bu byte id'leri üzerinde çalışır.

    Temel çalışma mantığı:
        1. Metin UTF-8 byte dizisine çevrilir.
        2. Her byte başlangıçta ayrı bir token olarak kabul edilir.
           Bu nedenle base vocabulary her zaman 0-255 arası 256 byte tokenından oluşur.
        3. Eğitim sırasında en sık yan yana gelen token çiftleri bulunur.
        4. Bu çiftler sırayla yeni token id'lerine merge edilir.
        5. Encode sırasında eğitimde öğrenilen merge kuralları aynı sırayla uygulanır.
        6. Decode sırasında token id'leri tekrar byte dizisine çevrilir ve UTF-8 decode edilir.

    Neden byte-level yaklaşım önemlidir?
        - Tüm UTF-8 karakterlerini temsil edebilir.
        - Türkçe karakterler, emoji, semboller ve daha önce görülmemiş karakterler için OOV problemi yaşamaz.
        - Character-level tokenizer'a göre daha iyi sıkıştırma sağlayabilir.
        - Word-level tokenizer'a göre daha esnek ve dil bağımsızdır.

    Not:
        Bu sınıf GPT-2 tokenizer'ın birebir kopyası değildir.
        Ama Byte-Level BPE algoritmasının temel mantığını sade, okunabilir ve test edilebilir şekilde göstermek için tasarlanmıştır.
    """

    def __init__(self, num_merges: int = 100) -> None:
        """
        ByteLevelBPETokenizer nesnesini başlatır.

        Args:
            num_merges:
                Eğitim sırasında öğrenilecek maksimum merge sayısıdır.
                
                Daha yüksek değer daha fazla sıkıştırma sağlayabilir; ancak küçük
                eğitim metinlerinde overfitting benzeri davranışlara ve daha uzun
                eğitim süresine yol açabilir.

        Raises:
            ValueError:
                num_merges değeri 1'den küçükse fırlatılır.
        """
        super().__init__(name="byte_level_bpe")

        # num_merges için minimum 1 değeri zorunludur. 
        # 0 veya negatif değerler mantıksızdır 
        # çünkü hiç merge olmaz ve tokenizer sadece byte seviyesinde kalır.
        if num_merges < 1:
            raise ValueError("num_merges must be at least 1")

        # num_merges, eğitim sırasında öğrenilecek maksimum merge sayısını belirler.
        self.num_merges = num_merges

        # Base vocabulary:
        # 0-255 arası her token id doğrudan bir byte değerini temsil eder.
        # Örneğin:
        #   97  -> b"a"
        #   98  -> b"b"
        #   195 -> "ü", "ç" gibi bazı Unicode karakterlerin UTF-8 byte parçalarından biri olabilir.
        #   128 -> Emoji veya diğer özel karakterlerin UTF-8 byte parçalarından biri olabilir.
        #   255 -> En yüksek byte değeri.
        # Bu mapping, encode ve decode işlemlerinde temel referans olarak kullanılır.

        # Byte-level yaklaşımın avantajı şudur:
        # Her metin önce UTF-8 byte dizisine çevrilebildiği için tokenizer temel düzeyde her karakteri temsil edebilir.
        self._id_to_bytes: dict[int, bytes] = {
            byte_id: bytes([byte_id]) for byte_id in range(256)
        }

        self._merge_lookup: dict[tuple[int, int], int] = {} # Merge edilen pair'i yeni token id'si ile eşleştiren bir lookup tablosu. 
        # Bu tablo encode sırasında hızlı erişim için kullanılır.   

        # Merge ile üretilecek yeni token id'leri 256'dan başlar.
        # Çünkü 0-255 arası id'ler base byte vocabulary için ayrılmıştır.
        self._next_token_id = 256

        # Eğitimde öğrenilen merge adımları sıralı tutulur.
        # BPE'de merge sırası kritik olduğu için encode sırasında bu liste aynı sırayla uygulanmalıdır.
        # Her merge adımı, hangi pair'in merge edildiği, yeni token id'si ve frekansı gibi bilgileri içerir.
        # Örnek:
        #   merge_steps = [ 
        #       ByteLevelBPEMerge(pair=(97, 98), merged_token_id=256, frequency=100),
        #       ByteLevelBPEMerge(pair=(256, 99), merged_token_id=257, frequency=50),
        #       ...
        #   ]
        # Bu yapı, eğitim sürecinde öğrenilen merge kurallarını ve sırasını açıkça saklar.
        self.merge_steps: list[ByteLevelBPEMerge] = []

        # Tokenizer'ın eğitilip eğitilmediğini açıkça takip eder.
        self._trained = False

    # ---------------------------------------------------------
    # TRAINING
    # ---------------------------------------------------------

    def train(self, text: str) -> None:
        """
        Verilen eğitim metni üzerinden Byte-Level BPE merge kurallarını öğrenir.

        Eğitim algoritması:
            1. Metin boş mu diye kontrol edilir.
            2. Önceki eğitim state'i temizlenir.
            3. Metin UTF-8 byte id listesine çevrilir.
            4. Yan yana gelen token çiftlerinin frekansları hesaplanır.
            5. En sık görülen pair seçilir.
            6. Bu pair için yeni token id oluşturulur.
            7. Pair'in byte karşılığı vocabulary'ye eklenir.
            8. Merge adımı merge_steps listesine kaydedilir.
            9. Pair, token dizisi üzerinde soldan sağa merge edilir.
            10. Bu işlem num_merges kadar veya merge edilecek pair kalmayana kadar devam eder.

        Deterministic seçim:
            Aynı frekansta birden fazla pair varsa seçim `(frequency, pair)` ile
            yapılır. Böylece aynı input metni için aynı merge sırası üretilir.

        Args:
            text:
                Eğitimde kullanılacak ham metin.

        Raises:
            ValueError:
                Eğitim metni boşsa veya yalnızca whitespace içeriyorsa.
        """
        # Eğitim metni boş veya sadece whitespace ise eğitim anlamsızdır ve bu durum açıkça hata olarak belirtilir.
        if not text or not text.strip():
            raise ValueError("Training text cannot be empty")

        # Önceki eğitim state'ini temizler. 
        # Bu, aynı tokenizer instance'ı birden fazla kez train edildiğinde eski merge kurallarının yeni eğitime sızmasını engeller.
        self._reset_training_state()

        # Metin UTF-8 byte id listesine çevrilir. Her karakter bir veya daha fazla byte'a karşılık gelebilir.
        # Örnek:
        #   text = "Merhaba"
        #   token_ids = [77, 101, 114, 104, 97, 98, 97] (ASCII karakterler tek byte ile temsil edilir)
        #   text = "😊"
        #   token_ids = [240, 159, 152, 138] (Emoji UTF-8 byte dizisi)
        #   text = "Türkçe"
        #   token_ids = [84, 195, 188, 114, 107, 195, 167, 101] (Türkçe karakterler UTF-8 byte dizisi)
        # Bu byte id'leri başlangıçta ayrı tokenlar olarak kabul edilir ve eğitim sürecinde merge edilerek yeni tokenlar oluşturulur.
        token_ids = list(text.encode("utf-8"))

        for _ in range(self.num_merges):
            # Token dizisindeki yan yana pair frekanslarını hesaplar.
            # Örnek:
            #   token_ids = [97, 98, 97, 98]
            #   Pairs: [(97, 98), (98, 97), (97, 98)]
            #   Result: Counter({(97, 98): 2, (98, 97): 1})
            # Bu frekanslar, hangi pair'in merge edileceğine karar vermede kullanılır.
            pair_frequencies = self._get_pair_frequencies(token_ids)

            # Eğer merge edilecek pair kalmazsa eğitim durdurulur.
            if not pair_frequencies:
                break

            # En sık pair'i seçer. 
            # Eğer birden fazla pair aynı frekansta ise deterministik seçim için pair'in kendisi de karşılaştırmaya dahil edilir.
            # Örnek:
            #   pair_frequencies = Counter({(97, 98): 2, (98, 97): 2})
            #   En sık pair'ler: [(97, 98), (98, 97)]
            #   Deterministik seçim için (frequency, pair) kullanılır:
            #   -   (2, (97, 98)) vs (2, (98, 97)) -> (2, (98, 97)) seçilir 
            # çünkü pair'ler karşılaştırıldığında (98, 97) > (97, 98) olur.
            # Bu sayede aynı input için aynı merge sırası öğrenilir ve sonuçlar stabil olur.
            best_pair, frequency = max(
                pair_frequencies.items(), # pair ve frekans bilgisi içeren dict items'ı
                key=lambda item: (item[1], item[0]), # Öncelikle frekansa göre, eşit frekansta ise pair'in kendisine göre sıralama yapar.
            )

            # Yeni token id'si oluşturulur. 
            # Base byte vocabulary 0-255 aralığını kullandığı için yeni tokenlar 256'dan başlar.
            # Örnek:
            #   İlk merge için best_pair = (97, 98) seçildiğinde merged_token_id = 256 olur.
            #   İkinci merge için best_pair = (256, 99) seçildiğinde merged_token_id = 257 olur.
            # Bu şekilde her merge adımında yeni token id'si sırayla atanır.
            merged_token_id = self._next_token_id # Yeni token id'si olarak sıradaki değeri alır.
            self._next_token_id += 1 # Sonraki merge için token id'si artırılır.    

            # Merge edilecek tokenların byte karşılıkları alınır.
            # Bu tokenlar base byte tokenı da olabilir, daha önce öğrenilmiş merge tokenı da.
            # Örnek:
            #   best_pair = (97, 98) -> left_bytes = b"a", right_bytes = b"b"
            #   best_pair = (256, 99) -> left_bytes = b"ab" (merge edilmiş tokenın byte karşılığı), right_bytes = b"c"
            # Bu byte karşılıkları, yeni merge tokenının byte karşılığını oluşturmak için birleştirilir.
            left_bytes = self._id_to_bytes[best_pair[0]] # Merge edilen pair'in ilk token id'sinin byte karşılığı alınır.
            right_bytes = self._id_to_bytes[best_pair[1]] # Merge edilen pair'in ikinci token id'sinin byte karşılığı alınır.

            # Yeni merge tokenının byte karşılığı, merge edilen iki tokenın byte'larının birleştirilmesiyle oluşturulur.
            # Örnek:
            #   left_bytes = b"a", right_bytes = b"b" -> merged_token_bytes = b"ab"
            #   left_bytes = b"ab", right_bytes = b"c" -> merged_token_bytes = b"abc"
            # Bu sayede merge tokenları, merge edilen tokenların byte'larının birleşimiyle temsil edilir.
            # Yeni merge tokenının byte karşılığı, token id'si ile eşleştirilerek saklanır.
            self._id_to_bytes[merged_token_id] = left_bytes + right_bytes

            # Merge adımı bilgisi saklanır. 
            # Bu, eğitim sürecinde öğrenilen merge kurallarını ve sırasını açıkça saklar.
            # Örnek:
            #   merge_steps = [
            #       ByteLevelBPEMerge(pair=(97, 98), merged_token_id=256, frequency=100),
            #       ByteLevelBPEMerge(pair=(256, 99), merged_token_id=257, frequency=50),
            #       ...
            #   ]
            # Bu yapı, eğitim sürecinde öğrenilen merge kurallarını ve sırasını açıkça saklar.
            merge_step = ByteLevelBPEMerge(
                pair=best_pair, # Merge edilen iki token id'si.
                merged_token_id=merged_token_id, # Bu pair birleştirildiğinde üretilecek yeni token id'si.
                frequency=frequency, # Eğitim anında bu pair'in token dizisi içinde kaç kez geçtiğini gösterir.
            )

            self.merge_steps.append(merge_step) # Öğrenilen merge adımını merge_steps listesine ekler.
            self._merge_lookup[best_pair] = merged_token_id # Merge edilen pair'i yeni token id'si ile eşleştiren bir lookup tablosu oluşturur. Bu tablo encode sırasında hızlı erişim için kullanılır.

            # Merge işlemi token dizisi üzerinde soldan sağa uygulanır.
            # Örnek:
            #   token_ids = [97, 98, 97, 98]
            #   best_pair = (97, 98), merged_token_id = 256
            #   Merge sonucu: [256, 256]
            # Merge işlemi sırasında aynı token birden fazla merge'e dahil edilmemelidir.
            # Eğitim token dizisi güncellenir.
            # Böylece sonraki iterasyonda artık yeni merge tokenları da pair frekanslarına dahil olur.
            token_ids = self._merge_pair(
                token_ids=token_ids, # Merge işlemi uygulanacak token id listesi.
                pair=best_pair, # Merge edilecek pair.
                new_token_id=merged_token_id, # Merge işlemi sonucunda oluşacak yeni token id.
            )

        # Eğitim tamamlandıktan sonra tokenizer'ın eğitildiği bilgisi güncellenir.
        self._trained = True

    # ---------------------------------------------------------
    # TOKENIZATION
    # ---------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        """
        Metni okunabilir Byte-Level BPE token listesine dönüştürür.

        Bu metod özellikle debug, karşılaştırma ekranları ve raporlama çıktıları
        için kullanışlıdır. Model girdisi için asıl temsil encode() metodunun
        döndürdüğü integer token id listesidir.

        Önemli:
            Byte-level tokenlar her zaman tek başına geçerli UTF-8 string olmak
            zorunda değildir. Örneğin bir Türkçe karakter veya emoji birden fazla
            byte'tan oluşabilir. Eğer bir token tek başına UTF-8 olarak decode
            edilemiyorsa raw bytes temsili döndürülür.

        Args:
            text:
                Tokenize edilecek ham metin.

        Returns:
            list[str]:
                Tokenların okunabilir string veya raw bytes temsilleri.

        Raises:
            ValueError:
                Tokenizer henüz train edilmemişse.
        """
        # Boş veya sadece whitespace içeren metinler için tokenize işlemi anlamsızdır ve bu durum açıkça boş liste döndürülerek belirtilir.
        if not text or not text.strip():
            return []

        # Tokenizer eğitilmemişse tokenize işlemi yapılamaz ve bu durum açıkça hata olarak belirtilir.
        if not self._trained:
            raise ValueError("Tokenizer has not been trained yet")

        # Metin önce encode() ile token id'lerine dönüştürülür, ardından her token id'nin byte karşılığı alınarak string token listesi oluşturulur.
        # Örnek:
        #   text = "ab"
        #   encode(text) -> [256] (merge edilmiş token id'si)
        #   _id_to_bytes[256] -> b"ab"
        #   decode b"ab" -> "ab"
        #   Sonuç: ["ab"]
        #   text = "abc"
        #   encode(text) -> [257] (merge edilmiş token id'si)
        #   _id_to_bytes[257] -> b"abc"
        #   decode b"abc" -> "abc"
        #   Sonuç: ["abc"]
        #   text = "a"
        #   encode(text) -> [97] (base byte token id'si)
        #   _id_to_bytes[97] -> b"a"
        #   decode b"a" -> "a"
        #   Sonuç: ["a"]
        token_ids = self.encode(text)

        tokens: list[str] = [] # Token id'lerinin string temsillerini saklayacak liste.

        for token_id in token_ids:
            # Her token id'nin byte karşılığı alınır. Eğer token id bilinmiyorsa hata fırlatılır.
            if token_id not in self._id_to_bytes:
                raise ValueError(f"Unknown token id: {token_id}")

            # Token id'sinin byte karşılığı alınır. 
            # Bu byte dizisi UTF-8 olarak decode edilmeye çalışılır. 
            # Eğer geçerli bir UTF-8 byte dizisi değilse, byte dizisinin temsilini string olarak ekler.
            token_bytes = self._id_to_bytes[token_id] # Token id'sinin byte karşılığı.

            try:
                tokens.append(token_bytes.decode("utf-8")) # Byte dizisi UTF-8 olarak decode edilir ve token listesine eklenir. 
            except UnicodeDecodeError:
                tokens.append(repr(token_bytes)) # Geçerli bir UTF-8 byte dizisi değilse, byte dizisinin temsilini string olarak ekler. Örnek: b'\xff\xfe' -> "b'\\xff\\xfe'"   

        return tokens

    # ---------------------------------------------------------
    # ENCODING
    # ---------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """
        Metni Byte-Level BPE token id listesine dönüştürür.

        Encode akışı:
            1. Tokenizer'ın train edilmiş olduğu doğrulanır.
            2. Metin UTF-8 byte id listesine çevrilir.
            3. Eğitimde öğrenilen merge kuralları sırayla uygulanır.
            4. Final token id listesi döndürülür.

        Merge sırası neden önemli?
            BPE'de her merge, sonraki merge adaylarını etkiler. Bu nedenle encode
            sırasında kurallar rastgele değil, train() sırasında öğrenildikleri
            sırayla uygulanmalıdır.

        Args:
            text:
                Encode edilecek ham metin.

        Returns:
            list[int]:
                Byte-Level BPE token id listesi.

        Raises:
            ValueError:
                Tokenizer henüz train edilmemişse.
        """
        # Encode işlemi için tokenizer'ın eğitilmiş olması gerekir. 
        # Eğer eğitim tamamlanmamışsa, encode işlemi yapılamaz ve bu durum açıkça hata olarak belirtilir.
        if not self._trained:
            raise ValueError("Tokenizer has not been trained yet")

        # Boş veya sadece whitespace içeren metinler için encode işlemi anlamsızdır ve bu durum açıkça boş liste döndürülerek belirtilir.
        if not text or not text.strip():
            return []

        # Metin UTF-8 byte id listesine çevrilir. Her karakter bir veya daha fazla byte'a karşılık gelebilir.
        # Örnek:
        #   text = "Merhaba"
        #   token_ids = [77, 101, 114, 104, 97, 98, 97] (ASCII karakterler tek byte ile temsil edilir)
        #   text = "😊"
        #   token_ids = [240, 159, 152, 138] (Emoji karakterler birden fazla byte ile temsil edilir)
        #   text = "Türkçe"
        #   token_ids = [84, 195, 188, 114, 107, 195, 167, 101] (Türkçe karakterler birden fazla byte ile temsil edilir)
        # Bu byte id'leri başlangıçta ayrı tokenlar olarak kabul edilir ve eğitim sürecinde merge edilerek yeni tokenlar oluşturulur.
        # Encode işlemi sırasında eğitimde öğrenilen merge kuralları sırayla uygulanır. Her merge adımında, token dizisi soldan sağa taranır ve eğer ardışık iki token, merge edilen pair ile eşleşirse bu iki token yerine merge token id'si eklenir.
        token_ids = list(text.encode("utf-8"))

        for merge_step in self.merge_steps:
            # Merge işlemi sırasında aynı token birden fazla merge'e dahil edilmemelidir.
            # Bu davranış BPE merge uygulamasının deterministic olmasını sağlar.
            # Örnek:
            #   token_ids = [1, 1, 1]
            #   merge_step.pair = (1, 1), merge_step.merged_token_id = 256
            #   Soldan sağa non-overlapping merge sonucu: [256, 1]
            #   Burada ortadaki token hem ilk çiftte hem ikinci çiftte kullanılamaz.
            token_ids = self._merge_pair(
                token_ids=token_ids, # Merge işlemi uygulanacak token id listesi.
                pair=merge_step.pair, # Merge edilecek pair.
                new_token_id=merge_step.merged_token_id, # Merge işlemi sonucunda oluşacak yeni token id.
            ) 

        return token_ids

    # ---------------------------------------------------------
    # DECODING
    # ---------------------------------------------------------

    def decode(self, token_ids: list[int]) -> str:
        """
        Token id listesini tekrar string'e dönüştürür.

        Decode akışı:
            1. Her token id'nin byte karşılığı alınır.
            2. Byte parçaları sırayla birleştirilir.
            3. Ortaya çıkan byte dizisi UTF-8 olarak decode edilir.

        Bu tokenizer whitespace bilgisini ayrıca saklamaz; ancak byte-level BPE
        tokenları orijinal byte dizisini koruduğu için encode/decode round-trip
        doğru token id'leriyle lossless çalışabilir.

        Args:
            token_ids:
                Decode edilecek token id listesi.

        Returns:
            str:
                Token id'lerden geri oluşturulan metin.

         Raises:
            ValueError:
                Tokenizer henüz train edilmemişse.
            ValueError:
                Bilinmeyen token id verilirse.
            ValueError:
                Token id'ler geçerli UTF-8 byte dizisi oluşturmuyorsa.
        """
        # Decode işlemi için tokenizer'ın eğitilmiş olması gerekir. 
        # Eğer eğitim tamamlanmamışsa, decode işlemi yapılamaz ve bu durum açıkça hata olarak belirtilir.
        if not self._trained:
            raise ValueError("Tokenizer has not been trained yet")

        byte_parts: list[bytes] = [] # Token id'lerinin byte karşılıklarını saklayacak liste.

        for token_id in token_ids:
            # Her token id'nin byte karşılığı alınır. Eğer token id bilinmiyorsa hata fırlatılır.
            if token_id not in self._id_to_bytes:
                raise ValueError(f"Unknown token id: {token_id}")

            # Token id'sinin byte karşılığı alınır ve byte_parts listesine eklenir.
            # Örnek:
            #   token_ids = [256, 257]
            #   _id_to_bytes[256] -> b"ab"
            #   _id_to_bytes[257] -> b"abc"
            #   byte_parts = [b"ab", b"abc"]
            # Bu byte parçaları daha sonra birleştirilerek tek bir byte dizisi oluşturulur ve UTF-8 olarak decode edilir.
            byte_parts.append(self._id_to_bytes[token_id])

        # Byte parçaları birleştirilir.
        # Örnek:
        #   byte_parts = [b"ab", b"abc"]
        #   raw_bytes = b"".join(byte_parts) -> b"ababc"
        # Bu birleşik byte dizisi UTF-8 olarak decode edilmeye çalışılır. Eğer geçerli bir UTF-8 byte dizisi değilse, decode işlemi başarısız olur ve hata fırlatılır.
        raw_bytes = b"".join(byte_parts)

        try:
            return raw_bytes.decode("utf-8") # Birleşik byte dizisi UTF-8 olarak decode edilir ve sonuç döndürülür.
        except UnicodeDecodeError as exc:
            raise ValueError(
                "Token ids do not form a valid UTF-8 byte sequence"
            ) from exc

    # ---------------------------------------------------------
    # INTERNAL HELPERS
    # ---------------------------------------------------------

    def _reset_training_state(self) -> None:
        """
        Eğitim state'ini başlangıç durumuna döndürür.

        Aynı tokenizer instance'ı birden fazla kez train edilebilir.
        Bu durumda eski merge kurallarının, eski vocabulary genişlemelerinin veya eski token id sayaçlarının yeni eğitime karışmaması gerekir.

        Bu metod:
            - base byte vocabulary'yi yeniden oluşturur
            - next token id değerini 256'ya çeker
            - merge_steps listesini temizler
            - trained state'i False yapar
        """
        # Base byte vocabulary her zaman 256 token içerir. Her token id doğrudan bir byte değerini temsil eder.
        # Örneğin:
        #   97  -> b"a"
        #   195 -> Türkçe karakterlerin UTF-8 byte parçalarından biri olabilir.
        #   128 -> Emoji veya diğer özel karakterlerin UTF-8 byte parçalarından biri olabilir.
        #   255 -> En yüksek byte değeri.
        # Bu mapping, encode ve decode işlemlerinde temel referans olarak kullanılır.
        self._id_to_bytes = {
            byte_id: bytes([byte_id]) for byte_id in range(256) # Base byte vocabulary 0-255 arası her token id doğrudan bir byte değerini temsil eder. 
        }

        self._next_token_id = 256 # Merge ile oluşacak yeni token id'leri 256'dan başlar.

        self.merge_steps = [] # Eğitimde öğrenilen merge adımları sıralı tutulur. 
        # BPE'de merge sırası kritik olduğu için encode sırasında bu liste aynı sırayla uygulanmalıdır. 
        # Her merge adımı, hangi pair'in merge edildiği, yeni token id'si ve frekansı gibi bilgileri içerir. 
        # Bu yapı, eğitim sürecinde öğrenilen merge kurallarını ve sırasını açıkça saklar. 

        self._merge_lookup = {} # Merge edilen pair'i yeni token id'si ile eşleştiren bir lookup tablosu oluşturur. 
        # Bu tablo encode sırasında hızlı erişim için kullanılır. Örnek: {(97, 98): 256, (256, 99): 257, ...}
        # Bu lookup tablosu, encode işlemi sırasında merge edilen pair'lerin hızlı bir şekilde yeni token id'lerine dönüştürülmesini sağlar.

        self._trained = False # Tokenizer'ın eğitilip eğitilmediğini açıkça takip eder. 
        # Eğitim state'i sıfırlandığında tokenizer'ın eğitilmediği durumu temsil eder.

    def _get_pair_frequencies(self, token_ids: list[int]) -> Counter[tuple[int, int]]:
        """
        Token dizisindeki ardışık token çiftlerinin frekanslarını hesaplar.

        BPE training aşamasında en sık görülen token çiftini bulmak gerekir.
        Bu metod, mevcut token dizisini soldan sağa tarar ve yan yana gelen
        tüm pair'leri Counter ile sayar.

         Örnek:
            token_ids = [97, 98, 97, 98]

            Ardışık pair'ler:
                (97, 98)
                (98, 97)
                (97, 98)

            Result:
                Counter({
                    (97, 98): 2,
                    (98, 97): 1
                })

        Args:
            token_ids:
                Pair frekansı hesaplanacak token id listesi.

        Returns:
            Counter[tuple[int, int]]:
                Pair -> frekans bilgisini içeren Counter nesnesi.
        """
        # zip(token_ids, token_ids[1:]) ifadesi, token_ids listesini kendisiyle bir pozisyon kaydırarak eşleştirir ve yan yana gelen token çiftlerini oluşturur.
        # Örnek:
        #   token_ids = [97, 98, 97, 98]
        #   token_ids[1:] = [98, 97, 98]
        #   zip(token_ids, token_ids[1:]) -> [(97, 98), (98, 97), (97, 98)]
        # Bu çiftler Counter ile sayılarak her bir pair'in frekansı hesaplanır.
        #  Bu frekanslar, eğitim sürecinde hangi pair'in merge edileceğine karar vermede kullanılır.
        return Counter(zip(token_ids, token_ids[1:]))

    def _merge_pair(
        self,
        token_ids: list[int],
        pair: tuple[int, int],
        new_token_id: int,
    ) -> list[int]:
        """
        Belirli bir token çiftini yeni token id ile değiştirir.

        Bu metod tek bir BPE merge operasyonunu uygular.
        Token listesi soldan sağa taranır. Ardışık iki token hedef pair ile
        eşleşirse bu iki token yerine new_token_id eklenir.

        Örnek:
            token_ids = [1, 2, 1, 2, 3]
            pair = (1, 2)
            new_token_id = 256

            Output:
                [256, 256, 3]

        Non-overlapping merge davranışı:
            Aynı token aynı merge adımında birden fazla pair'e dahil edilemez.

            Örnek:
                token_ids = [1, 1, 1]
                pair = (1, 1)

            Soldan sağa merge sonucu:
                [256, 1]

            Ortadaki token hem ilk çiftte hem ikinci çiftte kullanılamaz.
            Bu davranış deterministic ve beklenen BPE merge davranışıdır.

        Args:
            token_ids:
                Merge uygulanacak token id listesi.

            pair:
                Birleştirilecek ardışık token çifti.

            new_token_id:
                Pair eşleştiğinde output'a yazılacak yeni token id.

        Returns:
            list[int]:
                Merge uygulanmış yeni token id listesi.
        """
        output: list[int] = [] # Merge işlemi sonucunda oluşacak yeni token id listesi.
        index = 0 # token_ids listesinde gezinmek için kullanılan indeks.

        while index < len(token_ids):
            # Mevcut token ve bir sonraki token pair ile eşleşiyor mu kontrol edilir.
            if (
                index < len(token_ids) - 1 # Son token'da pair kontrolü yapılmaz.
                and token_ids[index] == pair[0] # İlk token pair'in ilk elemanına eşleşiyor mu?
                and token_ids[index + 1] == pair[1] # İkinci token pair'in ikinci elemanına eşleşiyor mu?
            ):
                output.append(new_token_id) # Pair eşleştiğinde yeni token id'si output listesine eklenir.
                index += 2 # Pair eşleştiği için iki token atlanır.
            else:
                output.append(token_ids[index]) # Pair eşleşmiyorsa mevcut token olduğu gibi output listesine eklenir.
                index += 1 # Sonraki token'a geçilir.

        return output

    # ---------------------------------------------------------
    # VOCABULARY
    # ---------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """
        Mevcut vocabulary boyutunu döndürür.

        Vocabulary iki bölümden oluşur:
            1. Base byte vocabulary:
                0-255 arası 256 temel byte tokenı.

            2. Learned merge vocabulary:
                train() sırasında öğrenilen merge tokenları.
                Bu tokenlar 256'dan başlayan id'lerle eklenir.

        Eğitim öncesi:
            vocab_size = 256

        Eğitim sonrası:
            vocab_size = 256 + öğrenilen merge token sayısı

        Not:
            num_merges maksimum merge sayısını belirtir. Eğitim metni çok kısaysa
            veya merge edilecek pair kalmazsa öğrenilen merge sayısı num_merges'ten
            daha az olabilir.

        Returns:
            int:
                Vocabulary içinde bulunan toplam token sayısı.
        """
        return len(self._id_to_bytes)