from __future__ import annotations

from tokenizer_workshop.tokenizers.base import BaseTokenizer
from tokenizer_workshop.tokenizers.registry import register_tokenizer
from tokenizer_workshop.trainers import BPETrainer, MergeStep


@register_tokenizer("byte_bpe")
class ByteBPETokenizer(BaseTokenizer):
    """
    Byte seviyesinde çalışan bir BPE tokenizer.
    Utf-8 byte'ları üzerinde çalışan, BPE merge'leri öğrenen ve uygulayan bir tokenizer.

    Eğitsel amaç:
    - ByteTokenizer ile SimpleBPETokenizer'ın fikirlerini birleştirir.
    - Metnin önce UTF-8 byte'larına dönüştürülüp ardından BPE merge rules ile sıkıştırılabileceğini gösterir.
    - Production-grade tokenizer'ların (örneğin GPT-2, GPT-4) neden byte-level BPE
      tercih ettiğini somutlaştırır: byte tabanı her dili/emoji'yi kaybı olmadan
      taşıyabilirken BPE merge'leri dizinin uzunluğunu kısaltır.

    Temel fikir:
    1. Metin UTF-8 byte'larına çevrilir (0-255 arası değerler).
    2. Her byte, benzersiz tek karakterlik bir sembole map edilir (chr(b)).
       Bu sayede mevcut BPETrainer'ı (string üzerinde çalışan) hiç değiştirmeden
       kullanilabilinir. Sembol tercihi keyfidir; önemli olan map'in bijective olması.
    3. BPETrainer, sembol dizisi üzerinde byte çiftlerini (ve sonra merged çiftleri)
       birleştirmeyi öğrenir.
    4. Encoding sırasında aynı dönüşüm ve merge rules sırayla uygulanır.
    5. Decoding sırasında token id'ler sembollere, semboller byte'lara, byte'lar
       tekrar UTF-8 metne çevrilir.

    SimpleBPETokenizer'dan önemli farkı:
    - SimpleBPETokenizer vocabulary'yi training text'indeki characters üzerinden kurar.
      Bu yüzden training sırasında görülmemiş bir karakter encode edilemez.
    - ByteBPETokenizer'ın base vocabulary'si her zaman 256 byte'ı içerir. Bu yüzden
      training sırasında görülmemiş karakterler bile (farklı dil, emoji vb.) encode
      edilebilir; sadece merge faydası azalır.

    Sınır:
    - Çok byte'lı karakterler birden fazla base token olarak başlar; yeterli merge
      öğrenilmezse dizi uzun kalır.
    - Decode aşaması, byte dizisinin geçerli UTF-8 oluşturmasını gerektirir.
    """

    def __init__(self, num_merges: int = 10) -> None:
        super().__init__(name="byte_bpe")

        # num_merges, tokenizer'ın öğrenmeye çalışacağı merge sayısını belirler. 
        # Eğitim sırasında öğrenilen merge rules'un sayısı, tokenizer'ın ne kadar etkili bir şekilde metni sıkıştırabileceğini belirler. 
        # Çok yüksek bir değer, eğitim süresini uzatır ve küçük metinler için overfitting'e neden olabilir. 
        # Çok düşük bir değer ise sınırlı sıkıştırma sağlar.
        # Eğitim sırasında öğrenilen merge rules, encoding sırasında uygulanır ve tokenization'ın ne kadar etkili olduğunu belirler.
        # Bu parametre, eğitim sırasında öğrenilen merge rules'un sayısını kontrol eder.
        if num_merges < 1: # En az 1 merge öğrenilmesi gerekir; aksi takdirde BPE'nin anlamı kalmaz.
            raise ValueError("num_merges must be at least 1.")

        self.num_merges = num_merges # Eğitim sırasında öğrenilecek merge sayısını saklar.
        self.trainer = BPETrainer() # BPE merge rules öğrenmek için kullanılan trainer'ı başlatır.

        # Training sırasında öğrenilen merge steps'leri, öğrenildikleri sırayla tutar.
        # MergeStep, bir merge işleminin detaylarını (birleştirilen çift ve ortaya çıkan token) tutan bir veri yapısıdır.
        # Bu liste, eğitim sırasında öğrenilen merge rules'un sırasını korur, bu da encoding sırasında aynı sırayla uygulanmalarını sağlar.
        # Merge steps, encoding sırasında sırayla uygulanır; bu yüzden öğrenildikleri sırayı korumak önemlidir.
        self.merge_steps: list[MergeStep] = [] # Eğitim sırasında öğrenilen merge steps'leri saklamak için bir liste oluşturur.

        # Byte'lar 0-255 arasında olduğu için, chr(b) ile her byte'ı tek karakterlik bir sembole map eder.
        base_tokens = [chr(b) for b in range(256)]

        # Final learned vocabulary için token <-> id mapping'leri oluşturur.
        # Token id'leri, token'ların vocab_tokens listesindeki indekslerine göre atanır.

        # Token -> id mapping'i oluşturur. Her token'ın id'si, base_tokens listesindeki indeksine karşılık gelir.
        self._stoi: dict[str, int] = {token: idx for idx, token in enumerate(base_tokens)}
        # id -> token mapping'i oluşturur. Her id, base_tokens listesindeki indeksine karşılık gelir.
        self._itos: dict[int, str] = {idx: token for token, idx in self._stoi.items()}

        # Tokenizer'ın eğitilip eğitilmediğini takip eder. 
        # Eğitim tamamlanmadan encode/decode işlemi yapılamaz.
        # Encoding veya decoding işlemleri, tokenizer eğitilmeden önce çağrılırsa hata verir.  
        self._is_trained = False 

    # ---------------------------------------------------------------------
    # Byte <-> sembol dönüşümü
    # ---------------------------------------------------------------------
    @staticmethod
    def _bytes_to_symbols(data: bytes) -> list[str]:
        """
        Byte dizisini, her byte'ı tek karakterlik bir sembole çevirerek
        string listesine dönüştürür.
        Örneğin, byte değeri 65 (ASCII 'A') için chr(65) = 'A' olur.
        Bu dönüşüm, trainer'ın string sembollerle çalışmasına olanak tanır.
        Ayrıca byte'ların kayıpsız bir şekilde temsil edilmesini sağlar.

        Neden bu dönüşüm?
        Projenin BPETrainer'ı string sembolleri üzerinde çalışır. Byte'ları doğrudan
        integer olarak verseydik trainer'ı değiştirmek gerekirdi. chr(b) her byte'ı
        benzersiz bir Unicode code point'ine map eder; bu map bijective olduğu için
        kayıpsız geri dönüşüm mümkündür.
        """
        # Bu sayede trainer'ı değiştirmeden byte'lar üzerinde BPE merge'leri öğrenebilir.

        # Byte'ların 0-255 arası değerler olduğunu varsayarsak, chr(b) her byte'ı benzersiz bir karaktere dönüştürür.
        # Bu dönüşüm, trainer'ın string sembollerle çalışmasına olanak tanır.
        # Aynı zamanda byte'ların kayıpsız bir şekilde temsil edilmesini sağlar.
        # Örneğin, byte dizisi b'\x41\x42\x43' (ASCII 'ABC') için bu fonksiyon ['A', 'B', 'C'] döndürür.
        # Bu dönüşüm, encoding sırasında byte'ların tek karakterlik sembollere dönüştürülmesini ve 
        # decoding sırasında sembollerin tekrar byte'lara dönüştürülmesini sağlar.
        return [chr(b) for b in data]  

    @staticmethod
    def _symbols_to_bytes(symbols: list[str]) -> bytes:
        """
        Merged sembol dizisini tekrar byte dizisine çevirir.

        Merged tokens birden fazla base sembolün birleşmesiyle oluştuğundan,
        her sembol içindeki her karakter tek bir byte'a karşılık gelir.
        """
        # Sembolleri byte'lara dönüştürür.
        # Her sembol bir veya daha fazla karakter içerebilir; her karakter tek bir byte'ı temsil eder.
        # Örneğin, sembol "AB" iki karakter içerir ve bunlar sırasıyla byte değerleri 65 ('A') ve 66 ('B')'ye karşılık gelir.

        # Bu fonksiyon, her sembolün içindeki karakterleri tek tek ele alır ve ord() ile byte değerlerine dönüştürür.
        # Bu dönüşüm, encoding sırasında sembollerin byte'lara dönüştürülmesini ve decoding sırasında byte'ların tekrar sembollere dönüştürülmesini sağlar.

        byte_values: list[int] = [] # Sembollerin içindeki karakterlerin byte değerlerini saklamak için bir liste oluşturur.

        # Her sembolün içindeki karakterleri tek tek ele alır 
        for symbol in symbols: 
            # Her karakter tek bir byte'ı temsil eder, 
            # sembolün her karakteri için ord() ile byte değerini alır ve byte_values listesine ekler.
            for ch in symbol: 
                # ord() fonksiyonu, bir karakterin Unicode code point'ini (integer) döndürür. 
                # Bu integer değeri byte olarak kullanılır. 
                # Örneğin, ord('A') = 65, ord('B') = 66, ord('C') = 67 gibi. 
                # Bu sayede sembollerin içindeki karakterler byte değerlerine dönüştürülür ve byte dizisi oluşturulabilir. 
                byte_values.append(ord(ch)) # sembolün içindeki her karakterin byte değerini byte_values listesine ekler.

        # Byte değerlerini bytes() fonksiyonu ile byte dizisine dönüştürür ve döndürür.
        return bytes(byte_values)

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------
    def train(self, text: str) -> None:
        """
        Tokenizer'ı raw text üzerinde train eder.

        Training flow:
        1. Metin UTF-8 byte'larına çevrilir.
        2. Byte'lar tek karakterlik sembollere map edilir.
        3. BPETrainer bu semboller üzerinde merge rules öğrenir.
        4. Final vocabulary şu iki kaynaktan oluşturulur:
           - 256 base byte token'ı (her zaman eksiksiz dahil edilir)
           - training sırasında öğrenilen merged tokens

        Neden 256 base token'ı her zaman dahil ediyoruz?
        Training text'inde bazı byte değerleri hiç görünmeyebilir. Ama tokenizer
        daha sonra farklı bir metinle (örneğin bir emoji veya farklı dil) encode
        edilmek istenirse, bu byte'ların da id'si olmalıdır. 256 byte'ı sabit
        tabana almak tokenizer'ı robust kılar.
        """
        
        # Eğitim metni boş olamaz; 
        # bu durumda öğrenilecek merge rules olmaz ve tokenizer'ın anlamı kalmaz.

        # Bu kontrol, kullanıcıya geçerli bir eğitim metni sağlaması gerektiğini açıkça belirtir.
        # Eğitim metni boşsa, bu durumun oluşmaması gerekir çünkü eğitim metni sağlanmadan önce train() çağrılmaz.
        if not text: 
            raise ValueError("Training text cannot be empty.")

        # 1) Metni byte sembollerine dönüştür ve trainer'ın kabul ettiği string
        #    formuna getir. SimpleBPETokenizer da trainer'a raw text veriyor;
        #    orada list(text) yapıldığında karakterler elde ediliyor. Biz burada
        #    sembolleri birleştirdiğimizde trainer aynı şekilde sembolleri çıkarır.
        symbols = self._bytes_to_symbols(text.encode("utf-8")) # Metni UTF-8 byte'larına çevirir ve ardından byte'ları tek karakterlik sembollere dönüştürür.
        training_text = "".join(symbols) # Sembolleri tek bir string olarak birleştirir. 
        # Trainer, string üzerinde çalışır, bu yüzden sembolleri birleştirmek gerekir.

        self.merge_steps = self.trainer.train(training_text, num_merges=self.num_merges) # BPETrainer ile merge rules öğrenir ve merge steps'leri saklar.
        # Merge steps, encoding sırasında sırayla uygulanır; bu yüzden öğrenildikleri sırayı korumak önemlidir.
        # Merge steps, token dizisini sırayla sıkıştırır; bu yüzden öğrenildikleri sırayı korumak önemlidir.

        # Base vocabulary zaten __init__'te kuruldu; sadece yeni merged tokens
        # eklenir. Duplicate'ler atlanır (nadiren de olsa byte sembolüyle çakışabilir).
        next_id = len(self._stoi) # Yeni token'lara atanacak id'lerin başlangıç noktası, mevcut vocabulary'nin boyutudur.

        # Merge steps'leri sırayla ele alır ve her merge step'in ortaya çıkan merged token'ını vocabulary'ye ekler.
        # Merge step'inin ortaya çıkan merged token'ını alır. Eğer bu token zaten vocabulary'de yoksa, yeni bir id atar ve mapping'leri günceller.
        # Bu süreç, eğitim sırasında öğrenilen merged tokens'ı base byte token'larının üzerine ekler.
        # Merge'ler sırayla uygulandığı için, önce "g" ve "e" merge'lenirse "ge" token'ı oluşur; sonra "ge" ve "ç" merge'lenirse "geç" token'ı oluşur.
        # Merge'ler farklı sırayla uygulanırsa farklı tokenizasyon sonuçları ortaya çıkabilir; bu yüzden sıranın korunması önemlidir.
        # # Merge'ler sırayla uygulanır; bu yüzden öğrenildikleri sırayı korumak önemlidir.
        # Merge'ler, token dizisini sırayla sıkıştırır; bu yüzden öğrenildikleri sırayı korumak önemlidir.
        for step in self.merge_steps: # Her merge step'i sırayla ele alır.
            token = step.merged_token # Merge step'inin ortaya çıkan merged token'ını alır.
    
            if token not in self._stoi: # Eğer merged token zaten vocabulary'de yoksa, yeni bir id atar ve mapping'leri günceller.
                self._stoi[token] = next_id # Merged token'a yeni bir id atar.
                self._itos[next_id] = token # Yeni id'nin karşılık geldiği token'ı _itos mapping'ine ekler.
                next_id += 1 # Sonraki token için id'yi artırır.

    # ---------------------------------------------------------------------
    # Encoding
    # ---------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        """
        Text'i, öğrenilmiş byte-level BPE merge rules kullanarak token ids'e encode eder.

        Encoding flow:
        1. Metin UTF-8 byte'larına çevrilir.
        2. Byte'lar tek karakterlik sembollere map edilir.
        3. Öğrenilen her merge step sırayla uygulanır.
        4. Final tokens integer ids'e dönüştürülür.

        Important teaching point:
        Base vocabulary tüm byte'ları içerdiği için, training sırasında hiç
        görülmemiş bir karakter bile (örneğin yeni bir emoji) encode edilebilir.
        Sadece ilgili merge rules öğrenilmediği için çıktı dizisi daha uzun olur.
        Bu durum, byte-level BPE'nin esnekliğini gösterir: yeni karakterler encode edilebilir, 
        ancak merge'ler öğrenilmediği sürece sıkıştırma sağlanmaz.
        """

        # encode() çağrıldığında merge_steps'in boş olması da bu durumun oluşmaması gerekir 
        # çünkü merge_steps, train() sırasında doldurulur.
        # Bu kontrol, kullanıcıya tokenizer'ın eğitilmesi gerektiğini açıkça belirtir.
        # Eğer tokenizer eğitilmeden önce encode() çağrılırsa, 
        # bu durumun oluşmaması gerekir çünkü tokenizer'ın train() metodu çağrılmadan önce encode() çağrılmaz.
        if not self.merge_steps:
            raise ValueError("Tokenizer has not been trained yet.")

        # Metni byte sembollerine dönüştürür.
        # Metin önce UTF-8 byte'larına çevrilir, 
        # ardından byte'lar tek karakterlik sembollere map edilir.
        # Bu sayede, eğitim sırasında hiç görülmemiş karakterler bile encode edilebilir; 
        # çünkü base vocabulary tüm byte'ları içerir.
        tokens = self._bytes_to_symbols(text.encode("utf-8"))

        # Merge rules, öğrenildikleri sırayla birebir uygulanır.
        # Bu sıranın korunması önemlidir 
        # çünkü farklı sırayla uygulanan merge'ler farklı tokenizasyon sonuçları verebilir.
        # Merge'ler, token dizisini sırayla sıkıştırır; bu yüzden öğrenildikleri sırayı korumak önemlidir.
        for step in self.merge_steps: # Her merge step'i sırayla uygulanır.
            # merge_pair, token dizisinde belirtilen pair'ı merged_token ile birleştirir ve yeni token dizisini döndürür.
            tokens = self.trainer.merge_pair(tokens, step.pair, step.merged_token) 
            # Bu işlem, token dizisini sırayla sıkıştırır; bu yüzden öğrenildikleri sırayı korumak önemlidir.
            # Örneğin, "g" ve "e" merge'lenerek "ge" token'ı oluşturulmuşsa, bu merge step'i uygulandığında "g" ve "e" yan yana geldiğinde "ge" token'ı oluşur.
            # Merge'ler sırayla uygulandığı için, önce "g" ve "e" merge'lenirse "ge" token'ı oluşur; sonra "ge" ve "ç" merge'lenirse "geç" token'ı oluşur.
            # Merge'ler farklı sırayla uygulanırsa farklı tokenizasyon sonuçları ortaya çıkabilir; bu yüzden sıranın korunması önemlidir.

        token_ids: list[int] = [] # Merge'ler uygulandıktan sonra elde edilen token'ların id'lerini saklamak için bir liste oluşturur.

        # Her token için, token'ın id'sini _stoi mapping'inden alır ve token_ids listesine ekler.
        for token in tokens:
            # Eğer token _stoi'da yoksa, bu durumun oluşmaması gerekir 
            # çünkü tokenizer'ın eğitilmeden önce encode() çağrılmaz ve train() sırasında tüm token'lar _stoi'ya eklenir.
            if token not in self._stoi:
                # Bu durumun oluşmaması gerekir çünkü base vocabulary tüm byte'ları
                # içeriyor. Yine de defensive bir kontrol olarak bırakılmıştır.
                raise ValueError(f"Unknown token encountered during encoding: {token!r}")

            token_ids.append(self._stoi[token]) # token'ın id'sini _stoi mapping'inden alır ve token_ids listesine ekler.

        # Final token id'lerini döndürür. Bu id'ler, modelin input'u olarak kullanılabilir.
        return token_ids

    def tokenize(self, text: str) -> list[str]:
        """
        CompareManager ile uyumlu olması için eklenmiş wrapper metottur.

        encode() integer token id döndürür,
        fakat compare sistemi string token listesi bekler.

        Bu yüzden:
        - encode() çağrılır
        - id'ler tekrar token string'lerine çevrilir
        """

        token_ids = self.encode(text)

        # id -> token (string) dönüşümü
        return [self._itos[token_id] for token_id in token_ids]

    # ---------------------------------------------------------------------
    # Decoding
    # ---------------------------------------------------------------------
    def decode(self, token_ids: list[int]) -> str:
        """
        Token ids'i tekrar text'e decode eder.

        Decoding flow:
        1. Her id, karşılık gelen sembol dizisine (bir veya daha fazla byte) çevrilir.
        2. Tüm semboller birleştirilip byte dizisine dönüştürülür.
        3. Byte dizisi UTF-8 olarak decode edilir.

        Önemli öğretici nokta:
        Byte-level BPE'nin cazip yanı şudur: merged tokens rastgele byte sınırlarında
        kesilmiş olsa bile birleştirildiğinde orijinal byte dizisini aynen geri verir.
        Yani "ge" ve "çerli" merge'leri olmasa bile, tokenizer "geçerli" kelimesini
        byte düzeyinde her zaman doğru decode eder.

        eğitim sırasında hiç görülmemiş karakterler bile decode edilebilir; 
        çünkü base vocabulary tüm byte'ları içerir. 
        Ancak, merge'ler öğrenilmediği sürece çıktı dizisi daha uzun olur.
        Bu durum, byte-level BPE'nin esnekliğini gösterir: yeni karakterler decode edilebilir, 
        ancak merge'ler öğrenilmediği sürece sıkıştırma sağlanmaz.
        """
      
        # Bu kontrol, tokenizer'ın eğitilmeden önce decode işlemi yapılamayacağını açıkça belirtir.
        # Tokenizer eğitilmeden önce decode() çağrılırsa, bu durumun oluşmaması gerekir 
        # çünkü tokenizer'ın train() metodu çağrılmadan önce decode() çağrılmaz.
        if not self.merge_steps:
            raise ValueError("Tokenizer has not been trained yet.")

        symbols: list[str] = [] # Token id'lerin karşılık geldiği sembolleri saklamak için bir liste oluşturur.

        # Her token id için, token id'nin karşılık geldiği sembolü _itos mapping'inden alır ve symbols listesine ekler.
        for token_id in token_ids:
            # Eğer token_id _itos'da yoksa, bu durumun oluşmaması gerekir 
            # çünkü tokenizer'ın eğitilmeden önce decode() çağrılmaz ve 
            # train() sırasında tüm token'lar _itos'a eklenir.
            if token_id not in self._itos:
                raise ValueError(
                    f"Unknown token id encountered during decoding: {token_id}"
                )
            symbols.append(self._itos[token_id]) # token id'nin karşılık geldiği sembolü _itos mapping'inden alır ve symbols listesine ekler.

        byte_seq = self._symbols_to_bytes(symbols) # Sembolleri byte'lara dönüştürür.
        # Byte dizisini UTF-8 olarak decode eder ve döndürür.

        # Byte dizisi geçerli bir UTF-8 oluşturmazsa, bu durumun oluşmaması gerekir
        # çünkü encoding sırasında metin önce UTF-8 byte'larına çevrilir ve 
        # ardından byte'lar tek karakterlik sembollere map edilir.
        try:
            return byte_seq.decode("utf-8") # Byte dizisini UTF-8 olarak decode eder ve döndürür.
        except UnicodeDecodeError as exc:
            raise ValueError(
                "Token ids do not form a valid UTF-8 byte sequence."
            ) from exc

    # ---------------------------------------------------------------------
    # Vocabulary
    # ---------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        """
        Tokenizer'ın şu anda bildiği token sayısını döndürür.

        Buna şunlar dahildir:
        - 256 base byte token
        - training sırasında öğrenilen merged BPE tokens

        Dolayısıyla training sırasında tüm merge'ler benzersizse vocab_size
        yaklaşık olarak 256 + num_merges olur.
        """
        return len(self._stoi) # Vocabulary'deki token sayısını döndürür. 
        # Bu sayı, 256 base byte token'ı ve training sırasında öğrenilen merged tokens'ı içerir.