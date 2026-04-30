from __future__ import annotations

import re
import math
from collections import Counter

from tokenizer_workshop.tokenizers.base import BaseTokenizer
from tokenizer_workshop.tokenizers.registry import register_tokenizer


@register_tokenizer("unigram")
class UnigramTokenizer(BaseTokenizer):
    """
    Basitleştirilmiş Unigram Language Model tabanlı tokenizer.

    Bu tokenizer, SentencePiece kütüphanesinde kullanılan Unigram yaklaşımının
    eğitim amaçlı sadeleştirilmiş bir versiyonudur.

    Unigram tokenization, klasik tokenizer’lardan (word, char, regex vb.)
    farklı olarak deterministik değil, olasılıksal bir parçalama yaklaşımı kullanır.

    Temel mantık:
        1. Eğitim verisinden olası subword (alt parça) adayları çıkarılır. 
        2. Bu adayların frekansları hesaplanır.
        3. En sık geçen adaylar vocabulary (sözlük) olarak seçilir.
        4. Her token için bir olasılık (probability) atanır.
        5. Tokenization sırasında kelime, en yüksek olasılığa sahip parçalamaya bölünür.
        6. En yüksek toplam olasılığa sahip olan parçalama seçilir.

    Bu implementasyon:
        - Eğitim sırasında frekansa dayalı probabilistic vocab kurar
        - Encode sırasında greedy + probabilistic yaklaşım kullanır
        - Olasılık hesaplaması için Laplace smoothing (add-one) kullanır

    Örnek:
        "tokenizer" kelimesi şu şekillerde bölünebilir:

            ["token", "izer"]
            ["tok", "en", "izer"]
            ["tokenizer"]

        Model, bu seçenekler arasından en yüksek toplam olasılığa sahip olanı seçer.

    Notlar:
        - Bu implementasyon production-level SentencePiece değildir.
        - Eğitim yapılmadan encode/decode çalıştırılamaz.
        - Eğitim öncesinde tokenize() metodu fallback olarak basic regex tokenization kullanır.
        - decode işlemi whitespace'i birebir geri üretmez (lossy olabilir).
    """

    # UNK token'ı her zaman vocab'da bulunur ve id'si 0'dır
    UNKNOWN_TOKEN = "[UNK]" # UNK token'ı, bilinmeyen veya eğitim sırasında görülmemiş parçaları temsil eder

    def __init__(
        self,
        vocab_size: int = 100,
        max_subword_length: int = 12,
    ) -> None:
        super().__init__(name="unigram")

        # vocab_size en az 2 olmalıdır (1 UNK + en az 1 gerçek token)   
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")

        # max_subword_length, eğitim sırasında çıkarılacak substring'lerin maksimum uzunluğunu belirler.
        if max_subword_length < 1:
            raise ValueError("max_subword_length must be at least 1")

        # UNK token'ı her zaman vocab'da bulunur ve id'si 0'dır, 
        # bu yüzden target_vocab_size = vocab_size - 1 gerçek token'ı içerir.    
        self.target_vocab_size = vocab_size - 1
        self.max_subword_length = max_subword_length 

        # Eğitim sırasında oluşturulacak vocab ve log probability tabloları
        self._token_to_id: dict[str, int] = {} # token -> id mapping
        self._id_to_token: dict[int, str] = {} # id -> token mapping
        self._token_logprob: dict[str, float] = {} # token -> log probability mapping

        # Eğitim yapılıp yapılmadığını takip eder
        self._trained = False

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------

    def train(self, text: str) -> None:
        """
        Verilen metin üzerinden unigram vocabulary oluşturur.

        Bu metod tokenizer'ın "öğrenme" aşamasıdır. 

        İşleyiş adımları:

            1. Girdi doğrulama:
                Metnin boş veya sadece whitespace olmadığından emin olunur.

            2. Basic tokenization:
                Metin regex ile kelime + noktalama seviyesinde parçalanır.

            3. Subword aday üretimi:
                Her kelimenin tüm olası substring parçaları çıkarılır.

            4. Frekans hesaplama:
                Her substring’in kaç kez geçtiği sayılır.

            5. Vocabulary seçimi:
                En sık geçen tokenlar seçilerek hedef vocab boyutuna ulaşılır.

            6. [UNK] token eklenir:
                Bilinmeyen parçalar için özel token.

            7. Olasılık hesaplama:
                Her token için frekansa dayalı log-probability hesaplanır.

        Neden substring?
            Unigram yaklaşımı en iyi parçalamayı öğrenmeye çalışır.
            Bu yüzden mümkün olan tüm parçalar aday olarak değerlendirilir.

        Neden log-probability?
            - Çok küçük olasılıkların çarpımı sayısal hatalara yol açabilir
            - Eğitim sırasında görülen token'ların olasılıklarını öğrenir, böylece encode sırasında daha iyi parçalamalar yapabilir.
            - Encode sırasında, kelimenin tüm parçalamalarını değerlendirirken, her parçanın log-probability'sini toplar ve en yüksek toplam log-probability'ye sahip parçalamayı seçer.
                - log kullanarak çarpma → toplama dönüşür
            - daha stabil hesaplama sağlar
            - greedy seçim yaparken en yüksek toplam olasılığı bulmak kolaylaşır
            - Laplace smoothing (add-one) ile sıfır frekans problemini önler
            - Olasılıkların normalize edilmesine gerek kalmaz, sadece sıralama için yeterlidir
            - Decode sırasında, token id'lerini token'lara çevirirken, eğer token id'si bilinmeyen bir token'a karşılık geliyorsa [UNK] token'ını kullanır. 
            - Eğitim sırasında, token'ların frekanslarına dayalı olarak log-probability'ler hesaplanır. 
              Bu, encode sırasında hangi parçaların daha olası olduğunu belirlemek için kullanılır. 
              Daha sık görülen parçalar daha yüksek log-probability'ye sahip olur ve encode sırasında tercih edilir.

        Raises:
            ValueError:
                Eğer metin boşsa veya yalnızca boşluklardan oluşuyorsa.
        """
        # Eğitim için metin doğrulaması yapılır. 
        # Boş veya sadece whitespace içeren metinler kabul edilmez.
        if not text or not text.strip():
            raise ValueError("Training text cannot be empty")

        # Basic tokenization ile metin kelime ve noktalama seviyesinde parçalanır.
        # Bu, eğitim sırasında substring adaylarının çıkarılacağı temel birimlerin belirlenmesini sağlar.
        # Örneğin, "Hello, world!" metni ["hello", ",", "world", "!"] şeklinde tokenize edilir.
        # Bu parçalar üzerinden substring adayları çıkarılacaktır.
        # Basic tokenization, eğitim öncesi tokenize() metodunun fallback olarak da kullanılır.
        words = self._basic_tokenize(text)

        # Subword adaylarının frekanslarını saymak için bir Counter kullanılır.
        # Her kelimenin tüm olası substring parçaları çıkarılır ve bu parçaların kaç kez geçtiği sayılır.
        # Örneğin, "hello" kelimesi için "h", "he", "hel", "hell", "hello", "e", "el", "ell", "l", "ll", "o" gibi substring'ler çıkarılır ve her birinin frekansı sayılır.
        # Bu frekanslar, hangi substring'lerin daha sık geçtiğini belirlemek ve hedef vocab boyutuna göre en sık geçenleri seçmek için kullanılır.
        candidate_counter: Counter[str] = Counter()

        # Her kelime için, tüm olası substring'ler çıkarılır ve frekansları sayılır.
        for word in words:
            # Kelimenin her pozisyonundan başlayarak max_subword_length kadar uzunlukta substring'ler çıkarılır.
            for i in range(len(word)):
                # i'den başlayarak max_subword_length kadar uzunlukta substring'ler çıkarılır.
                # Örneğin, "hello" kelimesi için i=0 noktasından başlayarak j=1, j=2, j=3, j=4, j=5 noktalarına kadar substring'ler çıkarılır: "h", "he", "hel", "hell", "hello".
                # i=1 noktasından başlayarak j=2, j=3, j=4, j=5 noktalarına kadar substring'ler çıkarılır: "e", "el", "ell", "ello".
                for j in range(i + 1, min(len(word), i + self.max_subword_length) + 1): # i'den başlayarak j'ye kadar olan substring'ler çıkarılır, ancak j'nin i + max_subword_length'ı geçmemesine dikkat edilir.
                    # word[i:j] substring'i candidate olarak sayılır.
                    # Örneğin, "hello" kelimesi için i=0, j=1 → "h", i=0, j=2 → "he", i=0, j=3 → "hel", ... gibi substring'ler çıkarılır.
                    # Bu substring'lerin frekansları candidate_counter'da sayılır.
                    candidate_counter[word[i:j]] += 1 # candidate_counter[word[i:j]] değeri, word[i:j] substring'inin kaç kez geçtiğini temsil eder. Her substring çıkarıldığında bu değer 1 artırılır.

        # en sık geçenleri al
        # candidate_counter.most_common(n) metodu, en sık geçen n tane token'ı ve frekanslarını döndürür.
        # self.target_vocab_size - 1, UNK token'ı için yer bırakmak amacıyla gerçek token'ların sayısını belirler.
        # Örneğin, target_vocab_size 100 ise, en sık geçen 99 token seçilir ve UNK token'ı ile birlikte toplam 100 token'lık bir vocab oluşturulur.
        # most_common, token'ların frekanslarına göre sıralanmış bir liste döndürür.
        # Bu liste, encode sırasında hangi token'ların daha olası olduğunu belirlemek için kullanılır.
        # En sık geçen token'lar, encode sırasında daha yüksek log-probability'ye sahip olacak ve bu nedenle tercih edilecektir.
        most_common = candidate_counter.most_common(self.target_vocab_size)

        # UNK token'ı her zaman vocab'da bulunur ve id'si 0'dır, 
        # bu yüzden vocab listesi UNK token'ı ile başlar ve ardından en sık geçen token'lar eklenir.
        # Örneğin, vocab = ["[UNK]", "the", "and", "to", ...] şeklinde bir liste oluşturulur.
        # Bu vocab, encode ve decode işlemleri sırasında token'ların id'lere ve id'lerin token'lara çevrilmesi için kullanılır.
        # UNK token'ı, eğitim sırasında görülmeyen veya encode sırasında karşılaşılan bilinmeyen parçalar için kullanılır.
        # most_common listesindeki token'lar, encode sırasında daha yüksek log-probability'ye sahip olacak ve bu nedenle tercih edilecektir.
        vocab = [self.UNKNOWN_TOKEN] + [token for token, _ in most_common]

        # Laplace smoothing (add-one) için toplam frekans hesaplanır.
        # Bu, sıfır frekans problemini önlemek için gereklidir.
        # total_freq, tüm token'ların frekanslarının toplamını temsil eder.
        # most_common listesindeki token'ların frekansları toplanır ve 1 eklenir (add-one smoothing için).
        # Bu toplam frekans, log-probability hesaplamasında payda olarak kullanılır.
        # Örneğin, eğer most_common'da "the" token'ı 500 kez, "and" token'ı 300 kez geçiyorsa ve target_vocab_size 100 ise, total_freq = 500 + 300 + ... + 1 şeklinde hesaplanır.   
        total_freq = sum(freq for _, freq in most_common) + 1

        # Laplace smoothing, encode sırasında karşılaşılan bilinmeyen token'ların log-probability'sinin sıfır olmamasını sağlar, böylece bu token'lar da encode sırasında değerlendirilebilir hale gelir.
        # Bu adım, eğitim sırasında modelin hangi parçaların daha önemli olduğunu öğrenmesini sağlar ve encode sırasında daha iyi parçalamalar yapmasına yardımcı olur.
        # Eğer total_freq çok küçükse, log-probability'ler çok düşük olabilir, bu yüzden add-one smoothing ile toplam frekans artırılır ve daha stabil log-probability hesaplaması sağlanır.

        # log probability hesaplar
        # Her token için log-probability hesaplanır.
        # Log-probability, token'ın frekansına dayalı olarak hesaplanır ve Laplace smoothing (add-one) uygulanır.
        # Örneğin, eğer "the" token'ı 500 kez geçiyor ve total_freq 10000 ise, log-probability = log((500 + 1) / 10000) şeklinde hesaplanır.
        # Log-probability'ler, encode sırasında hangi token'ların daha olası olduğunu belirlemek için kullanılır. 
        # Daha sık görülen token'lar daha yüksek log-probability'ye sahip olur ve encode sırasında tercih edilir.
        self._token_logprob = {
            token: math.log((candidate_counter[token] + 1) / total_freq) # token'ın log-probability'si, token'ın frekansına dayalı olarak hesaplanır ve Laplace smoothing (add-one) uygulanır. Eğer token most_common'da yoksa, candidate_counter[token] değeri 0 olur ve log-probability = log(1 / total_freq) şeklinde hesaplanır.
            for token in vocab
        }

        # Mapping oluşturma
        # Bu mapping'ler, encode ve decode işlemleri sırasında token'ların id'lere ve id'lerin token'lara çevrilmesi için kullanılır.
        # Encode sırasında, token'lar ID'lere çevrilirken token_to_id mapping'i kullanılır.
        # Eğer token vocab'da yoksa UNK token'ının ID'si (0) kullanılır.
        # Decode sırasında, ID'ler token'lara çevrilirken id_to_token mapping'i kullanılır.

        # Token -> ID ve ID -> Token mapping'leri oluşturulur.
        # Token'lar vocab listesinde sıralanır ve her token'a bir ID atanır.
        # UNK token'ı her zaman ID 0'a sahip olur, ardından en sık geçen token'lar sırasıyla ID 1, 2, ... şeklinde atanır.
        self._token_to_id = {
            token: idx for idx, token in enumerate(vocab) # token -> id mapping'i oluşturulur. 
            # Örneğin, "[UNK]" token'ı ID 0'a, "the" token'ı ID 1'e, "and" token'ı ID 2'ye atanır.
        }

        # ID -> Token mapping'i token_to_id mapping'inden türetilir.
        # Bu, decode işlemi sırasında token ID'lerini token'lara çevirmek için kullanılır.
        # Örneğin, ID 0 için token "[UNK]", ID 1 için token "the" gibi mapping'ler oluşturulur.
        self._id_to_token = {
            idx: token for token, idx in self._token_to_id.items() # id -> token mapping'i oluşturulur.
            # Örneğin, ID 0 için token "[UNK]", ID 1 için token "the" gibi mapping'ler oluşturulur.
        }

        # Eğitim tamamlandıktan sonra _trained flag'i True olarak ayarlanır, 
        # böylece encode ve decode işlemleri yapılabilir hale gelir.
        # Eğitim yapılmadan encode veya decode işlemi yapılmaya çalışılırsa ValueError hatası fırlatılır.
        self._trained = True

    # ---------------------------------------------------------
    # ENCODE
    # ---------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """
        Metni token id'lerine çevirir.

        İşleyiş:
            1. Metin tokenize edilir (subword seviyesinde)
            2. Her token vocabulary içindeki id ile eşlenir

        Eğer token vocabulary'de yoksa:
            → [UNK] token id'si kullanılır

        Raises:
            ValueError:
                Eğer tokenizer henüz eğitilmemişse.
        """
        # Encode işlemi için tokenizer'ın eğitilmiş olması gerekir.
        if not self._trained:
            raise ValueError("Tokenizer has not been trained yet")

        # Metin tokenize edilir (subword seviyesinde).
        # tokenize() metodu, metni subword seviyesinde parçalara böler.
        # Örneğin, "tokenizer" kelimesi ["token", "izer"] şeklinde tokenize edilebilir.
        tokens = self.tokenize(text)

        # Her token vocabulary içindeki id ile eşlenir.
        # Eğer token vocabulary'de yoksa, UNK token id'si (0) kullanılır.
        return [
            self._token_to_id.get(token, self._token_to_id[self.UNKNOWN_TOKEN]) # token'ın id'si, eğer token vocab'da yoksa UNK token'ının id'si (0) kullanılır
            for token in tokens
        ]

    # ---------------------------------------------------------
    # DECODE
    # ---------------------------------------------------------

    def decode(self, token_ids: list[int]) -> str:
        """
        Token id listesini tekrar string'e çevirir.

        İşleyiş:
            - Her id, token karşılığına çevrilir
            - Tokenlar birleştirilir

        Önemli:
            Bu basitleştirilmiş versiyon whitespace bilgisini korumaz.
            Bu nedenle decode işlemi birebir orijinal metni üretmeyebilir.

        Raises:
            ValueError:
                Eğer tokenizer eğitilmemişse
            ValueError:
                Eğer bilinmeyen bir token id verilirse
        """
        # Decode işlemi için tokenizer'ın eğitilmiş olması gerekir.
        if not self._trained:
            raise ValueError("Tokenizer has not been trained yet")

        # Her id, token karşılığına çevrilir.
        # Eğer token id'si bilinmeyen bir token'a karşılık geliyorsa, UNK token'ı kullanılır.
        tokens: list[str] = []

        # Token id'leri üzerinden token'lar elde edilir.
        for tid in token_ids:
            # Eğer token id'si vocabulary'de yoksa, ValueError hatası fırlatılır.
            if tid not in self._id_to_token:
                raise ValueError("Unknown token id")

            # Token id'si token karşılığına çevrilir.
            # Örneğin, token id 0 için token "[UNK]", token id 1 için token "the" gibi mapping'ler kullanılarak token'lar elde edilir.
            # Eğer token id'si UNK token'ına karşılık geliyorsa, token olarak UNK token'ı kullanılır.
            # Diğer token id'leri için, token karşılığı doğrudan kullanılır.
            token = self._id_to_token[tid]

            # Token'lar birleştirilir.
            # Bu basitleştirilmiş versiyon whitespace bilgisini korumaz, 
            # bu nedenle token'lar doğrudan birleştirilir.
            # Örneğin, token'lar ["token", "izer"] ise, decode işlemi "tokenizer" sonucunu verir.
            # Eğer token UNK token'ı ise, decode sonucunda "[UNK]" ifadesi yer alır.
            if token == self.UNKNOWN_TOKEN:
                tokens.append(token)
            else:
                tokens.append(token)

        # Token'lar birleştirilir ve sonuç döndürülür.
        # Bu basitleştirilmiş versiyon whitespace bilgisini korumaz, 
        # bu nedenle token'lar doğrudan birleştirilir.
        # Örneğin, token'lar ["token", "izer"] ise, decode işlemi "tokenizer" sonucunu verir.
        # Eğer token'lar arasında whitespace veya diğer karakterler varsa, bu bilgiler decode sonucunda kaybolur.
        # Bu, decode işleminin birebir orijinal metni üretmeyebileceği anlamına gelir, 
        # çünkü whitespace ve diğer karakterler korunmaz.
        return "".join(tokens)

    # ---------------------------------------------------------
    # TOKENIZE (VITERBI-LIKE)
    # ---------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        """
        Metni unigram subword tokenlara böler.

        Davranış:
            - Boş metin → []
            - Eğitilmemiş tokenizer → basic tokenization
            - Eğitilmiş tokenizer → Viterbi segmentasyonu

        Returns:
            list[str]:
                Subword token listesi
        """
        # tokenize() metodu, metni subword seviyesinde parçalara böler.
        # Eğer metin boşsa veya sadece whitespace içeriyorsa, boş bir liste döndürülür.
        if not text or not text.strip():
            return []

        # Eğer tokenizer henüz eğitilmemişse, basic tokenization yapılır.
        if not self._trained:
            return self._basic_tokenize(text)

        # Eğitilmiş tokenizer için Viterbi segmentasyonu yapılır.
        # Metin önce basic tokenization ile kelime ve noktalama seviyesinde parçalanır, 
        # ardından her kelime Viterbi segmentasyonu ile subword tokenlara bölünür.

        # Örneğin, "tokenizer" kelimesi önce ["tokenizer"] olarak tokenize edilir, 
        # ardından Viterbi segmentasyonu ile ["token", "izer"] şeklinde bölünebilir.

        # Viterbi segmentasyonu, kelimenin tüm olası parçalamalarını değerlendirir ve en yüksek toplam log-probability'ye sahip olan parçalamayı seçer.
        # Bu adım, eğitim sırasında modelin hangi parçaların daha önemli olduğunu öğrenmesini sağlar
        # Encode sırasında daha iyi parçalamalar yapabilmek için eğitim sırasında öğrenilen log-probability'leri kullanır.
        words = self._basic_tokenize(text)

        output_tokens: list[str] = [] 

        # Her kelime için Viterbi segmentasyonu yapılır ve sonuçlar birleştirilir.
        # Örneğin, "Hello, world!" metni ["hello", ",", "world", "!"] olarak tokenize edilir, ardından Viterbi segmentasyonu ile ["hello"], [","], ["world"], ["!"] şeklinde bölünebilir.
        # Örneğin, "tokenizer" kelimesi ["tokenizer"] olarak tokenize edilir, ardından Viterbi segmentasyonu ile ["token", "izer"] şeklinde bölünebilir.
        # Viterbi segmentasyonu, kelimenin tüm olası parçalamalarını değerlendirir ve en yüksek toplam log-probability'ye sahip olan parçalamayı seçer.
        # Bu, encode sırasında daha iyi parçalamalar yapabilmek için eğitim sırasında öğrenilen log-probability'leri kullanır.
        # Bu adım, eğitim sırasında modelin hangi parçaların daha önemli olduğunu öğrenmesini sağlar ve encode sırasında daha iyi parçalamalar yapmasına yardımcı olur.
        for word in words:
            output_tokens.extend(self._viterbi_segment(word)) # Viterbi segmentasyonu sonucunda elde edilen token'lar output_tokens listesine eklenir.

        # tokenize() metodu, metni subword seviyesinde parçalara böler ve sonuç olarak token listesi döndürür.
        # Bu basitleştirilmiş versiyon whitespace bilgisini korumaz, bu nedenle token'lar doğrudan birleştirilir.
        # Örneğin, "tokenizer" kelimesi ["token", "izer"] şeklinde tokenize edilir, ancak "Hello, world!" metni ["hello", ",", "world", "!"] şeklinde tokenize edilir ve whitespace bilgisi kaybolur.
        return output_tokens

    # ---------------------------------------------------------
    # VITERBI SEGMENTATION
    # ---------------------------------------------------------

    def _viterbi_segment(self, word: str) -> list[str]:
        """
        Bir kelimeyi en yüksek olasılıklı subword dizisine böler.

        Bu metod Viterbi-benzeri dinamik programlama kullanır.

        Yapı:

            dp[i]:
                word[:i] için elde edilebilecek en yüksek skor

            backpointer[i]:
                en iyi bölünmeyi sağlayan önceki indeks

        Algoritma:
            Her i noktası için:
                - önceki j noktalarına bakılır
                - word[j:i] substring’i değerlendirilir
                - vocabulary’de varsa skor hesaplanır

        Örnek:
            "tokenizer"

            olası bölünmeler:
                ["token", "izer"]
                ["tok", "en", "izer"]
                ["tokenizer"]

            en yüksek skor seçilir

        Neden önemli?
            Bu yöntem Unigram tokenizer'ın kalbidir.
            Deterministik değil → en iyi olasılığı arar.

        Returns:
            list[str]:
                En iyi subword parçalama

        Fallback:
            Eğer parçalama yapılamazsa → [UNK]
        """
        # Kelimenin uzunluğu n olarak belirlenir.
        n = len(word)

        # dp ve backpointer tabloları oluşturulur.
        dp = [-float("inf")] * (n + 1)
        backpointer = [0] * (n + 1)

        # Başlangıç durumu: boş string için skor 0'dır.
        # Bu, dp tablosunun ilk elemanını başlatır ve algoritmanın çalışması için temel oluşturur.
        # dp[0] = 0 olarak atanır, çünkü boş string için skor 0'dır (hiçbir token kullanmadan).
        dp[0] = 0

        # dp[i], word[:i] için elde edilebilecek en yüksek skoru temsil eder.
        # Başlangıçta tüm dp değerleri -inf olarak atanır, çünkü henüz herhangi bir parçalama değerlendirilmemiştir.

        # backpointer[i], en iyi bölünmeyi sağlayan önceki indeks'i temsil eder.
        # Başlangıçta tüm backpointer değerleri 0 olarak atanır.

        # Algoritma, her i noktası için önceki j noktalarına bakarak word[j:i] substring'ini değerlendirir.
        # Eğer substring vocabulary'de varsa, dp[j] + token_logprob[piece] skorunu hesaplar ve dp[i] ile karşılaştırır.
        # Eğer bu skor dp[i]'den büyükse, dp[i] güncellenir ve backpointer[i] j olarak atanır.
        # Bu şekilde, dp tablosu doldurulur ve en yüksek skora sahip parçalama bulunur.

        # Örneğin, "tokenizer" kelimesi için i=5 noktasında "token" substring'i değerlendirilir ve eğer "token" vocabulary'de varsa, dp[0] + log-probability("token") skoru hesaplanır ve dp[5] ile karşılaştırılır.
        # Eğer bu skor dp[5]'ten büyükse, dp[5] güncellenir ve backpointer[5] 0 olarak atanır.
        # Bu süreç tüm i noktaları için tekrarlanır ve sonunda en yüksek skora sahip parçalama bulunur.
        # Bu yöntem Unigram tokenizer'ın kalbidir, çünkü deterministik değil, en iyi olasılığı arar.

        # Her i noktası için önceki j noktalarına bakarak word[j:i] substring'ini değerlendirir.
        for i in range(1, n + 1):
            # i noktası için önceki j noktalarına bakılır.
            # j, i noktaları arasındaki substring değerlendirilir.
            # Örneğin, i=5 için j=0, j=1, j=2, j=3, j=4 noktalarına bakılır ve word[j:5] substring'leri değerlendirilir
            # Bu substring'ler, "tokenizer" kelimesi için "t", "to", "tok", "toke", "token" gibi substring'ler olabilir.
            for j in range(max(0, i - self.max_subword_length), i):
                # word[j:i] substring'i değerlendirilir.
                # Örneğin, i=5 ve j=0 için word[0:5] = "token" substring'i değerlendirilir.
                piece = word[j:i]

                # Eğer substring vocabulary'de varsa, dp[j] + token_logprob[piece] skorunu hesaplar ve dp[i] ile karşılaştırır.
                # Eğer bu skor dp[i]'den büyükse, dp[i] güncellenir ve backpointer[i] j olarak atanır.
                if piece in self._token_logprob:
                    # Skor hesaplanır: dp[j] + self._token_logprob[piece]
                    # dp[j], word[:j] için elde edilebilecek en yüksek skoru temsil eder, self._token_logprob[piece] ise piece token'ının log-probability'sini temsil eder.
                    # Bu skor, word[:i] için piece token'ını eklediğimizde elde edilebilecek toplam skoru temsil eder.
                    # Örneğin, i=5 ve j=0 için piece = "token" ise, skor dp[0] + log-probability("token") olarak hesaplanır.
                    score = dp[j] + self._token_logprob[piece]

                    # Eğer bu skor dp[i]'den büyükse, dp[i] güncellenir ve backpointer[i] j olarak atanır.
                    # Bu şekilde, dp tablosu doldurulur ve en yüksek skora sahip parçalama bulunur.
                    # Örneğin, i=5 için dp[5] = -inf ise ve score = 0.5 ise, dp[5] güncellenir ve backpointer[5] 0 olarak atanır.
                    if score > dp[i]:
                        dp[i] = score # dp[i] güncellenir, çünkü piece token'ını eklediğimizde elde edilebilecek toplam skor daha yüksek olur.
                        backpointer[i] = j # backpointer[i] j olarak atanır, çünkü piece token'ını eklediğimizde elde edilebilecek toplam skor daha yüksek olur ve bu parçalama için j noktasına geri dönmemiz gerekir.

        # parçalama yoksa UNK
        # Eğer dp[n] hala -inf ise, bu kelime için geçerli bir parçalama bulunamamıştır, bu nedenle [UNK] token'ı döndürülür.
        if dp[n] == -float("inf"):
            return [self.UNKNOWN_TOKEN]

        # backtrack
        tokens: list[str] = []
        i = n

        # En iyi segmentation bulunduğunda, backpointer dizisi kullanılarak sondan başa gidilir.
        # Parçalar ters sırayla toplandığı için en sonunda reverse edilir.
        # Örneğin, "tokenizer" kelimesi için i=10 noktasında backpointer[10] = 5 ise, piece = word[5:10] = "izer" token'ı elde edilir ve i 5'e güncellenir.
        # Ardından i=5 noktasında backpointer[5] = 0 ise, piece = word[0:5] = "token" token'ı elde edilir ve i 0'a güncellenir.
        # Bu süreç, en iyi parçalama bulunana kadar devam eder ve elde edilen token'lar ters sırayla toplandığı için en sonunda reverse edilir.
        # Bu yöntem, Unigram tokenizer'ın kalbidir, çünkü deterministik değil, en iyi olasılığı arar.    
        while i > 0:
            # backpointer dizisi kullanılarak sondan başa gidilir.
            j = backpointer[i] # backpointer[i], i noktasına gelmek için hangi j noktasından gelindiğini temsil eder. Bu, en iyi parçalamayı sağlayan önceki indeks'i gösterir.
            piece = word[j:i] # piece token'ı, word[j:i] substring'i olarak elde edilir. Bu, i noktasına gelmek için j noktasından eklenen token'ı temsil eder.

            # Eğer piece token'ı vocabulary'de yoksa, bu durum beklenmedik bir durumdur, çünkü dp[i] = -inf olmalıydı ve zaten [UNK] token'ı döndürülmeliydi.
            # Ancak bu kontrol eklenir, çünkü teorik olarak böyle bir durum oluşmamalıdır. Eğer böyle bir durum oluşursa, [UNK] token'ı döndürülür.
            # Bu durum, eğitim sırasında token'ların log-probability'lerinin doğru hesaplanmaması veya Viterbi segmentasyonu sırasında dp tablosunun yanlış doldurulması gibi durumlarda ortaya çıkabilir.
            if piece not in self._token_to_id:
                return [self.UNKNOWN_TOKEN]

            # Parçalar ters sırayla toplandığı için en sonunda reverse edilir.
            # Örneğin, "tokenizer" kelimesi için i=10 noktasında backpointer[10] = 5 ise, piece = word[5:10] = "izer" token'ı elde edilir ve i 5'e güncellenir.
            # Ardından i=5 noktasında backpointer[5] = 0 ise, piece = word[0:5] = "token" token'ı elde edilir ve i 0'a güncellenir.
            tokens.append(piece)
            i = j

        # Parçalar ters sırayla toplandığı için en sonunda reverse edilir.
        # Örneğin, "tokenizer" kelimesi için tokens listesi ["izer", "token"] şeklinde olabilir, bu nedenle reverse edilerek ["token", "izer"] şeklinde döndürülür.
        return list(reversed(tokens))

    # ---------------------------------------------------------
    # UTILS
    # ---------------------------------------------------------

    def _basic_tokenize(self, text: str) -> list[str]:
        """
        Basit regex tabanlı tokenization uygular.

        İşleyiş:
            - Metin lowercase yapılır
            - Kelimeler ve noktalama ayrılır

        Regex:
            \\w+        → kelimeler
            [^\\w\\s]   → noktalama / semboller

        Örnek:
            "Hello, world!"
            → ["hello", ",", "world", "!"]

        Returns:
            list[str]
        """
        # Basit regex tabanlı tokenization uygular.
        # Metin lowercase yapılır ve kelimeler ile noktalama ayrılır.
        return re.findall(r"\w+|[^\w\s]", text.lower())

    @property
    def vocab_size(self) -> int:
        """
        Vocabulary boyutunu döndürür.

        Eğitim öncesi:
            → 0

        Eğitim sonrası:
            → [UNK] + öğrenilen subword'ler
        """
        # Vocabulary boyutunu döndürür.
        # Eğitim öncesi, vocab oluşturulmadığı için 0 döndürülür.
        # Eğitim sonrası, vocab oluşturulduğu için [UNK] token'ı ve öğrenilen subword'ler dahil olmak üzere toplam token sayısı döndürülür.
        return len(self._token_to_id)