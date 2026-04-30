from __future__ import annotations

import tempfile
from pathlib import Path

from tokenizer_workshop.tokenizers.base import BaseTokenizer
from tokenizer_workshop.tokenizers.registry import register_tokenizer


@register_tokenizer("sentencepiece")
class SentencePieceTokenizer(BaseTokenizer):
    """
    SentencePiece tabanlı tokenizer.

    Bu sınıf, Google tarafından geliştirilen SentencePiece kütüphanesini
    sarmalayarak projeye gerçek bir production-grade subword tokenizer kazandırır.

    ---------------------------------------------------------
    📌 SentencePiece NEDİR?
    ---------------------------------------------------------

    SentencePiece, metni doğrudan karakter dizisi olarak ele alır ve
    whitespace'e bağımlı olmadan subword tokenization yapar.

    Önemli fark:
        Word tokenizer:
            "hello world" -> ["hello", "world"]

        SentencePiece:
            "hello world" -> ["▁hello", "▁world"]

    Buradaki "▁" (underscore):
        → kelime başlangıcını temsil eder

    Bu sayede:
        - whitespace kaybı yaşanmaz
        - decode işlemi daha doğru yapılır

    ---------------------------------------------------------
    📌 DESTEKLENEN MODEL TÜRLERİ
    ---------------------------------------------------------

        unigram:
            - probabilistic model
            - en modern ve güçlü yaklaşım

        bpe:
            - merge tabanlı sıkıştırma
            - GPT-2 benzeri

        char:
            - karakter seviyesinde tokenization

        word:
            - whitespace tabanlı (baseline)

    ---------------------------------------------------------
    📌 TASARIM KARARLARI
    ---------------------------------------------------------

    - SentencePieceTrainer her çağrıda sıfırdan model üretir
    - Model geçici dosyada oluşturulur (tmp dir)
    - Model memory'e load edilir (processor.load)

    Bu sayede:
        → disk clutter olmaz
        → test ortamı temiz kalır

    ---------------------------------------------------------
    📌 ÖNEMLİ NOTLAR
    ---------------------------------------------------------

    - Bu tokenizer external dependency içerir
    - sentencepiece kurulmadan çalışmaz
    - decode işlemi whitespace-aware çalışır (UnigramTokenizer’dan farkı)

    Kurulum:
        uv add sentencepiece
    """

    def __init__(
        self,
        vocab_size: int = 100,
        model_type: str = "unigram",
        character_coverage: float = 1.0,
    ) -> None:
        """
        SentencePieceTokenizer constructor.

        Parameters:
            vocab_size:
                Öğrenilecek toplam token sayısı.
                Bu sayı:
                    - subword parçalar
                    - özel tokenlar (UNK vb.)
                dahil olmak üzere toplam vocabulary boyutudur.

            model_type:
                Kullanılacak algoritma tipi.
                Seçenekler:
                    - unigram
                    - bpe
                    - char
                    - word

            character_coverage:
                Unicode coverage oranı.
                Genelde:
                    - İngilizce: 1.0
                    - Japonca/Çince: 0.9995

        Raises:
            ValueError:
                - vocab_size < 2
                - invalid model_type
                - invalid coverage
        """
        super().__init__(name="sentencepiece")

        # Parametre doğrulamaları
        # vocab_size en az 2 olmalıdır (1 token UNK için, 1 token gerçek içerik için)
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")

        # model_type geçerli seçeneklerden biri olmalıdır
        if model_type not in {"unigram", "bpe", "char", "word"}:
            raise ValueError(
                "model_type must be one of: unigram, bpe, char, word"
            )

        # character_coverage 0 ile 1 arasında olmalıdır
        if not 0 < character_coverage <= 1.0:
            raise ValueError("character_coverage must be between 0 and 1")

        # Bu tokenizer'ın çalışması için sentencepiece kütüphanesi gereklidir, 
        # bu nedenle import hatası durumunda kullanıcıya nasıl kurulacağını belirten bir mesaj sağlanır.
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise ImportError(
                "SentencePieceTokenizer requires the 'sentencepiece' package. "
                "Install it with: uv add sentencepiece"
            ) from exc

        self.vocab_size_target = vocab_size # Kullanıcı tarafından istenen vocab size (model eğitilirken bu hedeflenir)
        self.model_type = model_type # Kullanılacak SentencePiece algoritması (unigram, bpe, char, word)
        self.character_coverage = character_coverage # Unicode karakter coverage oranı (örneğin Japonca için 0.9995)

        self._spm = spm # SentencePiece modülü (import sırasında doğrulandı)

        # Processor: encode/decode işlemlerini yapar
        self._processor = spm.SentencePieceProcessor() 

        # Eğitim yapılıp yapılmadığını takip eder (tokenize/encode/decode öncesi kontrol için)
        # Eğitim yapılmadan tokenize/encode/decode işlemi yapılmaya çalışılırsa ValueError fırlatılır
        self._trained = False

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------

    def train(self, text: str) -> None:
        """
        SentencePiece modelini verilen metin üzerinde eğitir.

        ---------------------------------------------------------
        📌 WORKFLOW
        ---------------------------------------------------------

        1. Input validation
        2. Temporary training file oluşturma
        3. SentencePieceTrainer ile model üretme
        4. Modeli memory'e yükleme

        ---------------------------------------------------------
        📌 NEDEN TEMP FILE?
        ---------------------------------------------------------

        SentencePiece API doğrudan string almaz.
        Input dosya ister.

        Bu yüzden:
            → geçici dosya oluşturuyoruz
            → işlem bitince otomatik siliniyor

        ---------------------------------------------------------
        📌 TRAIN PARAMETRELERİ
        ---------------------------------------------------------

        bos_id = -1:
            → BOS token kapalı

        eos_id = -1:
            → EOS token kapalı

        pad_id = -1:
            → padding yok

        unk_id = 0:
            → unknown token sabit index

        hard_vocab_limit = False:
            → küçük datasetlerde crash engeller

        Raises:
            ValueError:
                boş input verilirse
                eğitim yapılmadan tokenize/encode/decode yapılmaya çalışılırsa
        """
        # Input validation: boş veya sadece whitespace içeren metin kabul edilmez
        # Eğitim için anlamlı veri gereklidir.
        # Eğer boş veya sadece whitespace içeren metin verilirse, bu da eğitim sürecinin başarısız olmasına veya anlamsız bir model oluşturmasına yol açar.
        if not text or not text.strip():
            raise ValueError("Training text cannot be empty")

        # SentencePieceTrainer doğrudan string input alamadığı için geçici bir dosya oluşturulur, eğitim bu dosya üzerinden yapılır, ardından model memory'e yüklenir.
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # SentencePiece input file ister
            # Geçici dosya oluşturulur, eğitim bu dosya üzerinden yapılır, ardından model memory'e yüklenir.
            input_path = tmp_path / "sentencepiece_input.txt"

            # Model dosyası için prefix belirlenir (SentencePiece bu prefix'i kullanarak .model ve .vocab dosyalarını oluşturur)
            model_prefix = tmp_path / "sentencepiece_model"

            # Model dosyası için tam path belirlenir (processor.load için)
            model_path = tmp_path / "sentencepiece_model.model"

            # Eğitim için gerekli parametreler SentencePieceTrainer'a verilir, 
            # eğitim tamamlandıktan sonra model memory'e yüklenir.
            input_path.write_text(text, encoding="utf-8")

            # SentencePiece training başlatılır
            # SentencePieceTrainer her çağrıda sıfırdan model üretir, 
            # model geçici dosyada oluşturulur, model memory'e load edilir (processor.load)
            # Bu sayede disk clutter olmaz, test ortamı temiz kalır.
            self._spm.SentencePieceTrainer.train(
                input=str(input_path), # Eğitim için kullanılan geçici dosyanın path'i
                model_prefix=str(model_prefix), # Model dosyası için prefix )
                vocab_size=self.vocab_size_target, # Kullanıcı tarafından istenen vocab size (model eğitilirken bu hedeflenir)
                model_type=self.model_type, # Kullanılacak SentencePiece algoritması (unigram, bpe, char, word)
                character_coverage=self.character_coverage, # Unicode karakter coverage oranı (örneğin Japonca için 0.9995)
                bos_id=-1, # BOS token kapalı
                eos_id=-1, # EOS token kapalı
                pad_id=-1, # padding yok
                unk_id=0, # unknown token sabit index
                hard_vocab_limit=False, # küçük datasetlerde crash engeller
            )

            # Eğitim tamamlandıktan sonra model memory'e yüklenir, 
            # böylece tokenize/encode/decode işlemleri için hazır hale gelir.
            self._processor.load(str(model_path))

        # Eğitim tamamlandıktan sonra _trained flag'i True yapılır, 
        # böylece tokenize/encode/decode işlemleri için hazır hale gelir.
        self._trained = True

    # ---------------------------------------------------------
    # TOKENIZE
    # ---------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        """
        Metni SentencePiece tokenlarına böler.

        SentencePiece tokenlarında kelime başı boşluk genellikle özel
        underline karakteriyle gösterilir:

            "▁"

        Örnek:
            "Merhaba dünya"
            → ["▁Merhaba", "▁dünya"]

        "▁" karakteri:
            → whitespace boundary marker

        Bu sayede decode sırasında:
            → boşluklar doğru şekilde geri oluşturulur
        """
        # Input validation: boş veya sadece whitespace içeren metin kabul edilmez
        # Tokenize işlemi için anlamlı veri gereklidir.
        if not text or not text.strip():
            return []

        # Eğitim yapılmadan tokenize işlemi yapılmaya çalışılırsa ValueError fırlatılır,
        # Tokenize işlemi için modelin eğitilmiş ve processor'a yüklenmiş olması gerekir.
        if not self._trained:
            raise ValueError("Tokenizer has not been trained yet")

        # SentencePieceProcessor kullanılarak metin tokenlara bölünür,
        # tokenize işlemi için modelin eğitilmiş ve processor'a yüklenmiş olması gerekir, 
        # eğitim yapılmadan tokenize işlemi yapılmaya çalışılırsa ValueError fırlatılır.
        return list(self._processor.encode(text, out_type=str))

    # ---------------------------------------------------------
    # ENCODE
    # ---------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """
        Metni token ID listesine çevirir.

        SentencePiece:
            string → token → id mapping'i internal olarak yapar.
        """
        # Input validation: boş veya sadece whitespace içeren metin kabul edilmez
        # Encode işlemi için anlamlı veri gereklidir.
        if not self._trained:
            raise ValueError("Tokenizer has not been trained yet")

        # Eğer boş veya sadece whitespace içeren metin verilirse, encode işlemi için anlamlı veri sağlanmaz, 
        # bu da anlamsız token id'lerin üretilmesine veya eğitim yapılmadan encode işlemi yapılmaya çalışılırsa ValueError fırlatılmasına yol açar.
        if not text or not text.strip():
            return []

        # SentencePieceProcessor kullanılarak metin token ID'lerine çevrilir,
        # encode işlemi için modelin eğitilmiş ve processor'a yüklenmiş olması gerekir, 
        # eğitim yapılmadan encode işlemi yapılmaya çalışılırsa ValueError fırlatılır.
        return list(self._processor.encode(text, out_type=int))

    # ---------------------------------------------------------
    # DECODE
    # ---------------------------------------------------------

    def decode(self, token_ids: list[int]) -> str:
        """
        SentencePiece token id listesini tekrar string'e dönüştürür.

        SentencePiece:
            whitespace bilgisi korunduğu için decode sonucu
            genellikle orijinal metne çok yakındır.

        Bu, UnigramTokenizer'dan önemli bir farktır.
        """
        # Eğer token_ids listesi boşsa, decode işlemi için anlamlı veri sağlanmaz, 
        # bu da anlamsız bir string'in üretilmesine veya eğitim yapılmadan decode işlemi yapılmaya çalışılırsa ValueError fırlatılmasına yol açar.
        if not token_ids:
            return ""

        # Eğitim yapılmadan decode işlemi yapılmaya çalışılırsa ValueError fırlatılır,
        # Decode işlemi için modelin eğitilmiş ve processor'a yüklenmiş olması gerekir, 
        # eğitim yapılmadan decode işlemi yapılmaya çalışılırsa ValueError fırlatılır.
        if not self._trained:
            raise ValueError("Tokenizer has not been trained yet")

        return self._processor.decode(token_ids)

    # ---------------------------------------------------------
    # VOCAB
    # ---------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """
        Eğitilmiş SentencePiece modelinin vocabulary boyutunu döndürür.

        Eğitim öncesi:
            → 0

        Eğitim sonrası:
            → SentencePiece model size
        """
        if not self._trained:
            return 0

        return int(self._processor.get_piece_size())