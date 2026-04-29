from __future__ import annotations

from typing import Any

from tokenizer_workshop.tokenizers.base import BaseTokenizer
from tokenizer_workshop.tokenizers.registry import register_tokenizer


@register_tokenizer("pretrained")
class PreTrainedTokenizerWrapper(BaseTokenizer):
    """
    Hugging Face pretrained tokenizer'larını tokenizer-workshop projesindeki ortak tokenizer kontratına adapte eden wrapper sınıfıdır.

    Bu sınıf doğrudan yeni bir tokenizer algoritması öğretmez; bunun yerine
    dışarıda eğitilmiş, production-ready tokenizer'ları proje ekosistemine dahil eder.

    Projedeki temel tokenizer kontratı:

        tokenize(text) -> list[str]
        encode(text)   -> list[int]
        decode(ids)    -> str
        vocab_size     -> int

    Hugging Face tokenizer'ları ise daha geniş bir API sunar. Bu wrapper,
    o API'yi sadeleştirerek CompareManager, raporlama sistemi ve factory akışı ile uyumlu hale getirir.

    Bu sınıfın amacı:
        - Dış dünyadaki hazır tokenizer'ları projeye dahil etmek
        - CompareManager ile uyumlu tokenize() çıktısı üretmek
        - encode/decode akışını standartlaştırmak
        - Eğitim gerektirmeyen tokenizer'lar için ortak bir arayüz sağlamak

    Neden bu wrapper gerekli?

    1. Mimari tutarlılık sağlar:
        Projedeki custom tokenizer'lar ile pretrained tokenizer'lar aynı arayüzden çağrılabilir.

    2. Karşılaştırma yapılmasını kolaylaştırır:
        WordTokenizer, ByteLevelBPETokenizer, UnigramTokenizer gibi custom
        tokenizer'larla BERT, GPT-2 veya RoBERTa tokenizer'ları aynı tabloda
        karşılaştırılabilir.

    3. Gerçek dünya bağlantısı kurar:
        Eğitim amaçlı yazılan tokenizer'lar ile modern transformer modellerinde
        kullanılan tokenizer'lar arasındaki farkları ölçülebilir hale getirir.

    4. Training gerektirmez:
        Pretrained tokenizer zaten vocabulary, normalization, pre-tokenization kurallarıyla gelir. 

    Projedeki tokenizer'lar genellikle şu ortak davranışı sağlar:

        tokenizer.tokenize(text) -> list[str]

    Ancak Hugging Face tokenizer'ları daha zengin bir API sunar:

        tokenizer.tokenize(text)
        tokenizer.encode(text)
        tokenizer.decode(token_ids)
        tokenizer.convert_ids_to_tokens(ids)
        tokenizer.vocab_size

    Bu wrapper, Hugging Face tokenizer API'sini proje mimarisine uygun hale getirir.

    Önemli not:
        Bu sınıf tokenizer'ı sıfırdan eğitmez.
        Hazır eğitilmiş tokenizer modelini kullanır.

    Örnek kullanım:
        tokenizer = PreTrainedTokenizerWrapper(
            model_name="bert-base-uncased"
        )

        tokens = tokenizer.tokenize("Hello world!")
        ids = tokenizer.encode("Hello world!")
        text = tokenizer.decode(ids)

    Gerekli paket:
        uv add transformers
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        *,
        use_fast: bool = True,
        add_special_tokens: bool = False,
        **tokenizer_kwargs: Any,
    ) -> None:
        """
         PreTrainedTokenizerWrapper nesnesini oluşturur.

        Args:
            model_name:
                Hugging Face Hub üzerinde bulunan tokenizer/model adıdır.

                Örnekler:
                    - "bert-base-uncased"
                    - "bert-base-multilingual-cased"
                    - "gpt2"
                    - "roberta-base"
                    - "distilbert-base-uncased"

                Türkçe veya çok dilli metinler için:
                    - "bert-base-multilingual-cased"
                    - "xlm-roberta-base"

            use_fast:
                Hugging Face'in Rust tabanlı fast tokenizer implementasyonunu
                kullanıp kullanmayacağını belirler.

                Genellikle True tercih edilir çünkü:
                    - daha hızlıdır
                    - offset mapping gibi gelişmiş özellikleri destekler
                    - production kullanımına daha yakındır

            add_special_tokens:
                Encode sırasında [CLS], [SEP], <s>, </s> gibi özel tokenların
                otomatik eklenip eklenmeyeceğini belirler.

                Örneğin BERT için:
                    add_special_tokens=True  -> [CLS] text [SEP]
                    add_special_tokens=False -> sadece metnin kendi tokenları

                Tokenizer karşılaştırması yaparken genellikle False daha doğrudur.
                Çünkü amaç model input formatını değil, ham tokenization davranışını
                analiz etmektir.

            **tokenizer_kwargs:
                AutoTokenizer.from_pretrained() metoduna geçirilecek ek
                parametrelerdir.

                Örnek:
                    cache_dir="..."
                    revision="..."
                    trust_remote_code=False

        Raises:
            ValueError:
                model_name boş verilirse.

            ImportError:
                transformers paketi kurulu değilse.

            RuntimeError:
                Hugging Face tokenizer yüklenemezse.
        """
        super().__init__(name="pretrained")

        # Model adı boş veya sadece whitespace ise hata fırlatılır.
        if not model_name or not model_name.strip():
            raise ValueError("model_name cannot be empty")

        self.model_name = model_name # Hugging Face model adı, örneğin "bert-base-uncased"
        self.use_fast = use_fast # Rust tabanlı fast tokenizer'ı kullanma tercihi
        self.add_special_tokens = add_special_tokens # Encode sırasında özel tokenların eklenip eklenmeyeceği

        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "PreTrainedTokenizerWrapper requires the 'transformers' package. "
                "Install it with: pip install transformers"
            ) from exc

        try:
            # Hugging Face Hub'dan belirtilen model adıyla tokenizer'ı yükler.
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, # Hugging Face model adı
                use_fast=use_fast, # Rust tabanlı fast tokenizer'ı kullanma tercihi
                **tokenizer_kwargs, # AutoTokenizer.from_pretrained()'a geçirilecek ek parametreler
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load pretrained tokenizer: {model_name!r}"
            ) from exc

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------

    def train(self, text: str) -> None:
        """
        Pretrained tokenizer'lar için eğitim işlemi yapılmaz.

        Bu metod yalnızca proje içindeki ortak tokenizer akışına uyum sağlamak için vardır.

        Custom tokenizer'lar:
            train(text) ile vocabulary veya merge rule öğrenebilir.

        Pretrained tokenizer:
            Vocabulary ve tokenization kuralları zaten önceden öğrenilmiştir.

        Args:
            text:
                Uyumluluk amacıyla alınır; kullanılmaz.
        """
        return None

    # ---------------------------------------------------------
    # TOKENIZE
    # ---------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        """
        Metni pretrained tokenizer'ın token string çıktısına dönüştürür.

        Bu metod özellikle:
            - CompareManager çıktıları
            - rapor üretimi
            - token preview gösterimi
            - debug çıktıları
        için kullanılır.

        Args:
            text:
                Tokenize edilecek ham metin.

        Returns:
            Token string listesi.

        Raises:
            ValueError:
                text boşsa veya yalnızca whitespace içeriyorsa.
        """
        self._validate_text(text)

        return self._tokenizer.tokenize(text)

    # ---------------------------------------------------------
    # ENCODE
    # ---------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """
        Metni pretrained vocabulary'ye göre token id listesine çevirir.

        tokenize() token string döndürürken, encode() model input'una daha yakın
        olan integer token id listesini döndürür.

        Args:
            text:
                Encode edilecek ham metin.

        Returns:
            Token id listesi.

        Raises:
            ValueError:
                text boşsa veya yalnızca whitespace içeriyorsa.
        """
        # Ortak text validasyonu yapılır. 
        # Pretrained tokenizer'lar da boş veya sadece whitespace içeren metinleri tokenize edemez.
        self._validate_text(text)

        return list(
            self._tokenizer.encode(
                text,
                add_special_tokens=self.add_special_tokens,
            )
        )

    # ---------------------------------------------------------
    # DECODE
    # ---------------------------------------------------------

    def decode(self, token_ids: list[int]) -> str:
        """
        Token id listesini tekrar string forma dönüştürür.

        Decode işlemi birebir orijinal metni döndürmeyebilir. Bunun sebebi:
            - tokenizer normalization uygulayabilir
            - lowercasing yapılmış olabilir
            - special tokenlar atlanabilir
            - bazı tokenizer'lar whitespace'i farklı restore edebilir

        Bu yüzden reconstruction kontrolü yapılırken pretrained tokenizer'larda
        birebir string eşitliği her zaman beklenmemelidir.

        Args:
            token_ids:
                Decode edilecek token id listesi.

        Returns:
            Decode edilmiş metin.

        Raises:
            ValueError:
                token_ids boşsa.
            ValueError:
                token_ids sadece integer değerlerden oluşmuyorsa.
        """
        if not token_ids:
            raise ValueError("token_ids cannot be empty")

        # token_ids listesindeki her elemanın integer olup olmadığı kontrol edilir.
        # Eğer token_ids içinde integer olmayan bir eleman varsa, ValueError fırlatılır.
        if not all(isinstance(token_id, int) for token_id in token_ids):
            raise ValueError("token_ids must contain only integers")

        return str(
            self._tokenizer.decode(
                token_ids, # Decode edilecek token id listesi.
                skip_special_tokens=not self.add_special_tokens, # Eğer add_special_tokens False ise, decode sırasında special tokenlar atlanır. Eğer True ise, special tokenlar decode edilir.
            )
        )

    # ---------------------------------------------------------
    # ID <-> TOKEN HELPERS
    # ---------------------------------------------------------

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        """
        Token id listesini token string listesine çevirir.

        Bu metod raporlama tarafında özellikle faydalıdır. Çünkü encode()
        integer id döndürür; fakat kullanıcıya gösterilecek çıktıda tokenların
        string temsilini görmek daha anlaşılırdır.

        Args:
            token_ids:
                Token id listesi.

        Returns:
            Token string listesi.
        """
        # Eğer token_ids boşsa, boş bir liste döndürülür. 
        # Bu durum raporlama sırasında token preview için önemlidir; 
        # çünkü encode() boş liste döndürebilir ve 
        # bu durumda convert_ids_to_tokens()'un da boş liste döndürmesi beklenir.
        if not token_ids:
            return []

        # token_ids listesindeki her elemanın integer olup olmadığı kontrol edilir.
        # Eğer token_ids içinde integer olmayan bir eleman varsa, ValueError fırlatılır.
        if not all(isinstance(token_id, int) for token_id in token_ids):
            raise ValueError("token_ids must contain only integers")

        # Hugging Face tokenizer'ların convert_ids_to_tokens() metodu, verilen token id listesine karşılık gelen token string listesini döndürür.
        return list(self._tokenizer.convert_ids_to_tokens(token_ids))

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """
        Token string listesini token id listesine dönüştürür.

        Args:
            tokens:
                Token string listesi.

        Returns:
            Token id listesi.
        """
        # Eğer tokens boşsa, boş bir liste döndürülür. 
        # Bu durum raporlama sırasında token preview için önemlidir;
        # çünkü tokenize() boş liste döndürebilir ve 
        # bu durumda convert_tokens_to_ids()'un da boş liste döndürmesi beklenir.
        if not tokens:
            return []

        # tokens listesindeki her elemanın string olup olmadığı kontrol edilir.
        # Eğer tokens içinde string olmayan bir eleman varsa, ValueError fırlatılır.
        if not all(isinstance(token, str) for token in tokens):
            raise ValueError("tokens must contain only strings")

        # Hugging Face tokenizer'ların convert_tokens_to_ids() metodu, verilen token string listesine karşılık gelen token id listesini döndürür.
        return list(self._tokenizer.convert_tokens_to_ids(tokens))

    # ---------------------------------------------------------
    # INTERNAL HELPERS
    # ---------------------------------------------------------

    @staticmethod
    def _validate_text(text: str) -> None:
        """
        Ortak input text validasyonu yapar.

        Bu validasyon sayesinde tokenize() ve encode() metodları aynı hata
        davranışına sahip olur.

        Args:
            text:
                Kontrol edilecek input metni.

        Raises:
            ValueError:
                text boşsa veya yalnızca whitespace karakterlerinden oluşuyorsa.
        """
        # Eğer text boşsa veya sadece whitespace karakterlerinden oluşuyorsa, ValueError fırlatılır.
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

    # ---------------------------------------------------------
    # VOCAB
    # ---------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """
        Pretrained tokenizer vocabulary boyutunu döndürür.

        Returns:
            Vocabulary içindeki token sayısı.
        """
        return int(self._tokenizer.vocab_size)

    @property
    def special_tokens(self) -> dict[str, str | list[str]]:
        """
        Tokenizer'ın özel tokenlarını döndürür.

        Örnek BERT special tokens:
            {
                "unk_token": "[UNK]",
                "sep_token": "[SEP]",
                "pad_token": "[PAD]",
                "cls_token": "[CLS]",
                "mask_token": "[MASK]"
            }

        Returns:
            Özel token sözlüğü.
        """
        # Hugging Face tokenizer'ların special_tokens_map özelliği, tokenizer'ın özel tokenlarının adlarını ve string temsillerini içeren bir sözlük döndürür.
        return dict(self._tokenizer.special_tokens_map)

    @property
    def backend_tokenizer_name(self) -> str:
        """
        Kullanılan pretrained tokenizer model adını döndürür.

        Bu bilgi özellikle raporlama ve debug çıktılarında faydalıdır.

        Returns:
            Hugging Face model/tokenizer adı.
        """
        return self.model_name