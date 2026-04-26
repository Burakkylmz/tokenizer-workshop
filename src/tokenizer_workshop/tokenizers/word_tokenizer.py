from __future__ import annotations
import re

from tokenizer_workshop.tokenizers.base import BaseTokenizer
from tokenizer_workshop.tokenizers.registry import register_tokenizer


@register_tokenizer("word")
class WordTokenizer(BaseTokenizer):
    """"
    Metni kelime ve noktalama işaretleri bazında tokenize eder.
    - Char ve Byte tokenizer'dan farkı burada temel amaç 'Kelimelerin' tokenizasyonunu sağlamaktır.
    - Token sayısının azalmasına karşın bilinmeyen kelimelerin (özel karakter vb.) riskinin arttığını gösterir.

    How to:
        - Metin, regex kullanılarak kelime ve noktalama işaretleri olarak parçalara bölünür.
        - Her benzersiz token bir id ile eşlenir.
        örn. Merhaba Dünya! -> 'Merhaba', 'Dünya', '!' => [0, 1, 2] (encode)

    ByteTokenizer ile CharTokenizer kısmından farklılıklar:
        - ByteTokenizer vocablary'si sabit. (0, 256)
        - WordTokenizer eğitim verisine bağlı olduğundan değişkendir.
        - CharTokenizer her karakter için ayrı token sayar.
        - WordTokenizer daha büyük birimler üretir ve bu sayede sequence length kısalır.
        - WordTokenizer train'de görülmeyen kelimeleri encode edilemez..!

    Sınırlamalar:
        - Train setinde olmayan kelimeler encode (kodlama) kısmında hata verir.
        - Büyük küçük harf duyarlılığı vardır. Bu yüzden eğer nlp, llm vb. projede kullanılacaksa öncesinde text cleaning ile veri temizleme ve düzenleme işlemleri yapılmadılır.

    Best practise:
        - En yaygın kullanım örneği olarak nltk (natural language toolkit) kütüphanesi örnek gösterilebilir. Bu uygulamada nltk.word_tokenize() ile yapılan kelime tokenizasyon fonksiyonun çalışma mantığı açıklanmaya çalışılacaktır.
    """

    def __init__(self) -> None:
        super().__init__(name='word')
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        self._is_trained = False


    def train(self, text:str) -> None:
        """
        - Train setinden vocabluary oluşturulur.
        - Text önce regex ile kelimelere ve noktalama işaretlerine bölünür. Ardından benzersiz tokenlar sıralanarak her birine id atanır. (nltk.word_tokenizer mantığı)
        - Sıralamanın nedeni Deterministik bir vocabulary üretmek için. Aynı metin her seferinde aynı id ile eşleşmelidir.
        - örnek -> 'merhaba dünya merhaba' => {'dünya': '0', 'merhaba': '1'}
        """
        if not text:
            raise ValueError('Training text cannot be empty.')

        tokens = re.findall(r"\w+|[^\w\s]", text) # kelimeler ve noktalama işaretleri ayrı ayrı tokenlar olarak alınır.
        unique_tokens = sorted(set(tokens)) # set kullanılmasının sebebi 0(1) formatta daha hızlı hash işlemi yapılabilmesidir.  # benzersiz tokenların sıralanması

        self._token_to_id = {token: idx for idx, token in enumerate(unique_tokens)}
        self._id_to_token = {idx: token for token, idx in self._token_to_id.items()}
        self._is_trained = True


    def encode(self, text:str) -> list[int]:
        """
        Text'i token id listesine dönüştürür.
        örn. -> 'merhaba dünya' => [1, 0]
        - !!! Train set'te görünmeyen bir kelimeyle karşılaşırsa hata verir..!
        """
        if not self._is_trained:
            raise ValueError('Tokenizer has not been trained yet.')

        tokens = re.findall(r"\w+|[^\w\s]", text) # regex ile kullanılan \w ifadesi unicode farkındalıklı çalışır. yani türkçe harfleri de karakter olarak tanır.
        result = []
        for token in tokens:
            if token not in self._token_to_id:
                raise ValueError(f'Unknown token encountered during encoding. {token} \nBu token eğitim sırasında görülmedi. (OOV hatası).')
            result.append(self._token_to_id[token])
        return result

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
        return [self._id_to_token[token_id] for token_id in token_ids]


    def decode(self, token_ids:list[int]) -> str:
        """
        Token id listesini tekrar metne dönüştürür.
        örn. -> [1, 0] -> "merhaba dünya"
        !!! boşluk bilgisi kaybolduğunda kelimeler tek boşluk kalacak şekilde birleştirilir. (.split() fonksiyonu matnığı) word tokenizer'ların sınırlamalarından biridir.
        """
        if not self._is_trained:
            raise ValueError('Tokenizer has not been trained yet.')

        tokens = []
        for token_id in token_ids:
            if token_id not in self._id_to_token:
                raise ValueError(f'Unknown token id encountered during decoding.: {token_id}')
            tokens.append(self._id_to_token[token_id])

        return " ".join(tokens)

    @property
    def vocab_size(self) -> int:
        """
        Vocabulary boyutunu döndürür.
        ByteTokenizer'dan farklı olarak bu değer sabir değildir. Train text'teki benzersiz token sayısına bağlıdır.
        """
        return len(self._token_to_id)