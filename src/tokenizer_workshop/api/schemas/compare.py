"""
compare.py

Bu modül, tokenizer karşılaştırma API'si için kullanılan Pydantic modellerini içerir.

Amacı:
    - Çoklu tokenizer karşılaştırma endpoint'inin request ve response modellerini tanımlamak
    - API katmanında veri doğrulama ve standardizasyon sağlamak
    - Frontend ve rapor üretimi için tutarlı veri yapıları oluşturmak
    - Testlerde response contract'ını doğrulamak

İçerdiği modeller:
    - CompareRequest: Çoklu tokenizer karşılaştırma isteği için kullanılan model
    - CompareItemResponse: Tek bir tokenizer'ın karşılaştırma sonucunu temsil eden model
    - CompareResponse: Çoklu tokenizer karşılaştırma sonucunu temsil eden model
"""

from pydantic import BaseModel, ConfigDict, Field

from tokenizer_workshop.api.schemas.base import BaseTokenizerListRequest


class CompareRequest(BaseTokenizerListRequest):
    """
    Çoklu tokenizer karşılaştırma endpoint'i için kullanılan request modelidir.

    Bu model, aynı ham metnin birden fazla tokenizer üzerinden çalıştırılmasını
    sağlar. Amaç, farklı tokenizer stratejilerinin aynı input üzerinde nasıl
    farklı token dizileri ürettiğini gözlemlemektir.

    Bu modelin sağladığı bilgiler:
        - Karşılaştırma yapılacak ham metin
        - Çalıştırılacak tokenizer'ların isimleri

    Kullanım alanları:
        - API endpoint'ine gelen karşılaştırma isteklerinin doğrulanması
        - Frontend'den çoklu tokenizer seçimiyle karşılaştırma yapılması
        - Testlerde karşılaştırma request'lerinin standardize edilmesi
        - Raporlama servislerine karşılaştırma isteği verisi sağlanması

    Kullanım amacı:
        - Character, word, byte, BPE, regex, ngram gibi tokenizer'ları
          aynı metin üzerinde karşılaştırmak
        - Token sayısı, vocabulary çeşitliliği ve segmentation davranışını analiz etmek
        - UI tarafında çoklu tokenizer seçimiyle karşılaştırmalı çıktı üretmek
        - Raporlama servisleri için standart compare payload'u oluşturmak

    Miras aldığı alanlar:
        BaseTokenizerListRequest:
            text:
                Tokenize edilecek ham metin.

            tokenizer_names:
                Çalıştırılacak tokenizer adları.

    Örnek:
        {
            "text": "Merhaba dünya! Tokenizer karşılaştırması yapıyorum.",
            "tokenizer_names": ["char", "byte", "word"]
        }

    Not:
        Bu model, karşılaştırma için gerekli minimum bilgileri içerir. 
        Eğer latency, compression ratio, unknown token rate, reconstruction veya 
        pairwise comparison gibi gelişmiş analizler için ek parametreler gerekiyorsa, 
        bu model ayrı bir CompareAdvancedRequest modeliyle genişletilebilir.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "text": "Merhaba dünya! Tokenizer karşılaştırması yapıyorum.",
                    "tokenizer_names": ["char", "byte", "word"],
                },
                {
                    "text": "Byte Pair Encoding ile character tokenization farkını görmek istiyorum.",
                    "tokenizer_names": ["char", "bpe", "byte_bpe"],
                },
                {
                    "text": "Hello world! Tokenization is fun.",
                    "tokenizer_names": ["word", "white_space", "punctuation", "ngram"],
                },
            ] # JSON schema örnekleri, API dokümantasyonunda ve testlerde kullanılmak üzere sağlanır. 
            
        } # ConfigDict, Pydantic modellerinde model yapılandırması için kullanılan bir yapıdır.
    ) # Bu örnekler, API kullanıcılarının doğru request formatını anlamalarına yardımcı olur ve 
    # aynı zamanda testlerde doğrulama için referans olarak hizmet eder.


class CompareItemResponse(BaseModel):
    """
    Compare response içinde tek bir tokenizer'a ait sonucu temsil eder.

    Compare endpoint'i birden fazla tokenizer çalıştırdığı için response içinde
    her tokenizer için ayrı bir sonuç nesnesi döner. Bu model, o tekil sonucu
    standartlaştırır.


    Bu modelin sağladığı bilgiler:
        - Hangi tokenizer'ın çalıştığı
        - Üretilen token listesi
        - Toplam token sayısı
        - İlgili tokenizer'ın vocabulary büyüklüğü veya benzersiz token sayısı

    Kullanım alanları:
        - API response'unda her tokenizer için karşılaştırma sonuçlarını tutarlı şekilde döndürmek
        - Frontend'de tokenizer sonuçlarını tablo veya kart şeklinde göstermek
        - Raporlama servislerine veri sağlamak
        - Testlerde response contract doğrulama için kullanmak
        - Debug ve analiz amaçlı çıktılar üretmek

    Kullanım amacı:
        - Character, word, byte, BPE, regex, ngram gibi tokenizer'ların aynı metin üzerinde 
        nasıl farklı token dizileri ürettiğini gözlemlemek
        - Token sayısı, vocabulary çeşitliliği ve segmentation davranışını analiz etmek
        - UI tarafında tokenizer sonuçlarını detaylı şekilde göstermek
        - Raporlama servisleri için tokenizer sonuçlarını standart formatta sağlamak

    Not:
        Bu model intentionally sade tutulmuştur. Daha gelişmiş metrikler
        gerekiyorsa ayrı bir MetricsResponse veya DetailedCompareItemResponse
        modeliyle genişletilebilir.

    Response yapısı:
        {
            "tokenizer_name": "char",
            "tokens": ["M", "e", "r", "h", "a", "b", "a"],
            "token_count": 7,
            "vocab_size": 3
        }
    """

    tokenizer_name: str = Field(
        ...,
        description=(
            "Sonucu üreten tokenizer adıdır. "
            "Bu değer genellikle TokenizerFactory veya TokenizerRegistry "
            "üzerinden çözümlenen kayıtlı tokenizer ismine karşılık gelir."
        ), # Field, Pydantic modellerinde alan tanımlamak için kullanılan bir yapıdır. 
        # description, API dokümantasyonunda bu alanın ne anlama geldiğini açıklamak için kullanılır.
        examples=["char", "word", "byte_bpe"], # API dokümantasyonunda ve testlerde kullanılmak üzere sağlanır.
    ) # Bu alan, frontend ve rapor üretimi tarafından hangi tokenizer'ın hangi sonuçları ürettiğini göstermek için kullanılır.

    tokens: list[str] = Field(
        ...,
        description=(
            "İlgili tokenizer tarafından üretilen token listesidir. "
            "Tokenlar string olarak döndürülür; böylece UI, rapor üretimi ve "
            "debug çıktıları tarafından doğrudan gösterilebilir."
        ),
        examples=[
            ["M", "e", "r", "h", "a", "b", "a"],
            ["Merhaba", "dünya", "!"],
            ["Hello", "world", "!", "Tokenization", "is", "fun", "."],
        ],
    )

    token_count: int = Field(
        ...,
        ge=0, # token sayısı negatif olamaz, bu yüzden ge=0 (greater or equal to 0) kullanılır.
        description=(
            "Tokenizer tarafından üretilen toplam token sayısıdır. "
            "Bu metrik, tokenizer'ın segmentasyon granularity seviyesini "
            "anlamak için temel göstergelerden biridir. Daha yüksek değer, "
            "genellikle daha küçük parçalara bölme anlamına gelir."
        ),
        examples=[7], # API dokümantasyonunda ve testlerde kullanılmak üzere sağlanır.
    )

    vocab_size: int = Field(
        ...,
        ge=0,
        description=(
            "Tokenizer'ın vocabulary büyüklüğünü veya ilgili çalıştırma sonucunda "
            "gözlenen benzersiz token sayısını temsil eder. "
            "Custom tokenizer implementasyonuna göre bu değer gerçek vocabulary "
            "boyutu ya da output içindeki unique token sayısı olabilir."
        ),
        examples=[3, 256, 30522],
    )


class CompareResponse(BaseModel):
    """
    Çoklu tokenizer karşılaştırma endpoint'inin response modelidir.

    Bu model, aynı input metni üzerinde çalıştırılan tüm tokenizer sonuçlarını
    tek bir standart response altında toplar.

    Bu modelin sağladığı bilgiler:
    - Karşılaştırma yapılan orijinal ham metin
    - Başarıyla çalıştırılan tokenizer sayısı
    - Her tokenizer için CompareItemResponse listesi

    Response yapısı:
        text:
            Karşılaştırma yapılan orijinal ham metin.

        total_tokenizers:
            Başarıyla çalıştırılan tokenizer sayısı.

        results:
            Her tokenizer için CompareItemResponse listesi.

    Kullanım alanları:
        - API response standardizasyonu
        - Frontend compare ekranı
        - Tokenizer sonuçlarının tablo halinde gösterimi
        - Raporlama servislerine veri sağlama
        - Testlerde response contract doğrulama
        - Debug ve analiz amaçlı çıktılar üretmek
    
    Kullanım amacı:
        - Character, word, byte, BPE, regex, ngram gibi tokenizer'ların aynı metin üzerinde nasıl farklı token dizileri ürettiğini gözlemlemek
        - Token sayısı, vocabulary çeşitliliği ve segmentation davranışını analiz etmek
        - UI tarafında çoklu tokenizer sonuçlarını detaylı şekilde göstermek
        - Raporlama servisleri için çoklu tokenizer sonuçlarını standart formatta sağlamak

    Not:
        Bu response modeli temel karşılaştırma çıktısı için tasarlanmıştır.
        Eğer latency, compression ratio, unknown token rate, reconstruction
        veya pairwise comparison gibi gelişmiş analizler döndürülecekse
        response modeli ayrı bir detaylı schema ile genişletilebilir.
    """

    text: str = Field(
        ...,
        description=(
            "Karşılaştırma yapılan orijinal ham metindir. "
            "Response içinde tekrar döndürülmesi, frontend ve rapor üretimi "
            "tarafında context kaybını önler."
        ),
        examples=["Merhaba dünya!"], 
    )

    total_tokenizers: int = Field(
        ...,
        ge=0,
        description=(
            "Karşılaştırma sırasında başarıyla çalıştırılan tokenizer sayısıdır. "
            "Bu değer genellikle results listesinin uzunluğu ile aynı olmalıdır."
        ),
        examples=[3],
    )

    results: list[CompareItemResponse] = Field(
        ...,
        description=(
            "Her tokenizer için üretilen karşılaştırma sonuçlarının listesidir. "
            "Liste içindeki her item, tokenizer adı, token listesi, token sayısı "
            "ve vocabulary bilgisi içerir."
        ),
    )