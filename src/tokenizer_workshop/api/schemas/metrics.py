"""
metrics.py

Tokenizer metrikleri için response modellerini içerir.

Bu dosyanın amacı:
------------------
- Tokenizer'ların tokenization sonuçlarını analiz etmek için gerekli metrikleri tanımlamak
- API response'larında bu metrikleri standart bir formatta sunmak
- Frontend ve diğer client'ların bu metrikleri nasıl alacağını ve kullanacağını netleştirmek
- Testlerde metrik hesaplama doğruluğunu kontrol etmek

Bu dosyada tanımlanan modeller:
-------------------------------
1. TopTokenResponse: En sık görülen tokenları ve frekanslarını temsil eder.
2. MetricsResponse: Tek bir tokenizer için hesaplanan çeşitli metrikleri temsil eder.
3. EvaluationResponse: Tek bir tokenizer'ın tokenization sonuçları ve metriklerini temsil eder.
4. PairwiseComparisonResponse: İki tokenizer arasındaki token farklılıklarını ve örtüşme oranını temsil eder.
5. TokenizerComparisonResult: Çoklu tokenizer'ların karşılaştırma sonuçlarını ve metriklerini tek bir response altında toplar.

Bu modeller yalnızca veri taşımaz.
Aynı zamanda:
- validation sağlar
- OpenAPI/Swagger dokümantasyonunu zenginleştirir
- API contract'ını okunabilir hale getirir
- Test yazmayı kolaylaştırır
- Frontend ve diğer client'ların API ile nasıl iletişim kuracağını netleştirir

Tokenizer'ların performansını ve davranışını detaylı şekilde analiz etmek için kritik öneme sahiptir.
Tokenizer'ların tokenization stratejilerini, verimliliklerini ve çıktı kalitelerini değerlendirmek için kullanılır.
Tokenizer'ların aynı metin üzerinde nasıl farklı sonuçlar ürettiğini anlamak için de önemlidir.
UI tarafında tokenizer sonuçlarını detaylı şekilde göstermek ve raporlama servisleri için veri sağlamak amacıyla kullanılır.
"""

from pydantic import BaseModel, Field


class TopTokenResponse(BaseModel):
    """
    Token frekans analizi sonucunda en sık görülen tokenları temsil eder.

    Bu model, token değerini ve o tokenın metin içinde kaç kez geçtiğini içerir.
    Bu bilgiler, tokenizer'ın metni nasıl segment ettiğini anlamak için kritik bir sinyaldir.

    Bu model genellikle:
        - Zipf dağılımı analizi
        - Token yoğunluk incelemesi
        - Vocabulary davranışı gözlemi
    gibi analizlerde kullanılır.

    Not:
        Token frekansları, tokenizer'ın metni nasıl segment ettiğini
        anlamak için kritik bir sinyaldir.
    """

    token: str = Field(
        ...,
        description="Token string değeri.",
        examples=["the", "a", "##ing"],
    )

    count: int = Field(
        ...,
        ge=0,
        description="Token'ın metin içinde kaç kez geçtiğini gösterir.",
        examples=[42],
    )


class MetricsResponse(BaseModel):
    """
    Tek bir tokenizer için hesaplanan detaylı performans ve davranış metriklerini içerir.

    Bu model, tokenizer'ın tokenization sonuçlarını çeşitli açılardan analiz etmek için kullanılır.

    Bu model tokenizer'ı sadece 'çalışıyor mu?' seviyesinden çıkarıp
    'nasıl çalışıyor?' ve 'ne kadar iyi çalışıyor?' seviyesine taşır.

    Metrikler 3 ana kategoriye ayrılabilir:

    1. Yapısal metrikler:
        - token_count
        - unique_token_count
        - token_length_distribution

    2. İstatistiksel metrikler:
        - unique_ratio
        - average_token_length
        - avg_chars_per_token

    3. Performans metrikleri:
        - latency_seconds
        - latency_per_token
        - efficiency_score
        - compression_ratio

    4. Kalite metrikleri:
        - unknown_count
        - reconstruction_match
    """

    # ---------------------------------------------------------
    # BASIC COUNTS
    # ---------------------------------------------------------

    token_count: int = Field(
        ...,
        ge=0,
        description=(
            "Toplam üretilen token sayısıdır. "
            "Düşük değer → daha kompakt temsil, "
            "yüksek değer → daha granular segmentasyon."
        ),
        examples=[7],
    )

    unique_token_count: int = Field(
        ...,
        ge=0,
        description=(
            "Üretilen tokenlar içindeki benzersiz token sayısıdır. "
            "Vocabulary çeşitliliğini gösterir."
        ),
        examples=[5],
    )

    unique_ratio: float = Field(
        ...,
        ge=0,
        description=(
            "unique_token_count / token_count oranıdır. "
            "1'e yakınsa her token benzersizdir (düşük tekrar), "
            "0'a yakınsa yoğun tekrar vardır."
        ),
        examples=[0.71],
    )

    # ---------------------------------------------------------
    # TOKEN LENGTH ANALYSIS
    # ---------------------------------------------------------

    average_token_length: float = Field(
        ...,
        ge=0,
        description=(
            "Tokenların ortalama uzunluğunu gösterir. "
            "Karakter bazlı tokenizer'larda düşüktür, "
            "word/ngram tokenizer'larda yüksektir."
        ),
        examples=[4.2],
    )

    min_token_length: int = Field(
        ...,
        ge=0,
        description="En kısa token uzunluğu.",
        examples=[1],
    )

    max_token_length: int = Field(
        ...,
        ge=0,
        description="En uzun token uzunluğu.",
        examples=[12],
    )

    avg_chars_per_token: float = Field(
        ...,
        ge=0,
        description=(
            "Toplam karakter sayısının token sayısına oranıdır. "
            "Compression davranışını anlamak için kullanılır."
        ),
        examples=[4.7],
    )

    # ---------------------------------------------------------
    # UNKNOWN TOKENS
    # ---------------------------------------------------------

    unknown_count: int = Field(
        ...,
        ge=0,
        description=(
            "Tokenizer tarafından bilinmeyen ([UNK] vb.) token sayısıdır. "
            "Pretrained tokenizer'larda kritik bir kalite göstergesidir."
        ),
        examples=[2],
    )

    unknown_rate: float = Field(
        ...,
        ge=0,
        description=(
            "unknown_count / token_count oranıdır. "
            "Yüksek olması tokenizer'ın input'u iyi temsil edemediğini gösterir."
        ),
        examples=[0.12],
    )

    # ---------------------------------------------------------
    # PERFORMANCE METRICS
    # ---------------------------------------------------------

    latency_seconds: float = Field(
        ...,
        ge=0,
        description=(
            "Tokenization işleminin toplam süresidir (saniye cinsinden). "
            "Benchmark ve performans karşılaştırmalarında kullanılır."
        ),
        examples=[0.00012],
    )

    latency_per_token: float = Field(
        ...,
        ge=0,
        description=(
            "Token başına ortalama işlem süresidir. "
            "Tokenizer'ın ölçeklenebilirliğini ölçmek için önemlidir."
        ),
        examples=[0.00001],
    )

    # ---------------------------------------------------------
    # EFFICIENCY / COMPRESSION
    # ---------------------------------------------------------

    efficiency_score: float = Field(
        ...,
        ge=0,
        description=(
            "Tokenizer verimliliğini temsil eden türetilmiş bir metriktir. "
            "Genellikle chars_per_token veya benzeri formüllerden türetilir. "
            "Yüksek değer → daha iyi sıkıştırma."
        ),
        examples=[6.5],
    )

    compression_ratio: float = Field(
        ...,
        ge=0,
        description=(
            "Metnin ne kadar kompakt temsil edildiğini gösterir. "
            "Genellikle len(text) / token_count olarak hesaplanır."
        ),
        examples=[4.7],
    )

    # ---------------------------------------------------------
    # DISTRIBUTION ANALYSIS
    # ---------------------------------------------------------

    top_tokens: list[TopTokenResponse] = Field(
        default_factory=list,
        description=(
            "En sık görülen tokenların listesi. "
            "Token frekans dağılımını anlamak için kullanılır."
        ),
    )

    token_length_distribution: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Token uzunluklarının dağılımını gösterir. "
            "Örnek: {'1': 10, '2': 5, '5': 2}"
        ),
    )

    # ---------------------------------------------------------
    # RECONSTRUCTION
    # ---------------------------------------------------------

    reconstructed_text: str | None = Field(
        default=None,
        description=(
            "Tokenlardan yeniden oluşturulan metin. "
            "Tüm tokenizer'lar için birebir eşleşme garanti değildir."
        ),
    )

    reconstruction_match: bool | None = Field(
        default=None,
        description=(
            "Reconstructed text ile orijinal metnin birebir eşleşip eşleşmediğini gösterir. "
            "Byte-level tokenizer'larda genellikle True, "
            "subword tokenizer'larda False olabilir."
        ),
    )


class EvaluationResponse(BaseModel):
    """
    Tek bir tokenizer için tüm evaluation çıktısını temsil eder.

    Bu model, tokenizer'ın tokenization sonuçlarını ve hesaplanan metrikleri tek bir yapıda toplar.

    Bu model:
        - tokenizer output (tokens)
        - hesaplanan metrikler (metrics)
    birleşimini sağlar.

    Compare pipeline içinde her tokenizer için bir EvaluationResponse üretilir.
    """

    tokenizer_name: str = Field(
        ...,
        description="Değerlendirilen tokenizer adı.",
        examples=["char", "byte_bpe", "word"],
    )

    tokens: list[str] = Field(
        ...,
        description=(
            "Tokenizer tarafından üretilen token listesi. "
            "Bu liste sıralıdır ve segmentation davranışını doğrudan gösterir."
        ),
    )

    metrics: MetricsResponse = Field(
        ...,
        description="Bu tokenizer için hesaplanan detaylı metrikler.",
    )


class PairwiseComparisonResponse(BaseModel):
    """
    İki tokenizer arasındaki karşılaştırmayı temsil eder.

    Bu model, iki tokenizer'ın tokenization sonuçları arasındaki farkları ve benzerlikleri analiz etmek için kullanılır.

    Bu model, tokenizer'ların sadece bireysel performansını değil,
    birbirlerine göre farklarını analiz etmeyi sağlar. 

    Kullanım amacı:
        - Token overlap analizi
        - Semantic segmentation farklarını inceleme
        - Tokenization stratejileri arasındaki benzerlik ölçümü
        - Tokenizer sonuçlarının tablo halinde gösterimi
        - Raporlama servislerine veri sağlama
        - Testlerde response contract doğrulama
        - Debug ve analiz amaçlı çıktılar üretmek

    Not:
        - Bu model, iki tokenizer'ın aynı metin üzerinde nasıl farklı token dizileri ürettiğini gözlemlemek için kritik bir yapıdır.
        - Token overlap analizi, tokenizer'ların benzer segmentasyon stratejileri kullanıpunmadığını anlamak için önemlidir.
        - Semantic segmentation farkları, tokenizer'ların metni nasıl farklı şekilde parçaladığını anlamak için önemlidir.  
    """

    left_name: str = Field(..., description="Karşılaştırmanın sol tarafındaki tokenizer adı.")
    right_name: str = Field(..., description="Karşılaştırmanın sağ tarafındaki tokenizer adı.")
    common_tokens: list[str] = Field(
        default_factory=list,
        description="Her iki tokenizer tarafından ortak üretilen tokenlar.",
    )
    unique_to_left: list[str] = Field(
        default_factory=list,
        description="Sadece sol tokenizer'a özgü tokenlar.",
    )
    unique_to_right: list[str] = Field(
        default_factory=list,
        description="Sadece sağ tokenizer'a özgü tokenlar.",
    )
    overlap_ratio: float = Field(
        ...,
        ge=0,
        description=(
            "İki tokenizer arasındaki token örtüşme oranıdır. "
            "1'e yakın → benzer segmentation, "
            "0'a yakın → tamamen farklı segmentation."
        ),
    )


class TokenizerComparisonResult(BaseModel):
    """
    Çoklu tokenizer karşılaştırmasının nihai ve zenginleştirilmiş çıktısını temsil eder.

    Bu model, aynı input metni üzerinde çalıştırılan tüm tokenizer sonuçlarını tek bir standart response altında toplar.

    Bu model sistemin en üst seviyedeki çıktısıdır ve şunları içerir:
        1. Source text
        2. Her tokenizer için evaluation
        3. Tokenizer'lar arası pairwise karşılaştırmalar

    Bu yapı:
        - API response
        - rapor üretimi (TXT / MD / PDF)
        - UI görselleştirme
        - analiz pipelineları için merkezi veri modelidir.
    için merkezi veri modelidir.
    """

    source_text: str = Field(
        ...,
        description="Karşılaştırma yapılan ham metin.",
    )

    evaluations: list[EvaluationResponse] = Field(
        ...,
        description="Her tokenizer için detaylı evaluation sonuçları.",
    )

    pairwise_comparisons: list[PairwiseComparisonResponse] = Field(
        default_factory=list,
        description="Tokenizer çiftleri arasındaki karşılaştırmalar.",
    )