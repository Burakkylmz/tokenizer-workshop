"""
tokenize.py

Tekli tokenization endpoint'i için request ve response modellerini içerir.

Bu schema dosyası, kullanıcıdan gelen ham metni ve seçilen tokenizer adını
standartlaştırır. Aynı zamanda API response'unun frontend, test ve raporlama
tarafında tutarlı şekilde tüketilmesini sağlar.

Bu dosyada iki ana model vardır:
1. TokenizeRequest
2. TokenizeResponse

TokenizeRequest:
- text: Tokenize edilecek ham metin 
- tokenizer_name: Kullanılacak tokenizer'ın adı

TokenizeResponse:
- tokenizer_name: Kullanılan tokenizer'ın adı   

- tokens: Token listesi
- token_count: Toplam token sayısı
- vocab_size: Benzersiz token sayısı

Bu modeller yalnızca veri taşımaz.

Aynı zamanda:
- validation sağlar
- OpenAPI/Swagger dokümantasyonunu zenginleştirir
- API contract'ını okunabilir hale getirir
- Test yazmayı kolaylaştırır
- Frontend ve diğer client'ların API ile nasıl iletişim kuracağını netleştirir
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tokenizer_workshop.api.schemas.base import BaseTextRequest


class TokenizeRequest(BaseTextRequest):
    """
    Tekli tokenizer çalıştırma işlemi için kullanılan request modelidir.

    Bu model, tek bir ham metnin yalnızca bir tokenizer ile işlenmesini sağlar.

    Bu model, BaseTextRequest sınıfından miras alır. Bu nedenle otomatik olarak şu alanlara sahiptir:
        text: str

    BaseTextRequest üzerinden gelen davranışlar:    
    - text alanı zorunludur.    
    - text boş olamaz.    
    - text yalnızca whitespace karakterlerinden oluşamaz.
    - text başındaki ve sonundaki boşluklardan temizlenir.
    - text normalize edilir (örneğin, Unicode normalization)
    - text'in uzunluğu belirli bir sınırı aşamaz (örneğin, 10.000 karakter)
    - text'in içeriği belirli kurallara uyar (örneğin, kontrol karakterleri içermez)
    
    Ek alanlar:
    - tokenizer_name: Çalıştırılacak tokenizer adı. 
    Bu alan, API'ye gönderilen veride tokenizer'ı belirtmek için kullanılır. 
    Bu alanın doğrulanması ve normalize edilmesi için field_validator kullanılır.   

    Kullanım amacı:
        - Kullanıcının seçtiği tokenizer ile hızlı tokenization yapmak
        - UI tarafında tekli tokenizer çıktısı göstermek
        - Compare endpoint'ine göre daha sade bir analiz akışı sağlamak
        - Debug ve eğitim amaçlı token çıktısını doğrudan incelemek

    Miras aldığı alanlar:
        BaseTextRequest:
            text:
                Tokenize edilecek ham metin.

    Not:
        - Bu model, tek bir tokenizer için tasarlanmıştır. Çoklu tokenizer karşılaştırması  
        için CompareRequest modeli kullanılmalıdır.
        - Bu model, tokenizer'ın tokenize() metoduna doğrudan input olarak verilecek veriyi temsil eder.
        - Bu model, tokenizer'ın tokenize() metodundan dönecek çıktıyı değil, 
        yalnızca tokenize() metoduna girdi olarak verilecek ham metni ve tokenizer adını temsil eder.

    Örnek:
        {
            "text": "Merhaba dünya!",
            "tokenizer_name": "char"
        }
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "text": "Merhaba dünya!",
                    "tokenizer_name": "char",
                },
                {
                    "text": "This is a tokenizer test.",
                    "tokenizer_name": "word",
                },
                {
                    "text": "Tokenizer behavior changes depending on strategy.",
                    "tokenizer_name": "byte_level_bpe",
                },
            ]
        }
    )

    tokenizer_name: str = Field(
        ...,
        min_length=1,
        description=(
            "Tokenization işlemi için kullanılacak tokenizer adıdır. "
            "Bu değer backend tarafında TokenizerFactory veya TokenizerRegistry "
            "üzerinden çözümlenir. Büyük/küçük harf farkı normalize edilir."
        ),
        examples=["char", "byte", "word", "byte_level_bpe"],
    )

    @field_validator("tokenizer_name")
    @classmethod
    def validate_tokenizer_name(cls, value: str) -> str:
        """
        tokenizer_name alanını normalize eder.

        Yapılan işlemler:
            1. Baştaki ve sondaki boşluklar temizlenir.
            2. Değer lowercase hale getirilir.
            3. Boş veya yalnızca whitespace içeren değerler reddedilir.

        Bu validasyon sayesinde:
            - " Char " -> "char"
            - "WORD"   -> "word"
            - " byte " -> "byte"
        gibi inputlar backend tarafında tutarlı şekilde işlenir.

        Args:
            value:
                Kullanıcıdan gelen tokenizer adı.

        Returns:
            Normalize edilmiş tokenizer adı.

        Raises:
            ValueError:
                tokenizer_name boşsa veya yalnızca whitespace içeriyorsa.

        Not:    
            Bu validasyon, API'ye gelen verinin backend tarafında doğru şekilde işlenmesini sağlar.
            Ayrıca, kullanıcı hatalarını tolere ederek daha iyi bir kullanıcı deneyimi sunar.
            Bu validasyon sayesinde, frontend tarafında kullanıcıya geri bildirim vermek ve hataları düzeltmek daha kolay hale gelir.
        """
        # tokenizer_name alanı için normalize işlemi yapılır. 
        # Bu sayede kullanıcı inputundaki küçük hatalar tolere edilir.
        cleaned_value = value.strip().lower()

        if not cleaned_value:
            raise ValueError("tokenizer_name boş olamaz.")

        return cleaned_value


class TokenizeResponse(BaseModel):
    """
    Tekli tokenization endpoint'inin response modelidir.

    Bu model, bir tokenizer'ın bir input metni üzerinde ürettiği temel çıktıyı
    standartlaştırır.

    Response içeriği:
        tokenizer_name:
            Kullanılan tokenizer adı.

        tokens:
            Tokenizer tarafından üretilen sıralı token listesi.

        token_count:
            Üretilen toplam token sayısı.

        vocab_size:
            Output içindeki benzersiz token sayısı.

    Kullanım amacı:
        - API response standardizasyonu
        - Frontend token preview ekranı
        - Testlerde response contract doğrulama
        - Raporlama servislerine temel tokenization çıktısı sağlama
        - Debug ve eğitim amaçlı çıktılar üretmek

    Not:
        - Bu model, tek bir tokenizer'ın çıktısını temsil eder. Çoklu tokenizer karşılaştırması  
        için CompareResponse modeli kullanılmalıdır.
        - Bu model, tokenizer'ın tokenize() metodundan dönecek çıktıyı temsil eder. 
        Yani, API'ye gelen ham metni tokenize() metoduna verdikten sonra elde edilen sonuç bu modelde temsil edilir.
    """

    tokenizer_name: str = Field(
        ...,
        description=(
            "Tokenization sırasında kullanılan tokenizer adıdır. "
            "Genellikle request içinde gönderilen tokenizer_name alanının "
            "normalize edilmiş halidir."
        ),
        examples=["char", "word", "byte_level_bpe"],
    )

    tokens: list[str] = Field(
        ...,
        description=(
            "Tokenization sonucu oluşan sıralı token listesidir. "
            "Token sırası korunur; bu nedenle çıktı doğrudan segmentasyon "
            "davranışını incelemek için kullanılabilir."
        ),
        examples=[
            ["M", "e", "r", "h", "a", "b", "a"],
            ["Merhaba", "dünya", "!"],
        ],
    )

    token_count: int = Field(
        ...,
        ge=0,
        description=(
            "Toplam token sayısıdır. "
            "Bu değer normalde len(tokens) sonucuna eşit olmalıdır. "
            "Tokenizer'ın metni ne kadar parçalı ya da kompakt temsil ettiğini "
            "gösteren temel metriktir."
        ),
        examples=[7],
    )

    vocab_size: int = Field(
        ...,
        ge=0,
        description=(
            "Bu response bağlamında output içindeki benzersiz token sayısını temsil eder. "
            "Genellikle len(set(tokens)) ile hesaplanır. "
            "Not: Eğer gerçek tokenizer vocabulary boyutu döndürülmek istenirse, "
            "ayrı bir alan olarak tokenizer_vocab_size kullanılması daha doğru olur."
        ),
        examples=[6],
    )