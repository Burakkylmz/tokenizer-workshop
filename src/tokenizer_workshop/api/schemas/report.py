"""
report.py

report.py

Report endpoint'i için request ve response modellerini içerir.

Bu dosyanın amacı:
------------------
- Rapor üretme endpoint'inin request contract'ını tanımlamak
- Rapor üretme endpoint'inin response contract'ını tanımlamak
- Kullanıcının hangi metin, hangi tokenizer'lar ve hangi format ile rapor istediğini
  açık ve doğrulanabilir bir model üzerinden almak
- API çıktısının frontend, CLI veya başka client'lar tarafından tutarlı şekilde
  tüketilmesini sağlamak

Bu dosyada iki ana model vardır:
-------------------------------
1. ReportRequest
   - Client tarafından API'ye gönderilen veriyi temsil eder.

2. ReportResponse
   - API tarafından client'a döndürülen rapor sonucunu temsil eder.

Bu modeller yalnızca veri taşımaz.
Aynı zamanda:
- validation sağlar
- OpenAPI/Swagger dokümantasyonunu zenginleştirir
- API contract'ını okunabilir hale getirir
- Test yazmayı kolaylaştırır
- Frontend ve diğer client'ların API ile nasıl iletişim kuracağını netleştirir
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tokenizer_workshop.api.schemas.base import BaseTokenizerListRequest


class ReportRequest(BaseTokenizerListRequest):
    """
    Tokenizer raporu üretmek için kullanılan request modelidir.

    Bu model, BaseTokenizerListRequest sınıfından miras alır.
    Bu nedenle otomatik olarak şu alanlara sahiptir:

        text: str
        tokenizer_names: list[str]

    BaseTokenizerListRequest üzerinden gelen davranışlar:
    ----------------------------------------------------
    - text alanı zorunludur.
    - text boş olamaz.
    - text yalnızca whitespace karakterlerinden oluşamaz.
    - text başındaki ve sonundaki boşluklardan temizlenir.
    - tokenizer_names en az bir tokenizer adı içermelidir.
    - tokenizer isimleri trim edilir.
    - tokenizer isimleri lowercase hale getirilir.
    - duplicate tokenizer isimleri engellenir.

    Bu modele özel alanlar:
    -----------------------
    - format
    - mode

    Kullanım amacı:
    ---------------
    Client, tokenizer raporu üretmek istediğinde bu modeli kullanarak API'ye
    gerekli bilgileri gönderir.

    Örnek request body:

        {
            "text": "Merhaba dünya! Tokenizer raporu oluştur.",
            "tokenizer_names": ["char", "word"],
            "format": "txt",
            "mode": "evaluate"
        }

    Bu request şunu ifade eder:
    ---------------------------
    - Verilen text analiz edilecek.
    - "char" ve "word" tokenizer'ları kullanılacak.
    - Rapor txt formatında üretilecek.
    - Rapor evaluate modunda hazırlanacak.

    Neden BaseTokenizerListRequest'ten türedi?
    ------------------------------------------
    Çünkü rapor endpoint'i de text tabanlıdır ve birden fazla tokenizer ile çalışır.

    Eğer bu model doğrudan BaseModel'den türeseydi:
    - text alanı tekrar tanımlanacaktı
    - tokenizer_names alanı tekrar tanımlanacaktı
    - validation tekrar yazılacaktı
    - farklı endpoint'ler arasında contract tutarsızlığı oluşabilecekti

    Bu yüzden ortak alanlar base class'a alınmış, bu model ise sadece rapor
    üretimine özel alanları tanımlamıştır.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "text": "Merhaba dünya! Tokenizer raporu oluştur.",
                    "tokenizer_names": ["char", "word"],
                    "format": "txt",
                    "mode": "evaluate",
                },
                {
                    "text": "This is a sample input for tokenizer comparison.",
                    "tokenizer_names": ["word", "bpe", "byte_bpe"],
                    "format": "md",
                    "mode": "compare",
                },
            ]
        }
    )

    format: Literal["txt", "md"] = Field(
        default="txt",
        description=(
            "Üretilecek rapor formatı. "
            "Desteklenen değerler: 'txt' ve 'md'. "
            "'txt' düz metin rapor üretir. "
            "'md' Markdown formatında rapor üretir. "
            "Varsayılan değer 'txt' olarak belirlenmiştir."
        ),
        examples=["txt", "md"],
    )

    mode: Literal["compare", "evaluate"] = Field(
        default="evaluate",
        description=(
            "Rapor üretiminde kullanılacak analiz modu. "
            "'evaluate' modu tokenizer sonuçlarını metrik odaklı değerlendirir. "
            "'compare' modu tokenizer çıktıları arasındaki farkları karşılaştırmaya odaklanır. "
            "Varsayılan değer 'evaluate' olarak belirlenmiştir."
        ),
        examples=["evaluate", "compare"],
    )


class ReportResponse(BaseModel):
    """
     Rapor üretme endpoint'inin döndürdüğü response modelidir.

    Bu model, report endpoint'i başarılı şekilde çalıştığında client'a döndürülen
    verinin yapısını temsil eder.

    Response modelinin amacı:
    -------------------------
    - Üretilen rapor içeriğini client'a döndürmek
    - Raporun hangi formatta üretildiğini belirtmek
    - İndirme işlemleri için önerilen dosya adını sağlamak

    Alanlar:
    --------
    report:
        Üretilen raporun string içeriğidir.

    format:
        Raporun üretildiği formatı belirtir.
        Desteklenen değerler:
        - txt
        - md

    filename:
        Client tarafında dosya indirme işlemi yapılacaksa kullanılabilecek
        önerilen dosya adıdır.

    Örnek response:

        {
            "report": "TOKENIZER COMPARISON REPORT\\n...",
            "format": "txt",
            "filename": "tokenizer_report.txt"
        }

    Frontend açısından önemi:
    -------------------------
    Frontend bu response'u aldıktan sonra:

    - report alanını ekranda gösterebilir
    - format alanına göre syntax highlighting veya preview seçebilir
    - filename alanını indirilebilir dosya adı olarak kullanabilir

    CLI açısından önemi:
    --------------------
    CLI client bu response'u aldıktan sonra:

    - report içeriğini terminale yazdırabilir
    - filename ile dosyaya kaydedebilir
    - format bilgisine göre output davranışını değiştirebilir

    Neden response modeli gerekli?
    ------------------------------
    Response modeli kullanılmazsa endpoint düzensiz dict döndürebilir.
    Bu da:
    - client tarafında belirsizlik oluşturur
    - OpenAPI dokümantasyonunu zayıflatır
    - test yazmayı zorlaştırır
    - uzun vadede contract kırılmalarına neden olabilir

    Bu model sayesinde endpoint çıktısı açık, tahmin edilebilir ve test edilebilir olur.
    """

    report: str = Field(
        ...,
        description=(
            "Üretilen raporun string içeriği. "
            "Bu alan txt veya markdown formatındaki rapor metnini taşır. "
            "Client bu alanı doğrudan ekranda gösterebilir veya dosyaya yazabilir."
        ),
    )

    format: Literal["txt", "md"] = Field(
        ...,
        description=(
            "Üretilen raporun formatı. "
            "Bu değer request içinde gönderilen format ile uyumlu olmalıdır. "
            "Desteklenen değerler: 'txt' ve 'md'."
        ),
        examples=["txt", "md"],
    )

    filename: str = Field(
        ...,
        description=(
            "İndirme işlemleri için önerilen dosya adı. "
            "Dosya uzantısı rapor formatıyla uyumlu olmalıdır. "
            "Örneğin format 'txt' ise filename genellikle '.txt' ile, "
            "format 'md' ise filename genellikle '.md' ile biter."
        ),
        examples=["tokenizer_report.txt", "tokenizer_report.md"],
    )