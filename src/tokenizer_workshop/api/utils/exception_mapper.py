"""
exception_mapper.py

API layer ile service layer arasındaki exception mapping fonksiyonlarını içerir.

Bu dosyanın amacı:
------------------
- Service katmanında fırlatılan domain-specific exception'ları HTTP seviyesine map etmek
- API layer'da tutarlı HTTP response'ları sağlamak
- Hata yönetimini merkezi hale getirmek
- Service katmanının FastAPI bağımlılığını ortadan kaldırmak
- API kullanıcılarına kontrollü hata mesajları sunmak

Bu dosyada tek bir fonksiyon vardır:
- map_service_exception: Service katmanında fırlatılan exception'ları HTTPException'a çevirir.

Bu fonksiyon, API endpoint'lerinde try-except bloklarında kullanılarak service katmanında oluşan hataların uygun HTTP response'lara dönüştürülmesini sağlar. 

Tasarımsal olarak, bu fonksiyon API layer ile service layer arasındaki error translation boundary'sini temsil eder. 
Service katmanı domain-specific exception'lar fırlatırken, API layer bu exception'ları HTTPException'a çevirir. 
Bu ayrım sayesinde service katmanı FastAPI bağımsız olur ve test edilebilirlik artar.
"""

from fastapi import HTTPException, status

from tokenizer_workshop.api.services import (
    UnsupportedTokenizerError,
    TokenizationServiceError,
)


def map_service_exception(exc: Exception) -> None:
    """
    Service katmanında fırlatılan exception'ları HTTP seviyesine map eder.

    Bu fonksiyon, API layer ile business logic (service layer) arasındaki
    hata çeviri (error translation) boundary'sini temsil eder.

    Amaç:
        - Domain exception'larını HTTP response'a dönüştürmek
        - Tutarlı status code kullanımı sağlamak
        - Internal error'ları kullanıcıya kontrollü şekilde expose etmek
        - Logging ile observability sağlamak

    Tasarım prensibi:
        Service layer:
            - domain-specific exception fırlatır

        API layer:
            - bu exception'ları HTTPException'a çevirir

    Bu ayrım sayesinde:
        ✔ Service katmanı FastAPI bağımlı olmaz
        ✔ Test edilebilirlik artar
        ✔ Hata yönetimi merkezi hale gelir
    """

    # ---------------------------------------------------------
    # CLIENT ERRORS (4xx)
    # ---------------------------------------------------------

    if isinstance(exc, UnsupportedTokenizerError):
        """
        Kullanıcı geçersiz tokenizer adı göndermiştir.

        Bu bir client hatasıdır çünkü:
            - input hatalıdır
            - sistem doğru çalışmaktadır
        """
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    # ---------------------------------------------------------
    # SERVER ERRORS (5xx)
    # ---------------------------------------------------------

    if isinstance(exc, TokenizationServiceError):
        """
        Tokenization sırasında beklenmeyen bir hata oluşmuştur.

        Bu bir server hatasıdır çünkü:
            - input doğru olabilir
            - işlem sırasında failure oluşmuştur
        """
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    raise exc