# Protocol:
# belirli bir davranışı (metot imzasını) tanımlamak için kullanılır
# Bu sayede bir sınıfın hangi metodu sağlaması gerektiğini belirtiriz.
from typing import Protocol


class Tokenizer(Protocol):
    """
    Tokenizer protocol'ü, tokenize işlemi yapabilen tüm sınıflar için
    ortak bir sözleşme (contract) tanımlar.

    Bu protocol'ün amacı:
    - CompareManager gibi yapılarda concrete bir sınıfa bağımlılığı azaltmak
    - tokenize davranışı olan tüm sınıfları ortak bir tip altında toplamak
    - type hint kullanımını daha profesyonel ve anlaşılır hale getirmek

    Yani burada önemli olan sınıfın adı değil,
    `tokenize(text: str) -> list[str]` davranışını sağlamasıdır.

    Bu sayede aşağıdaki gibi farklı sınıflar aynı yapıyla kullanılabilir:
    - WordTokenizer
    - CharTokenizer
    - ByteTokenizer
    - ByteBPETokenizer
    - gelecekte eklenecek başka tokenizer'lar

    Avantajları:
    - daha gevşek bağlı (loosely coupled) tasarım sağlar
    - test yazmayı kolaylaştırır
    - yeni tokenizer eklemeyi kolaylaştırır
    - compare logic'i belirli sınıflara bağımlı kalmaz
    """

    def tokenize(self, text: str) -> list[str]:
        """
        Verilen metni token listesine dönüştürmelidir.

        Args:
            text (str):
                Tokenize edilecek ham metin.

        Returns:
            list[str]:
                Tokenize işlemi sonucunda elde edilen token listesi.
        """
        ...