from __future__ import annotations

# Protocol:
# Belirli bir davranışı (yani sağlanması gereken metot imzalarını) tanımlamak için kullanılır.
# Amaç, CompareManager gibi sınıfların belirli concrete class'lara bağımlı olmasını önlemektir.
#
# önemli olan şey:
# "Bu nesne WordTokenizer mı, CharTokenizer mı?" sorusu değil, "Bu nesne tokenize davranışını sağlıyor mu?" sorusudur.
from typing import Protocol, runtime_checkable


# ============================================================
# TokenizerProtocol
# ============================================================
# Bu protocol, tokenize işlemi yapabilen tüm tokenizer sınıfları için temel sözleşmeyi tanımlar.
#
# Burada yalnızca tokenize davranışı zorunludur.
# Çünkü bazı tokenizer'lar eğitim (train) gerektirmez.
#
# Örnek olarak şu sınıflar bu protocol'e uyabilir:
# - WordTokenizer
# - CharTokenizer
# - ByteTokenizer
# - ByteBPETokenizer
#
# Avantajı:
# CompareManager artık belirli sınıf isimlerine değil, ortak davranışa bağımlı hale gelir.
@runtime_checkable
class TokenizerProtocol(Protocol):
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

    Bu protocol, compare sisteminde kullanılacak tokenizer'ların
    en azından tokenize(text: str) -> list[str] davranışını
    sağlamasını bekler.
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


# ============================================================
# TrainableTokenizerProtocol
# ============================================================
# Bu protocol, tokenize edebilmenin yanında train() desteği de olan
# tokenizer'ları temsil eder.
#
# Bu ayrım neden önemli?
# Çünkü her tokenizer eğitilebilir olmak zorunda değildir.
#
# Örneğin:
# - CharTokenizer çoğu zaman train gerektirmez
# - ByteTokenizer çoğu zaman train gerektirmez
# - ByteBPETokenizer ise train gerektirebilir
#
# Böylece CompareManager içinde:
# "Bu tokenizer train edilebilir mi?" sorusunu daha temiz ele alabiliriz.
@runtime_checkable
class TrainableTokenizerProtocol(TokenizerProtocol, Protocol):
    """
    TrainableTokenizer protocol'ü, tokenize işlemi yapabilmenin yanında
    train() desteği de olan tokenizer'ları temsil eder.

    Bu protocol'ün amacı:
    - tokenize edebilmenin yanında eğitim (train) desteği de olan tokenizer'ları tanımlamak
    - CompareManager gibi yapılarda train edilebilir tokenizer'ları özel olarak ele almak

    Örneğin bazı tokenizer'lar (örneğin CharTokenizer, ByteTokenizer) çoğu zaman train gerektirmezken,
    bazıları (örneğin ByteBPETokenizer) train gerektirebilir.

    Bu sayede CompareManager içinde:
    "Bu tokenizer train edilebilir mi?" sorusunu daha temiz ele alabiliriz.

    Bu protocol'e uyan sınıflar hem tokenize etmeli
    hem de train edilebilmelidir.
    """

    def train(self, text: str) -> None:
        """
        Tokenizer'ı verilen eğitim metni üzerinde eğitmelidir.

        Args:
            text (str):
                Tokenizer'ı eğitmek için kullanılacak ham metin.
        """
        ...