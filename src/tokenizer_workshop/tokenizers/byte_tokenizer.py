from __future__ import annotations

from tokenizer_workshop.tokenizers.base import BaseTokenizer


class ByteTokenizer(BaseTokenizer):
    """
    UTF-8 kodlamasını temel alan basit bir byte-level tokenizer.

    Eğitsel amaç:
    - Metnin karakter seviyesi yerine byte seviyesi üzerinden de temsil edilebileceğini gösterir.
    - Tokenizer vocabulary'sinin önceden sabit tanımlanabileceğini öğretir.
    - Byte-level tokenization yaklaşımının diller arasında neden daha dayanıklı olduğunu anlamayı kolaylaştırır.

    Temel fikir:
    Metin UTF-8 byte'larına dönüştürülür ve her byte bir token id olarak kullanılır.

    Örnek:
        text = "abc"
        utf-8 byte'ları -> [97, 98, 99]

    CharTokenizer'dan önemli farkı:
    - CharTokenizer vocabulary'yi eğitim metninden öğrenir.
    - ByteTokenizer ise boyutu sabit olan 256 elemanlı bir vocabulary kullanır (0 ile 255 arası).

    Sınır:
    - Byte-level tokenization oldukça dayanıklı olsa da daha uzun token dizileri üretebilir.
    - UTF-8 içindeki çok byte'lı karakterler birden fazla byte token'a bölünür.
    """

    def __init__(self) -> None:
        super().__init__(name="byte")
        self._is_trained = False

    def train(self, text: str) -> None:
        """
        Tokenizer'ı kullanıma hazır olarak işaretler.

        Vocabulary sabit ise neden burada yine de train() metodu var?
        Çünkü bu projedeki tüm tokenizer'lar aynı arayüzü takip eder.
        Ortak bir arayüz kullanmak projeyi hem öğretmeyi hem de genişletmeyi kolaylaştırır.

        Bu tokenizer'da metinden yeni bir vocabulary öğrenilmez.
        Byte vocabulary her zaman 0 ile 255 arasındaki tam sayılardan oluşur.
        """
        if not text:
            raise ValueError("Training text cannot be empty.")

        self._is_trained = True

    def encode(self, text: str) -> list[int]:
        """
        Metni UTF-8 byte token id'lerine dönüştürür.

        Örnek:
            "abc" -> [97, 98, 99]

        Önemli öğretici nokta:
        CharTokenizer'ın aksine burada öğrenilmiş bir karakter vocabulary'sine ihtiyaç yoktur.
        UTF-8 kodlaması zaten bize byte tabanlı bir temsil verir.
        """
        if not self._is_trained:
            raise ValueError("Tokenizer has not been trained yet.")

        # UTF-8 kodlaması bir bytes nesnesi döndürür.
        # Bunu list[int] haline çevirdiğimizde token id'leri doğrudan elde ederiz.
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        """
        Byte token id'lerini tekrar metne dönüştürür.

        Örnek:
            [97, 98, 99] -> "abc"

        Önemli öğretici nokta:
        Decode işlemi ancak byte dizisi geçerli bir UTF-8 yapısı oluşturuyorsa çalışır.
        Rastgele byte id'leri her zaman başarılı şekilde decode edilemeyebilir.
        """
        if not self._is_trained:
            raise ValueError("Tokenizer has not been trained yet.")

        for token_id in token_ids:
            if not 0 <= token_id <= 255:
                raise ValueError(
                    f"Invalid byte token encountered during decoding: {token_id}"
                )

        try:
            return bytes(token_ids).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Token ids do not form a valid UTF-8 byte sequence.") from exc

    @property
    def vocab_size(self) -> int:
        """
        Byte vocabulary boyutunu döndürür.

        Bir byte toplam 256 farklı değer alabilir: 0 ile 255 arası.
        """
        return 256