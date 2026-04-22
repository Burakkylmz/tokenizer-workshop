from __future__ import annotations

from tokenizer_workshop.tokenizers.base import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    """
    Basit bir character-level tokenizer.

    Educational purpose:
    - tokenization'ın en temel formunu gösterir.
    - raw text üzerinden bir vocabulary'nin nasıl oluşturulduğunu öğretir.
    - text, token ids ve reconstruction arasındaki ilişkiyi görünür hale getirir.

    Core idea:
    Her benzersiz character, vocabulary içinde bir token haline gelir.

    Example:
        text = "aba"

        unique characters -> ["a", "b"]
        stoi -> {"a": 0, "b": 1}
        encode("aba") -> [0, 1, 0]
        decode([0, 1, 0]) -> "aba"

    Limitation:
    Bu tokenizer yalnızca training sırasında gördüğü characters'ı bilir.
    Daha sonra yeni bir character gelirse, encoding başarısız olur.
    Bu davranış v1 için bilinçli olarak seçilmiştir çünkü coverage problemini görünür kılar.
    """

    def __init__(self) -> None:
        super().__init__(name="char")

        # _stoi: string-to-integer mapping
        # encoding sırasında characters'ı token ids'e dönüştürmek için kullanılır.
        self._stoi: dict[str, int] = {}

        # _itos: integer-to-string mapping
        # decoding sırasında orijinal text'i yeniden oluşturmak için kullanılır.
        self._itos: dict[int, str] = {}

    def train(self, text: str) -> None:
        """
        raw text üzerinden character vocabulary oluşturur.

        Neden bu kadar basit bir tokenizer için bile training var?
        Çünkü bir tokenizer'ın learned veya predefined bir vocabulary'ye ihtiyacı vardır.
        Bu versiyonda vocabulary doğrudan training text içinden oluşturulur.

        Design choice:
        Yalnızca set(text) yerine sorted(set(text)) kullanıyoruz
        çünkü vocabulary sırasının run'lar arasında deterministic olmasını istiyoruz.
        Bu da experiment'ların reproducible olmasını sağlar ve öğretmeyi kolaylaştırır.
        """
        if not text:
            raise ValueError("Training text cannot be empty.")

        # training corpus içindeki tüm benzersiz characters'ı topluyoruz.
        # Example: "merhaba" -> {"m", "e", "r", "h", "a", "b"}
        #
        # Bunları sıralıyoruz ki token ids tutarlı şekilde atansın.
        # sorting olmazsa mapping sırası kararsız olabilir.
        unique_chars = sorted(set(text))

        # forward mapping'i oluştur: character -> token id
        self._stoi = {char: idx for idx, char in enumerate(unique_chars)}

        # reverse mapping'i oluştur: token id -> character
        #
        # Bunu _stoi üzerinden türetiyoruz ki iki mapping her zaman tutarlı kalsın.
        self._itos = {idx: char for char, idx in self._stoi.items()}

    def encode(self, text: str) -> list[int]:
        """
        text'i bir token ids listesine dönüştürür.

        Example:
            vocabulary: {"a": 0, "b": 1}
            encode("aba") -> [0, 1, 0]

        Important teaching point:
        encoding işlemi ancak her character vocabulary içinde varsa başarılı olur.
        Görülmemiş bir character gelirse tahmin yürütmek yerine error fırlatırız.
        """
        if not self._stoi:
            raise ValueError("Tokenizer has not been trained yet.")

        # output'u list[int] olarak tutuyoruz çünkü tokenizer'lar genelde
        # set veya dictionary değil, sıralı bir ids dizisi üretir.
        token_ids: list[int] = []

        for char in text:
            # v1 içinde unknown characters doğrudan hard error olarak ele alınır.
            # İleride <UNK> strategy eklenebilir, ama şimdilik learners'ın
            # coverage limitation'ı net biçimde görmesini istiyoruz.
            if char not in self._stoi:
                raise ValueError(
                    f"Unknown character encountered during encoding: {char!r}"
                )

            token_ids.append(self._stoi[char])

        return token_ids

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
        return [str(token_id) for token_id in token_ids]

    def decode(self, token_ids: list[int]) -> str:
        """
        token ids'i tekrar text'e dönüştürür.

        Example:
            reverse vocabulary: {0: "a", 1: "b"}
            decode([0, 1, 0]) -> "aba"

        Bu method önemlidir çünkü tokenization inspectable olmalıdır.
        Öğrenci ids'ten tekrar text'e dönemiyorsa mapping soyut kalır.
        """
        if not self._itos:
            raise ValueError("Tokenizer has not been trained yet.")

        chars: list[str] = []

        for token_id in token_ids:
            # Eğer bir token id reverse vocabulary içinde yoksa,
            # input'ta veya tokenizer state'inde bir problem vardır.
            if token_id not in self._itos:
                raise ValueError(
                    f"Unknown token id encountered during decoding: {token_id}"
                )

            chars.append(self._itos[token_id])

        # character sequence'i tekrar string haline getiriyoruz.
        return "".join(chars)

    @property
    def vocab_size(self) -> int:
        """
        tokenizer'ın şu anda bildiği benzersiz characters sayısını döndürür.

        Bu bir character-level tokenizer olduğu için vocab size,
        training sırasında toplanan benzersiz characters sayısına eşittir.
        """
        return len(self._stoi)