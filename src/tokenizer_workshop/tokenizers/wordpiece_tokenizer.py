from __future__ import annotations

import re
from collections import Counter

from tokenizer_workshop.tokenizers.base import BaseTokenizer
from tokenizer_workshop.tokenizers.registry import register_tokenizer


@register_tokenizer("wordpiece")
class WordPieceTokenizer(BaseTokenizer):
    """
    WordPiece tabanlı subword tokenizer implementasyonu.

    Bu tokenizer, özellikle BERT ailesinde kullanılan WordPiece yaklaşımının
    sadeleştirilmiş ve eğitim amaçlı bir versiyonudur. Amaç, production seviyesinde
    birebir HuggingFace/BERT uyumluluğu sağlamak değil; WordPiece mantığını projedeki
    diğer tokenizer türleriyle aynı arayüz altında karşılaştırılabilir hale getirmektir.

    WordPiece yaklaşımının temel fikri:
        - Metin önce temel kelime/noktalama parçalarına ayrılır.
        - Eğitim sırasında kelimelerden olası alt-parça adayları üretilir.
        - En sık görülen adaylar vocabulary içine alınır.
        - Tokenization sırasında her kelime greedy longest-match stratejisiyle en uzun uygun alt parçalara bölünür.

    WordPiece formatında:
        - Kelimenin başında yer alan parça normal yazılır.
        - Kelimenin devamı olan parçalar "##" prefix'i ile gösterilir.

    Örnek:
        "tokenization" -> ["token", "##ization"]

    Bu sayede tokenizer:
        - Tam kelime eşleşmelerini destekler.
        - Bilinmeyen kelimeleri daha küçük alt parçalara ayırabilir.
        - Vocabulary boyutunu kontrol altında tutar.
        - Word-level tokenizer'a göre daha esnek temsil sağlar.

    Not:
        Bu implementasyon BERT'in gerçek WordPiece eğitim algoritmasının birebir
        kopyası değildir. Burada frekans temelli sade bir aday seçimi yapılır.
        Projenin amacı açısından bu yaklaşım; test edilebilir, okunabilir ve
        karşılaştırma pipeline'ına kolay entegre edilebilir bir WordPiece modeli sunar.
    """

    # [UNK] token'ı, vocabulary içinde bulunmayan kelimeler veya alt parçalar için kullanılır.
    UNKNOWN_TOKEN = "[UNK]"

    def __init__(
        self,
        vocab_size: int = 100,
        max_subword_length: int = 12,
    ) -> None:
        super().__init__(name="wordpiece")

        # Vocabulary boyutu en az 2 olmalıdır çünkü [UNK] token'ı her zaman eklenir.
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")

        # Çok uzun alt parçaların vocabulary'yi gereksiz büyütmesini engellemek için bu parametre kullanılır.
        if max_subword_length < 1:
            raise ValueError("max_subword_length must be at least 1")

        # Vocabulary boyutu ve alt parça uzunluğu sınırlamaları eğitim sırasında dikkate alınır.
        self.target_vocab_size = vocab_size 
        self.max_subword_length = max_subword_length 

        # Tokenizer'ın eğitim sürecinde oluşturulan vocabulary'yi tutacak yapılar.
        self._token_to_id: dict[str, int] = {} # Token-string'den numeric id'ye eşleme sözlüğü. 
        self._id_to_token: dict[int, str] = {} # Numeric id'den token-string'e eşleme sözlüğü. 
        self._is_trained = False # Tokenizer'ın eğitim durumunu takip eder. Eğitim tamamlanmadan encode/decode işlemleri yapılamaz.

    def train(self, text: str) -> None:
        """
        Verilen eğitim metni üzerinden WordPiece vocabulary oluşturur.

        Eğitim akışı:
            1. Metin boşluk, kelime ve noktalama seviyesinde normalize edilerek tokenlara ayrılır.
            2. Her kelime için olası WordPiece adayları üretilir.
            3. Aday parçaların frekansları hesaplanır.
            4. En sık görülen adaylar hedef vocabulary boyutuna göre seçilir.
            5. Vocabulary içine her zaman [UNK] token'ı eklenir.

        [UNK] token'ı:
            Eğitim sonrası tokenize/encode sırasında vocabulary içinde karşılığı bulunamayan
            kelimeler veya alt parçalar için kullanılır.

        Raises:
            ValueError:
                Eğitim metni boş, None ya da yalnızca whitespace karakterlerinden oluşuyorsa.
        """

        # Boş veya yalnızca whitespace içeren metinler eğitim için uygun değildir.
        if not text or not text.strip():
            raise ValueError("Training text cannot be empty")

        # Metni normalize ederek kelime ve noktalama tokenlarına ayırır.
        words = self._basic_tokenize(text)

        # Tokenizer'ın eğitim sürecinde kelimelerden WordPiece adayları üretilir ve frekansları hesaplanır.
        if not words:
            raise ValueError("Training text cannot be empty")

        # Kelimelerden üretilen WordPiece adaylarının frekanslarını saymak için Counter kullanılır.
        # Kelimelerden üretilen adaylar, vocabulary oluşturulurken en sık görülenler seçilir.
        # Counter, kelime parçalarının ne kadar yaygın olduğunu belirlemek için kullanılır.
        # Aday parçaların frekanslarına göre sıralama yapılır ve hedef vocabulary boyutuna göre en sık görülenler seçilir.
        # Bu sayede, eğitim metninde sıkça görülen kelime parçaları vocabulary içine alınırken, nadir parçalar elenerek vocabulary boyutu kontrol altında tutulur.
        # [UNK] token'ı her zaman vocabulary içinde yer alır, bu nedenle hedef vocabulary boyutundan 1 çıkarılır.
        candidate_counter: Counter[str] = Counter() 

        # Kelimelerden WordPiece adayları üretilir ve frekansları sayılır.
        for word in words:
            candidate_counter.update(self._generate_wordpiece_candidates(word))

        # En sık görülen adaylar hedef vocabulary boyutuna göre seçilir. 
        # [UNK] token'ı her zaman eklenir, bu nedenle hedef boyuttan 1 çıkarılır.
        most_common_tokens = [
            token # En sık görülen token'lar, vocabulary oluşturulurken seçilir.
            for token, _ in candidate_counter.most_common(
                max(0, self.target_vocab_size - 1)
            ) 
        ]

        # Vocabulary oluşturulurken [UNK] token'ı her zaman eklenir, 
        # ardından en sık görülen aday token'lar eklenir.
        vocab = [self.UNKNOWN_TOKEN, *most_common_tokens]

        # Tokenizer'ın eğitim sürecinde oluşturulan vocabulary'yi tutacak yapılar doldurulur.
        self._token_to_id = {
            token: index
            for index, token in enumerate(vocab)
        }

        # Tokenizer'ın eğitim sürecinde oluşturulan vocabulary'yi tutacak yapılar doldurulur.
        self._id_to_token = {
            index: token
            for token, index in self._token_to_id.items()
        }

        self._is_trained = True

    def encode(self, text: str) -> list[int]:
        """
        Metni WordPiece token id listesine dönüştürür.

        Bu method önce metni tokenize eder, ardından her token'ı vocabulary içindeki
        numeric id karşılığına çevirir. Vocabulary içinde bulunmayan tokenlar [UNK]
        token id'si ile temsil edilir.

        Not:
            encode işlemi için tokenizer'ın önceden train edilmiş olması gerekir.
            Çünkü token-id eşleşmeleri eğitim sırasında oluşturulan vocabulary üzerinden yapılır.

        Raises:
            ValueError:
                Tokenizer henüz train edilmeden encode çağrılırsa.
        """
        #encode işlemi için tokenizer'ın önceden train edilmiş olması gerekir. 
        # Eğitim tamamlanmadan encode/decode işlemleri yapılamaz.
        if not self._is_trained:
            raise ValueError("Tokenizer has not been trained yet")

        # Metni tokenize eder. 
        # Tokenizer'ın eğitim durumuna göre basic tokenization veya WordPiece tokenization uygulanır.
        tokens = self.tokenize(text)

        return [
            self._token_to_id.get(token, self._token_to_id[self.UNKNOWN_TOKEN])
            for token in tokens
        ] # Her token'ı vocabulary içindeki numeric id karşılığına çevirir.
        # Vocabulary içinde bulunmayan tokenlar [UNK] token id'si ile temsil edilir

    def decode(self, token_ids: list[int]) -> str:
        """
        Token id listesini tekrar okunabilir metin formatına dönüştürür.

        Decode sırasında:
            - Her id vocabulary üzerinden string token'a çevrilir.
            - "##" ile başlayan parçalar önceki token'ın devamı olarak birleştirilir.
            - [UNK] tokenları korunur.
            - Kelime başı tokenları boşluk ile ayrılır.

        Örnek:
            ["token", "##ization"] -> "tokenization"

        Not:
            WordPiece decode işlemi her zaman orijinal metni birebir geri üretmeyebilir.
            Çünkü temel tokenization aşamasında lowercase uygulanır ve bazı spacing bilgileri
            korunmaz. Bu methodun amacı daha çok okunabilir bir rekonstrüksiyon üretmektir.

        Raises:
            ValueError:
                Tokenizer henüz train edilmeden decode çağrılırsa.
                Vocabulary içinde bulunmayan bir token id ile karşılaşılırsa.
        """
        # Decode işlemi için tokenizer'ın önceden train edilmiş olması gerekir. 
        # Eğitim tamamlanmadan encode/decode işlemleri yapılamaz.
        if not self._is_trained:
            raise ValueError("Tokenizer has not been trained yet")

        pieces: list[str] = [] # Decode sırasında oluşturulacak token parçalarını tutar. 
        # "##" ile başlayan parçalar önceki token'ın devamı olarak birleştirilir, 
        # kelime başı tokenları ise boşluk ile ayrılır.

        # Her id vocabulary üzerinden string token'a çevrilir.
        for token_id in token_ids:
            # Vocabulary içinde bulunmayan bir token id ile karşılaşılırsa ValueError fırlatılır.
            if token_id not in self._id_to_token:
                raise ValueError(f"Unknown token id encountered: {token_id}")

            token = self._id_to_token[token_id]

            # "##" ile başlayan parçalar önceki token'ın devamı olarak birleştirilir,
            # kelime başı tokenları ise boşluk ile ayrılır.
            if token == self.UNKNOWN_TOKEN:
                # [UNK] tokenları korunur.
                pieces.append(self.UNKNOWN_TOKEN)
            elif token.startswith("##"):
                if pieces:
                    pieces[-1] = pieces[-1] + token[2:]
                else:
                    pieces.append(token[2:])
            else:
                pieces.append(token)

        return " ".join(pieces)

    def tokenize(self, text: str) -> list[str]:
        """
        Metni WordPiece string token listesine dönüştürür.

        Davranış:
            - Boş veya yalnızca whitespace içeren metinlerde boş liste döndürür.
            - Tokenizer henüz train edilmemişse basic tokenization sonucu döndürülür.
            - Train edilmişse her kelime greedy WordPiece algoritmasıyla alt parçalara ayrılır.

        Greedy yaklaşım:
            Her kelime için mümkün olan en uzun vocabulary eşleşmesi aranır.
            Eşleşme bulunursa token listeye eklenir ve kelimenin kalan kısmı işlenir.
            Hiçbir uygun parça bulunamazsa kelime [UNK] olarak temsil edilir.

        Returns:
            WordPiece token string listesi.
        """

        # Boş veya yalnızca whitespace içeren metinlerde boş liste döndürür.
        if not text or not text.strip():
            return []

        # Tokenizer henüz train edilmemişse basic tokenization sonucu döndürülür.
        words = self._basic_tokenize(text)

        # Train edilmişse her kelime greedy WordPiece algoritmasıyla alt parçalara ayrılır.
        if not self._is_trained:
            return words

        # Greedy yaklaşım: Her kelime için mümkün olan en uzun vocabulary eşleşmesi aranır.
        output_tokens: list[str] = []

        # Her kelime için greedy WordPiece algoritmasıyla alt parçalara ayrılır.
        for word in words:
            output_tokens.extend(self._greedy_wordpiece_tokenize(word))

        return output_tokens

    @property
    def vocab_size(self) -> int:
        """
        Mevcut vocabulary boyutunu döndürür.
        """
        return len(self._token_to_id)

    def _basic_tokenize(self, text: str) -> list[str]:
        """
        Metni basit şekilde kelime/noktalama tokenlarına ayırır.
        """
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def _generate_wordpiece_candidates(self, word: str) -> list[str]:
        """
        Tek bir kelimeden WordPiece vocabulary adayları üretir.

        Kelimenin başından başlayan parçalar normal token olarak eklenir.
        Kelimenin içinden başlayan parçalar ise "##" prefix'i ile eklenir.
        Bu prefix, parçanın bağımsız bir kelime başlangıcı değil, önceki parçanın devamı
        olduğunu belirtir.

        Örnek:
            word = "token"

            Üretilebilecek adaylardan bazıları:
                "t", "to", "tok", "toke", "token"
                "##o", "##ok", "##oke", "##oken"
                "##k", "##ke", "##ken"

        max_subword_length:
            Çok uzun alt-parçaların vocabulary'yi gereksiz büyütmesini engellemek için
            aday uzunluğu bu değerle sınırlandırılır.
        """
        candidates: list[str] = []

        # Kelimenin başından başlayan parçalar normal token olarak eklenir.
        # Kelimenin içinden başlayan parçalar ise "##" prefix'i ile eklenir.
        for start in range(len(word)):
            max_end = min(len(word), start + self.max_subword_length)

            for end in range(start + 1, max_end + 1):
                piece = word[start:end]

                # Bu prefix, parçanın bağımsız bir kelime başlangıcı değil, önceki parçanın devamı olduğunu belirtir.
                if start == 0:
                    candidates.append(piece)
                else:
                    candidates.append(f"##{piece}")

        return candidates

    def _greedy_wordpiece_tokenize(self, word: str) -> list[str]:
        """
        Tek bir kelimeyi vocabulary kullanarak greedy longest-match stratejisiyle böler.

        Algoritma:
            1. Eğer kelimenin tamamı vocabulary içinde varsa doğrudan tek token döndürülür.
            2. Aksi halde kelimenin başından başlanır.
            3. Mevcut pozisyondan itibaren mümkün olan en uzun parça denenir.
            4. Parça kelime başında değilse "##" prefix'i ile aranır.
            5. Vocabulary içinde eşleşme bulunursa token listeye eklenir.
            6. Hiçbir eşleşme bulunamazsa kelime [UNK] olarak temsil edilir.

        Bu yaklaşım WordPiece tokenization için kritik öneme sahiptir çünkü kısa parçalar
        yerine mümkün olan en uzun anlamlı parçayı seçerek daha kompakt token dizileri üretir.

        Returns:
            Kelimeye karşılık gelen WordPiece token listesi.
            Eğer kelime parçalanamıyorsa ["[UNK]"] döndürülür.
        """
        if word in self._token_to_id:
            return [word]

        tokens: list[str] = [] 
        start = 0

        # Kelimenin başından başlanır. Mevcut pozisyondan itibaren mümkün olan en uzun parça denenir.
        while start < len(word):
            # Mevcut pozisyondan itibaren mümkün olan en uzun parça denenir. 
            # Parça kelime başında değilse "##" prefix'i ile aranır.
            end = min(len(word), start + self.max_subword_length)
            matched_token: str | None = None

            #  Mevcut pozisyondan itibaren mümkün olan en uzun parça denenir. 
            # Parça kelime başında değilse "##" prefix'i ile aranır.
            while end > start:
                piece = word[start:end]

                if start > 0:
                    piece = f"##{piece}"

                if piece in self._token_to_id:
                    matched_token = piece
                    break

                end -= 1

            # Hiçbir eşleşme bulunamazsa kelime [UNK] olarak temsil edilir.
            if matched_token is None:
                return [self.UNKNOWN_TOKEN]

            tokens.append(matched_token)

            # Parça kelime başında değilse "##" prefix'i ile aranır. 
            # Eğer eşleşen token "##" ile başlıyorsa, start pozisyonu end'e kaydırılır.
            if matched_token.startswith("##"):
                start = end
            else:
                start = end

        return tokens