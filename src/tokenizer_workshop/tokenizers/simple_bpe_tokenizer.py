from __future__ import annotations

from tokenizer_workshop.tokenizers.base import BaseTokenizer
from tokenizer_workshop.trainers import BPETrainer, MergeStep


class SimpleBPETokenizer(BaseTokenizer):
    """
    Basit bir character-based BPE tokenizer.

    Educational purpose:
    - BPE training ile gerçek tokenization davranışı arasında bağlantı kurar.
    - Öğrenilen merge rules'un encoding sırasında nasıl uygulandığını gösterir.
    - BPE tokenization'ın şu iki şeye birden bağlı olduğunu görünür kılar:
        1) öğrenilmiş merge rules
        2) bu rules'un hangi sırayla öğrenildiği

    Important scope note:
    Bu sınıf bilinçli olarak basit bir character-based BPE tokenizer olarak tasarlanmıştır.
    Production-grade byte-level BPE sistemlerini birebir taklit etmeyi hedeflemez.
    """

    def __init__(self, num_merges: int = 10) -> None:
        super().__init__(name="simple_bpe")

        if num_merges < 1:
            raise ValueError("num_merges must be at least 1.")

        self.num_merges = num_merges
        self.trainer = BPETrainer()

        # Training sırasında öğrenilen merge steps'leri, öğrenildikleri sırayla tutar.
        self.merge_steps: list[MergeStep] = []

        # Final learned vocabulary için token <-> id mapping'leri.
        self._stoi: dict[str, int] = {}
        self._itos: dict[int, str] = {}

    def train(self, text: str) -> None:
        """
        Tokenizer'ı raw text üzerinde train eder.

        Training flow:
        1. BPETrainer ile merge rules öğrenilir.
        2. Final vocabulary şu iki kaynaktan oluşturulur:
           - base character tokens
           - training sırasında öğrenilen merged tokens

        Neden hem base characters hem de merged tokens tutuluyor?
        Çünkü encoding sırasında bazı symbol'ler merge edilmeden kalabilir.
        Bu yüzden tokenizer, raw characters'ı da id'lere map edebilmeye devam etmelidir.
        """
        if not text:
            raise ValueError("Training text cannot be empty.")

        self.merge_steps = self.trainer.train(text, num_merges=self.num_merges)

        # Base vocabulary, training sırasında görülen benzersiz characters'dan başlar.
        base_tokens = sorted(set(text))

        # Merged tokens, öğrenildikleri sırayla vocabulary'ye eklenir.
        merged_tokens = [step.merged_token for step in self.merge_steps]

        # Duplicate token'ları engelliyoruz ama order'ı koruyoruz.
        vocab_tokens: list[str] = []
        for token in base_tokens + merged_tokens:
            if token not in vocab_tokens:
                vocab_tokens.append(token)

        self._stoi = {token: idx for idx, token in enumerate(vocab_tokens)}
        self._itos = {idx: token for token, idx in self._stoi.items()}

    def encode(self, text: str) -> list[int]:
        """
        Text'i, öğrenilmiş BPE merge rules kullanarak token ids'e encode eder.

        Encoding flow:
        1. Character tokens ile başlar.
        2. Öğrenilen her merge step sırayla uygulanır.
        3. Final tokens integer ids'e dönüştürülür.

        Important teaching point:
        BPE sadece "bir şekilde common chunks bulmak" değildir.
        Kesin merge order önemlidir ve final segmentation sonucunu etkiler.
        """
        if not self._stoi:
            raise ValueError("Tokenizer has not been trained yet.")

        # Raw character sequence ile başlıyoruz.
        tokens = list(text)

        # Merge rules, öğrenildikleri sırayla birebir uygulanır.
        for step in self.merge_steps:
            tokens = self.trainer.merge_pair(tokens, step.pair, step.merged_token)

        token_ids: list[int] = []

        for token in tokens:
            if token not in self._stoi:
                raise ValueError(f"Unknown token encountered during encoding: {token!r}")
            token_ids.append(self._stoi[token])

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
        Token ids'i tekrar text'e decode eder.

        Bu tokenizer'da tokens string olduğu için (character veya merged chunks),
        decoding işlemi temelde bu string parçalarını birleştirmekten ibarettir.
        """
        if not self._itos:
            raise ValueError("Tokenizer has not been trained yet.")

        tokens: list[str] = []

        for token_id in token_ids:
            if token_id not in self._itos:
                raise ValueError(
                    f"Unknown token id encountered during decoding: {token_id}"
                )
            tokens.append(self._itos[token_id])

        return "".join(tokens)

    @property
    def vocab_size(self) -> int:
        """
        Tokenizer'ın şu anda bildiği token sayısını döndürür.

        Buna şunlar dahildir:
        - original character tokens
        - training sırasında öğrenilen merged BPE tokens
        """
        return len(self._stoi)