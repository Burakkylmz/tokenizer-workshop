from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MergeStep:
    """
    Öğrenilmiş tek bir BPE merge rule'unu temsil eder.

    Example:
        pair=("a", "b"), merged_token="ab", frequency=5

    Educational note:
    Burada frequency bilgisini tutuyoruz çünkü learners'ın
    training sırasında neden belirli bir merge'in seçildiğini incelemesini kolaylaştırır.
    """

    pair: tuple[str, str]
    merged_token: str
    frequency: int


class BPETrainer:
    """
    Basit bir character-level BPE trainer.

    Educational purpose:
    - Byte Pair Encoding tarzı merging mantığının nasıl çalıştığını öğretir.
    - "most frequent adjacent pair" fikrini görünür hale getirir.
    - training logic ile tokenizer encode/decode logic'ini birbirinden ayırır.

    Important note:
    Bu ilk versiyon byte token'lar yerine character token'lardan başlar.
    Bu tercih, daha gerçekçi byte-level BPE tasarımlarına geçmeden önce
    algorithm'i anlamayı kolaylaştırır.
    """

    def __init__(self) -> None:
        self.merge_steps: list[MergeStep] = []

    def train(self, text: str, num_merges: int) -> list[MergeStep]:
        """
        raw text üzerinden BPE merge rules öğrenir.

        Args:
            text: raw text olarak training corpus.
            num_merges: Öğrenilecek maksimum merge operation sayısı.

        Returns:
            Öğrenildikleri kesin sırayla bir MergeStep listesi döndürür.

        Why order matters:
        BPE merge'ler sadece bir rules set'i değildir. Öğrenilmiş bir sırayla uygulanırlar.
        Bu sıra final tokenization sonucunu doğrudan etkiler.
        """
        if not text:
            raise ValueError("Training text cannot be empty.")

        if num_merges < 1:
            raise ValueError("num_merges must be at least 1.")

        # En temel symbol sequence ile başlıyoruz: her character bir token.
        tokens = list(text)
        learned_merges: list[MergeStep] = []

        for _ in range(num_merges):
            pair_stats = self.get_pair_stats(tokens)

            # Eğer geriye adjacent pair kalmadıysa, daha fazla merging yapmak mümkün değildir.
            if not pair_stats:
                break

            # En sık geçen adjacent pair'i seçiyoruz.
            #
            # Tie-breaker:
            # Önce frequency'ye, sonra pair'in lexicographic sırasına bakıyoruz.
            # Bu sayede training deterministic olur ve yeniden üretmek kolaylaşır.
            best_pair = min(
                pair_stats.items(),
                key=lambda item: (-item[1], item[0]),
            )[0]
            best_frequency = pair_stats[best_pair]

            merged_token = "".join(best_pair)

            learned_merges.append(
                MergeStep(
                    pair=best_pair,
                    merged_token=merged_token,
                    frequency=best_frequency,
                )
            )

            tokens = self.merge_pair(tokens, best_pair, merged_token)

        self.merge_steps = learned_merges
        return learned_merges

    @staticmethod
    def get_pair_stats(tokens: list[str]) -> dict[tuple[str, str], int]:
        """
        adjacent token pair'lerini sayar.

        Example:
            tokens = ["a", "b", "a", "b"]

            adjacent pairs:
            ("a", "b"), ("b", "a"), ("a", "b")

            result:
            {
                ("a", "b"): 2,
                ("b", "a"): 1,
            }

        Bu function, BPE training'in çekirdeğidir:
        hangi local pattern'in en sık göründüğünü bize söyler.
        """
        stats: dict[tuple[str, str], int] = {}

        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            stats[pair] = stats.get(pair, 0) + 1

        return stats

    @staticmethod
    def merge_pair(
        tokens: list[str],
        pair_to_merge: tuple[str, str],
        merged_token: str,
    ) -> list[str]:
        """
        Eşleşen her adjacent pair'i merged token ile değiştirir.

        Example:
            tokens = ["a", "b", "a", "b"]
            pair_to_merge = ("a", "b")
            merged_token = "ab"

            result -> ["ab", "ab"]

        Important detail:
        Soldan sağa tarıyoruz ve greedy merge uyguluyoruz.
        Bir pair merge edildikten sonra overlapping merge'leri önlemek için onun üzerinden atlıyoruz.
        """
        if len(pair_to_merge) != 2:
            raise ValueError("pair_to_merge must contain exactly two tokens.")

        merged_tokens: list[str] = []
        i = 0

        while i < len(tokens):
            # Mevcut pozisyon ile bir sonraki pozisyonun
            # target pair'i oluşturup oluşturmadığını kontrol ediyoruz.
            if (
                i < len(tokens) - 1
                and tokens[i] == pair_to_merge[0]
                and tokens[i + 1] == pair_to_merge[1]
            ):
                merged_tokens.append(merged_token)
                i += 2
            else:
                merged_tokens.append(tokens[i])
                i += 1

        return merged_tokens