from __future__ import annotations

"""
compare.py

Tokenizer Workshop CLI entry point.

Bu dosya intentionally thin tutulur:
- CLI akışı burada yönetilmez
- Comparison pipeline burada çalıştırılmaz
- Sadece bağımlılıklar oluşturulur ve CLIController başlatılır

Asıl sorumluluklar:
    cli/controller.py  → kullanıcı akışı
    cli/runner.py      → comparison execution
    cli/menu.py        → terminal çıktıları
    cli/input.py       → input parsing
"""

from typing import Any

from tokenizer_workshop.cli import CLIComparisonRunner, CLIController
from tokenizer_workshop.comparisons.compare_manager import CompareManager


# ============================================================
# CONSTANTS (SABİT VERİLER)
# ============================================================

# Eğitim için kullanılacak metin
# Özellikle BPE gibi tokenizer'lar için train() gereklidir
TRAIN_TEXT = """
Hello world! Tokenization is fun.
Tokenization helps language models process text.
Byte pair encoding can merge frequent byte patterns.
"""

# Karşılaştırma için kullanılacak kısa test metni
COMPARE_TEXT = "Hello world! Tokenization is fun."

# Raporun kaydedileceği dosya yolu
REPORT_PATH = "report.md"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

TOKENIZER_CONFIG: dict[str, dict[str, Any]] = {
    "word": {},
    "char": {},
    "byte": {},
    "byte_bpe": {"num_merges": 10},
    "simple_bpe": {"num_merges": 10},
    "regex": {},
    "regex_bpe": {},
    "ngram": {"n": 2},
    "wordpiece": {"vocab_size": 100},
    "unigram": {"vocab_size": 100},
    "sentencepiece": {"vocab_size": 100},
    "white_space": {},
    "punctuation": {},
    "subword": {"subword_size": 3},
    "morpheme": {},
    "byte_level_bpe": {"num_merges": 10},
    "pretrained": {"model_name": "gpt2"},
}


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:
    """
    Uygulamanın giriş noktası (entry point).

    Bu fonksiyon, tokenizer evaluation pipeline'ını baştan sona çalıştırır.

    ------------------------------------------------------------
    Pipeline adımları:
    ------------------------------------------------------------

    1. CompareManager oluşturulur
    2. Tokenizer nesneleri hazırlanır
    3. Gerekli tokenizer'lar train edilir
    4. Tokenizer'lar aynı metin üzerinde karşılaştırılır
    5. Sonuç terminale yazdırılır

    ------------------------------------------------------------
    Tasarım kararı:
    ------------------------------------------------------------

    Bu dosya özellikle "ince" tutulmuştur (thin entry point).

    Yani:
    - İş mantığı burada değil
    - CompareManager içinde

    Bu yaklaşım sayesinde:
    - test yazmak kolaylaşır
    - kod tekrar kullanılabilir olur
    - mimari daha temiz olur
    """

    # ============================================================
    # 1. MANAGER OLUŞTUR
    # ============================================================
    # CompareManager, tüm karşılaştırma sürecini yöneten ana sınıftır.
    # Bu sınıfın görevi:
    # - tokenize işlemini yönetmek
    # - evaluation sonuçlarını toplamak
    # - tokenizer'lar arasındaki karşılaştırmaları yapmak
    # - sonuçları raporlamak
    # CompareManager, karşılaştırma sürecinin merkezi kontrol noktasıdır.
    # Bu sınıfın içinde tüm karşılaştırma mantığı yer alır.
    # Böylece main() fonksiyonu sadece bu sınıfı kullanarak süreci başlatır ve sonuçları yazdırır.
    # Bu tasarım, kodun daha modüler, test edilebilir ve genişletilebilir olmasını sağlar.
    manager = CompareManager()

    # ============================================================
    # 2. TOKENIZER OLUŞTUR
    # ============================================================
    # Tokenizer'lar, metni token'lara bölen sınıflardır.
    # Her tokenizer'ın kendine özgü bir tokenize() metodu vardır.
    # Tokenizer'lar, farklı tokenization stratejileri uygularlar:
        # - WordTokenizer: Metni kelimelere böler
        # - CharTokenizer: Metni karakterlere böler
        # - ByteTokenizer: Metni byte'lara böler
        # - ByteLevelBPETokenizer: Byte'lara dayalı BPE tokenization yapar
        # - SimpleBPETokenizer: Basit BPE tokenization yapar
        # - RegexTokenizer: Regex desenlerine göre token'lara böler
        # - RegexBPETokenizer: Regex tabanlı BPE tokenization yapar
        # - NGramTokenizer: N-gram tokenization yapar
        # - WordPieceTokenizer: WordPiece tokenization yapar
        # - UnigramTokenizer: Unigram tokenization yapar
        # - SentencePieceTokenizer: SentencePiece tokenization yapar
        # - WhiteSpaceTokenizer: Metni boşluklara göre böler
        # - PunctuationTokenizer: Noktalama işaretlerini ayrı token yapar
        # - SubwordTokenizer: Subword tokenization yapar
        # - MorphemeTokenizer: Morfolojik analiz yapar
        # - PretrainedTokenizer: Önceden eğitilmiş bir tokenizer'ı kullanır
    # Tokenizer'lar, farklı tokenization stratejileri uygulayarak metni token'lara bölerler.
    # Bu tokenizer'lar, karşılaştırma sürecinde farklı tokenization sonuçları üreterek birbirleriyle karşılaştırılırlar.
    # Bu tokenizer'lar, CompareManager tarafından yönetilir ve karşılaştırma sürecinde kullanılırlar.
    runner = CLIComparisonRunner(
        manager=manager, # CompareManager instance'ını runner'a verilir
        train_text=TRAIN_TEXT, # Eğitim metni runner'a verilir
        report_path=REPORT_PATH, # Raporun kaydedileceği dosya yolu runner'a verilir
    )

    # ============================================================
    # 3. CONTROLLER OLUŞTUR VE ÇALIŞTIR
    # ============================================================
    # CLIController, kullanıcı arayüzünü yöneten sınıftır.
    # Bu sınıfın görevi:
    # - Kullanıcıdan input almak
    # - Runner'ı tetiklemek
    # - Sonuçları terminale yazdırmak
    # CLIController, kullanıcı etkileşimlerini yönetir ve runner'ı kullanarak karşılaştırma sürecini başlatır.
    # CLIController, kullanıcı arayüzü ile karşılaştırma mantığını birbirinden ayırır.
    # Bu tasarım, kodun daha modüler, test edilebilir ve genişletilebilir olmasını sağlar.
    ontroller = CLIController(
        runner=runner, # CLIComparisonRunner instance'ını controller'a verilir
        tokenizer_config=TOKENIZER_CONFIG, # Tokenizer konfigürasyonu controller'a verilir
        default_compare_text=COMPARE_TEXT, # Karşılaştırma için kullanılacak metin controller'a verilir
    )

    controller.run() # CLIController'un run() metodu çağrılarak kullanıcı arayüzü başlatılır ve karşılaştırma süreci tetiklenir


# ============================================================
# ENTRY POINT CHECK
# ============================================================

# Bu blok sayesinde:
# Dosya doğrudan çalıştırıldığında main() tetiklenir
# Import edildiğinde otomatik çalışmaz
if __name__ == "__main__":
    main()


# ============================================================
# RUN KOMUTU
# ============================================================

# Terminalden çalıştırmak için:
# uv run python -m tokenizer_workshop.compare