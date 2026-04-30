from __future__ import annotations

"""
compare.py

Uygulamanın CLI entry point'i.

Bu dosya intentionally thin tutulur:
- İş mantığı burada olmaz
- Orchestration burada olur
- Asıl logic CompareManager + services katmanındadır

Yeni mimari:
    TokenizerFactory → Registry → Discovery → Tokenizer
"""

from tokenizer_workshop.comparisons.compare_manager import CompareManager

from tokenizer_workshop.tokenizers.base import BaseTokenizer
from tokenizer_workshop.api.services.tokenizer_factory import TokenizerFactory


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


# ============================================================
# HELPER FUNCTIONS
# ============================================================

TOKENIZER_CONFIG = {
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
}


def build_tokenizers() -> dict[str, BaseTokenizer]:
    """
    Karşılaştırmada kullanılacak tokenizer instance'larını üretir.

    Yeni tokenizer eklemek için sadece TOKENIZER_CONFIG'e ekleme yapmak yeterlidir.
    """

    return {
        name: TokenizerFactory.create(name, **config)
        for name, config in TOKENIZER_CONFIG.items()
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
    # 2. TOKENIZER'LARI OLUŞTUR
    # ============================================================
    # build_tokenizers fonksiyonu, karşılaştırmada kullanılacak tokenizer nesnelerini oluşturur.
    # Bu fonksiyonun amacı:
    # - tokenizer oluşturma logic'ini main() içinden çıkarmak
    # - kodu daha modüler hale getirmek
    # - ileride yeni tokenizer eklemeyi kolaylaştırmak
    # Bu fonksiyon, her tokenizer için bir nesne oluşturur ve bunları bir sözlük (dict) içinde döner.
    # Böylece main() fonksiyonu sadece bu sözlüğü alır ve karşılaştırma sürecine dahil eder.
    # Bu tasarım, tokenizer oluşturma logic'ini tek bir yerde toplar ve main() fonksiyonunu daha temiz tutar.
    # Ayrıca yeni tokenizer eklemek istediğimizde sadece bu fonksiyonu güncelleyerek kolayca yapabiliriz.
    # Örneğin:
    # def build_tokenizers():
    #     return {
    #         "word": WordTokenizer(),
    #         "byte_bpe": ByteBPETokenizer(num_merges=10),
    #         "char": CharTokenizer(),
    #     }
    # Böylece main() fonksiyonu değişmeden yeni tokenizer'lar eklenebilir.
    # Bu fonksiyon, karşılaştırmada kullanılacak tokenizer'ların merkezi oluşturulma noktasıdır.
    # Bu sayede kodun organizasyonu daha iyi olur ve genişletilebilirlik artar.
    tokenizers = build_tokenizers()

    # ============================================================
    # 3. TOKENIZER'LARI EĞİT
    # ============================================================
    # Not:
    # Tüm tokenizer'lar train() desteklemeyebilir.
    # Bu yüzden CompareManager içindeki train_tokenizers kullanılır.
    # Bu metod, verilen tokenizer'lar arasında train() destekleyenleri tespit eder ve sadece onlara eğitim metniyle train() çağırır.
    # Böylece main() fonksiyonu, hangi tokenizer'ın train() desteklediği konusunda endişelenmeden sadece tüm tokenizer'ları train_tokenizers metoduna verir.
    # CompareManager içindeki train_tokenizers metodu, her tokenizer'ı kontrol eder:
    # for name, tokenizer in tokenizers.items():
    #     if hasattr(tokenizer, "train") and callable(getattr(tokenizer, "train")):
    #         tokenizer.train(train_text)
    # Bu sayede main() fonksiyonu sadece tokenizers sözlüğünü verir ve gerisini CompareManager halleder.
    manager.train_tokenizers(
        tokenizers=tokenizers,
        train_text=TRAIN_TEXT,
    )

    # ============================================================
    # 4. KARŞILAŞTIRMA
    # ============================================================
    # compare_multiple:
    # artık birden fazla tokenizer destekler
    # Bu metod, verilen metin üzerinde tüm tokenizer'ları çalıştırır ve 
    # sonuçları tek bir ComparisonResult içinde döner.
    # Bu sayede main() fonksiyonu sadece bu tek metodu çağırarak tüm karşılaştırma sürecini başlatır.
    # Böylece main() fonksiyonu çok ince (thin) kalır ve tüm mantık CompareManager içinde yer alır.
    result = manager.compare_multiple(
        text=COMPARE_TEXT,
        tokenizers=tokenizers,
    )

    # ============================================================
    # 5. SONUCU YAZDIR
    # ============================================================
    # print_comparison_result:
    # karşılaştırma sonucunu ekrana yazdırır
    manager.print_comparison_result(result, save_path="report.md")


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