from __future__ import annotations

# runner modülü:
# tüm tokenizer'ları alır ve sample text'ler üzerinde çalıştırır
from tokenizer_workshop.comparisons.runner import run_all_samples_across_tokenizers

# report modülü:
# elde edilen sonuçları ekrana düzgün formatta basar
from tokenizer_workshop.comparisons.report import print_all_sample_results

# protocol:
# tokenize davranışını sağlayan tüm sınıflar için ortak tip tanımı sunar
from tokenizer_workshop.comparisons.protocols import Tokenizer

# model:
# iki tokenizer arasındaki karşılaştırma sonucunu taşıyan veri modelidir
from tokenizer_workshop.comparisons.comparison_models import ComparisonResult


class CompareManager:
    """
    CompareManager sınıfı, tokenizer karşılaştırma sürecini yönetir.

    Bu sınıf iki farklı kullanım senaryosunu destekler:
    1. Mevcut runner/report akışı ile tüm tokenizer'ları toplu çalıştırmak
    2. Aynı metin üzerinde iki tokenizer'ı özel olarak karşılaştırmak

    Böylece sistem hem mevcut yapıyı korur
    hem de daha modüler ve genişletilebilir hale gelir.
     """

    def __init__(self):
        """
        Constructor (başlatıcı fonksiyon)

        results:
        - tokenizer'ların çalıştırılması sonucu elde edilen çıktıları tutar
        - başlangıçta None olarak atanır (henüz çalıştırılmadı)
        """
        self.results = None

    def compare(
        self,
        text: str,
        tokenizer_a: Tokenizer,
        tokenizer_b: Tokenizer,
    ) -> ComparisonResult:
        """
        Aynı metni iki farklı tokenizer ile tokenize eder
         ve sonucu yapılandırılmış bir ComparisonResult nesnesi olarak döndürür.

        Args:
            text (str):
                Karşılaştırılacak metin.

            tokenizer_a:
                İlk tokenizer nesnesi.
                Örn: WordTokenizer()

            tokenizer_b:
                İkinci tokenizer nesnesi.
                Örn: ByteBPETokenizer()

        Returns:
            ComparisonResult:
                Karşılaştırma sonucunu içeren veri modeli.
        """

        tokens_a = tokenizer_a.tokenize(text) # tokenizer_a ile metni tokenize eder
        tokens_b = tokenizer_b.tokenize(text) # tokenizer_b ile metni tokenize eder

        set_a = set(tokens_a) # tokenizer_a'nın ürettiği token'ları set'e dönüştürür
        set_b = set(tokens_b) # tokenizer_b'nin ürettiği token'ları set'e dönüştürür

        unique_to_a = sorted(set_a - set_b) # sadece tokenizer_a'ya özgü token'ları bulur ve sıralar
        unique_to_b = sorted(set_b - set_a) # sadece tokenizer_b'ye özgü token'ları bulur ve sıralar
        common_tokens = sorted(set_a & set_b) # her iki tokenizer'da ortak olan token'ları bulur ve sıralar

        return ComparisonResult(
            text=text,
            tokenizer_a_name=tokenizer_a.__class__.__name__,
            tokenizer_b_name=tokenizer_b.__class__.__name__,
            tokens_a=tokens_a,
            tokens_b=tokens_b,
            token_count_a=len(tokens_a),
            token_count_b=len(tokens_b),
            unique_to_a=unique_to_a,
            unique_to_b=unique_to_b,
            common_tokens=common_tokens,
         )

    def run(self) -> None:
        """
        Tüm tokenizer'ları sample text'ler üzerinde çalıştırır.

        Bu fonksiyon:
        - runner içindeki fonksiyonu çağırır
        - dönen sonuçları self.results içine kaydeder
        """
        self.results = run_all_samples_across_tokenizers() # tüm tokenizer'ları çalıştırır ve sonuçları kaydeder

    def report(self):
        """
        run() ile elde edilen toplu sonuçları ekrana yazdırır.

        ÖNEMLİ:
        - Eğer run() çağrılmadan report() çağrılırsa hata verir
        - Bu kontrol, hatalı kullanımın önüne geçmek için konulmuştur
        """
        if self.results is None:
            raise ValueError("Run must be called before report.")

        print_all_sample_results(self.results) # sonuçları ekrana basmak için kullanılır


    def execute(self):
        """
        Tüm pipeline'ı tek seferde çalıştırır.

        Sırasıyla:
        1. run()  → tokenizer'ları çalıştır
        2. report() → sonuçları yazdır

        Bu metod sayesinde dışarıdan tek satırla tüm süreç çalıştırılabilir.
        """
        self.run() # tokenizer'ları çalıştırır
        self.report() # sonuçları ekrana basar

    def print_comparison_result(self, result: ComparisonResult) -> None:
        """
        ComparisonResult nesnesini terminalde okunabilir biçimde yazdırır.

        Compare işlemi ile sunum işlemini ayırmak için ayrı tutulur.
        Bu sayede:
        - compare() sadece veri üretir
        - print_comparison_result() sadece çıktı sunar
        """

        print("=" * 70)
        print("TOKENIZER COMPARISON")
        print("=" * 70)

        print("\nTEXT:")
        print(result.text)

        print("\n" + "-" * 70)
        print(f"{result.tokenizer_a_name} TOKENS:")
        print(result.tokens_a)

        print("\n" + "-" * 70)
        print(f"{result.tokenizer_b_name} TOKENS:")
        print(result.tokens_b)

        print("\n" + "-" * 70)
        print("TOKEN COUNTS:")
        print(f"{result.tokenizer_a_name}: {result.token_count_a}")
        print(f"{result.tokenizer_b_name}: {result.token_count_b}")

        print("\n" + "-" * 70)
        print(f"UNIQUE TO {result.tokenizer_a_name}:")
        print(result.unique_to_a)

        print("\n" + "-" * 70)
        print(f"UNIQUE TO {result.tokenizer_b_name}:")
        print(result.unique_to_b)

        print("\n" + "-" * 70)
        print("COMMON TOKENS:")
        print(result.common_tokens)

        print("=" * 70)