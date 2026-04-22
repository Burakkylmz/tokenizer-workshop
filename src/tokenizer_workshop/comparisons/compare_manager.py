from __future__ import annotations

import os

from collections import Counter

# combinations:
# Verilen bir liste içindeki elemanlardan ikili kombinasyonlar üretir.
# Örnek:
# ["word", "char", "byte"] → 
# ("word", "char"), ("word", "byte"), ("char", "byte")
# Tokenizer'lar arasında pairwise (ikili) karşılaştırma üretmek için kullanılacaktır.
from itertools import combinations

from time import perf_counter

# runner modülü:
# Mevcut sistemde tüm tokenizer'ları sample text'ler üzerinde çalıştıran akışı içerir.
# tüm tokenizer'ları alır ve sample text'ler üzerinde çalıştırır
from tokenizer_workshop.comparisons.runner import run_all_samples_across_tokenizers

# report modülü:
# runner tarafından üretilen sonuçları terminale düzenli şekilde yazdırır.
from tokenizer_workshop.comparisons.report import print_all_sample_results

# protocol:
# tokenize davranışını sağlayan tüm sınıflar için ortak tip tanımı sunar
# CompareManager'ın concrete class'lara değil, tokenizer davranışına bağımlı olmasını sağlar.
from tokenizer_workshop.comparisons.protocols import (
    TokenizerProtocol,
    TrainableTokenizerProtocol,
)

# veri modelleri:
# Çoklu tokenizer evaluation framework'ünde kullanilan ana modeller
from tokenizer_workshop.comparisons.models import (
    ComparisonResult,
    PairwiseComparison,
    TokenizerEvaluation,
    TokenizerMetrics,
)


class CompareManager:
    """
    CompareManager sınıfı, tokenizer evaluation ve comparison sürecinin
    merkezi yönetim noktasıdır (orchestrator).

    Bu sınıfın temel görevi:
    - tokenizer'ları çalıştırmak
    - çıktıları analiz etmek
    - metrikleri hesaplamak
    - tokenizer'ları birbirleriyle karşılaştırmak
    - sonucu yapılandırılmış veri olarak döndürmek

    Bu sınıf iki farklı kullanım senaryosunu destekler:

    1. Mevcut runner/report akışı:
       - run()
       - report()
       - execute()

    Bu yapı, projedeki mevcut toplu örnek çalıştırma davranışını korur.

    2. Doğrudan tokenizer karşılaştırması:
       - compare()
       - compare_multiple()
       - print_comparison_result()

    Bu yapı ise aynı metin üzerinde bir veya birden fazla tokenizer'ı
    analiz edip yapılandırılmış sonuç döndürür.

     Böylece sınıf hem backward-compatible kalır
    hem de daha modüler ve genişletilebilir bir hale gelir.
    """

    def __init__(self) -> None:
        """
        CompareManager oluşturulduğunda başlangıçta toplu run sonucu yoktur.

       self.results:
            runner üzerinden üretilen toplu sample sonuçlarını tutar.
            Başlangıçta None'dır; run() çağrılınca dolar.
        """
        self.results = None

    # ============================================================
    # PUBLIC API — TRAIN
    # ============================================================
    def train_tokenizers(
        self,
        tokenizers: dict[str, TokenizerProtocol],
        train_text: str,
    ) -> None:
        """
        Eğitim desteği olan tokenizer'ları verilen metin üzerinde eğitir.
        
        Bu fonksiyon, TrainableTokenizerProtocol'ünü sağlayan tokenizer'ları
        otomatik olarak tespit eder ve onları eğitir.
    
        Böylece CompareManager içinde:
        "Bu tokenizer train edilebilir mi?" sorusunu daha temiz ele alabiliriz.
        
        Bu metod neden gerekli?
        Çünkü bazı tokenizer'lar doğrudan tokenize edebilirken (CharTokenizer),
        bazıları önce train() çağrısı gerektirir.
        Örneğin ByteBPETokenizer gibi yapılar genellikle eğitim ister.

        Bu metodun yaptığı şey:
        - tüm tokenizer'ları dolaşır
        - train() desteği olanları tespit eder
        - yalnızca train edilebilen tokenizer'lara train_text gönderir

        Args:
            tokenizers (dict[str, TokenizerProtocol]):
                Eğitilecek tokenizer nesnelerinin sözlüğü. 
                Anahtarlar tokenizer isimleri, değerler tokenizer nesneleri.
                    Örn: {  "byte_bpe": ByteBPETokenizer(), "word": WordTokenizer() }
            train_text (str):
                Eğitim için kullanılacak metin.

        Raises:
            ValueError:
                Eğitim metni boşsa hata verir.

        Returns:
            None
        """
        # Eğitim metni boşsa hata verir
        # boş metinle eğitim yapmak anlamsızdır
        if not train_text.strip():
            raise ValueError("Training text cannot be empty.")

        # Tokenizer'ları dolaşır ve train() desteği olanları eğitir
        for tokenizer_name, tokenizer in tokenizers.items():
            if isinstance(tokenizer, TrainableTokenizerProtocol): # train() desteği var mı kontrol eder
                tokenizer.train(train_text) # Eğitim desteği olan tokenizer'ları eğitir

    # ============================================================
    # PUBLIC API — OLD TWO-TOKENIZER COMPARE
    # ============================================================
    def compare(
        self,
        text: str,
        tokenizer_a: TokenizerProtocol,
        tokenizer_b: TokenizerProtocol,
    ) -> ComparisonResult:
        """
        Aynı metni iki farklı tokenizer ile tokenize eder
        ve sonucu yapılandırılmış bir ComparisonResult nesnesi olarak döndürür.

        Bu metod backward compatibility için korunur.
        Yani mevcut kullanım alışkanlığını bozmaz.

        İçeride aslında compare_multiple(...) metodunu kullanır.
        Böylece gerçek comparison logic tek yerde tutulur.

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
        return self.compare_multiple(
            text=text, # karşılaştırılacak metin
            tokenizers={
                tokenizer_a.__class__.__name__: tokenizer_a, # tokenizer_a'nın sınıf adını anahtar olarak kullanır
                tokenizer_b.__class__.__name__: tokenizer_b, # tokenizer_b'nin sınıf adını anahtar olarak kullanır
            }, # tokenizers sözlüğü oluşturur ve compare_multiple'a iletir
        ) # compare_multiple metodunu kullanarak karşılaştırmayı gerçekleştirir

    # ============================================================
    # PUBLIC API — MULTI TOKENIZER COMPARE
    # ============================================================
    def compare_multiple(
        self,
        text: str,
        tokenizers: dict[str, TokenizerProtocol],
    ) -> ComparisonResult:
        """
        Birden fazla tokenizer'ı aynı metin üzerinde karşılaştırır.

        Bu metod framework'ün asıl güçlü tarafıdır.
        Sistem artık yalnızca iki tokenizer ile sınırlı değildir.

        Bu metod sırasıyla:
        1. Her tokenizer için tokenize işlemi yapar
        2. Her tokenizer için metrikleri hesaplar
        3. TokenizerEvaluation nesneleri üretir
        4. Tüm tokenizer çiftleri için pairwise karşılaştırma oluşturur
        5. Bunları tek bir ComparisonResult içinde döndürür

        Neden bu önemli?

        Eski sistem:
            sadece 2 tokenizer

        Yeni sistem:
            N adet tokenizer (scalable)

        Args:
            text:
                Tokenize edilip karşılaştırılacak metin.
            tokenizers:
                Ad -> tokenizer nesnesi eşlemesi.
                Örnek:
                {
                    "word": WordTokenizer(),
                    "char": CharTokenizer(),
                    "byte_bpe": ByteBPETokenizer(num_merges=10),
                }

        Returns:
            ComparisonResult:
                Tüm tokenizer evaluation sonuçlarını ve pairwise karşılaştırmaları içeren ana sonuç nesnesi.

        Raises:
            ValueError:
                Eğer text boşsa veya tokenizer sözlüğü boşsa hata verir.
        """
        # Girdi doğrulaması
        if not text.strip():
            raise ValueError("Comparison text cannot be empty.")

        # En az bir tokenizer sağlanmalıdır
        if not tokenizers:
            raise ValueError("At least one tokenizer must be provided.")

        # Her tokenizer için evaluation sonuçlarını tutacak liste
        evaluations: list[TokenizerEvaluation] = []

        # --------------------------------------------------------
        # 1. Her tokenizer için evaluation üret
        # --------------------------------------------------------
        # Her tokenizer'ı dolaşır, tokenize eder, metrikleri hesaplar ve evaluation nesnesi oluşturur
        for tokenizer_name, tokenizer in tokenizers.items():
            # tokenize işlemi yapar ve sonucu normalize eder
            start = perf_counter()

            # tokenize işlemi yapar ve sonucu normalize eder
            tokens = self._tokenize(
                tokenizer_name=tokenizer_name, # hata mesajlarında kullanılacak tokenizer adı
                tokenizer=tokenizer, # tokenize işlemini yapacak nesne
                text=text, # tokenize edilecek metin
            )

            # tokenize işlemi süresini ölçer
            latency_seconds = perf_counter() - start

            # tokenize sonucu üzerinden metrikleri hesaplar
            metrics = self._calculate_metrics(
                text=text, # karşılaştırılacak metin
                tokens=tokens, # tokenizer tarafından üretilen token listesi
                latency_seconds=latency_seconds, # tokenize işlemi süresi
                tokenizer=tokenizer, # tokenizer nesnesi (metrics hesaplamada kullanılabilir)
            )

            # TokenizerEvaluation nesnesi oluşturur ve evaluations listesine ekler
            evaluation = TokenizerEvaluation(
                name=tokenizer_name, # tokenizer'ın görünen adı
                tokens=tokens, # tokenizer tarafından üretilen token listesi
                metrics=metrics, # hesaplanan metrikler
            )
            # evaluation nesnesini evaluations listesine ekler
            evaluations.append(evaluation)

        # --------------------------------------------------------
        # 2. Pairwise comparison üret
        # --------------------------------------------------------
        # Tüm tokenizer çiftleri arasında ikili karşılaştırmalar üretir
        # Her karşılaştırma için ortak token'lar, 
        # sadece sol tarafta olan token'lar ve 
        # sadece sağ tarafta olan token'lar hesaplanır
        # Bu sayede tokenizer'ların birbirinden nasıl ayrıştığına dair detaylı bilgi elde edilir
        # pairwise_comparisons listesi oluşturur
        # pairwise_comparisons listesi, tüm ikili tokenizer karşılaştırmalarını içerir
        # Örnek:
        # evaluations içinde şu tokenizer'lar varsa:
        # - word
        # - char
        # - byte_bpe
        # Bu metod şu karşılaştırmaları üretir:
        # - word <-> char
        # - word <-> byte_bpe
        # - char <-> byte_bpe
        # Her karşılaştırma için:
        # - ortak token var mı?
        # - sadece sol tarafta olan token'lar var mı?
        # - sadece sağ tarafta olan token'lar var mı?
        # Bu bilgiler tokenizer'ların birbirinden nasıl ayrıştığını anlamamıza yardımcı olur
        pairwise_comparisons = self._build_pairwise_comparisons(evaluations)

        # --------------------------------------------------------
        # 3. Ana sonucu döndür
        # --------------------------------------------------------
        # Tüm bilgileri tek bir ComparisonResult nesnesi içinde döndürür
        # ComparisonResult, karşılaştırmada hangi metin kullanıldı, 
        # her tokenizer ne üretti ve tokenizer'lar birbirinden nasıl ayrıştı gibi bilgileri içerir
        # Böylece CompareManager.compare_multiple(...) tek bir güçlü veri modeli döndürür
        # Sonrasında bu veri:
        # - terminal çıktısına dönüştürülebilir
        # - testlerde doğrulanabilir
        # - JSON'a çevrilebilir
        # - report formatter'a gönderilebilir
        # ComparisonResult nesnesi oluşturur ve döndürür
        # ComparisonResult, tüm tokenizer evaluation sonuçlarını ve 
        # pairwise karşılaştırmaları içeren ana sonuç nesnesidir
        # ComparisonResult sayesinde tüm karşılaştırma sonucunu tek bir nesne içinde toplanmış olunur
        # Böylece karşılaştırma sonucunu yönetmek, sunmak ve test etmek çok daha kolay hale gelir
        # ComparisonResult, karşılaştırma sonucunu tek bir güçlü veri modeli olarak döndürür
        return ComparisonResult(
            source_text=text, # karşılaştırmada kullanılan orijinal metin
            evaluations=evaluations, # her tokenizer için oluşturulmuş değerlendirme sonuçları
            pairwise_comparisons=pairwise_comparisons, # tokenizer çiftleri arasındaki karşılaştırma sonuçları
        )

    # ============================================================
    # INTERNAL HELPERS
    # ============================================================
    def _tokenize(
        self,
        tokenizer_name: str,
        tokenizer: TokenizerProtocol,
        text: str,
    ) -> list[str]:
        """
        Verilen tokenizer ile tokenize işlemi yapar ve sonucu normalize eder.

        Bu yardımcı metodun amacı:
        - tokenize çağrısını tek yerde toplamak
        - hatalı dönüş tiplerini erken yakalamak
        - farklı tokenizer çıktılarının tek formatta işlenmesini sağlamak

        Neden normalize ediyoruz?
        Çünkü bazı tokenizer'lar str değil int veya başka türde elemanlar döndürebilir.
        Özellikle byte tabanlı tokenizer'larda bu görülebilir.
        Biz pairwise comparison ve yazdırma tarafını sade tutmak için
        tüm token'ları str'e çeviriyoruz.

        Args:
            tokenizer_name:
                Hata mesajlarında kullanılacak tokenizer adı.
            tokenizer:
                Tokenize işlemini yapacak nesne.
            text:
                Tokenize edilecek metin.

        Returns:
            list[str]:
                Normalize edilmiş token listesi.

        Raises:
            TypeError:
                Tokenizer liste döndürmezse hata verir.
        """
        tokens = tokenizer.tokenize(text)

        if not isinstance(tokens, list):
            raise TypeError(
                f"Tokenizer '{tokenizer_name}' must return list[str], "
                f"but returned {type(tokens).__name__}."
            )

        return [str(token) for token in tokens]

    def _calculate_metrics(
        self,
        text: str,
        tokens: list[str],
        latency_seconds: float,
        tokenizer: TokenizerProtocol,
    ) -> TokenizerMetrics:
        """
        Bir tokenizer'ın token listesi üzerinden temel metrikleri hesaplar.

        Hesaplanan metrikler:
        - toplam token sayısı
        - unique token sayısı
        - unique token oranı
        - ortalama token uzunluğu
        - minimum token uzunluğu
        - maksimum token uzunluğu
        - compression ratio
        - top 5 token
        - average chars per token
        - unknown token sayısı
        - unknown token oranı
        - efficiency score

        compression_ratio nedir?
        Basitçe metin uzunluğunun token sayısına oranıdır.
        Çok kaba ama faydalı bir gösterge olarak kullanılabilir.

        Args:
            text:
                Orijinal kaynak metin.
            tokens:
                Tokenizer tarafından üretilen token listesi.
            latency_seconds:
                Tokenize işleminin süresi saniye cinsinden.
            tokenizer:
                Tokenize işlemini yapan tokenizer nesnesi.

        Returns:
            TokenizerMetrics:
                Hesaplanan metriklerin düzenli modeli.
        """
        token_count = len(tokens)
        unique_token_count = len(set(tokens))
        unique_ratio = unique_token_count / token_count if token_count else 0.0

        token_lengths = [len(token) for token in tokens]

        average_token_length = (
            sum(token_lengths) / token_count if token_count > 0 else 0.0
        )
        min_token_length = min(token_lengths) if token_lengths else 0
        max_token_length = max(token_lengths) if token_lengths else 0

        avg_chars_per_token = len(text) / token_count if token_count else 0.0
        compression_ratio = avg_chars_per_token

        unknown_count = tokens.count("<unk>")
        unknown_rate = unknown_count / token_count if token_count else 0.0

        token_counter = Counter(tokens)
        top_tokens = token_counter.most_common(5)

        token_length_distribution = dict(sorted(Counter(token_lengths).items()))

        efficiency_score = avg_chars_per_token * (1 - unknown_rate)

        latency_per_token = latency_seconds / token_count if token_count else 0.0

        reconstructed_text: str | None = None
        reconstruction_match: bool | None = None

        # decode desteği varsa reconstruct denemesi yap
        if hasattr(tokenizer, "decode"):
            try:
                # Bazı tokenizer'larda decode token listesi değil id listesi isteyebilir.
                # Bu nedenle burada güvenli davranıyoruz.
                reconstructed_text = None
                reconstruction_match = None
            except Exception:
                reconstructed_text = None
                reconstruction_match = None

        return TokenizerMetrics(
            token_count=token_count,
            unique_token_count=unique_token_count,
            unique_ratio=unique_ratio,
            average_token_length=average_token_length,
            min_token_length=min_token_length,
            max_token_length=max_token_length,
            avg_chars_per_token=avg_chars_per_token,
            unknown_count=unknown_count,
            unknown_rate=unknown_rate,
            latency_per_token=latency_per_token,
            latency_seconds=latency_seconds,
            efficiency_score=efficiency_score,
            compression_ratio=compression_ratio,
            top_tokens=top_tokens,
            token_length_distribution=token_length_distribution,
            reconstructed_text=reconstructed_text,
            reconstruction_match=reconstruction_match,
        )

    def _build_pairwise_comparisons(
        self,
        evaluations: list[TokenizerEvaluation],
    ) -> list[PairwiseComparison]:
        """
        Tüm tokenizer'lar arasında ikili karşılaştırmalar üretir.

        Örnek:
        Eğer evaluations içinde şu tokenizer'lar varsa:
        - word
        - char
        - byte_bpe

        Bu metod şu karşılaştırmaları üretir:
        - word <-> char
        - word <-> byte_bpe
        - char <-> byte_bpe

        Her karşılaştırma için:
        - ortak token'lar
        - sadece sol tarafta olan token'lar
        - sadece sağ tarafta olan token'lar
        hesaplanır.

        Args:
            evaluations:
                Her tokenizer için daha önce hesaplanmış evaluation listesi.

        Returns:
            list[PairwiseComparison]:
                Tüm ikili tokenizer karşılaştırmaları.
        """
        pairwise_results: list[PairwiseComparison] = []

        for left, right in combinations(evaluations, 2):
            left_set = set(left.tokens)
            right_set = set(right.tokens)

            common_tokens = sorted(left_set & right_set)
            unique_to_left = sorted(left_set - right_set)
            unique_to_right = sorted(right_set - left_set)

            union = left_set | right_set
            overlap_ratio = len(common_tokens) / len(union) if union else 0.0

            comparison = PairwiseComparison(
                left_name=left.name,
                right_name=right.name,
                common_tokens=common_tokens,
                unique_to_left=unique_to_left,
                unique_to_right=unique_to_right,
                overlap_ratio=overlap_ratio,
            )
            pairwise_results.append(comparison)

        return pairwise_results

    # ============================================================
    # EXISTING RUNNER / REPORT PIPELINE
    # ============================================================
    def run(self) -> None:
        """
        Mevcut runner akışını çalıştırır.

        Bu metod:
        - comparisons.runner içindeki toplu çalıştırma fonksiyonunu çağırır
        - dönen sonucu self.results içinde saklar

        Böylece eski kullanım biçimi korunmuş olur.
        """
        # comparisons.runner içindeki toplu çalıştırma fonksiyonunu çağırır
        # Tüm sample'lar üzerinde tüm tokenizer'ları çalıştırır ve sonuçları döner
        self.results = run_all_samples_across_tokenizers()

    def report(self) -> None:
        """
        run() ile üretilen toplu sonuçları ekrana yazdırır.

        Raises:
            ValueError:
                Eğer önce run() çağrılmadıysa hata verir.
        """
        # herhangi bir sonuç yoksa hata verir
        if self.results is None:
            raise ValueError("Run must be called before report.")

        # comparisons.report içindeki yazdırma fonksiyonunu çağırır
        # run() ile üretilen sonuçları ekrana yazdırır
        print_all_sample_results(self.results)

    def execute(self) -> None:
        """
        Mevcut toplu pipeline'ı tek seferde çalıştırır.

        Sırasıyla:
        1. run()
        2. report()

        Bu metod, dışarıdan tek çağrıyla tüm mevcut compare akışını
        çalıştırmak isteyen kullanım senaryoları için korunur.
        """
        # sırasıyla run() ve report() metodlarını çağırır
        # böylece tüm mevcut compare akışını tek seferde çalıştırır

        self.run() # önce run() ile tüm sample'ları çalıştırır ve sonuçları self.results içinde saklar
        self.report() # sonra report() ile self.results içindeki sonuçları ekrana yazdırır

    # ============================================================
    # PRESENTATION / PRINTING
    # ============================================================
    def build_report(self, result: ComparisonResult) -> str:
        """
        ComparisonResult nesnesini terminalde düzenli ve okunabilir bir rapor
        halinde yazdırır.

        Bu metodun amacı:
        - compare logic ile output/presentation logic'ini ayırmak
        - çoklu tokenizer karşılaştırma sonucunu kullanıcıya anlaşılır şekilde sunmak
        - özellikle token sayıları ve pairwise farkları kolay okunur hale getirmek

        Yazdırılan bölümler:
        1. Başlık
        2. Kaynak metin
        3. Özet tablo
        4. Highlights
        5. Interpretation
        6. Pairwise karşılaştırmalar
        """
        WIDTH = 120
        lines = []
        append = lines.append

        # ============================================================
        # 1. RAPOR BAŞLIĞI
        # ============================================================
        append("\n" + "=" * WIDTH)
        append("TOKENIZER EVALUATION REPORT".center(WIDTH))
        append("=" * WIDTH)

        # ============================================================
        # 2. KAYNAK METİN
        # ============================================================
        append("\nSOURCE TEXT")
        append("-" * WIDTH)
        append(result.source_text)

        # ============================================================
        # 3. ÖZET TABLO
        # ============================================================
        append("\nSUMMARY TABLE")
        append("-" * WIDTH)

        # Tablo başlıkları
        headers = [
            "Tokenizer",
            "Token",
            "Uniq",
            "Uniq Ratio",
            "Avg Len",
            "Min Len",
            "Max Len",
            "Avg Chars/Token",
            "Unknown",
            "Latency",
            "Eff.",
        ]

        # Her kolonun minimum genişliğini başlık uzunluklarına göre başlatıyoruz
        col_widths = [len(header) for header in headers]

        # Önce satırları hazırlıyoruz
        rows = []
        for evaluation in result.evaluations:
            metrics = evaluation.metrics

            row = [
                evaluation.name,
                str(metrics.token_count),
                str(metrics.unique_token_count),
                f"{metrics.unique_ratio:.2f}",
                f"{metrics.average_token_length:.2f}",
                str(metrics.min_token_length),
                str(metrics.max_token_length),
                f"{metrics.avg_chars_per_token:.2f}",
                f"{metrics.unknown_rate:.2f}",
                f"{metrics.latency_seconds:.6f}",
                f"{metrics.efficiency_score:.2f}",
            ]
            rows.append(row)

            # Kolon genişliklerini içeriklere göre güncelliyoruz
            for i, value in enumerate(row):
                col_widths[i] = max(col_widths[i], len(value))

        # Satır formatlayıcı yardımcı fonksiyon
        def format_row(row: list[str]) -> str:
            return " | ".join(
                value.ljust(col_widths[index])
                for index, value in enumerate(row)
            )

        # Başlık satırı
        append(format_row(headers))

        # Ayırıcı satır
        append("-+-".join("-" * width for width in col_widths))

        # Veri satırları
        for row in rows:
            append(format_row(row))

        # ============================================================
        # 4. HIGHLIGHTS
        # ============================================================
        append("\nHIGHLIGHTS")
        append("-" * WIDTH)

        lowest = min(result.evaluations, key=lambda evaluation: evaluation.metrics.token_count)
        highest = max(result.evaluations, key=lambda evaluation: evaluation.metrics.token_count)
        best_eff = max(result.evaluations, key=lambda evaluation: evaluation.metrics.efficiency_score)
        highest_unique = max(result.evaluations, key=lambda evaluation: evaluation.metrics.unique_token_count)
        fastest = min(result.evaluations, key=lambda evaluation: evaluation.metrics.latency_seconds)

        append(f"Lowest Token Count   : {lowest.name} ({lowest.metrics.token_count})")
        append(f"Highest Token Count  : {highest.name} ({highest.metrics.token_count})")
        append(f"Best Efficiency      : {best_eff.name} ({best_eff.metrics.efficiency_score:.2f})")
        append(f"Highest Unique Count : {highest_unique.name} ({highest_unique.metrics.unique_token_count})")

        latency_us = fastest.metrics.latency_seconds * 1_000_000
        append(f"Fastest Tokenizer    : {fastest.name} ({latency_us:.0f}µs)")

        # ============================================================
        # 5. INTERPRETATION
        # ============================================================
        append("\nINTERPRETATION")
        append("-" * WIDTH)

        append(
            f"The '{lowest.name}' tokenizer yielded the lowest token count, "
            f"suggesting a coarse-grained segmentation strategy."
        )

        append(
            f"The '{highest.name}' tokenizer produced the highest number of tokens, "
            f"indicating a fine-grained segmentation."
        )

        append(
            f"The '{best_eff.name}' tokenizer achieved the highest efficiency score, "
            f"({best_eff.metrics.efficiency_score:.2f}), balancing token count and unknown-token behavior."
        )

        append(
            f"The '{highest_unique.name}' tokenizer generated the most unique tokens "
            f"({highest_unique.metrics.unique_token_count}), suggesting higher diversity."
        )

        append(
            f"The '{fastest.name}' tokenizer completed tokenization in the shortest time "
            f"({fastest.metrics.latency_seconds:.6f}s), making it the fastest option for this sample."
        )

        append(
            f"The '{best_eff.name}' tokenizer produced the highest characters-per-token ratio "
            f"({best_eff.metrics.avg_chars_per_token:.2f}), indicating larger token chunks."
        )

        append(
            "\nOverall, these observations highlight how different tokenization strategies "
            "affect segmentation granularity, token diversity, and processing efficiency."
        )

        # TOKENIZER DETAILS
        append("\nTOKENIZER DETAILS")
        append("-" * WIDTH)

        for evaluation in result.evaluations:
            metrics = evaluation.metrics

            append(f"\n[{evaluation.name}]")
            append(f"Tokens                 : {evaluation.tokens}")
            append(f"Token Count            : {metrics.token_count}")
            append(f"Unique Token Count     : {metrics.unique_token_count}")
            append(f"Unique Ratio           : {metrics.unique_ratio:.2f}")
            append(f"Average Token Length   : {metrics.average_token_length:.2f}")
            append(f"Min Token Length       : {metrics.min_token_length}")
            append(f"Max Token Length       : {metrics.max_token_length}")
            append(f"Avg Chars / Token      : {metrics.avg_chars_per_token:.2f}")
            append(f"Unknown Count          : {metrics.unknown_count}")
            append(f"Unknown Rate           : {metrics.unknown_rate:.2f}")
            append(f"Latency                : {metrics.latency_seconds:.6f}s")
            append(f"Efficiency Score       : {metrics.efficiency_score:.2f}")
            append(f"Top-5 Tokens           : {metrics.top_tokens}")
            append(f"Token Length Dist.     : {metrics.token_length_distribution}")

            if metrics.reconstructed_text is not None:
                append(f"Reconstructed Text     : {metrics.reconstructed_text}")
                append(f"Reconstruction Match   : {metrics.reconstruction_match}")

        # OVERALL RANKING
        append("\nOVERALL RANKING")
        append("-" * WIDTH)

        ranking = sorted(
            result.evaluations,
            key=lambda evaluation: (
                evaluation.metrics.efficiency_score,
                -evaluation.metrics.latency_seconds,
            ),
            reverse=True,
        )

        for index, evaluation in enumerate(ranking, start=1):
            append(
                f"{index}. {evaluation.name} "
                f"(efficiency={evaluation.metrics.efficiency_score:.2f}, "
                f"latency={evaluation.metrics.latency_seconds:.6f}s)"
            )

        # ============================================================
        # 6. PAIRWISE COMPARISONS
        # ============================================================
        append("\nPAIRWISE COMPARISONS")
        append("-" * WIDTH)

        if not result.pairwise_comparisons:
            append("No pairwise comparison available.")
        else:
            for comparison in result.pairwise_comparisons:
                append(f"\n[{comparison.left_name} <-> {comparison.right_name}]")
                append(
                    f"Common Tokens ({len(comparison.common_tokens)})      : "
                    f"{comparison.common_tokens}"
                )
                append(
                    f"Only In {comparison.left_name} ({len(comparison.unique_to_left)}) : "
                    f"{comparison.unique_to_left}"
                )
                append(
                    f"Only In {comparison.right_name} ({len(comparison.unique_to_right)}) : "
                    f"{comparison.unique_to_right}"
                )
                append(
                    f"Overlap Ratio : {comparison.overlap_ratio:.2f}"
                )

        # ============================================================
        # 7. RAPOR SONU
        # ============================================================
        append("\n" + "=" * WIDTH)
        append("END OF REPORT".center(WIDTH))
        append("=" * WIDTH)

        report = "\n".join(lines)
        return report


    def print_report(self, report: str) -> None:
        """Raporu terminale yazdırır."""
        print(report)


    def save_report(self, report: str, path: str) -> None:
        """Raporu belirtilen dosya yoluna kaydeder."""
        # Eğer dosya uzantısı .md ise raporu markdown formatında kaydeder
        if path.endswith(".md"):
            # Markdown formatında kaydetmek için raporu üçlü tırnak içine alır
            with open(path, "w", encoding="utf-8") as f:
                f.write("```\n" + report + "\n```")
        else: # Diğer uzantılar için düz metin olarak kaydeder
            with open(path, "w", encoding="utf-8") as f:
                f.write(report)


    def print_comparison_result(
        self,
        result: ComparisonResult,
        save_path: str | None = None,
    ) -> None:
        """Karşılaştırma sonuçlarını yazdırır ve isteğe bağlı olarak kaydeder."""
        report = self.build_report(result)
        self.print_report(report)

        if save_path:
            self.save_report(report, save_path)
