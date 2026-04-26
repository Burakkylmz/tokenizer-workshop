from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from tokenizer_workshop.api.services.tokenizer_factory import TokenizerFactory
from tokenizer_workshop.evaluators import TokenizationMetrics, evaluate_tokenizer
from tokenizer_workshop.tokenizers.base import BaseTokenizer
from tokenizer_workshop.utils import load_sample_texts

# Bir tokenizer factory fonksiyonunun tip karşılığıdır.
#
# Neden doğrudan tokenizer instance değil factory kullanıyoruz?
# Çünkü tokenizer'lar train() sonrası state tutabilir.
# Örneğin:
#   - vocabulary
#   - merges
#   - token_to_id / id_to_token
#
# Her evaluation için fresh instance üretmek state leakage riskini azaltır.
TokenizerFactoryFn = Callable[[], BaseTokenizer]


@dataclass(frozen=True)
class ComparisonResult:
    """
    Tek bir tokenizer evaluation sonucunu temsil eder.

    Bu model runner katmanının sade veri taşıyıcısıdır.

    Attributes:
        label:
            Sonucun hangi tokenizer / sample / sweep grubuna ait olduğunu belirtir.

            Örnek:
                "word"
                "byte_bpe_20_merges"
                "regex | sample_en"

        metrics:
            evaluate_tokenizer(...) tarafından üretilen ölçüm sonucudur.

            İçinde şunlar bulunabilir:
                - token_count
                - vocab_size
                - compression ratio
                - roundtrip sonucu
                - latency / performance metrikleri
    """

    label: str
    metrics: TokenizationMetrics


def build_default_tokenizer_factories() -> list[tuple[str, TokenizerFactoryFn]]:
    """
    Varsayılan tokenizer factory listesini döndürür.

    Bu fonksiyon artık tokenizer class'larını doğrudan import etmez.
    Bunun yerine merkezi TokenizerFactory kullanır.

    Eski yaklaşım:
        from tokenizer_workshop.tokenizers import WordTokenizer
        ("word", WordTokenizer)

    Yeni yaklaşım:
        ("word", lambda: TokenizerFactory.create("word"))

    Bu değişiklik neden önemli?

    - tokenizers/__init__.py içinde tokenizer class export etmeye gerek kalmaz.
    - registry/discovery mimarisi korunur.
    - yeni tokenizer eklendiğinde runner.py değiştirilmez.
    - tokenizer oluşturma sorumluluğu tek noktaya taşınır.

    Returns:
        list[tuple[str, TokenizerFactoryFn]]:
            Her eleman:
                - label
                - tokenizer üreten callable factory
    """
    tokenizer_names = [
        "char",
        "byte",
        "bpe",
        "byte_bpe",
        "word",
        "regex",
        "regex_bpe",
    ]

    # lambda içinde name=tokenizer_name kullanılmasının sebebi:
    #
    # Python closure davranışında döngü değişkeni late-binding ile yakalanır.
    # Yani name=tokenizer_name yazmazsak tüm lambda'lar son değeri kullanabilir.
    #
    # Yanlış:
    #   lambda: TokenizerFactory.create(tokenizer_name)
    #
    # Doğru:
    #   lambda name=tokenizer_name: TokenizerFactory.create(name)
    #
    # Böylece her factory kendi tokenizer adını güvenli şekilde saklar.
    return [
        (
            tokenizer_name,
            lambda name=tokenizer_name: TokenizerFactory.create(name),
        )
        for tokenizer_name in tokenizer_names
    ]


class TokenizerComparator:
    """
    Sample text'ler üzerinde tokenizer karşılaştırmaları çalıştıran yardımcı sınıf.

    Bu sınıf runner fonksiyonları için daha nesne odaklı bir kullanım sağlar.

    Desteklediği senaryolar:
        1. Aynı text üzerinde tüm tokenizer'ları çalıştırmak
        2. Tüm sample text'ler üzerinde tüm tokenizer'ları çalıştırmak
        3. Aynı tokenizer'ı farklı sample text'ler üzerinde çalıştırmak
        4. BPE tokenizer'lar için farklı merge değerlerini karşılaştırmak

    Not:
        Bu sınıf comparison logic'i doğrudan kendi içinde büyütmez.
        Asıl işi aşağıdaki fonksiyonlara delege eder.
        Böylece hem fonksiyonel API hem de class-based API korunur.
    """

    def __init__(
        self,
        tokenizer_factories: list[tuple[str, TokenizerFactoryFn]] | None = None,
    ) -> None:
        """
        TokenizerComparator instance oluşturur.

        Args:
            tokenizer_factories:
                Dışarıdan özel tokenizer factory listesi verilebilir.

                Eğer verilmezse build_default_tokenizer_factories() kullanılır.

        Neden opsiyonel?
            Testlerde veya özel benchmark senaryolarında sadece belirli
            tokenizer'ları çalıştırmak isteyebiliriz.
        """
        self.tokenizer_factories = (
            tokenizer_factories or build_default_tokenizer_factories()
        )

    def run_single_text(
        self,
        text: str,
        train_text: str | None = None,
    ) -> list[ComparisonResult]:
        """
        Aynı text üzerinde tüm tokenizer'ları çalıştırır.

        Args:
            text:
                Evaluation yapılacak metin.

            train_text:
                Opsiyonel eğitim metni.
                Verilmezse evaluate_tokenizer genellikle text'i training text olarak kullanır.

        Returns:
            list[ComparisonResult]:
                Her tokenizer için bir evaluation sonucu.
        """
        return run_same_text_across_tokenizers(
            text=text,
            tokenizer_factories=self.tokenizer_factories,
            train_text=train_text,
        )

    def run_all_samples(self) -> dict[str, list[ComparisonResult]]:
        """
        config üzerinden yüklenen tüm sample text'lerde tüm tokenizer'ları çalıştırır.

        Returns:
            dict[str, list[ComparisonResult]]:
                Key:
                    sample adı

                Value:
                    ilgili sample üzerinde üretilmiş tokenizer sonuçları
        """
        return run_all_samples_across_tokenizers(
            tokenizer_factories=self.tokenizer_factories,
        )

    def run_across_samples(
        self,
        tokenizer_factory: TokenizerFactoryFn,
        tokenizer_label: str,
    ) -> list[ComparisonResult]:
        """
        Aynı tokenizer'ı tüm sample text'ler üzerinde çalıştırır.

        Bu analiz şu soruya cevap verir:
            "Aynı tokenizer farklı metin türlerinde nasıl davranıyor?"

        Örnek:
            ByteTokenizer Türkçe, İngilizce ve emoji içeren metinlerde
            nasıl token sayısı üretiyor?
        """
        return run_same_tokenizer_across_samples(
            tokenizer_factory=tokenizer_factory,
            tokenizer_label=tokenizer_label,
        )

    def run_simple_bpe_sweep(
        self,
        text: str,
        merge_values: list[int],
        train_text: str | None = None,
    ) -> list[ComparisonResult]:
        """
        Simple BPE için farklı num_merges değerlerini karşılaştırır.

        Bu analiz şu soruya cevap verir:
            "Merge sayısı arttıkça token sayısı ve compression nasıl değişiyor?"
        """
        return run_bpe_merge_sweep(
            text=text,
            merge_values=merge_values,
            tokenizer_name="bpe",
            label_prefix="simple_bpe",
            train_text=train_text,
        )

    def run_byte_bpe_sweep(
        self,
        text: str,
        merge_values: list[int],
        train_text: str | None = None,
    ) -> list[ComparisonResult]:
        """
        Byte BPE için farklı num_merges değerlerini karşılaştırır.

        Byte-level BPE, UTF-8 byte seviyesinde çalıştığı için özellikle:
            - Türkçe karakterler
            - emoji
            - multilingual input
            - unseen character senaryoları

        üzerinde anlamlı karşılaştırmalar sağlar.
        """
        return run_bpe_merge_sweep(
            text=text,
            merge_values=merge_values,
            tokenizer_name="byte_bpe",
            label_prefix="byte_bpe",
            train_text=train_text,
        )


def run_same_text_across_tokenizers(
    text: str,
    tokenizer_factories: list[tuple[str, TokenizerFactoryFn]] | None = None,
    train_text: str | None = None,
) -> list[ComparisonResult]:
    """
    Aynı text üzerinde birden fazla tokenizer'ı karşılaştırır.

    Bu compare tipi şu soruya cevap verir:
        "Aynı input, farklı tokenizer stratejilerinde nasıl davranıyor?"

    Örnek:
        "Hello world!" input'u:
            - char tokenizer'da çok fazla küçük token üretir
            - word tokenizer'da daha az token üretir
            - byte_bpe tokenizer'da subword/byte merge davranışı gösterir

    Args:
        text:
            Karşılaştırılacak kaynak metin.

        tokenizer_factories:
            Opsiyonel tokenizer factory listesi.
            Verilmezse default tokenizer set'i kullanılır.

        train_text:
            Opsiyonel eğitim metni.
            Bazı tokenizer'lar training gerektirir.

    Returns:
        list[ComparisonResult]:
            Her tokenizer için evaluation sonucu.
    """
    _validate_text(text)

    factories = tokenizer_factories or build_default_tokenizer_factories()
    results: list[ComparisonResult] = []

    for label, factory in factories:
        tokenizer = factory()

        metrics = evaluate_tokenizer(
            tokenizer=tokenizer,
            text=text,
            train_text=train_text,
        )

        results.append(
            ComparisonResult(
                label=label,
                metrics=metrics,
            )
        )

    return results


def run_all_samples_across_tokenizers(
    tokenizer_factories: list[tuple[str, TokenizerFactoryFn]] | None = None,
) -> dict[str, list[ComparisonResult]]:
    """
    Tüm sample text'ler üzerinde tokenizer karşılaştırması çalıştırır.

    config.yaml veya utility layer üzerinden gelen sample text'ler kullanılır.

    Dönüş formatı:
        {
            "sample_tr": [ComparisonResult, ...],
            "sample_en": [ComparisonResult, ...],
        }

    Bu compare tipi şu soruya cevap verir:
        "Tokenizer'lar farklı veri örneklerinde tutarlı davranıyor mu?"
    """
    sample_texts = load_sample_texts()

    return {
        sample_name: run_same_text_across_tokenizers(
            text=text,
            tokenizer_factories=tokenizer_factories,
        )
        for sample_name, text in sample_texts.items()
    }


def run_same_tokenizer_across_samples(
    tokenizer_factory: TokenizerFactoryFn,
    tokenizer_label: str,
) -> list[ComparisonResult]:
    """
    Aynı tokenizer'ı farklı sample text'ler üzerinde çalıştırır.

    Bu compare tipi şu soruya cevap verir:
        "Aynı tokenizer farklı metin türlerinde nasıl performans gösteriyor?"

    Örneğin:
        - Türkçe text
        - İngilizce text
        - emoji içeren text
        - teknik text

    üzerinde aynı tokenizer'ın token_count, compression ve roundtrip davranışı
    karşılaştırılabilir.
    """
    sample_texts = load_sample_texts()
    results: list[ComparisonResult] = []

    for sample_name, text in sample_texts.items():
        tokenizer = tokenizer_factory()

        metrics = evaluate_tokenizer(
            tokenizer=tokenizer,
            text=text,
        )

        results.append(
            ComparisonResult(
                label=f"{tokenizer_label} | {sample_name}",
                metrics=metrics,
            )
        )

    return results


def run_simple_bpe_merge_sweep(
    text: str,
    merge_values: list[int],
    train_text: str | None = None,
) -> list[ComparisonResult]:
    """
    Backward-compatible Simple BPE sweep fonksiyonu.

    Eski kullanım bozulmasın diye korunur.
    İçeride generic run_bpe_merge_sweep(...) fonksiyonunu kullanır.
    """
    return run_bpe_merge_sweep(
        text=text,
        merge_values=merge_values,
        tokenizer_name="bpe",
        label_prefix="simple_bpe",
        train_text=train_text,
    )


def run_byte_bpe_merge_sweep(
    text: str,
    merge_values: list[int],
    train_text: str | None = None,
) -> list[ComparisonResult]:
    """
    Backward-compatible Byte BPE sweep fonksiyonu.

    Eski kullanım bozulmasın diye korunur.
    İçeride generic run_bpe_merge_sweep(...) fonksiyonunu kullanır.
    """
    return run_bpe_merge_sweep(
        text=text,
        merge_values=merge_values,
        tokenizer_name="byte_bpe",
        label_prefix="byte_bpe",
        train_text=train_text,
    )


def run_bpe_merge_sweep(
    text: str,
    merge_values: list[int],
    tokenizer_name: str,
    label_prefix: str,
    train_text: str | None = None,
) -> list[ComparisonResult]:
    """
    BPE tabanlı tokenizer'lar için farklı merge değerlerini karşılaştırır.

    Bu fonksiyon generic yapıdadır.
    Hem SimpleBPE hem ByteBPE için kullanılabilir.

    Args:
        text:
            Evaluation yapılacak metin.

        merge_values:
            Denenecek num_merges değerleri.

            Örnek:
                [1, 5, 10, 20]

        tokenizer_name:
            TokenizerFactory üzerinden üretilecek tokenizer adı.

            Örnek:
                "bpe"
                "byte_bpe"

        label_prefix:
            Raporlarda kullanılacak label prefix.

            Örnek:
                "simple_bpe"
                "byte_bpe"

        train_text:
            Opsiyonel eğitim metni.

    Returns:
        list[ComparisonResult]:
            Her merge değeri için evaluation sonucu.

    Not:
        Tokenizer constructor num_merges destekliyorsa parametreli üretilir.
        Desteklemiyorsa default factory creation'a fallback yapılır.
    """
    _validate_text(text)

    if not merge_values:
        raise ValueError("merge_values cannot be empty.")

    results: list[ComparisonResult] = []

    for num_merges in merge_values:
        tokenizer = _create_tokenizer_with_optional_num_merges(
            tokenizer_name=tokenizer_name,
            num_merges=num_merges,
        )

        metrics = evaluate_tokenizer(
            tokenizer=tokenizer,
            text=text,
            train_text=train_text,
        )

        results.append(
            ComparisonResult(
                label=f"{label_prefix}_{num_merges}_merges",
                metrics=metrics,
            )
        )

    return results


def _create_tokenizer_with_optional_num_merges(
    tokenizer_name: str,
    num_merges: int,
) -> BaseTokenizer:
    """
    num_merges destekleyen tokenizer'lar için parametreli instance üretir.

    Neden gerekli?
        TokenizerFactory.create("byte_bpe") default constructor kullanır.
        Ancak sweep testlerinde farklı num_merges değerleri denenmelidir.

    Bu helper:
        1. Registry üzerinden tokenizer instance'larını alır.
        2. İlgili tokenizer'ın class bilgisini çıkarır.
        3. Constructor num_merges kabul ediyorsa parametreli instance üretir.
        4. Kabul etmiyorsa default TokenizerFactory.create(...) akışına döner.

    Bu sayede:
        - factory/discovery mimarisi bozulmaz
        - BPE sweep esnekliği korunur
        - non-BPE tokenizer'lar yanlışlıkla num_merges ile kırılmaz
    """
    registry = TokenizerFactory.get_registry()

    if tokenizer_name not in registry:
        return TokenizerFactory.create(tokenizer_name)

    tokenizer_or_class = registry[tokenizer_name]

    # get_registry() şu an instance döndürüyor.
    # Eğer ileride class döndürecek şekilde değişirse bu kod onu da destekler.
    tokenizer_cls = (
        tokenizer_or_class.__class__
        if not isinstance(tokenizer_or_class, type)
        else tokenizer_or_class
    )

    try:
        return tokenizer_cls(num_merges=num_merges)
    except TypeError:
        return TokenizerFactory.create(tokenizer_name)


def _validate_text(text: str) -> None:
    """
    Compare edilecek text'in boş olmadığını doğrular.

    Boş veya sadece whitespace içeren text ile evaluation yapmak anlamlı değildir.
    Bu yüzden erken hata verilir.
    """
    if not text or not text.strip():
        raise ValueError("Comparison text cannot be empty.")