from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from tokenizer_workshop.evaluators import TokenizationMetrics, evaluate_tokenizer
from tokenizer_workshop.tokenizers import (
    BaseTokenizer,
    ByteBPETokenizer,
    ByteTokenizer,
    CharTokenizer,
    SimpleBPETokenizer,
    WordTokenizer,
)
from tokenizer_workshop.utils import load_sample_texts

TokenizerFactory = Callable[[], BaseTokenizer]


@dataclass(frozen=True)
class ComparisonResult:
    """
    Tek bir comparison run sonucunu temsil eder.

    field'ler:
    - label: Sonucun hangi compare grubuna ait olduğunu açıklar.
    - metrics: İlgili tokenizer evaluation sonucu.
    """

    label: str
    metrics: TokenizationMetrics


def build_default_tokenizer_factories(
    simple_bpe_num_merges: int = 20,
    byte_bpe_num_merges: int = 20,
) -> list[tuple[str, TokenizerFactory]]:
    """
    Projede kullandığımız varsayılan tokenizer factory listesini döndürür.

    Neden instance değil de factory döndürüyoruz?
    Çünkü her evaluation için temiz bir tokenizer instance üretmek istiyoruz.
    Böylece state carry-over riskini azaltırız.
    """
    return [
        ("char", CharTokenizer),
        ("byte", ByteTokenizer),
        (
            f"simple_bpe_{simple_bpe_num_merges}_merges",
            lambda: SimpleBPETokenizer(num_merges=simple_bpe_num_merges),
        ),
        (
            f"byte_bpe_{byte_bpe_num_merges}_merges", # Label, örneğin "byte_bpe_20_merges".
            lambda: ByteBPETokenizer(num_merges=byte_bpe_num_merges), # Factory, örneğin lambda: ByteBPETokenizer(num_merges=20).
        ),
        ('word', WordTokenizer),
    ]


def run_same_text_across_tokenizers(
    text: str,
    tokenizer_factories: list[tuple[str, TokenizerFactory]] | None = None,
    train_text: str | None = None,
) -> list[ComparisonResult]:
    """
    Aynı text üzerinde birden fazla tokenizer'ı karşılaştırır.

    Bu compare tipi şu soruya cevap verir:
    "Aynı input, farklı tokenizer'larda nasıl davranıyor?"
    """
    if not text:
        raise ValueError("Comparison text cannot be empty.")

    factories = tokenizer_factories or build_default_tokenizer_factories()
    results: list[ComparisonResult] = []

    for label, factory in factories:
        tokenizer = factory()
        metrics = evaluate_tokenizer(
            tokenizer=tokenizer,
            text=text,
            train_text=train_text,
        )
        results.append(ComparisonResult(label=label, metrics=metrics))

    return results


def run_all_samples_across_tokenizers(
    tokenizer_factories: list[tuple[str, TokenizerFactory]] | None = None,
) -> dict[str, list[ComparisonResult]]:
    """
    config.yaml içinde tanımlı tüm sample text'ler üzerinde
    aynı-text-across-tokenizers compare çalıştırır.

    Dönüş formatı:
        {
            "data/sample_tr.txt": [...],
            "data/sample_en.txt": [...],
            ...
        }
    """
    sample_texts = load_sample_texts()
    all_results: dict[str, list[ComparisonResult]] = {}

    for sample_name, text in sample_texts.items():
        all_results[sample_name] = run_same_text_across_tokenizers(
            text=text,
            tokenizer_factories=tokenizer_factories,
        )

    return all_results


def run_same_tokenizer_across_samples(
    tokenizer_factory: TokenizerFactory,
    tokenizer_label: str,
) -> list[ComparisonResult]:
    """
    Aynı tokenizer'ı farklı sample text'ler üzerinde çalıştırır.

    Bu compare tipi şu soruya cevap verir:
    "Aynı tokenizer, farklı text türlerinde nasıl davranıyor?"
    """
    sample_texts = load_sample_texts()
    results: list[ComparisonResult] = []

    for sample_name, text in sample_texts.items():
        tokenizer = tokenizer_factory()
        metrics = evaluate_tokenizer(
            tokenizer=tokenizer,
            text=text,
        )
        results.append(ComparisonResult(label=f"{tokenizer_label} | {sample_name}", metrics=metrics))

    return results


def run_simple_bpe_merge_sweep(
    text: str,
    merge_values: list[int],
    train_text: str | None = None,
) -> list[ComparisonResult]:
    """
    Aynı text üzerinde farklı num_merges değerleri ile
    SimpleBPETokenizer compare çalıştırır.

    Bu compare tipi şu soruya cevap verir:
    "num_merges değiştikçe tokenization davranışı nasıl değişiyor?"
    """
    if not text:
        raise ValueError("Comparison text cannot be empty.")

    if not merge_values:
        raise ValueError("merge_values cannot be empty.")

    results: list[ComparisonResult] = []

    for num_merges in merge_values:
        tokenizer = SimpleBPETokenizer(num_merges=num_merges)
        metrics = evaluate_tokenizer(
            tokenizer=tokenizer,
            text=text,
            train_text=train_text,
        )
        results.append(
            ComparisonResult(
                label=f"simple_bpe_{num_merges}_merges",
                metrics=metrics,
            )
        )

    return results


def run_byte_bpe_merge_sweep(
    text: str, # Karşılaştırma için kullanılacak metin.
    merge_values: list[int], # Karşılaştırma için kullanılacak num_merges değerleri listesi.
    train_text: str | None = None, # Opsiyonel olarak training metni; verilmezse text kullanılır.
) -> list[ComparisonResult]:
    """
    Aynı text üzerinde farklı num_merges değerleri ile
    ByteBPETokenizer compare çalıştırır.

    Bu compare tipi şu soruya cevap verir:
    "num_merges değiştikçe tokenization davranışı nasıl değişiyor?"
    """
    # Bu fonksiyon, run_simple_bpe_merge_sweep ile benzer şekilde çalışır ancak ByteBPETokenizer'ı kullanır.
    # Byte-level BPE'nin davranışı, karakter-level BPE'den farklı olabilir çünkü byte'lar karakterlerden daha düşük seviyeli birimlerdir.
    
    # run_simple_bpe_merge_sweep fonksiyonunda olduğu gibi, her num_merges değeri için yeni bir ByteBPETokenizer instance'ı oluşturulur ve evaluate edilir.
    # Sonuçlar, num_merges değerine göre etiketlenmiş ComparisonResult instance'ları olarak saklanır ve döndürülür.

    # Bu fonksiyon, ByteBPETokenizer'ın num_merges parametresinin tokenization performansı üzerindeki etkisini anlamak isteyenler için yararlı olabilir.
    # Byte-level BPE'nin karakter-level BPE'ye göre farklı bir tokenization davranışı sergileyebileceği göz önünde bulundurularak, bu karşılaştırma, num_merges parametresinin etkisini daha iyi anlamamıza yardımcı olabilir.
    # Ayrıca, bu fonksiyon, ByteBPETokenizer'ın farklı num_merges değerleriyle nasıl performans gösterdiğini görselleştirmek veya raporlamak isteyenler için de kullanılabilir.
    
    if not text: # Karşılaştırma için kullanılacak metnin boş olup olmadığını kontrol eder.
        raise ValueError("Comparison text cannot be empty.")

    if not merge_values: # Karşılaştırma için kullanılacak merge_values listesinin boş olup olmadığını kontrol eder.
        raise ValueError("merge_values cannot be empty.")

    results: list[ComparisonResult] = [] # Sonuçları saklamak için boş bir liste oluşturur.

    for num_merges in merge_values: # merge_values listesindeki her num_merges değeri için döngü başlatır.
        tokenizer = ByteBPETokenizer(num_merges=num_merges) # Her num_merges değeri için yeni bir ByteBPETokenizer instance'ı oluşturur.
        metrics = evaluate_tokenizer(
            tokenizer=tokenizer, # Tokenizer'ı evaluate ederken kullanacağı tokenizer instance'ını belirtir.
            text=text, # Evaluate edilecek metni belirtir.
            train_text=train_text, # Opsiyonel olarak training metnini belirtir; verilmezse text kullanılır.
        ) # Tokenizer'ı evaluate eder ve sonuçları metrics değişkenine atar.
        results.append(
            ComparisonResult(
                label=f"byte_bpe_{num_merges}_merges", # Sonuç için açıklayıcı bir label oluşturur, örneğin "byte_bpe_20_merges".
                metrics=metrics, # Evaluate edilen metrikleri belirtir.
            ) # ComparisonResult instance'ı oluşturur ve results listesine ekler.
        ) # Tüm merge_values için bu işlemi tekrarlar ve sonunda results listesini döndürür.

    return results