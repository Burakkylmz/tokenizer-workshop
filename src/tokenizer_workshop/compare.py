from __future__ import annotations

# Compare işlemlerini yöneten ana sınıf
from tokenizer_workshop.comparisons.compare_manager import CompareManager

# Karşılaştırmada kullanılacak örnek tokenizer'lar
from tokenizer_workshop.tokenizers.word_tokenizer import WordTokenizer
from tokenizer_workshop.tokenizers.byte_bpe_tokenizer import ByteBPETokenizer


def main() -> None:
    """
    Uygulamanın giriş noktası (entry point)

    Bu fonksiyon:
    - CompareManager nesnesini oluşturur
    - karşılaştırılacak örnek metni belirler
    - iki tokenizer örneği üretir
    - aynı metin üzerinde tokenizer karşılaştırmasını çalıştırır
    - sonucu terminale okunabilir biçimde yazdırır

    Not:
    Bu dosya, compare akışını başlatan sade bir entry point olarak tasarlanmıştır.
    İş mantığı CompareManager içinde tutulur.
    """

    # Karşılaştırma sürecini yönetecek nesne oluşturulur
    manager = CompareManager()

    # Aynı metin üzerinde karşılaştırılacak örnek input
    train_text = """
    Hello world! Tokenization is fun.
    Tokenization helps language models process text.
    Byte pair encoding can merge frequent byte patterns.
    """
    
    compare_text = "Hello world! Tokenization is fun."

    # Karşılaştırılacak tokenizer nesneleri oluşturulur
    tokenizer_a = WordTokenizer()
    tokenizer_b = ByteBPETokenizer(num_merges=10) # ByteBPETokenizer için örnek bir parametre (merge sayısı)

    # Compare işleminden önce tokenizer'lar aynı metin üzerinde eğitilir
    tokenizer_a.train(train_text)
    tokenizer_b.train(train_text)


    # İki tokenizer aynı metin üzerinde karşılaştırılır
    result = manager.compare(
        text=compare_text,
        tokenizer_a=tokenizer_a,
        tokenizer_b=tokenizer_b,
    )

    # Elde edilen karşılaştırma sonucu terminale yazdırılır
    manager.print_comparison_result(result)


# Python script doğrudan çalıştırıldığında burası tetiklenir
if __name__ == "__main__":
    main()

# For Run:
# uv run python -m tokenizer_workshop.compare