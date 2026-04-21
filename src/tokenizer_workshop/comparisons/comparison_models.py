# dataclass:
# sınıfı daha kısa ve temiz şekilde veri tutan bir yapıya dönüştürmek için kullanılır
from dataclasses import dataclass


@dataclass
class ComparisonResult:
    """
    ComparisonResult sınıfı, iki tokenizer arasındaki karşılaştırma sonucunu
    düzenli bir şekilde tutmak için kullanılır.

    Bu sınıfın amacı:
    - karşılaştırılan metni saklamak
    - her iki tokenizer'ın adını saklamak
    - üretilen token listelerini saklamak
    - token sayılarını saklamak
    - sadece birinci tokenizer'a özgü token'ları saklamak
    - sadece ikinci tokenizer'a özgü token'ları saklamak
    - ortak token'ları saklamak

    Böylece compare işlemi sonucunda oluşan tüm bilgiler
    tek bir veri yapısı içinde düzenli biçimde tutulmuş olur.

    Neden dataclass kullandık?
    - __init__ metodunu otomatik üretir
    - veri taşıyan sınıflarda daha temiz görünür
    - okunabilirliği artırır
    - test etmeyi kolaylaştırır
    """

    # Karşılaştırmada kullanılan orijinal metin
    text: str

    # Birinci tokenizer'ın sınıf adı
    tokenizer_a_name: str

    # İkinci tokenizer'ın sınıf adı
    tokenizer_b_name: str

    # Birinci tokenizer'ın ürettiği token listesi
    tokens_a: list[str]

    # İkinci tokenizer'ın ürettiği token listesi
    tokens_b: list[str]

    # Birinci tokenizer'ın ürettiği toplam token sayısı
    token_count_a: int

    # İkinci tokenizer'ın ürettiği toplam token sayısı
    token_count_b: int

    # Sadece birinci tokenizer sonucunda bulunan token'lar
    unique_to_a: list[str]

    # Sadece ikinci tokenizer sonucunda bulunan token'lar
    unique_to_b: list[str]

    # Her iki tokenizer sonucunda ortak olan token'lar
    common_tokens: list[str]