"""
tokenizers package

Bu package, proje içinde kullanılan tokenizer altyapısını ve ortak tokenizer bileşenlerini expose eder.

Kritik Tasarım Kararı:
    Bu dosya içerisinde tokenizer class'ları (CharTokenizer, WordTokenizer vb.) DOĞRUDAN import edilmez.

Yani burada şu tarz importlar bilinçli olarak kullanılmaz:

    from .char_tokenizer import CharTokenizer
    from .word_tokenizer import WordTokenizer
    from .regex_tokenizer import RegexTokenizer

Bu yaklaşım:
    - her yeni tokenizer eklendiğinde bu dosyanın değiştirilmesini gerektirir
    - merkezi bağımlılık oluşturur (tight coupling)
    - bakım maliyetini artırır

Bunun yerine proje, registry + discovery tabanlı bir mimari kullanır.

Mimari akış:

    tokenizer module
        ↓
    @register_tokenizer("tokenizer_name")
        ↓
    TokenizerRegistry
        ↓
    TTokenizerFactory

Bu yaklaşımın avantajları:
    - yeni tokenizer eklemek için sadece yeni bir dosya yazmak yeterlidir
    - Yeni tokenizer eklemek için __init__.py dosyasını değiştirmek gerekmez.
    - Merkezi import bağımlılığı azaltılır.
    - Tokenizer sistemi plug-in benzeri genişletilebilir hale gelir.
    - Open/Closed Principle desteklenir.
    - TokenizerFactory sadece registry üzerinden çalışır.
    - Yeni tokenizer'lar sisteme otomatik dahil edilebilir. (Sistem otomatik genişler (extensible))

Discovery mekanizması:
    tokenizer_workshop.tokenizers.discovery modülü, tokenizers package altındaki
    tokenizer modüllerini otomatik olarak import eder.

    Import sırasında tokenizer class'ları üzerindeki @register_tokenizer(...)
    decorator'ları çalışır ve class'lar TokenizerRegistry içine kaydedilir.

Örnek:

    @register_tokenizer("subword")
    class SubwordTokenizer(BaseTokenizer):
        ...

Bu class ilgili modül import edildiğinde registry'ye eklenir.
Factory daha sonra bu tokenizer'ı şu şekilde oluşturabilir:

    TokenizerFactory.create("subword")

Bu dosyanın sorumluluğu:
    - tokenizer implementasyonlarını topluca import etmek değildir
    - ortak altyapı bileşenlerini public API olarak expose etmektir

Export edilen public bileşenler:
    BaseTokenizer:
        Tüm tokenizer implementasyonlarının uyması gereken temel base class.

   TokenizerRegistry:
        Tokenizer class'larının runtime'da tutulduğu merkezi registry.

    register_tokenizer:
        Tokenizer class'larını registry'ye ekleyen decorator.

    auto_import_tokenizers:
        Tokenizer modüllerini otomatik import eden discovery fonksiyonu.

Sonuç:
    Bu yapı sayesinde tokenizer sistemi daha genişletilebilir, bakım dostu
    ve düşük bağımlılıklı bir mimariye sahip olur.
"""

from tokenizer_workshop.tokenizers.base import BaseTokenizer
from tokenizer_workshop.tokenizers.registry import (
    TokenizerRegistry,
    register_tokenizer,
)
from tokenizer_workshop.tokenizers.discovery import auto_import_tokenizers

__all__ = [
    "BaseTokenizer",
    "TokenizerRegistry",
    "register_tokenizer",
    "auto_import_tokenizers",
]