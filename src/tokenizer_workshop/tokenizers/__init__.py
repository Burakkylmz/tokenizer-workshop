"""
tokenizers package

Bu package, projede kullanılan tüm tokenizer implementasyonlarını barındırır.

Kritik Tasarım Kararı:

Bu dosya içerisinde tokenizer class'ları (CharTokenizer, WordTokenizer vb.)
DOĞRUDAN import edilmez.

Neden?

Eski yaklaşım:
    from .char_tokenizer import CharTokenizer
    from .word_tokenizer import WordTokenizer

Bu yaklaşım:
    - her yeni tokenizer eklendiğinde bu dosyanın değiştirilmesini gerektirir
    - merkezi bağımlılık oluşturur (tight coupling)
    - bakım maliyetini artırır

Yeni yaklaşım (mevcut sistem):

    tokenizer module
        ↓
    @register_tokenizer(...)
        ↓
    TokenizerRegistry
        ↓
    TokenizerFactory

Bu sayede:

✔ Yeni tokenizer eklemek için sadece yeni bir dosya yazılır  
✔ Factory veya __init__ dosyasına dokunulmaz  
✔ Sistem otomatik genişler (extensible)  
✔ Open/Closed Principle sağlanır  

---

Discovery Mekanizması

tokenizer_workshop.tokenizers.discovery modülü:

- tokenizers package altındaki modülleri tarar
- tokenizer dosyalarını otomatik import eder
- import sırasında decorator'ların çalışmasını sağlar

Bu mekanizma sayesinde:

    RegexTokenizer yaz → @register_tokenizer ekle → sistem otomatik tanır

---

Bu dosyanın rolü nedir?

Bu dosya:

- public API yüzeyini tanımlar
- dış modüllerin erişebileceği temel yapı taşlarını expose eder
- tokenizer implementasyonlarını değil, altyapıyı export eder

---

Export edilenler:

BaseTokenizer:
    Tüm tokenizer'ların uyması gereken abstract base class.

TokenizerRegistry:
    Tokenizer class'larının runtime'da tutulduğu merkezi registry.

register_tokenizer:
    Tokenizer class'larını registry'ye eklemek için kullanılan decorator.

auto_import_tokenizers:
    Tokenizer modüllerini otomatik yükleyen discovery fonksiyonu.

---

Örnek kullanım:

Yeni tokenizer eklemek:

    @register_tokenizer("my_tokenizer")
    class MyTokenizer(BaseTokenizer):
        ...

Başka hiçbir yere dokunmana gerek yok.

---

Sonuç:

Bu yapı sayesinde sistem:

- plug-in benzeri çalışır
- genişletilebilir (extensible)
- düşük bağımlılıklıdır (loosely coupled)
- bakım dostudur (maintainable)
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