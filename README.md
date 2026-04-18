# tokenizer-workshop

## Project Overview
`tokenizer-workshop`, tokenization kavramını uygulamalı ve eğitim odaklı biçimde öğretmek için geliştirilen bir Python projesidir.

Bu projenin temel amacı, farklı tokenizer yaklaşımlarını adım adım inşa ederek şu temel soruya güçlü bir cevap vermektir:

**Bir metin, makine tarafından anlamlı parçalara nasıl ayrılır ve bu ayrım neden önemlidir?**

Bu repo, mevcut tokenizer kütüphanelerini sadece kullanmayı değil, onların altında yatan mantığı kavramayı hedefler.

## Problem Definition
### Problem
Tokenization, NLP ve LLM sistemlerinin en temel katmanlarından biridir; ancak çoğu eğitimde ya yüzeysel anlatılır ya da hazır kütüphaneler üzerinden black-box biçimde geçilir.

Bunun sonucu olarak öğrenciler genellikle:
- token nedir sorusuna eksik cevap verir,
- character-level, byte-level ve subword yaklaşımları ayırt etmekte zorlanır,
- BPE gibi yöntemlerin neden ortaya çıktığını tam anlayamaz,
- aynı metnin neden farklı tokenizer’larda farklı token sayısı ürettiğini açıklayamaz.

### Why this problem matters
LLM, fine-tuning, embedding, context window, maliyet ve latency gibi birçok kritik konu doğrudan tokenization ile ilişkilidir. Tokenization mantığını bilmeden LLM sistem tasarımı hakkında derin bir anlayış geliştirmek zordur.

### Target user / use case
Bu proje şu kullanıcılar için uygundur:
- AI / NLP öğrenen geliştiriciler
- LLM sistemlerini daha derin anlamak isteyen mühendisler
- teknik eğitim veren eğitmenler
- tokenizer kavramını kod yazarak öğrenmek isteyen öğrenciler

## Solution Approach
Bu proje, tokenization kavramını tek bir sınıf üzerinden değil; karşılaştırmalı ve aşamalı biçimde öğretir.

Projede temel olarak şu yaklaşımlar ele alınır:
- `CharTokenizer`
- `ByteTokenizer`
- `SimpleBPETokenizer`

Bu sayede öğrenci şu ilerleyişi net görür:
- karakter düzeyi temsil
- byte düzeyi temsil
- subword / merge tabanlı temsil

### Architecture summary
Proje, temiz bir `src/` yapısı altında geliştirilen Python paketinden oluşur. Uygulama ayarları `config.yaml` içinde, proje metadata ve dependency bilgileri ise `pyproject.toml` içinde tutulur. Geliştirme akışı `uv` ile yönetilir. Secret değerler repoya yazılmaz; gerektiğinde environment variable üzerinden okunur. Amaç, production-grade performans kovalamaktan çok okunabilir, test edilebilir ve eğitimde anlatılabilir bir tokenizer laboratuvarı oluşturmaktır.

## Tech Stack

| Component                | Choice                | Notes                               |
| ------------------------ | --------------------- | ----------------------------------- |
| Language                 | Python                | Python 3.10+                        |
| Environment & workflow   | uv                    | Dependency ve environment yönetimi  |
| Project metadata         | pyproject.toml        | Paket ve dependency merkezi         |
| App config               | YAML                  | `config.yaml` ile uygulama ayarları |
| Tokenizer implementation | Custom                | Eğitim odaklı özgün implementasyon  |
| UI / Interface           | CLI / script          | Sade kullanım                       |
| Evaluation               | Simple custom metrics | token count, vocab size, comparison |

## Project Structure
```text
src/
└── tokenizer_workshop/
    ├── tokenizers/
    ├── trainers/
    ├── evaluators/
    └── utils/

tests/
data/
README.md
config.yaml
pyproject.toml
```

### Folder descriptions
- `src/tokenizer_workshop/`: Ana uygulama kodu
- `tests/`: Test dosyaları
- `data/`: Örnek metinler ve küçük demo input’ları

## Setup
### Requirements
- Python version: **3.10+**
- Required tool: **uv**
- Optional secret: **GROQ_API_KEY**

### Installation
```bash
uv sync
```

### Environment Variables
Gerektiğinde sistem environment üzerinden aşağıdaki değer tanımlanabilir:

```env
GROQ_API_KEY=
```

Not: API key değerleri repoya yazılmamalıdır.

## Run Instructions
Proje entry point’ini çalıştırmak için:

```bash
uv run tokenizer-workshop
```

Testleri çalıştırmak için:

```bash
uv run pytest -v
```

## Example Input / Output
### Example input
```text
Merhaba dünya!
```

### Example output
```text
CharTokenizer -> character-level tokens
ByteTokenizer -> UTF-8 byte ids
SimpleBPETokenizer -> learned subword tokens
```

## Key Features
- Eğitim odaklı tokenizer tasarımı
- Karşılaştırmalı öğrenme yaklaşımı
- Character, byte ve BPE düzeylerini bir arada gösterme
- Test destekli geliştirme akışı
- Basit ama öğretici metrikler

## Limitations
Bu proje aşağıdaki sınırlılıklara sahiptir:
- production-grade tokenizer performansı hedeflemez
- büyük ölçekli veri ve optimizasyon problemi çözmez
- tüm gerçek dünya tokenizer davranışlarını birebir taklit etmeyi amaçlamaz

## Future Improvements
İleride aşağıdaki geliştirmeler yapılabilir:
- `WordTokenizer` eklenmesi
- `RegexTokenizer` eklenmesi
- `RegexBPETokenizer` eklenmesi
- `ByteBPETokenizer` eklenmesi
- merge trace / görselleştirme modülü
- notebook tabanlı eğitim materyalleri

## Project Status
**Status:** in progress

## Repository Workflow
- Geliştirme kontrollü ve adım adım ilerletilir.
- Büyük toplu değişikliklerden kaçınılır.


## Author
- Name: Burak
- Project Topic: Educational tokenizer workshop for learning tokenization step by step
