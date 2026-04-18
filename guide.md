# Guide

Bu doküman, `tokenizer-workshop` projesine katkı verecek öğrenciler için hazırlanmıştır. Amaç; projeyi locale almak, `uv` ile çalıştırmak, kendi branch'inde geliştirme yapmak, değişiklikleri push etmek ve Pull Request (PR) açmak için ortak bir çalışma standardı oluşturmaktır.

## 1. Projeye katkı yaklaşımı

Bu projede doğrudan `main` branch üzerinde geliştirme yapılmaz.
Her öğrenci kendi branch'inde çalışır ve işini tamamladıktan sonra `main` branch'e Pull Request açar.

Temel akış şöyledir:

1. Repoyu locale al
2. Projeyi ayağa kaldır
3. Kendi branch'ini oluştur
4. Geliştirmeyi yap
5. Testleri çalıştır
6. Commit al
7. Remote'a push et
8. Pull Request aç

---

## 2. Gereksinimler

Bu projede temel araçlar şunlardır:

- Git
- Python 3.10+
- `uv`

### `uv` kontrolü

PowerShell'de:

```powershell
uv --version
```

Eğer sürüm dönüyorsa hazırdır.

---

## 3. Repoyu locale alma

### Senaryo A — Repo'ya doğrudan erişimin varsa

```powershell
git clone <REPO_URL>
cd tokenizer-workshop
```

### Senaryo B — Repo'ya doğrudan yazma yetkin yoksa

Bu durumda önce GitHub üzerinden repo'yu **fork** et, sonra kendi fork'unu clone et:

```powershell
git clone <YOUR_FORK_URL>
cd tokenizer-workshop
```

---

## 4. Projeyi locale ayağa kaldırma

Repo klasörüne girdikten sonra önce bağımlılıkları senkronize et:

```powershell
uv sync
```

Bu komut:
- `.venv` oluşturur (yoksa)
- `pyproject.toml` ve `uv.lock` dosyasına göre bağımlılıkları kurar
- proje ortamını hazır hale getirir

Ardından proje girişini çalıştır:

```powershell
uv run tokenizer-workshop
```

Bu komut başarılı çalışıyorsa proje temel seviyede ayağa kalkmış demektir.

### Testleri çalıştırma

Tüm testleri çalıştırmak için:

```powershell
uv run pytest -v
```

Belirli bir test dosyasını çalıştırmak için:

```powershell
uv run pytest tests/test_char_tokenizer.py -v
```

---

## 5. Güncel kodu alma

Çalışmaya başlamadan önce local `main` branch'ini güncelle:

```powershell
git checkout main
git pull origin main
```

Eğer fork ile çalışıyorsan ve ana repo `upstream` olarak tanımlıysa:

```powershell
git checkout main
git pull upstream main
```

---

## 6. Kendi branch'ini açma

Her geliştirme ayrı branch'te yapılmalıdır.
Branch isimleri açık ve kısa olmalıdır.

Örnek branch isimleri:

- `feature/word-tokenizer`
- `feature/regex-tokenizer`
- `feature/regex-bpe-tokenizer`
- `feature/byte-bpe-tokenizer`
- `test/metrics-improvements`
- `docs/contribution-guide`

Branch oluşturmak ve geçmek için:

```powershell
git checkout -b feature/<your-work-name>
```

Örnek:

```powershell
git checkout -b feature/word-tokenizer
```

---

## 7. Geliştirme sırasında çalışma standardı

Katkı verirken şu prensiplere uy:

- Küçük ve kontrollü değişiklik yap
- Gereksiz dosya ekleme
- Kod ile birlikte test ekle
- Yorum satırları eğitim değerini artırıyorsa ekle
- Mevcut klasör yapısını bozma
- `main` branch'e doğrudan push atma

### Beklenen temel kontrol listesi

Kodunu push etmeden önce şunları kontrol et:

1. Proje çalışıyor mu?
2. İlgili testler geçiyor mu?
3. Yeni eklediğin dosyalar doğru klasörde mi?
4. Gerekliyse `__init__.py` güncellendi mi?
5. Değişiklik açıklanabilir ve küçük parçalara ayrılmış mı?

---

## 8. Dosya değişikliklerini kontrol etme

Durumu görmek için:

```powershell
git status
```

Yapılan değişiklikleri satır bazlı görmek için:

```powershell
git diff
```

---

## 9. Commit alma

Önce dosyaları stage et:

```powershell
git add .
```

Daha kontrollü gitmek istersen belirli dosyaları ekle:

```powershell
git add src/tokenizer_workshop/tokenizers/word_tokenizer.py
git add tests/test_word_tokenizer.py
```

Sonra commit al:

```powershell
git commit -m "Add word tokenizer and tests"
```

### Commit mesajı önerileri

- `Add word tokenizer and tests`
- `Add regex tokenizer implementation`
- `Add byte BPE tokenizer draft`
- `Improve tokenizer metrics tests`
- `Update contribution guide`

Commit mesajı kısa, net ve fiil ile başlamalıdır.

---

## 10. Remote'a push etme

İlk kez push ederken branch'i remote'a gönder:

```powershell
git push -u origin feature/<your-work-name>
```

Örnek:

```powershell
git push -u origin feature/word-tokenizer
```

Sonraki push'larda daha kısa yazabilirsin:

```powershell
git push
```

---

## 11. Pull Request açma

Push işleminden sonra GitHub'a git ve ilgili branch için PR aç.

### PR açarken dikkat edilmesi gerekenler

PR açıklaması şu 4 soruya cevap vermelidir:

1. Ne eklendi veya değişti?
2. Neden bu değişiklik yapıldı?
3. Hangi dosyalar etkilendi?
4. Hangi testler çalıştırıldı?

### PR açıklaması için örnek şablon

```md
## Summary
This PR adds the first implementation of the tokenizer.

## Changes
- Added tokenizer implementation
- Added tests
- Updated package exports if needed

## Validation
- Ran `uv run pytest -v`
- Verified local run with `uv run tokenizer-workshop`

## Notes
- This PR focuses only on the initial version
- Follow-up improvements can be added separately
```

### PR başlığı örnekleri

- `Add WordTokenizer implementation`
- `Add RegexTokenizer with tests`
- `Add ByteBPETokenizer baseline`
- `Improve tokenizer evaluation metrics`

---

## 12. PR sonrası revizyon süreci

PR açıldıktan sonra yorum gelebilir. Bu durumda:

1. İstenen değişikliği localde yap
2. Testleri tekrar çalıştır
3. Yeni commit al
4. Aynı branch'e tekrar push et

Örnek:

```powershell
git add .
git commit -m "Address PR review comments"
git push
```

Aynı PR otomatik güncellenir. Yeni PR açman gerekmez.

---

## 13. Fork ile çalışanlar için ek not

Eğer fork üzerinden çalışıyorsan, ana repo güncellendikçe kendi fork'un geride kalabilir.
Bu durumda ana repodan güncel kodu almak için önce `upstream` tanımlanır:

```powershell
git remote add upstream <MAIN_REPO_URL>
```

Sonra:

```powershell
git fetch upstream
git checkout main
git merge upstream/main
```

İstersen sonra kendi fork'una da gönder:

```powershell
git push origin main
```

---

## 14. Sık kullanılan komutlar

### Projeyi çalıştır

```powershell
uv run tokenizer-workshop
```

### Tüm testleri çalıştır

```powershell
uv run pytest -v
```

### Tek test dosyası çalıştır

```powershell
uv run pytest tests/test_simple_bpe_tokenizer.py -v
```

### Yeni branch aç

```powershell
git checkout -b feature/<your-work-name>
```

### Durumu kontrol et

```powershell
git status
```

### Commit al

```powershell
git add .
git commit -m "Your commit message"
```

### Push et

```powershell
git push -u origin feature/<your-work-name>
```

---

## 15. Katkı verirken kaçınılması gereken hatalar

- `main` branch üzerinde çalışmak
- Test çalıştırmadan push etmek
- Çok büyük ve dağınık PR açmak
- Tek PR içinde birden fazla bağımsız konu çözmek
- Gereksiz refactor yapmak
- İsimlendirme ve klasör yapısını bozmak
- Çalışmayan kodu “taslak” diye main'e taşımaya çalışmak

---

## 16. Beklenen minimum katkı kalitesi

Bir katkının kabul edilebilir olması için minimum beklenti:

- Kod localde çalışmalı
- İlgili testler yazılmış olmalı
- Mevcut yapıyla uyumlu olmalı
- PR açıklaması net olmalı
- Değişikliğin kapsamı anlaşılır olmalı

---

## 17. Son öneri

Bu projede amaç sadece kod yazmak değil, yazılan şeyi açıklayabilmek.
Bu yüzden katkı verirken şu soruya net cevap verebilmelisin:

**"Ben ne yaptım, neden yaptım ve nasıl doğruladım?"**

PR değerlendirmesinde en önemli nokta budur.
