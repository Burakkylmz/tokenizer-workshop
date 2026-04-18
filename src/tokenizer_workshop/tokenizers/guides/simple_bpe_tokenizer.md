# SimpleBPETokenizer

## 1. Purpose

`SimpleBPETokenizer`, bu projede **subword tokenization** fikrini öğretmek için yer alan tokenizer türüdür.

Bu sınıfın temel amacı, öğrencinin şu soruya net cevap verebilmesini sağlamaktır:

> Sık tekrar eden karakter çiftleri birleştirilirse, metin daha verimli biçimde temsil edilebilir mi?

Bu tokenizer, `CharTokenizer` ve `ByteTokenizer` arasında kavramsal bir köprü değil; onların üstüne çıkan daha güçlü bir adımdır. Çünkü burada artık yalnızca metni küçük parçalara ayırmak değil, **tekrar eden yapıları öğrenmek** söz konusudur.

---

## 2. Why This Tokenizer Exists

Bu tokenizer projede çok kritik bir boşluğu doldurur.

### a) Char-level yaklaşımın sınırlarını aşar
`CharTokenizer` metni anlaşılır biçimde tokenize eder, ama her karakteri ayrı tuttuğu için gereksiz uzun diziler üretebilir.

### b) Verimlilik fikrini görünür hale getirir
`SimpleBPETokenizer`, sık görülen komşu parçaları birleştirerek daha kısa token dizileri üretmeye çalışır.

### c) Modern tokenizer mantığına geçiş sağlar
Gerçek dünya tokenizer’larının önemli bölümü subword mantığıyla çalışır. Bu sınıf, bunun en sade ve öğretici halini sunar.

Başka bir deyişle, bu tokenizer’ın projedeki rolü şudur:

> Tokenization sadece bölmek değildir; bazen daha iyi temsil için parçaları birleştirmektir.

---

## 3. What “BPE” Means in This Project

BPE, burada **Byte Pair Encoding** fikrinden ilham alan bir birleştirme yaklaşımı olarak ele alınır.

Ancak bu projedeki sürüm özellikle sadeleştirilmiştir:

- byte-level değil
- character-level başlangıç kullanır
- regex pre-tokenization içermez
- özel token yönetimi içermez
- production parity hedeflemez

Bu yüzden adı `SimpleBPETokenizer`’dır.

Buradaki amaç, endüstriyel tokenizer sistemlerini birebir taklit etmek değil; **BPE mantığını öğretilebilir hale getirmektir**.

---

## 4. Core Idea

Bu tokenizer şu mantıkla çalışır:

1. Metni önce karakter dizisine ayır
2. Ardışık token çiftlerini say
3. En sık geçen çifti bul
4. Bu çifti yeni bir token olarak birleştir
5. Bu işlemi belirli sayıda tekrar et
6. Öğrenilen merge kurallarını encode sırasında sırayla uygula

Örnek:

```text
"abababa"
````

Başlangıç token’ları:

```text
["a", "b", "a", "b", "a", "b", "a"]
```

Ardışık çiftler:

```text
("a", "b") -> 3
("b", "a") -> 3
```

Tie-break kuralına göre ilk seçilen çift:

```text
("a", "b") -> "ab"
```

Birleştirme sonrası token dizisi:

```text
["ab", "ab", "ab", "a"]
```

Bu örnek öğrencinin şunu görmesini sağlar:

> BPE, sık tekrar eden yerel kalıpları daha büyük token’lara dönüştürür.

---

## 5. Separation of Responsibilities

Bu projede önemli bir mimari karar alınmıştır:

* `BPETrainer` öğrenme işini yapar
* `SimpleBPETokenizer` encode/decode davranışını yapar

Bu ayrım çok değerlidir çünkü iki farklı sorumluluğu net biçimde ayırır.

### `BPETrainer`

* pair frekanslarını hesaplar
* en iyi merge’i seçer
* merge sırasını üretir

### `SimpleBPETokenizer`

* merge kurallarını saklar
* vocab oluşturur
* encode sırasında merge’leri uygular
* decode sırasında string’i geri kurar

Bu tasarım, öğrencinin şu ayrımı öğrenmesini sağlar:

> “Model neyi öğreniyor?” ile “öğrenilen kurallar nasıl uygulanıyor?” aynı şey değildir.

Bu, sadece tokenizer için değil genel yazılım mimarisi açısından da çok önemli bir derstir.

---

## 6. Training Logic

`train()` metodu bu tokenizer’ın merkezidir.

Training sırasında şu adımlar gerçekleşir:

### a) Merge kuralları öğrenilir

`BPETrainer.train(text, num_merges=...)` çağrılır.

Buradan şu bilgi gelir:

* hangi pair seçildi
* neye dönüştü
* o anda frekansı neydi

Bu bilgiler `MergeStep` nesneleri olarak tutulur.

### b) Base vocabulary oluşturulur

Eğitim verisindeki benzersiz karakterler alınır.

### c) Learned merged tokens vocabulary’ye eklenir

Training sırasında öğrenilen yeni token’lar da vocab’e eklenir.

Bu çok önemli bir tasarım kararıdır.
Çünkü encode sırasında bazı karakterler birleşirken bazıları birleşmeyebilir. Bu nedenle tokenizer hem:

* base character token’ları
* merged token’ları

aynı anda tanımak zorundadır.

---

## 7. Why Merge Order Matters

Bu tokenizer’da en önemli kavramlardan biri **merge order**’dır.

BPE merge’leri sadece bir “kurallar kümesi” değildir.
Aynı pair’ler farklı sırayla uygulanırsa farklı tokenization çıktıları oluşabilir.

Bu yüzden merge’ler:

* öğrenildikleri sırayla saklanır
* encode sırasında aynı sırayla uygulanır

Bu çok kritik bir noktadır çünkü öğrenciler ilk başta genellikle şu hatalı sezgiye sahiptir:

> “Sık geçen bütün çiftleri öğrendik, sıranın çok önemi yoktur.”

Aslında sıranın önemi büyüktür.
Bu tokenizer, bu gerçeği görünür hale getirir.

---

## 8. Determinism and Tie-Breaking

BPE training sırasında bazen iki farklı pair aynı frekansta olabilir.

Örnek:

```text
"abababa"
```

Burada:

* `("a", "b")`
* `("b", "a")`

aynı sayıda görülebilir.

Bu durumda seçim belirsiz bırakılırsa farklı çalıştırmalarda farklı sonuçlar çıkabilir.
Bu da eğitim ve test açısından kötü olur.

Bu projede bu yüzden deterministik bir tie-break kuralı kullanılır:

* önce en yüksek frekans seçilir
* eşitlikte lexicographically küçük pair seçilir

Bu karar şu faydayı sağlar:

* aynı input → aynı merge sırası
* aynı merge sırası → aynı encode çıktısı
* tekrar üretilebilir deneyler

Bu, eğitim projeleri için özellikle çok önemlidir.

---

## 9. Encode Logic

`encode()` metodu şu akışla çalışır:

1. Metni karakter token’larına ayır
2. Öğrenilen merge adımlarını sırayla uygula
3. Son ortaya çıkan token’ları integer id’ye çevir

Örnek mantık:

```text
text -> char tokens -> merged tokens -> token ids
```

Burada öğrencinin anlaması gereken en kritik şey şudur:

> Encode sırasında tokenization “yeniden öğrenilmez”; sadece önceden öğrenilen merge kuralları uygulanır.

Yani training ile inference ayrıdır.

Bu, tokenizer mantığını anlamak açısından çok önemlidir.

---

## 10. Decode Logic

`decode()` metodu integer token id’lerini tekrar string parçalarına çevirir ve sonra bunları birleştirir.

Bu tokenizer’da token’lar string olduğu için decode süreci görece basittir:

```text
[id_ab, id_ab, id_a] -> ["ab", "ab", "a"] -> "ababa"
```

Burada dikkat edilmesi gereken şey şu:

Decode, token’ların “hangi granularity ile saklandığından” bağımsız biçimde çalışır.
Yani token tek karakter de olabilir, iki karakterlik merge de olabilir, daha büyük bir parça da olabilir.

Bu, subword mantığını öğretmek için çok faydalıdır.

---

## 11. Vocabulary Behavior

`SimpleBPETokenizer` için vocabulary şu iki parçadan oluşur:

### Base tokens

Training metnindeki benzersiz karakterler

### Merged tokens

Training sırasında öğrenilen yeni birleşik token’lar

Bu nedenle `vocab_size`, genellikle `CharTokenizer`’dan daha büyüktür.
Ama önemli nokta vocabulary boyutu değil, temsil gücüdür.

Bu tokenizer şu fikri öğretir:

> Daha büyük bir vocabulary bazen daha kısa token dizileri elde etmek için bilinçli bir trade-off olabilir.

Bu, tokenizer tasarımında çok temel bir mühendislik bakışıdır.

---

## 12. Compression Behavior

Bu tokenizer’ın en büyük avantajı, tekrar eden yapılarda token sayısını azaltabilmesidir.

Örnek:

```text
"abababa"
```

`CharTokenizer` ile:

* 7 karakter
* 7 token

`SimpleBPETokenizer` ile:

* bazı parçalar merge edilir
* toplam token sayısı düşebilir

Bu durum evaluation katmanında metriklerle de görünür hale gelir.

Bu çok değerlidir çünkü öğrenci artık sadece “algoritma doğru mu?” değil, şu soruyu da sorabilir:

> Bu tokenizer gerçekten daha verimli bir temsil üretiyor mu?

---

## 13. Strengths

`SimpleBPETokenizer`’ın güçlü yönleri şunlardır:

### a) Subword mantığını öğretir

Character-level yaklaşımın üstüne çıkar.

### b) Tekrar eden yapıları kullanır

Tokenization’da verimlilik sağlar.

### c) Merge order kavramını görünür kılar

Bu, gerçek tokenizer mantığına yaklaşmak için çok önemlidir.

### d) Test edilebilir ve gözlemlenebilir

Training adımları `MergeStep` üzerinden inspect edilebilir.

### e) Eğitim açısından idealdir

Gerçek dünya fikrini fazla karmaşıklaştırmadan anlatır.

---

## 14. Limitations

Bu tokenizer’ın bilinçli olarak kabul edilmiş sınırları vardır.

### a) Character-based başlar

Gerçek modern BPE sistemlerinin çoğu byte-level başlar.
Bu sınıf o kadar ileri gitmez.

### b) Regex pre-tokenization yoktur

Whitespace, punctuation, sayı ve kelime sınırları özel olarak ele alınmaz.

### c) Unknown token stratejisi yoktur

Training sonrası bilinmeyen token’lar için gelişmiş fallback davranışı bulunmaz.

### d) Save/load mekanizması yoktur

Tokenizer state’i şu an eğitim odaklı kullanım içindir.

### e) Büyük ölçek için optimize edilmemiştir

Amaç performans değil, açıklıktır.

Bu sınırlar eksiklik olarak değil, **kapsam kararı** olarak görülmelidir.

---

## 15. Comparison with Other Tokenizers

### SimpleBPETokenizer vs CharTokenizer

* `CharTokenizer` hiçbir şeyi birleştirmez
* `SimpleBPETokenizer` sık geçen parçaları birleştirir

Sonuç:

* `CharTokenizer` daha basit
* `SimpleBPETokenizer` daha verimli olabilir

### SimpleBPETokenizer vs ByteTokenizer

* `ByteTokenizer` sabit ve kapsayıcı byte uzayını kullanır
* `SimpleBPETokenizer` training ile yeni token’lar öğrenir

Sonuç:

* `ByteTokenizer` daha kapsayıcı
* `SimpleBPETokenizer` tekrar eden yapıları daha iyi kullanabilir

### SimpleBPETokenizer vs gerçek Regex/Byte BPE sistemleri

Bu sınıf, gerçek dünya tokenizer’larının sadeleştirilmiş bir versiyonudur.
Asıl amacı pedagojik açıklıktır.

Yani bu tokenizer, son durak değil; daha gelişmiş tokenizer’lara geçiş basamağıdır.

---

## 16. Design Decisions in This Project

Bu projede `SimpleBPETokenizer` için alınan temel kararlar şunlardır:

* character-level başlangıç tercih edilir
* merge öğrenme mantığı ayrı `BPETrainer` sınıfında tutulur
* merge sırası korunur
* deterministik tie-break uygulanır
* base token’lar ve merged token’lar birlikte vocabulary’ye alınır
* encode/decode davranışı açık ve inspectable tutulur

Bu kararların tamamı, öğretici değer ile mimari temizlik arasında denge kurmak için seçilmiştir.

---

## 17. Testing Perspective

Bu tokenizer için testlerde doğrulanan temel davranışlar şunlardır:

* invalid `num_merges` durumunda hata verilmesi
* boş text ile train edilmemesi
* training öncesi encode/decode kullanımının hata vermesi
* vocab’in training sonrası oluşması
* encode çıktısının integer id listesi olması
* decode sonrası orijinal metnin geri elde edilmesi
* merge step’lerin gerçekten öğrenilmesi
* tekrar eden yapılarda token sayısının azalabilmesi
* aynı input için aynı merge step sırasının üretilmesi

Bu testler çok değerlidir çünkü bu sınıf artık sadece mapping yapan bir tokenizer değil, **öğrenme davranışı olan bir tokenizer**dır.

---

## 18. When to Use

`SimpleBPETokenizer` şu durumlarda özellikle faydalıdır:

* subword tokenization öğretmek istediğinde
* merge mantığını göstermek istediğinde
* neden character-level yaklaşımın yeterli olmadığını anlatmak istediğinde
* tokenizer verimliliği üzerine düşünmek istediğinde
* modern LLM tokenizer mantığına giriş yapmak istediğinde

Şu durumlarda ise yeterli değildir:

* production-grade tokenizer parity gerektiğinde
* çok dilli ve daha karmaşık veri davranışı gerektiğinde
* regex boundary kontrolü istendiğinde
* byte-level robustness gerektiğinde

Bu durumlarda daha ileri yapılar gerekir.

---

## 19. Final Takeaway

`SimpleBPETokenizer`, bu projede tokenization eğitimini gerçek anlamda derinleştiren sınıftır.

Bu sınıfın verdiği en önemli ders şudur:

> İyi tokenization yalnızca metni bölmek değil, tekrar eden yapıları daha anlamlı ve daha verimli parçalara dönüştürmektir.

Bu fikir anlaşıldığında, öğrencinin modern tokenizer tasarımlarına bakışı ciddi biçimde değişir.

