# CharTokenizer

## 1. Purpose

`CharTokenizer`, metni **karakter seviyesinde** tokenize eden en temel tokenizer türüdür.

Bu tokenizer’ın projedeki amacı, tokenization kavramını en çıplak ve en anlaşılır haliyle göstermektir.  
Burada her benzersiz karakter bir token olarak kabul edilir.

Örnek:

```text
"merhaba" -> ["m", "e", "r", "h", "a", "b", "a"]
````

Bu yaklaşım özellikle eğitim açısından çok değerlidir çünkü öğrenci önce şu temel sorulara net cevap verebilir:

* token nedir?
* vocabulary nasıl oluşur?
* encode ne yapar?
* decode ne yapar?
* neden bir mapping tablosuna ihtiyaç vardır?

---

## 2. Why This Tokenizer Exists

Bu tokenizer gerçek dünya performansı için değil, **öğretim ve kavramsal netlik** için vardır.

Projede `CharTokenizer` şu rolü oynar:

* tokenization’ın başlangıç noktasıdır
* daha gelişmiş tokenizer’ları anlamak için referans davranış sağlar
* `ByteTokenizer` ve `SimpleBPETokenizer` ile karşılaştırma zemini oluşturur

Başka bir deyişle, bu sınıf olmadan öğrencinin daha ileri tokenizer’ları değerlendirmesi zorlaşır.
Çünkü önce “en basit hal” görülmeden “neden daha karmaşık yöntemlere ihtiyaç duyulduğu” tam anlaşılmaz.

---

## 3. Core Idea

Bu tokenizer’ın mantığı çok basittir:

1. Eğitim verisindeki tüm benzersiz karakterleri topla
2. Her karaktere bir integer kimlik ver
3. Metni bu kimliklere çevir
4. Gerekirse tekrar metne dönüştür

Örnek:

```text
text = "aba"

unique characters = ["a", "b"]
stoi = {"a": 0, "b": 1}
itos = {0: "a", 1: "b"}

encode("aba") -> [0, 1, 0]
decode([0, 1, 0]) -> "aba"
```

Burada iki şey çok önemlidir:

* `stoi`: string to integer
* `itos`: integer to string

Tokenizer sadece parçalamaz; gerektiğinde geri çevirebilmelidir.

---

## 4. Training Logic

`CharTokenizer` için “training” kavramı klasik makine öğrenmesi eğitimi değildir.
Buradaki training, metinden vocabulary çıkarmaktır.

Kodda bu iş şu mantıkla yapılır:

```python
unique_chars = sorted(set(text))
```

Bu satır iki önemli karar içerir:

### a) `set(text)`

Metindeki benzersiz karakterleri toplar.

### b) `sorted(...)`

Karakterleri deterministik sıraya koyar.

Bu neden önemli?

Çünkü tokenizer çıktısının tekrar üretilebilir olması gerekir.
Aynı metni iki kez eğittiğinde aynı karakter aynı id’yi almalıdır.

Eğer `sorted` kullanılmazsa, mapping sırası bazı durumlarda öngörülemez hale gelebilir ve eğitim çıktısı kararsız olabilir.

---

## 5. Encode Logic

`encode()` metodu, verilen metindeki her karakteri integer token id’ye çevirir.

Örnek:

```text
"merhaba" -> [id_m, id_e, id_r, id_h, id_a, id_b, id_a]
```

Bu aşamada önemli bir tasarım kararı alınmıştır:

Eğer tokenizer eğitim sırasında görmediği bir karakter ile karşılaşırsa, **sessizce geçmez** ve **uydurma çözüm üretmez**.
Doğrudan hata verir.

Bu karar eğitim açısından doğrudur çünkü şu problemi görünür kılar:

> Bir tokenizer, kapsamadığı karakterlerle karşılaştığında ne yapmalıdır?

Bu problem gerçek dünyada `unknown token`, `fallback`, `byte fallback` gibi yöntemlerle çözülür.
Ama burada amaç önce problemi çıplak biçimde göstermektir.

---

## 6. Decode Logic

`decode()` metodu integer token listesini tekrar metne çevirir.

Bu aşamada `_itos` mapping kullanılır:

```text
[0, 1, 0] -> "aba"
```

Burada dikkat edilmesi gereken nokta şu:

Decode işlemi, tokenizer’ın gerçekten iki yönlü çalıştığını gösterir.
Birçok öğrenci encode tarafını anlar ama decode tarafının neden gerekli olduğunu gözden kaçırır.

Oysa tokenizer’ın davranışını incelemek, debug etmek ve doğrulamak için decode çok önemlidir.

---

## 7. Vocabulary Behavior

`CharTokenizer` için vocabulary boyutu şudur:

> Eğitim verisindeki benzersiz karakter sayısı

Bu şu anlama gelir:

* vocabulary veri bağımlıdır
* farklı corpus farklı vocab üretir
* küçük metin küçük vocab üretir
* yeni karakterler yeni eğitim ihtiyacı doğurur

Bu davranış eğitim açısından öğreticidir çünkü öğrenciler tokenizer tasarımında “vocab sabit mi, öğrenilmiş mi?” sorusunu sormaya başlar.

---

## 8. Strengths

`CharTokenizer`’ın güçlü yönleri:

* kavramsal olarak çok nettir
* implementasyonu basittir
* encode/decode mantığını öğretir
* vocabulary oluşturmayı görünür kılar
* debugging kolaydır
* eğitim için çok uygundur

Bu yüzden projede ilk tokenizer olarak çok doğru bir seçimdir.

---

## 9. Limitations

Bu tokenizer’ın ciddi sınırları vardır:

### a) Sequence length çok uzayabilir

Her karakter ayrı token olduğu için metin çok uzun token dizilerine dönüşebilir.

### b) Yapısal tekrarları kullanmaz

Örneğin `"token"` kelimesini veya `"ing"` gibi sık parçaları özel olarak öğrenmez.

### c) Görülmeyen karakterlerde kırılır

Yeni karakter gelirse encode edemez.

### d) Gerçek dünya verimliliği düşüktür

Modern LLM sistemlerinde genellikle daha gelişmiş tokenizer’lar tercih edilir.

Yani bu tokenizer öğretici olarak güçlü, pratik verimlilik açısından sınırlıdır.

---

## 10. Comparison with Other Tokenizers

### CharTokenizer vs ByteTokenizer

* `CharTokenizer` karakterleri temel alır
* `ByteTokenizer` UTF-8 byte’larını temel alır

`ByteTokenizer` daha kapsayıcıdır çünkü her UTF-8 metni temsil edebilir.
Ama `CharTokenizer` kavramsal olarak daha kolay anlaşılır.

### CharTokenizer vs SimpleBPETokenizer

* `CharTokenizer` hiçbir şeyi birleştirmez
* `SimpleBPETokenizer` sık görülen parçaları birleştirir

Bu nedenle `SimpleBPETokenizer` bazı metinlerde daha kısa token dizileri üretebilir.

---

## 11. Design Decisions in This Project

Bu projede `CharTokenizer` için alınan önemli kararlar şunlardır:

* vocabulary metinden öğrenilir
* karakter sırası deterministik olacak şekilde kurulur
* eğitim yapılmadan encode/decode çalışmaz
* bilinmeyen karakterlerde hata verilir
* öğreticilik, performanstan daha önceliklidir

Bu kararlar production değil, eğitim amacıyla bilinçli şekilde seçilmiştir.

---

## 12. Testing Perspective

Bu tokenizer için testlerde doğrulanan temel davranışlar şunlardır:

* training sonrası vocab oluşması
* encode çıktısının integer listesi olması
* decode sonrası orijinal metnin geri elde edilmesi
* train edilmeden kullanımın hata vermesi
* bilinmeyen karakterlerde hata verilmesi
* aynı input için aynı vocab’ın oluşması

Bu testler sadece correctness değil, aynı zamanda tasarım sözleşmesini de korur.

---

## 13. When to Use

`CharTokenizer` şu durumlarda anlamlıdır:

* tokenization öğretmek istediğinde
* temel tokenizer mantığını göstermek istediğinde
* küçük ve şeffaf deneyler yapmak istediğinde
* encode/decode mapping yapısını açıklamak istediğinde

Ama şu durumlarda genellikle yeterli değildir:

* büyük ölçekli NLP sistemleri
* verimli sequence temsil ihtiyacı
* çok dilli ve karmaşık veri
* modern LLM pipeline’ları

---

## 14. Final Takeaway

`CharTokenizer`, bu projedeki en basit tokenizer olmasına rağmen en önemsiz tokenizer değildir.
Tam tersine, diğer bütün tokenizer’ları anlamak için gerekli temel kavramsal çerçeveyi sağlar.

Bu sınıfın asıl değeri şudur:

> Tokenization’ın özü, önce en sade haliyle burada görünür hale gelir.


