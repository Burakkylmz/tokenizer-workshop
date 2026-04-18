# ByteTokenizer

## 1. Purpose

`ByteTokenizer`, metni **UTF-8 byte seviyesinde** tokenize eden tokenizer türüdür.

Bu tokenizer’ın projedeki amacı, öğrencinin şu kritik farkı net biçimde görmesini sağlamaktır:

- Bir metin sadece karakterler üzerinden değil,
- daha düşük seviyeli bir temsil olan **byte dizileri** üzerinden de tokenize edilebilir.

Örnek:

```text
"abc" -> [97, 98, 99]
````

Bu yaklaşım özellikle önemlidir çünkü modern tokenizer sistemlerinde byte-level düşünce çok güçlü bir temel sağlar.

---

## 2. Why This Tokenizer Exists

`ByteTokenizer` projede iki büyük pedagojik boşluğu doldurur:

### a) CharTokenizer’ın sınırlarını görünür kılar

`CharTokenizer` anlaşılması kolaydır, ancak eğitim verisinde görülmeyen karakterlerde kırılır.

`ByteTokenizer` ise metni UTF-8 byte’larına çevirdiği için çok daha kapsayıcıdır.

### b) “Token = karakter” varsayımını kırar

Birçok öğrenci doğal olarak token kavramını karakter veya kelime ile özdeşleştirir.
`ByteTokenizer`, bu düşünceyi genişletir ve şunu gösterir:

> Bir tokenizer’ın temel birimi, insanın sezgisel olarak “karakter” dediği şey olmak zorunda değildir.

Bu, tokenizer kavramını daha derin anlamak için çok değerlidir.

---

## 3. Core Idea

Bu tokenizer’ın mantığı şudur:

1. Metni UTF-8 ile encode et
2. Ortaya çıkan byte dizisini integer listesine çevir
3. Her byte değerini bir token id olarak kullan
4. Decode sırasında byte dizisini tekrar metne çevir

Örnek:

```text
"text" -> b"text" -> [116, 101, 120, 116]
```

Burada çok önemli bir nokta var:

`ByteTokenizer` için token id üretmek adına ayrı bir öğrenilmiş character-to-id mapping kurmak zorunda değiliz.
Çünkü byte değerleri zaten doğal olarak `0..255` aralığında integer temsil taşır.

---

## 4. Why UTF-8 Matters

Bu tokenizer’ın temeli UTF-8’tir.

UTF-8 neden önemli?

* modern metin sistemlerinde yaygın standarttır
* İngilizce dışındaki karakterleri de taşıyabilir
* Türkçe gibi dillerdeki özel karakterleri temsil edebilir
* Unicode dünyasıyla uyumlu çalışır

Örnek olarak `"ğ"` gibi bir karakter, tek bir “harf” gibi görünse de UTF-8 seviyesinde birden fazla byte ile temsil edilebilir.

Bu da çok öğretici bir çıkarım doğurur:

> İnsan açısından tek karakter olan bir yapı, byte-level tokenizer açısından birden fazla token olabilir.

Bu fark, byte-level tokenization’ın anlaşılması için çok kritiktir.

---

## 5. Vocabulary Behavior

`ByteTokenizer` için vocabulary sabittir:

```text
0, 1, 2, ..., 255
```

Yani toplam vocabulary boyutu her zaman:

```text
256
```

Bu davranış `CharTokenizer`’dan çok farklıdır.

### CharTokenizer

* vocabulary veri bağımlıdır
* eğitim verisine göre değişir

### ByteTokenizer

* vocabulary sabittir
* veri değişse de byte uzayı değişmez

Bu fark eğitim açısından çok önemlidir, çünkü öğrenciler tokenizer tasarımında şu soruyu sormaya başlar:

> Vocab sonradan öğrenilen bir şey mi, yoksa baştan sabit tanımlanmış bir şey mi?

`ByteTokenizer` bu sorunun ikinci tarafını gösterir.

---

## 6. Training Logic

Bu tokenizer için “training” kavramı biraz farklıdır.

Gerçek anlamda yeni bir vocabulary öğrenilmez.
Çünkü vocabulary zaten sabittir.

Peki neden yine de `train()` metodu vardır?

Sebebi mimaridir.

Bu projede bütün tokenizer’lar ortak bir `BaseTokenizer` arayüzünü uygular.
Bu nedenle her tokenizer:

* `train`
* `encode`
* `decode`
* `vocab_size`

gibi ortak metotlara sahip olur.

`ByteTokenizer` içinde `train()` esas olarak şu mesajı verir:

> Bu tokenizer artık kullanıma hazır.

Yani burada training, “öğrenme” değil, “yaşam döngüsü tutarlılığı” açısından vardır.

Bu iyi bir tasarım kararıdır çünkü proje boyunca ortak arayüz korunur.

---

## 7. Encode Logic

`encode()` metodu şunu yapar:

```python
list(text.encode("utf-8"))
```

Bu satırın önemi büyüktür.

Çünkü burada öğrencinin şu farkı anlaması gerekir:

* `CharTokenizer` tek tek karakterleri bir sözlükte arıyordu
* `ByteTokenizer` ise doğrudan Python’un UTF-8 encoding mekanizmasını kullanıyor

Yani tokenization mantığı burada daha düşük seviyeli bir veri temsiline dayanır.

Örnek:

```text
"merhaba" -> [109, 101, 114, 104, 97, 98, 97]
```

ASCII karakterlerde bu çok sezgisel görünür.

Ama Türkçe veya farklı Unicode karakterlerde durum daha ilginç hale gelir:

```text
"ğ" -> birden fazla byte
```

Bu da öğrencinin çok önemli bir kavramı fark etmesini sağlar:

> Byte-level tokenization ile character-level tokenization aynı şey değildir.

---

## 8. Decode Logic

`decode()` metodu, byte token listesini tekrar metne çevirir.

Mantık şudur:

1. integer listeden `bytes(...)` oluştur
2. bunu `utf-8` ile decode et

Örnek:

```text
[97, 98, 99] -> b"abc" -> "abc"
```

Burada önemli bir risk vardır:

Her integer listesi geçerli bir UTF-8 byte dizisi oluşturmaz.

Bu yüzden `decode()` sırasında iki tür kontrol gerekir:

### a) Byte range kontrolü

Her token `0..255` aralığında mı?

### b) UTF-8 geçerliliği

Bu byte dizisi gerçekten decode edilebilir mi?

Bu kontrollü hata yaklaşımı eğitim açısından çok değerlidir.
Çünkü burada öğrenciler şu farkı görür:

> Token dizisi var diye her zaman geçerli metin elde edilemez.

---

## 9. Strengths

`ByteTokenizer`’ın güçlü yönleri:

### a) Çok kapsayıcıdır

UTF-8 ile temsil edilebilen her metni tokenize edebilir.

### b) Unknown character problemini azaltır

`CharTokenizer` gibi sadece eğitimde gördüğü karakterlerle sınırlı değildir.

### c) Vocabulary sabittir

Bu, tasarım açısından sade ve öngörülebilir bir yapı sağlar.

### d) Modern tokenizer mantığına daha yakındır

Gerçek dünya tokenizer’larının önemli bir kısmı byte-level düşünceden faydalanır.

Bu nedenle `ByteTokenizer`, eğitim açısından çok güçlü bir orta basamaktır:
ne çok basit ne de gereksiz karmaşıktır.

---

## 10. Limitations

Bu tokenizer’ın da belirgin sınırları vardır.

### a) Sequence length büyüyebilir

Özellikle çok byte’lı karakterlerde tek karakter birden fazla token’a dönüşebilir.

### b) Semantik yapı öğrenmez

Kelime, ek, kök, sık tekrar eden parça gibi daha anlamlı yapıları özel olarak modellemez.

### c) İnsan açısından yorumlamak daha zordur

`[109, 101, 114]` gibi çıktılar, karakter listesine göre daha az sezgiseldir.

### d) Tek başına verimlilik çözümü sunmaz

Kapsayıcıdır ama her zaman en kısa veya en verimli token dizisini üretmez.

Yani `ByteTokenizer`, kapsama konusunda güçlü ama kompresyon ve semantik parça yakalama konusunda sınırlıdır.

---

## 11. Comparison with Other Tokenizers

### ByteTokenizer vs CharTokenizer

* `CharTokenizer` karakter bazlıdır
* `ByteTokenizer` UTF-8 byte bazlıdır

`CharTokenizer` daha sezgiseldir.
`ByteTokenizer` daha kapsayıcıdır.

### ByteTokenizer vs SimpleBPETokenizer

* `ByteTokenizer` sadece ham byte dizisi üretir
* `SimpleBPETokenizer` tekrar eden yapıları birleştirerek daha kısa token dizileri oluşturabilir

Bu yüzden `SimpleBPETokenizer` verimlilik açısından bazı durumlarda daha avantajlı olabilir.

### ByteTokenizer vs gerçek byte-level BPE sistemleri

`ByteTokenizer`, byte seviyeli başlangıç fikrini öğretir ama merge tabanlı öğrenme içermez.
Gerçek byte-level BPE sistemleri bunun üstüne sık geçen byte çiftlerini birleştirir.

Bu nedenle `ByteTokenizer`, daha gelişmiş byte-level tokenizer’lar için kavramsal zemin hazırlar.

---

## 12. Design Decisions in This Project

Bu projede `ByteTokenizer` için alınan temel kararlar şunlardır:

* UTF-8 temel alınır
* vocabulary sabit 256 token olarak kabul edilir
* ortak tokenizer arayüzü korunur
* decode sırasında geçersiz byte ve geçersiz UTF-8 için kontrollü hata verilir
* öğretici açıklık, kısa kod ve davranış görünürlüğü ön plandadır

Bu kararlar production optimizasyonundan çok pedagojik netlik için seçilmiştir.

---

## 13. Testing Perspective

`ByteTokenizer` testlerinde doğrulanan temel davranışlar şunlardır:

* vocabulary boyutunun her zaman 256 olması
* training öncesi kullanımın hata vermesi
* ASCII metinlerin encode/decode roundtrip’inin doğru çalışması
* Türkçe karakterlerde de roundtrip’in korunması
* bazı karakterlerin birden fazla byte token’a dönüşmesi
* geçersiz byte değerlerinde hata verilmesi
* geçersiz UTF-8 dizilerinde hata verilmesi

Bu testler çok değerlidir çünkü byte-level mantığın gerçekten anlaşılıp anlaşılmadığını gösterir.

---

## 14. When to Use

`ByteTokenizer` şu durumlarda çok faydalıdır:

* Unicode ve çok dilli veri temsilini anlatırken
* tokenization’ın karakter seviyesinden daha alt seviyede de düşünülebileceğini göstermek istediğinde
* unknown token problemini tartışırken
* byte-level BPE gibi daha ileri yöntemlere geçiş yaparken

Ama şu durumlarda tek başına yeterli değildir:

* daha kısa token dizileri hedefleniyorsa
* tekrar eden alt parçaları öğrenmek isteniyorsa
* subword verimliliği gösterilmek isteniyorsa

Bu durumda BPE tabanlı yöntemler daha uygun olur.

---

## 15. Final Takeaway

`ByteTokenizer`, bu projede kritik bir eşik tokenizer’dır.

Çünkü bu sınıf öğrenciye şunu öğretir:

> Metin, insanın gördüğü karakter biçiminden daha temel bir seviyede temsil edilebilir.

Bu fark anlaşıldığında, modern tokenizer sistemlerinin neden byte seviyeli veya subword seviyeli tasarlandığı çok daha net hale gelir.

