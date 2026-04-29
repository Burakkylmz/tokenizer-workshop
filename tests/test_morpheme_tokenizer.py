from __future__ import annotations

import pytest

from tokenizer_workshop.api.services.tokenizer_factory import TokenizerFactory


# ---------------------------------------------------------
# INIT TESTS
# ---------------------------------------------------------

def test_morpheme_tokenizer_init_invalid_min_stem_length_raises_error() -> None:
    """
    Guardrail testi: min_stem_length en az 1 olmalıdır.

    Gerekçe:
        MorphemeTokenizer suffix ayırdıktan sonra geriye kalan kökün
        minimum uzunluğunu kontrol eder.

        Eğer min_stem_length < 1 olursa:
            - boş string kökler oluşabilir
            - anlamsız segmentasyonlar üretilebilir

    Bu test şunu garanti eder:
        - Geçersiz konfigürasyon erken aşamada reddedilir
        - Tokenizer invalid state ile oluşturulmaz

    Input:
        min_stem_length=0

    Beklenen:
        ValueError: "min_stem_length must be at least 1"

    Önlenen bug sınıfı:
        - min_stem_length=0 ile tokenizer oluşturulması
        - tokenize() sırasında boş string kökler oluşması
    """
    # bu test, TokenizerFactory üzerinden "morpheme" tokenizer'ı oluşturulurken min_stem_length parametresine 0 verilmesi durumunda ValueError bekler. 
    # Hata mesajında "min_stem_length must be at least 1" ifadesi aranır. 
    # Bu, MorphemeTokenizer'ın geçersiz bir min_stem_length değeriyle oluşturulmasını engeller ve 
    # tokenize() sırasında oluşabilecek boş string kökler gibi anlamsız segmentasyonları önler.
    
    with pytest.raises(ValueError, match="min_stem_length must be at least 1"):
        TokenizerFactory.create("morpheme", min_stem_length=0)


def test_morpheme_tokenizer_init_normalizes_suffix_list() -> None:
    """
    suffix listesi normalize edilmelidir.

    Gerekçe:
        - Kullanıcılar suffix listesine farklı formatlarda girebilirler (örneğin "S", "ness", "", "ness").
        - Tokenizer'ın bu farklı formatları tek bir standart formata dönüştürmesi beklenir.
        - Normalizasyon sayesinde tokenizer'ın davranışı tutarlı olur ve yanlış suffix ayrıştırmalarının önüne geçilir. 

    Beklenen:
        - lowercase dönüşümü yapılır
        - boş stringler atılır
        - duplicate suffixler temizlenir
        - uzunluklarına göre azalan sıralanır (longest-first)

    Bu davranış:
        - deterministik segmentasyon sağlar
        - yanlış parçalamayı (örneğin 's' vs 'ness') engeller

    Input:
        suffixes=["S", "ness", "", "ness"]  

    Beklenen:
        tokenizer.suffixes == ["ness", "s"]

    Bu test şunu garanti eder:
        - suffix listesi her zaman normalize edilmiş formatta saklanır
        - normalize edilmemiş suffix listesiyle oluşabilecek hatalar önlenir
        
    Önlenen bug sınıfı:
        - suffix listesinde büyük harflerin olması
        - boş stringlerin suffix listesinde yer alması
        - duplicate suffixlerin olması
        - suffixlerin uzunluk sırasına göre sıralanmaması (örneğin 's' önce gelmesi)
        - yanlış suffix ayrıştırması (örneğin 's' yerine 'ness' ayrıştırılması)
    """
    # Bu test, TokenizerFactory üzerinden "morpheme" tokenizer'ı oluşturulurken suffixes parametresine ["S", "ness", "", "ness"] verilmesi durumunda 
    # tokenizer.suffixes özelliğinin ["ness", "s"] olarak normalize edilmiş bir listeye dönüştürülmesini bekler. 
    # Normalizasyon süreci şu adımları içerir:
    # 1. Tüm suffixler lowercase'e dönüştürülür (örneğin "S" → "s").
    # 2. Boş stringler atılır (örneğin "" → atılır).
    # 3. Duplicate suffixler temizlenir (örneğin "ness" iki kez verilmişse bir tanesi atılır).
    # 4. Suffixler uzunluklarına göre azalan sırayla sıralanır (örneğin "ness" 4 karakter, "s" 1 karakter olduğu için "ness" önce gelir). 
    # Bu test, suffix listesi normalizasyonunun doğru çalıştığını ve tokenizer'ın tutarlı bir şekilde suffix ayrıştırması yapmasını garanti eder.
    
    tokenizer = TokenizerFactory.create(
        "morpheme",
        suffixes=["S", "ness", "", "ness"],
    )

    # suffixler lowercase + unique + sorted (length desc)
    assert tokenizer.suffixes == ["ness", "s"] # Bu assert, tokenizer.suffixes özelliğinin normalize edilmiş suffix listesiyle eşleştiğini doğrular.    


# ---------------------------------------------------------
# TRAIN TESTS
# ---------------------------------------------------------

def test_morpheme_tokenizer_train_with_empty_text_raises_error() -> None:
    """
    Boş metinle train() çağrıldığında ValueError beklenmelidir.

    Gerekçe:
        Vocabulary oluşturmak için en az bir token gerekir.

    Bu test şunu garanti eder:
        - Boş metinle train() çağrılması engellenir
        - Tokenizer'ın geçersiz bir state'e girmesi önlenir
    
    Input:
        ""
    
    Beklenen:
        ValueError: "Training text cannot be empty"

    Önlenen bug sınıfı:
        - Boş metinle train() çağrılması
        - train() sonrası vocab oluşturulamaması
        - tokenize() sırasında hatalar (örneğin, boş vocab nedeniyle tüm input bilinmeyen token olarak işlenebilir)
    """
    # Bu test, TokenizerFactory üzerinden "morpheme" tokenizer'ı oluşturulurken train() metoduna boş bir string ("") verilmesi durumunda ValueError bekler.
    # Hata mesajında "Training text cannot be empty" ifadesi aranır.

    tokenizer = TokenizerFactory.create("morpheme")

    with pytest.raises(ValueError, match="Training text cannot be empty"):
        tokenizer.train("")


def test_morpheme_tokenizer_train_with_whitespace_text_raises_error() -> None:
    """
    Sadece whitespace içeren metinle train() yapılmamalıdır.

    Input:
        "   "

    Beklenen:
        ValueError: "Training text cannot be empty"

    Gerekçe:
        Vocabulary oluşturmak için en az bir token gerekir.

    Bu test şunu garanti eder:
        - Sadece whitespace içeren metinle train() çağrılması engellenir
        - Tokenizer'ın geçersiz bir state'e girmesi önlenir

    Önlenen bug sınıfı:
        - Boş metinle train() çağrılması
        - train() sonrası vocab oluşturulamaması
        - tokenize() sırasında hatalar (örneğin, boş vocab nedeniyle tüm input bilinmeyen token olarak işlenebilir)
    """
    # Bu test, TokenizerFactory üzerinden "morpheme" tokenizer'ı oluşturulurken 
    # train() metoduna sadece whitespace içeren bir string ("   ") verilmesi durumunda ValueError bekler.
    # Hata mesajında "Training text cannot be empty" ifadesi aranır.

    tokenizer = TokenizerFactory.create("morpheme")

    with pytest.raises(ValueError, match="Training text cannot be empty"):
        tokenizer.train("   ")


def test_morpheme_tokenizer_train_builds_vocab() -> None:
    """
    train() sonrası vocabulary oluşturulmalıdır.

    Gerekçe:
        - Vocabulary, tokenize() ve encode() gibi işlemler için gereklidir.
        - train() metodu, verilen metni tokenize ederek token-to-id ve id-to-token mapping'lerini oluşturur.
        - Vocabulary'nin doğru oluşturulması, tokenizer'ın sonraki işlemlerinde doğru tokenizasyon ve encoding yapmasını sağlar.    

    Input:
        "Çocuklar okulda."

    Beklenen tokenlar:
        ["çocuk", "lar", "okul", "da", "."]

    vocab_size: 5

    Bu test şunu garanti eder:
        - train() metodu verilen metni doğru şekilde tokenize eder
        - Vocabulary doğru şekilde oluşturulur
        - tokenize() ve encode() gibi işlemler için gerekli altyapı sağlanır

    Önlenen bug sınıfı:
        - train() sonrası vocab oluşturulamaması
        - tokenize() ve encode() sırasında hatalar (örneğin, tüm tokenların bilinmeyen token olarak işlenmesi)
    """
    # Bu test, TokenizerFactory üzerinden "morpheme" tokenizer'ı oluşturulurken 
    # train() metoduna "Çocuklar okulda." metni verilmesi durumunda tokenizer'ın vocab_size özelliğinin 5 olduğunu doğrular.

    tokenizer = TokenizerFactory.create("morpheme")

    tokenizer.train("Çocuklar okulda.")

    assert tokenizer.vocab_size == 5 # Beklenen tokenlar: ["çocuk", "lar", "okul", "da", "."]   


def test_morpheme_tokenizer_vocab_is_deterministic() -> None:
    """
    Aynı input ile eğitilen tokenizer'lar aynı mapping'i üretmelidir.

    Bu test deterministik vocabulary davranışını doğrular.

    Gerekçe:
        - Aynı eğitim verisi aynı token-to-id ve id-to-token mapping'lerini üretmelidir.
        - Deterministik davranış, model geliştirme ve hata ayıklama süreçlerinde tutarlı sonuçlar sağlar.
        - Eğer tokenizer'lar aynı input ile farklı mapping'ler üretiyorsa, bu beklenmeyen davranışlara ve hatalara yol açabilir.    
    
    Input:
        "çocuklar çocuklar okulda"
    
    Beklenen:
        İki farklı tokenizer instance'ı aynı token-to-id ve id-to-token mapping'lerini üretmelidir.

    Bu test şunu garanti eder:
        - Aynı eğitim verisiyle eğitilen tokenizer'lar aynı vocabulary'yi oluşturur
        - Tokenizer'ın vocabulary oluşturma süreci deterministiktir
        - Model geliştirme ve hata ayıklama süreçlerinde tutarlı sonuçlar sağlanır  
    
    Önlenen bug sınıfı:
        - Aynı eğitim verisiyle eğitilen tokenizer'ların farklı mapping'ler üretmesi
        - Deterministik olmayan vocabulary oluşturma süreci
        - Beklenmeyen davranışlar ve hatalar (örneğin, aynı metni encode ederken farklı token id'ler üretilmesi)
    """
    # Bu test, TokenizerFactory üzerinden iki farklı "morpheme" tokenizer'ı oluşturulurken aynı metin ("çocuklar çocuklar okulda") ile train() metodunun çağrılması durumunda 
    # her iki tokenizer'ın da aynı token-to-id ve id-to-token mapping'lerini ürettiğini doğrular. 

    # Bu, tokenizer'ın vocabulary oluşturma sürecinin deterministik olduğunu garanti eder ve aynı eğitim verisiyle eğitilen tokenizer'ların tutarlı sonuçlar ürettiğini gösterir.   
    
    t1 = TokenizerFactory.create("morpheme")
    t2 = TokenizerFactory.create("morpheme")

    text = "çocuklar çocuklar okulda"

    t1.train(text)
    t2.train(text)

    assert t1._token_to_id == t2._token_to_id # t1 ve t2 tokenizer'larının token-to-id mapping'lerinin aynı olduğunu doğrular.
    assert t1._id_to_token == t2._id_to_token # t1 ve t2 tokenizer'larının id-to-token mapping'lerinin aynı olduğunu doğrular.


# ---------------------------------------------------------
# TOKENIZE TESTS
# ---------------------------------------------------------

def test_morpheme_tokenizer_splits_simple_suffix() -> None:
    """
    Basit suffix ayrıştırma yapılabilmelidir.

    Gerekçe:
        - MorphemeTokenizer'ın temel işlevi, kelimeleri kök ve ekler olarak ayrıştırmaktır.
        - Basit bir örnek üzerinden suffix ayrıştırmanın doğru çalıştığını doğrulamak önemlidir.
        - Eğer basit suffix ayrıştırması doğru çalışmazsa, daha karmaşık örneklerde de hatalar ortaya çıkabilir.

    Input:
        "books"

    Beklenen:
        ["book", "s"]

    Bu test şunu garanti eder:
        - MorphemeTokenizer'ın temel suffix ayrıştırma işlevi doğru çalışır
        - Basit suffixler doğru şekilde ayrıştırılır
        - Daha karmaşık örneklerde ortaya çıkabilecek hataların önüne geçilir

    Önlenen bug sınıfı:
        - Basit suffix ayrıştırmasının yanlış yapılması (örneğin, "books" → ["books"] olarak kalması)
        - Kök ve eklerin doğru şekilde ayrıştırılmaması (örneğin, "books" → ["boo", "ks"] gibi anlamsız ayrıştırmalar)
        - Daha karmaşık örneklerde de benzer hataların ortaya çıkması (örneğin, "evlerimizde" gibi kelimelerde yanlış ayrıştırmalar)    
    """
    # Bu test, TokenizerFactory üzerinden "morpheme" tokenizer'ı oluşturulurken tokenize() metoduna "books" verilmesi durumunda ["book", "s"] sonucunu bekler. 
    # Bu, MorphemeTokenizer'ın temel suffix ayrıştırma işlevinin doğru çalıştığını doğrular ve basit suffixlerin doğru şekilde ayrıştırıldığını gösterir. 
    # Eğer bu test başarısız olursa, tokenizer'ın suffix ayrıştırma logic'inde temel bir hata olduğu anlamına gelir ve daha karmaşık örneklerde de benzer hataların ortaya çıkması muhtemeldir.

    tokenizer = TokenizerFactory.create("morpheme")

    tokens = tokenizer.tokenize("books")

    assert tokens == ["book", "s"] # tokenize() metodunun "books" kelimesini ["book", "s"] olarak doğru şekilde ayrıştırdığını doğrular.


def test_morpheme_tokenizer_splits_turkish_plural() -> None:
    """
    Türkçe çoğul eki ayrıştırılabilmelidir.

    Gerekçe:
        - MorphemeTokenizer'ın Türkçe gibi eklemeli dillerdeki suffixleri doğru şekilde ayrıştırması beklenir.
        - "çocuklar" kelimesi, "çocuk" (kök) ve "lar" (çoğul eki) olarak ayrıştırılmalıdır.
        - Eğer bu temel örnek doğru şekilde ayrıştırılmazsa, diğer Türkçe kelimelerde de benzer hatalar ortaya çıkabilir.   

    Input:
        "çocuklar"

    Beklenen:
        ["çocuk", "lar"]

    Bu test şunu garanti eder:
        - MorphemeTokenizer'ın Türkçe suffix ayrıştırma işlevi doğru çalışır
        - Türkçe çoğul eki "lar" doğru şekilde ayrıştırılır
        - Diğer Türkçe kelimelerde de benzer hataların önüne geçilir

    Önlenen bug sınıfı:
        - Türkçe suffix ayrıştırmasının yanlış yapılması (örneğin, "çocuklar" → ["çocuklar"] olarak kalması)
        - Kök ve eklerin doğru şekilde ayrıştırılmaması (örneğin, "çocuklar" → ["çoc", "uklar"] gibi anlamsız ayrıştırmalar)
        - Diğer Türkçe kelimelerde de benzer hataların ortaya çıkması (örneğin, "evlerimizde" gibi kelimelerde yanlış ayrıştırmalar)
    """
    # Bu test, TokenizerFactory üzerinden "morpheme" tokenizer'ı oluşturulurken tokenize() metoduna "çocuklar" verilmesi durumunda ["çocuk", "lar"] sonucunu bekler. 
    # Bu, MorphemeTokenizer'ın Türkçe suffix ayrıştırma işlevinin doğru çalıştığını doğrular ve Türkçe çoğul eki "lar"ın doğru şekilde ayrıştırıldığını gösterir. 
    # Eğer bu test başarısız olursa, tokenizer'ın Türkçe suffix ayrıştırma logic'inde temel bir hata olduğu anlamına gelir ve diğer Türkçe kelimelerde de benzer hataların ortaya çıkması muhtemeldir.

    tokenizer = TokenizerFactory.create("morpheme")

    tokens = tokenizer.tokenize("çocuklar")

    assert tokens == ["çocuk", "lar"] # Bu assert, tokenize() metodunun "çocuklar" kelimesini ["çocuk", "lar"] olarak doğru şekilde ayrıştırdığını doğrular.    


def test_morpheme_tokenizer_handles_multiple_suffixes() -> None:
    """
    Birden fazla suffix greedy olarak ayrıştırılmalıdır.

    Gerekçe:
        - MorphemeTokenizer'ın bir kelime içinde birden fazla suffix'i doğru şekilde ayrıştırabilmesi beklenir.
        - "evlerimizde" kelimesi, "ev" (kök), "ler" (çoğul eki), "imiz" (iyelik eki) ve "de" (bulunma hali eki) olarak ayrıştırılmalıdır.
        - Eğer bu tür birden fazla suffix içeren kelimeler doğru şekilde ayrıştırılmazsa, eklemeli dillerdeki birçok kelimede benzer hatalar ortaya çıkabilir.

    Input:
        "evlerimizde"

    Beklenen:
        ["ev", "ler", "imiz", "de"]

    Bu test şunu garanti eder:
        - MorphemeTokenizer'ın bir kelime içinde birden fazla suffix'i doğru şekilde ayrıştırabilme yeteneği doğrulanır
        - "evlerimizde" kelimesi doğru şekilde ayrıştırılır 
        - Eklemeli dillerdeki diğer kelimelerde de benzer hataların önüne geçilir

    Önlenen bug sınıfı:
        - Birden fazla suffix içeren kelimelerin yanlış ayrıştırılması (örneğin, "evlerimizde" → ["evlerimizde"] olarak kalması)
        - Kök ve eklerin doğru şekilde ayrıştırılmaması (örneğin, "evlerimizde" → ["ev", "lerimizde"] gibi anlamsız ayrıştırmalar)
        - Eklemeli dillerdeki diğer kelimelerde de benzer hataların ortaya çıkması (örneğin, "kitaplarımızda" gibi kelimelerde yanlış ayrıştırmalar)
    """
    # Bu test, TokenizerFactory üzerinden "morpheme" tokenizer'ı oluşturulurken tokenize() metoduna "evlerimizde" verilmesi durumunda ["ev", "ler", "imiz", "de"] sonucunu bekler.  
    # Bu, MorphemeTokenizer'ın bir kelime içinde birden fazla suffix'i doğru şekilde ayrıştırabilme yeteneğini doğrular ve "evlerimizde" kelimesinin doğru şekilde ayrıştırıldığını gösterir.
    # Eğer bu test başarısız olursa, tokenizer'ın suffix ayrıştırma logic'inde birden fazla suffix'i doğru şekilde ayrıştıramadığı anlamına gelir ve bu durum daha karmaşık kelimelerde de benzer hataların ortaya çıkmasına yol açabilir.

    tokenizer = TokenizerFactory.create("morpheme")

    tokens = tokenizer.tokenize("evlerimizde")

    assert tokens == ["ev", "ler", "imiz", "de"] # Bu assert, tokenize() metodunun "evlerimizde" kelimesini ["ev", "ler", "imiz", "de"] olarak doğru şekilde ayrıştırdığını doğrular.    


def test_morpheme_tokenizer_respects_min_stem_length() -> None:
    """
    min_stem_length kuralı korunmalıdır.

    Gerekçe:
        - MorphemeTokenizer'ın suffix ayrıştırırken min_stem_length kuralına uyması beklenir.
        - "as" kelimesinde "s" suffix olarak ayrıştırılabilir, ancak geriye kalan "a" kökü sadece 1 karakter uzunluğundadır.
        - Eğer min_stem_length=2 ise, "a" kökü yeterince uzun olmadığı için "s" suffix olarak ayrıştırılmamalıdır.
        - Bu test, min_stem_length kuralının doğru şekilde uygulandığını doğrular ve tokenizer'ın anlamsız segmentasyonları önlemesini sağlar.  

    Input:
        "as"

    suffix = "s"
    stem = "a" (uzunluk 1)

    Eğer min_stem_length=2 ise split yapılmamalıdır.

    Beklenen:
        ["as"]

    Bu test şunu garanti eder:
        - MorphemeTokenizer'ın min_stem_length kuralına uygun şekilde suffix ayrıştırması doğrulanır
        - "as" kelimesinde "s" suffix olarak ayrıştırılmaz çünkü geriye kalan "a" kökü min_stem_length=2 kuralını ihlal eder
        - Anlamsız segmentasyonların önüne geçilir

    Önlenen bug sınıfı:
        - min_stem_length kuralının ihlal edilmesi (örneğin, "as" → ["a", "s"] olarak ayrıştırılması)
        - Anlamsız segmentasyonların oluşması (örneğin, "as" kelimesinin "a" ve "s" olarak ayrıştırılması, "a" kökünün anlamsız ve çok kısa olması)
    """
    # Bu test, TokenizerFactory üzerinden "morpheme" tokenizer'ı oluşturulurken min_stem_length parametresine 2 verilmesi durumunda tokenize() metoduna "as" verilmesi durumunda ["as"] sonucunu bekler. 
    # Bu, MorphemeTokenizer'ın min_stem_length kuralına uygun şekilde suffix ayrıştırması doğrular ve "as" kelimesinde "s" suffix olarak ayrıştırılmaz çünkü geriye kalan "a" kökü min_stem_length=2 kuralını ihlal eder. 
    # Eğer bu test başarısız olursa, tokenizer'ın min_stem_length kuralını doğru şekilde uygulamadığı anlamına gelir ve bu durum anlamsız segmentasyonların oluşmasına yol açabilir (örneğin, "as" kelimesinin "a" ve "s" olarak ayrıştırılması, "a" kökünün anlamsız ve çok kısa olması gibi).

    tokenizer = TokenizerFactory.create("morpheme", min_stem_length=2)

    tokens = tokenizer.tokenize("as")

    assert tokens == ["as"] # tokenize() metodunun "as" kelimesini ["as"] olarak ayrıştırdığını doğrular. 
    # "s" suffix olarak ayrıştırılmamalıdır çünkü geriye kalan "a" kökü min_stem_length=2 kuralını ihlal eder.


def test_morpheme_tokenizer_keeps_punctuation_separate() -> None:
    """
    Noktalama işaretleri ayrı token olarak korunmalıdır.

    Gerekçe:
        - MorphemeTokenizer'ın noktalama işaretlerini ayrı tokenlar olarak koruması beklenir.
        - "Çocuklar okulda." cümlesinde, nokta (".") ayrı bir token olarak kalmalıdır.
        - Eğer noktalama işaretleri ayrı tokenlar olarak korunmazsa, bu durum downstream işlemlerde (örneğin, dil modeli eğitimi) sorunlara yol açabilir 
        çünkü noktalama işaretleri genellikle özel anlam taşır ve ayrı tokenlar olarak işlenmeleri gerekir.

    Input:
        "Çocuklar okulda."

    Beklenen:
        ["çocuk", "lar", "okul", "da", "."]

    Bu test şunu garanti eder:
        - MorphemeTokenizer'ın noktalama işaretlerini ayrı tokenlar olarak koruması doğrulanır
        - "Çocuklar okulda." cümlesinde nokta (".") ayrı bir token olarak kalır
        - Downstream işlemlerde noktalama işaretlerinin özel anlam taşıması durumunda sorunların önüne geçilir

    Önlenen bug sınıfı:
        - Noktalama işaretlerinin ayrı tokenlar olarak korunmaması (örneğin, "Çocuklar okulda." → ["çocuk", "lar", "okul", "da"] olarak ayrıştırılması)
        - Noktalama işaretlerinin kök veya eklerle birleşerek yanlış tokenlar oluşturması (örneğin, "Çocuklar okulda." → ["çocuk", "lar", "okul", "da."] gibi ayrıştırmalar)    
    """
    # Bu test, TokenizerFactory üzerinden "morpheme" tokenizer'ı oluşturulurken tokenize() metoduna "Çocuklar okulda." verilmesi durumunda ["çocuk", "lar", "okul", "da", "."] sonucunu bekler. 
    # Bu, MorphemeTokenizer'ın noktalama işaretlerini ayrı tokenlar olarak koruması doğrular ve "Çocuklar okulda." cümlesinde nokta (".") ayrı bir token olarak kalır. 
    # Eğer bu test başarısız olursa, tokenizer'ın noktalama işaretlerini ayrı tokenlar olarak korumadığı anlamına gelir ve 
    # bu durum downstream işlemlerde (örneğin, dil modeli eğitimi) sorunlara yol açabilir çünkü noktalama işaretleri genellikle özel anlam taşır ve ayrı tokenlar olarak işlenmeleri gerekir.

    tokenizer = TokenizerFactory.create("morpheme")

    tokens = tokenizer.tokenize("Çocuklar okulda.")

    assert tokens[-1] == "." # tokenize() metodunun "Çocuklar okulda." cümlesindeki nokta (".") karakterini ayrı bir token olarak koruduğunu doğrular.  


def test_morpheme_tokenizer_handles_empty_and_whitespace() -> None:
    """
    Boş ve whitespace-only input için boş liste dönmelidir.

    Gerekçe:
        - MorphemeTokenizer'ın boş string veya sadece whitespace içeren stringler için boş bir liste döndürmesi beklenir.
        - Bu durum, tokenizer'ın geçersiz veya anlamsız inputlara karşı sağlam ve öngörülebilir davranmasını sağlar.
        - Eğer tokenizer boş veya sadece whitespace içeren inputlar için boş liste döndürmezse, downstream işlemlerde beklenmeyen hatalara veya 
        yanlış sonuçlara yol açabilir (örneğin, boş stringin bilinmeyen token olarak işlenmesi gibi).

    Input:
        ""
        "   \n\t"
    
    Beklenen:
        []

    Bu test şunu garanti eder:
        - MorphemeTokenizer'ın boş string veya sadece whitespace içeren stringler için boş bir liste döndürmesi doğrulanır
        - Tokenizer'ın geçersiz veya anlamsız inputlara karşı sağlam ve öngörülebilir davranması sağlanır
        - Downstream işlemlerde beklenmeyen hataların veya yanlış sonuçların önüne geçilir (örneğin, boş stringin bilinmeyen token olarak işlenmesi gibi)

    Önlenen bug sınıfı:
        - Boş string veya sadece whitespace içeren stringler için boş liste döndürülmemesi (örneğin, "" → [""] veya "   \n\t" → ["   \n\t"] gibi sonuçlar)
        - Boş veya sadece whitespace içeren inputların bilinmeyen token olarak işlenmesi (örneğin, "" → ["[UNK]"] veya "   \n\t" → ["[UNK]"] gibi sonuçlar)
    """
    # Bu test, TokenizerFactory üzerinden "morpheme" tokenizer'ı oluşturulurken tokenize() metoduna boş bir string ("") ve sadece whitespace içeren bir string ("   \n\t") verilmesi durumunda her iki durumda da boş bir liste ([]) sonucunu bekler. 
    # Bu, MorphemeTokenizer'ın boş string veya sadece whitespace içeren stringler için boş bir liste döndürmesi doğrular ve tokenizer'ın geçersiz veya anlamsız inputlara karşı sağlam ve öngörülebilir davranmasını sağlar. 
    # Eğer bu test başarısız olursa, tokenizer'ın boş veya sadece whitespace içeren inputlar için boş liste döndürmediği anlamına gelir ve downstream işlemlerde beklenmeyen hatalara veya yanlış sonuçlara yol açabilir 
    # (örneğin, boş stringin bilinmeyen token olarak işlenmesi gibi).

    tokenizer = TokenizerFactory.create("morpheme")

    assert tokenizer.tokenize("") == [] # tokenize() metodunun boş bir string ("") için boş bir liste ([]) döndürdüğünü doğrular.
    assert tokenizer.tokenize("   \n\t") == [] # tokenize() metodunun sadece whitespace içeren bir string ("   \n\t") için boş bir liste ([]) döndürdüğünü doğrular.


# ---------------------------------------------------------
# ENCODE TESTS
# ---------------------------------------------------------

def test_morpheme_tokenizer_encode_before_training_raises_error() -> None:
    """
    train() çağrılmadan encode() çalışmamalıdır.

    Gerekçe:
        - MorphemeTokenizer'ın encode() metodunun çalışabilmesi için öncelikle train() metodunun çağrılarak vocabulary'nin oluşturulması gerekir.
        - Eğer encode() çağrılmadan önce train() yapılmazsa, tokenizer'ın token-to-id mapping'i oluşturulmamış olur ve encode() işlemi başarısız olur.
        - Bu test, encode() metodunun train() çağrılmadan önce çalıştırılmasının engellendiğini doğrular ve tokenizer'ın geçersiz bir state'te encode() işlemi yapmasını önler.

    Input:
        "çocuklar"

    Beklenen:
        ValueError: "Tokenizer has not been trained yet"

    Bu test şunu garanti eder:
        - MorphemeTokenizer'ın encode() metodunun train() çağrılmadan önce çalıştırılmasının engellendiğini doğrular
        - Tokenizer'ın geçersiz bir state'te encode() işlemi yapması önlenir
        - Kullanıcıların encode() metodunu doğru sırayla kullanmaları sağlanır (önce train(), sonra encode())

    Önlenen bug sınıfı:
        - encode() metodunun train() çağrılmadan önce çalıştırılması
        - Tokenizer'ın geçersiz bir state'te encode() işlemi yapması (örneğin, token-to-id mapping'i oluşturulmadan encode() yapılması)
        - Kullanıcıların encode() metodunu yanlış sırayla kullanması (örneğin, encode() → train() gibi)
    """
    # Bu test, TokenizerFactory üzerinden "morpheme" tokenizer'ı oluşturulurken encode() metoduna "çocuklar" verilmesi durumunda ValueError bekler.
    # Hata mesajında "Tokenizer has not been trained yet" ifadesi aranır. 
    # Bu, MorphemeTokenizer'ın encode() metodunun train() çağrılmadan önce çalıştırılmasının engellendiğini doğrular ve tokenizer'ın geçersiz bir state'te encode() işlemi yapmasını önler.

    tokenizer = TokenizerFactory.create("morpheme")

    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.encode("çocuklar")


def test_morpheme_tokenizer_encode_returns_ids() -> None:
    """
    encode() integer id listesi döndürmelidir.

    Gerekçe:
        - MorphemeTokenizer'ın encode() metodunun, tokenize ettiği tokenları integer id'lere dönüştürerek bir liste döndürmesi beklenir.
        - Bu integer id'ler, model eğitimi ve diğer downstream işlemler için gereklidir.
        - Eğer encode() integer id listesi döndürmezse, bu durum downstream işlemlerde hatalara yol açabilir (örneğin, model eğitimi sırasında beklenen input formatının sağlanmaması gibi).

    Input:
        "çocuklar"
    
    Beklenen:
        encode() metodu bir liste döndürmelidir
        encode() tarafından döndürülen listenin tüm elemanları integer olmalıdır

    Bu test şunu garanti eder:
        - MorphemeTokenizer'ın encode() metodunun integer id listesi döndürmesi doğrulanır
        - Tokenların doğru şekilde integer id'lere dönüştürülmesi sağlanır  
        - Downstream işlemlerde beklenen input formatının sağlanması sağlanır (örneğin, model eğitimi sırasında integer id'lerin kullanılması gibi)

    Önlenen bug sınıfı:
        - encode() metodunun integer id listesi döndürmemesi (örneğin, encode() → ["çocuk", "lar"] gibi)
        - encode() tarafından döndürülen listenin elemanlarının integer olmaması (örneğin, encode() → ["0", "1"] gibi string id'ler döndürmesi)
        - Tokenların doğru şekilde integer id'lere dönüştürülmemesi (örneğin, encode() → [999, 1000] gibi beklenmeyen id'ler döndürmesi)
    """
    # Bu test, TokenizerFactory üzerinden "morpheme" tokenizer'ı oluşturulurken train() metoduna "çocuklar" verilmesi ve ardından encode() metoduna "çocuklar" verilmesi durumunda encode() metodunun bir liste döndürdüğünü ve bu listenin tüm elemanlarının integer olduğunu doğrular.
    # Bu, MorphemeTokenizer'ın encode() metodunun integer id listesi döndürmesi doğrular ve tokenların doğru şekilde integer id'lere dönüştürülmesini sağlar.
    # Eğer bu test başarısız olursa, tokenizer'ın encode() metodunun integer id listesi döndürmediği veya tokenların doğru şekilde integer id'lere dönüştürülmediği anlamına gelir ve bu durum downstream işlemlerde hatalara yol açabilir (örneğin, model eğitimi sırasında beklenen input formatının sağlanmaması gibi).

    tokenizer = TokenizerFactory.create("morpheme")

    tokenizer.train("çocuklar")

    encoded = tokenizer.encode("çocuklar")

    assert isinstance(encoded, list) # encode() metodunun bir liste döndürdüğünü doğrular.
    assert all(isinstance(i, int) for i in encoded) # encode() tarafından döndürülen listenin tüm elemanlarının integer olduğunu doğrular.


def test_morpheme_tokenizer_encode_unknown_token_raises_error() -> None:
    """
    Eğitimde görülmeyen token encode edilmemelidir.

    Gerekçe:
        - MorphemeTokenizer'ın encode() metodunun, eğitim sırasında görülmeyen tokenları encode etmeye çalıştığında ValueError raise etmesi beklenir.
        - Bu durum, tokenizer'ın bilinmeyen tokenları doğru şekilde işlemesini sağlar ve downstream işlemlerde beklenmeyen hataların önüne geçilir (örneğin, bilinmeyen tokenların yanlış id'lere dönüştürülmesi gibi).
        - Eğer encode() bilinmeyen tokenları encode etmeye çalıştığında hata raise etmezse, bu durum downstream işlemlerde hatalara yol açabilir (örneğin, model eğitimi sırasında beklenmeyen tokenların yanlış id'lere dönüştürülmesi gibi).

    Input:
        Eğitim: "çocuklar"

    Encode: "kitaplar"

    Beklenen:
        ValueError: "Unknown token: kitaplar"

    Bu test şunu garanti eder:
        - MorphemeTokenizer'ın encode() metodunun eğitim sırasında görülmeyen tokenları encode etmeye çalıştığında ValueError raise etmesi doğrulanır
        - Tokenizer'ın bilinmeyen tokenları doğru şekilde işlemesi sağlanır
        - Downstream işlemlerde beklenmeyen hataların önüne geçilir (örneğin, bilinmeyen tokenların yanlış id'lere dönüştürülmesi gibi)
        - Kullanıcıların encode() metodunu doğru şekilde kullanmaları sağlanır (örneğin, encode() → train() sırasının korunması gibi)

    Önlenen bug sınıfı:
        - encode() metodunun eğitim sırasında görülmeyen tokenları encode etmeye çalışması ve hata raise etmemesi
        - Bilinmeyen tokenların yanlış id'lere dönüştürülmesi (örneğin, "kitaplar" tokenının eğitim sırasında görülmemesine rağmen encode() → [999] gibi beklenmeyen id'ler döndürmesi)
        - Kullanıcıların encode() metodunu yanlış şekilde kullanması (örneğin, encode() → train() gibi)
    """
    # Bu test, TokenizerFactory üzerinden "morpheme" tokenizer'ı oluşturulurken train() metoduna "çocuklar" verilmesi ve ardından encode() metoduna "kitaplar" verilmesi durumunda ValueError bekler.
    # Hata mesajında "Unknown token: kitaplar" ifadesi aranır.

    tokenizer = TokenizerFactory.create("morpheme")

    tokenizer.train("çocuklar")

    with pytest.raises(ValueError, match="Unknown token"):
        tokenizer.encode("kitaplar")


# ---------------------------------------------------------
# DECODE TESTS
# ---------------------------------------------------------

def test_morpheme_tokenizer_decode_before_training_raises_error() -> None:
    """
    train() çağrılmadan decode() çalışmamalıdır.

    Gerekçe:
        - MorphemeTokenizer'ın decode() metodunun çalışabilmesi için öncelikle train() metodunun çağrılarak vocabulary'nin oluşturulması gerekir.
        - Eğer decode() çağrılmadan önce train() yapılmazsa, tokenizer'ın id-to-token mapping'i oluşturulmamış olur ve decode() işlemi başarısız olur.

    Input:
        [0, 1]

    Beklenen:
        ValueError: "Tokenizer has not been trained yet"

    Bu test şunu garanti eder:
        - MorphemeTokenizer'ın decode() metodunun train() çağrılmadan önce çalıştırılmasının engellendiğini doğrular
        - Tokenizer'ın geçersiz bir state'te decode() işlemi yapması önlenir
        - Kullanıcıların decode() metodunu doğru sırayla kullanmaları sağlanır (örneğin, decode() → train() gibi)

    Önlenen bug sınıfı:
        - decode() metodunun train() çağrılmadan önce çalıştırılması
        - Tokenizer'ın geçersiz bir state'te decode() işlemi yapması (örneğin, id-to-token mapping'i oluşturulmadan decode() yapılması)
        - Kullanıcıların decode() metodunu yanlış sırayla kullanması (örneğin, decode() → train() gibi)
    """
    # Bu test, TokenizerFactory üzerinden "morpheme" tokenizer'ı oluşturulurken decode() metoduna [0, 1] verilmesi durumunda ValueError bekler.
    # Hata mesajında "Tokenizer has not been trained yet" ifadesi aranır. 
    # Bu, MorphemeTokenizer'ın decode() metodunun train() çağrılmadan önce çalıştırılmasının engellendiğini doğrular ve tokenizer'ın geçersiz bir state'te decode() işlemi yapmasını önler.

    tokenizer = TokenizerFactory.create("morpheme")

    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.decode([0, 1])


def test_morpheme_tokenizer_decode_returns_joined_text() -> None:
    """
    decode() tokenları doğrudan birleştirir.

    Gerekçe:
        - MorphemeTokenizer'ın decode() metodunun, verilen token id'lerini karşılık gelen tokenlara dönüştürdükten sonra bu tokenları doğrudan birleştirerek tek bir string döndürmesi beklenir.
        - Bu durum, tokenizer'ın encode() ve decode() işlemleri arasında tutarlı bir şekilde çalışmasını sağlar.
        - Eğer decode() tokenları doğrudan birleştirmezse, bu durum downstream işlemlerde beklenmeyen sonuçlara yol açabilir (örneğin, decode() → "çocuklar" yerine "çocuk lar" gibi yanlış bir sonuç döndürmesi gibi).

    Input:
        "çocuklar"

    Beklenen:
        whitespace korunmaz

    Output:
        "çocuklar"

    Bu test şunu garanti eder:
        - MorphemeTokenizer'ın decode() metodunun tokenları doğrudan birleştirerek tek bir string döndürmesi doğrulanır
        - Tokenizer'ın encode() ve decode() işlemleri arasında tutarlı bir şekilde çalışması sağlanır
        - Downstream işlemlerde beklenmeyen sonuçların önüne geçilir (örneğin, decode() → "çocuklar" yerine "çocuk lar" gibi yanlış bir sonuç döndürmesi gibi)
        - Kullanıcıların decode() metodunun nasıl çalıştığını doğru şekilde anlamaları sağlanır (örneğin, decode() → tokenları doğrudan birleştirerek tek bir string döndürür gibi)

    Önlenen bug sınıfı:
        - decode() metodunun tokenları doğrudan birleştirmemesi (örneğin, decode() → "çocuk lar" gibi sonuçlar döndürmesi)
        - Tokenizer'ın encode() ve decode() işlemleri arasında tutarsızlık (örneğin, encode() → ["çocuk", "lar"] ve decode() → "çocuk lar" gibi sonuçlar döndürmesi)
        - Kullanıcıların decode() metodunun nasıl çalıştığını yanlış anlaması (örneğin, decode() → tokenları doğrudan birleştirerek tek bir string döndürür gibi)
    """
    # Bu test, TokenizerFactory üzerinden "morpheme" tokenizer'ı oluşturulurken train() metoduna "çocuklar" verilmesi ve ardından decode() metoduna encode("çocuklar") sonucunun verilmesi durumunda decode() metodunun "çocuklar" sonucunu döndürdüğünü doğrular.
    # Bu, MorphemeTokenizer'ın decode() metodunun tokenları doğrudan birleştirerek tek bir string döndürmesi doğrular ve tokenizer'ın encode() ve decode() işlemleri arasında tutarlı bir şekilde çalışmasını sağlar.
    # Eğer bu test başarısız olursa, tokenizer'ın decode() metodunun tokenları doğrudan birleştirmediği veya tokenizer'ın encode() ve decode() işlemleri arasında tutarsızlık olduğu anlamına gelir ve bu durum downstream işlemlerde beklenmeyen sonuçlara yol açabilir (örneğin, decode() → "çocuklar" yerine "çocuk lar" gibi yanlış bir sonuç döndürmesi gibi).

    tokenizer = TokenizerFactory.create("morpheme")

    tokenizer.train("çocuklar")

    decoded = tokenizer.decode(tokenizer.encode("çocuklar"))

    assert decoded == "çocuklar"


def test_morpheme_tokenizer_decode_unknown_id_raises_error() -> None:
    """
    Bilinmeyen token id decode edilmemelidir.

    Gerekçe:
        - MorphemeTokenizer'ın decode() metodunun, eğitim sırasında görülmeyen token id'lerini decode etmeye çalıştığında ValueError raise etmesi beklenir.
        - Bu durum, tokenizer'ın bilinmeyen token id'lerini doğru şekilde işlemesini sağlar ve downstream işlemlerde beklenmeyen hataların önüne geçilir (örneğin, bilinmeyen token id'lerinin yanlış tokenlara dönüştürülmesi gibi).
        - Eğer decode() bilinmeyen token id'lerini decode etmeye çalıştığında hata raise etmezse, bu durum downstream işlemlerde hatalara yol açabilir (örneğin, model eğitimi sırasında beklenmeyen token id'lerinin yanlış tokenlara dönüştürülmesi gibi).

    Input:
        Eğitim: "çocuklar"

    Decode: [999]

    Beklenen:
        ValueError: "Unknown token id: 999"

    Bu test şunu garanti eder:
        - MorphemeTokenizer'ın decode() metodunun eğitim sırasında görülmeyen token id'lerini decode etmeye çalıştığında ValueError raise etmesi doğrulanır
        - Tokenizer'ın bilinmeyen token id'lerini doğru şekilde işlemesi sağlanır
        - Downstream işlemlerde beklenmeyen hataların önüne geçilir (örneğin, bilinmeyen token id'lerinin yanlış tokenlara dönüştürülmesi gibi)
        - Kullanıcıların decode() metodunu doğru şekilde kullanmaları sağlanır (örneğin, decode() → train() sırasının korunması gibi)

    Önlenen bug sınıfı:
        - decode() metodunun eğitim sırasında görülmeyen token id'lerini decode etmeye çalışması ve hata raise etmemesi
        - Bilinmeyen token id'lerinin yanlış tokenlara dönüştürülmesi (örneğin, id 999'un eğitim sırasında görülmemesine rağmen decode() → "kitap" gibi beklenmeyen tokenlar döndürmesi)
        - Kullanıcıların decode() metodunu yanlış şekilde kullanması (örneğin, decode() → train() gibi)
    """
    # Bu test, TokenizerFactory üzerinden "morpheme" tokenizer'ı oluşturulurken train() metoduna "çocuklar" verilmesi ve ardından decode() metoduna [999] verilmesi durumunda ValueError bekler.
    # Hata mesajında "Unknown token id: 999" ifadesi aranır.

    tokenizer = TokenizerFactory.create("morpheme")

    tokenizer.train("çocuklar")

    with pytest.raises(ValueError, match="Unknown token id"):
        tokenizer.decode([999])


def test_morpheme_tokenizer_decode_empty_returns_empty_string() -> None:
    """
    Boş token listesi decode edildiğinde boş string dönmelidir.

    Gerekçe:
        - MorphemeTokenizer'ın decode() metodunun, boş bir token id listesi verildiğinde boş bir string döndürmesi beklenir.
        - Bu durum, tokenizer'ın geçersiz veya anlamsız inputlara karşı sağlam ve öngörülebilir davranmasını sağlar.
        - Eğer decode() boş token listesi için boş string döndürmezse, bu durum downstream işlemlerde beklenmeyen sonuçlara yol açabilir (örneğin, decode() → "" yerine "[UNK]" gibi yanlış bir sonuç döndürmesi gibi).

    Input:
        []

    Beklenen:
        ""

    Bu test şunu garanti eder:
        - MorphemeTokenizer'ın decode() metodunun boş bir token id listesi verildiğinde boş bir string döndürmesi doğrulanır
        - Tokenizer'ın geçersiz veya anlamsız inputlara karşı sağlam ve öngörülebilir davranması sağlanır
        - Downstream işlemlerde beklenmeyen sonuçların önüne geçilir (örneğin, decode() → "" yerine "[UNK]" gibi yanlış bir sonuç döndürmesi gibi)
        - Kullanıcıların decode() metodunun nasıl çalıştığını doğru şekilde anlamaları sağlanası sağlanır (örneğin, decode() → boş token listesi için boş string döndürür gibi)

    Önlenen bug sınıfı:
        - decode() metodunun boş bir token id listesi verildiğinde boş string döndürmemesi (örneğin, decode() → "[UNK]" gibi sonuçlar döndürmesi)
        - Tokenizer'ın geçersiz veya anlamsız inputlara karşı sağlam ve öngörülebilir davranmaması (örneğin, decode() → "" yerine "[UNK]" gibi yanlış bir sonuç döndürmesi gibi)
        - Kullanıcıların decode() metodunun nasıl çalıştığını yanlış anlaması (örneğin, decode() → boş token listesi için boş string döndürür gibi)
    """
    # Bu test, TokenizerFactory üzerinden "morpheme" tokenizer'ı oluşturulurken train() metoduna "çocuklar" verilmesi ve ardından decode() metoduna boş bir liste ([]) verilmesi durumunda decode() metodunun boş bir string ("") döndürdüğünü doğrular.
    # Bu, MorphemeTokenizer'ın decode() metodunun boş bir token id listesi verildiğinde boş bir string döndürmesi doğrular ve tokenizer'ın geçersiz veya anlamsız inputlara karşı sağlam ve öngörülebilir davranmasını sağlar.
    # Eğer bu test başarısız olursa, tokenizer'ın decode() metodunun boş bir token id listesi verildiğinde boş string döndürmediği anlamına gelir ve downstream işlemlerde beklenmeyen sonuçlara yol açabilir (örneğin, decode() → "" yerine "[UNK]" gibi yanlış bir sonuç döndürmesi gibi).

    tokenizer = TokenizerFactory.create("morpheme")

    tokenizer.train("çocuklar")

    assert tokenizer.decode([]) == "" # decode() metodunun boş bir token id listesi verildiğinde boş bir string ("") döndürdüğünü doğrular.


# ---------------------------------------------------------
# ROUNDTRIP TESTS
# ---------------------------------------------------------

def test_morpheme_tokenizer_roundtrip_single_word() -> None:
    """
    Tek kelime için encode → decode roundtrip korunmalıdır.

    Gerekçe:
        - MorphemeTokenizer'ın encode() ve decode() işlemleri arasında tek kelime için tutarlı bir şekilde çalışması beklenir.
        - "çocuklar" kelimesi encode edilip decode edildiğinde yine "çocuklar" sonucunu vermelidir, yani encode → decode roundtrip işlemi sırasında tek kelimenin doğru şekilde korunması sağlanmalıdır.
        - Bu durum, tokenizer'ın encode() ve decode() işlemleri arasında tutarlı bir şekilde çalıştığını doğrular ve downstream işlemlerde beklenmeyen sonuçların önüne geçilir (örneğin, encode() → ["çocuk", "lar"] ve decode() → "çocuk lar" gibi yanlış bir sonuç döndürmesi gibi).
        - Eğer encode() ve decode() işlemleri arasında tutarsızlık varsa, bu durum downstream işlemlerde hatalara yol açabilir (örneğin, model eğitimi sırasında beklenen token id'lerinin yanlış tokenlara dönüştürülmesi gibi).

    Input:
        "çocuklar"

    Beklenen:
        encode() → ["çocuk", "lar"]
        decode() → "çocuklar"
        
    Bu test şunu garanti eder:
        - MorphemeTokenizer'ın encode() ve decode() işlemleri arasında tek kelime için tutarlı bir şekilde çalışması doğrulanır
        - "çocuklar" kelimesi encode edilip decode edildiğinde yine "çocuklar" sonucunu vermelidir, yani encode → decode roundtrip işlemi sırasında tek kelimenin doğru şekilde korunması sağlanmalıdır
        - Tokenizer'ın encode() ve decode() işlemleri arasında tutarsızlık olması durumunda downstream işlemlerde hataların önüne geçilir (örneğin, encode() → ["çocuk", "lar"] ve decode() → "çocuk lar" gibi yanlış bir sonuç döndürmesi gibi)
        - Kullanıcıların encode() ve decode() işlemlerinin nasıl çalıştığını doğru şekilde anlamaları sağlanır (örneğin, encode() → ["çocuk", "lar"] ve decode() → "çocuklar" gibi)

    Önlenen bug sınıfı:
        - encode() ve decode() işlemleri arasında tek kelime için tutarsızlık (örneğin, encode() → ["çocuk", "lar"] ve decode() → "çocuk lar" gibi yanlış bir sonuç döndürmesi)
        - Tokenizer'ın encode() ve decode() işlemleri arasında tutarsızlık olması durumunda downstream işlemlerde hataların oluşması (örneğin, model eğitimi sırasında beklenen token id'lerinin yanlış tokenlara dönüştürülmesi gibi)
    """
    # Bu test, TokenizerFactory üzerinden "morpheme" tokenizer'ı oluşturulurken train() metoduna "çocuklar" verilmesi ve ardından decode() metoduna encode("çocuklar") sonucunun verilmesi durumunda decode() metodunun "çocuklar" sonucunu döndürdüğünü doğrular.
    # Bu, MorphemeTokenizer'ın encode() ve decode() işlemleri arasında tek kelime için tutarlı bir şekilde çalışması doğrular ve "çocuklar" kelimesi encode edilip decode edildiğinde yine "çocuklar" sonucunu vermelidir, yani encode → decode roundtrip işlemi sırasında tek kelimenin doğru şekilde korunması sağlanmalıdır.
    # Eğer bu test başarısız olursa, tokenizer'ın encode() ve decode() işlemleri arasında tek kelime için tutarsızlık olduğu anlamına gelir ve downstream işlemlerde hatalara yol açabilir (örneğin, model eğitimi sırasında beklenen token id'lerinin yanlış tokenlara dönüştürülmesi gibi).
    
    tokenizer = TokenizerFactory.create("morpheme")

    text = "çocuklar"

    tokenizer.train(text)

    decoded = tokenizer.decode(tokenizer.encode(text))

    assert decoded == text # encode → decode roundtrip işlemi sırasında tek kelimenin doğru şekilde korunmasını doğrular.


def test_morpheme_tokenizer_roundtrip_loses_whitespace() -> None:
    """
    Bu tokenizer whitespace bilgisini korumaz.

    Gerekçe:
        - MorphemeTokenizer'ın encode → decode roundtrip işlemi sırasında whitespace bilgisini korumadığı beklenir.
        - "çocuklar okulda" cümlesi encode edilip decode edildiğinde "çocuklarokulda" sonucunu vermelidir, yani whitespace karakterleri korunmaz.
        - Bu durum, tokenizer'ın whitespace karakterlerini tokenization sürecinde göz ardı ettiğini ve decode() metodunun tokenları doğrudan birleştirdiğini doğrular.
        - Eğer decode() whitespace bilgisini korursa, bu durum tokenizer'ın beklenmedik bir şekilde çalıştığını ve encode → decode roundtrip işlemi sırasında whitespace karakterlerini yanlış şekilde işlediğini gösterebilir.

    Input:
        "çocuklar okulda"

    Decode:
        "çocuklarokulda"

    Beklenen:
        - decode() metodunun whitespace bilgisini korumadığını doğrular (örneğin, decode() → "çocuklarokulda" gibi)
        - decode() metodunun whitespace bilgisini korumadığını doğrular (örneğin, decode() → "çocuklar okulda" gibi olmaması)

    Bu test şunu garanti eder:
        - MorphemeTokenizer'ın encode → decode roundtrip işlemi sırasında whitespace bilgisini korumadığı doğrulanır
        - Tokenizer'ın whitespace karakterlerini tokenization sürecinde göz ardı ettiği ve decode() metodunun tokenları doğrudan birleştirdiği doğrulanır
        - Kullanıcıların decode() metodunun nasıl çalıştığını doğru şekilde anlamaları sağlanır (örneğin, decode() → tokenları doğrudan birleştirerek tek bir string döndürür gibi)
        - Tokenizer'ın beklenmedik bir şekilde çalışması durumunda ortaya çıkabilecek sorunların önüne geçilir (örneğin, decode() → "çocuklar okulda" gibi olması durumunda downstream işlemlerde beklenmeyen sonuçlara yol açması gibi)
        - Eklemeli dillerdeki tokenization sürecinin beklenen şekilde çalıştığını doğrular (örneğin, "çocuklar okulda" cümlesinin encode → decode roundtrip işlemi sırasında whitespace karakterlerinin korunmaması gibi)

    Önlenen bug sınıfı:
        - decode() metodunun whitespace bilgisini koruması (örneğin, decode() → "çocuklar okulda" gibi olması)
        - Tokenizer'ın beklenmedik bir şekilde çalışması (örneğin, decode() → "çocuklar okulda" gibi olması durumunda downstream işlemlerde beklenmeyen sonuçlara yol açması gibi)
    """
    tokenizer = TokenizerFactory.create("morpheme")

    text = "çocuklar okulda"

    tokenizer.train(text)

    decoded = tokenizer.decode(tokenizer.encode(text))

    assert decoded == "çocuklarokulda" # decode() metodunun whitespace bilgisini korumadığını doğrular.
    assert decoded != text # decode() metodunun whitespace bilgisini korumadığını doğrular.


def test_morpheme_tokenizer_vocab_size_zero_before_training() -> None:
    """
    Eğitim öncesi vocab boş olmalıdır.

    Gerekçe:
        - MorphemeTokenizer'ın train() metodunu çağırmadan önce vocab'ının boş olduğunu doğrulamak önemlidir.
        - Bu durum, tokenizer'ın eğitim sırasında vocab'ını doğru şekilde oluşturduğunu ve başlangıçta herhangi bir token içerdiğini göstermediğini doğrular.
        - Eğer vocab başlangıçta boş değilse, bu durum tokenizer'ın beklenmedik bir state'te olduğunu ve train() metodunun vocab'ı doğru şekilde oluşturmadığını gösterebilir.

    Input:
        Eğitim öncesi herhangi bir input

    Beklenen:
        vocab_size == 0

    Bu test şunu garanti eder:
        - MorphemeTokenizer'ın train() metodunu çağırmadan önce vocab'ının boş olduğunu doğrular
        - Tokenizer'ın eğitim sırasında vocab'ını doğru şekilde oluşturduğunu ve başlangıçta herhangi bir token içerdiğini göstermediğini doğrular
        - Tokenizer'ın beklenmedik bir state'te olmadığını ve train() metodunun vocab'ı doğru şekilde oluşturduğunu gösterir

    Önlenen bug sınıfı:
        - vocab'ın eğitim öncesi boş olmaması (örneğin, vocab_size == 10 gibi başlangıçta tokenlar içermesi)
        - Tokenizer'ın beklenmedik bir state'te olması (örneğin, train() metodunun vocab'ı doğru şekilde oluşturmadığı ve başlangıçta tokenlar içerdiği gibi)
    """
    # Bu test, TokenizerFactory üzerinden "morpheme" tokenizer'ı oluşturulurken tokenizer.vocab_size özelliğinin 0 olduğunu doğrular. 
    # Bu, MorphemeTokenizer'ın train() metodunu çağırmadan önce vocab'ının boş olduğunu doğrular ve tokenizer'ın eğitim sırasında vocab'ını doğru şekilde oluşturduğunu ve başlangıçta herhangi bir token içerdiğini göstermediğini doğrular. 
    # Eğer bu test başarısız olursa, tokenizer'ın vocab'ının eğitim öncesi boş olmadığı anlamına gelir ve bu durum tokenizer'ın beklenmedik bir state'te olduğunu ve train() metodunun vocab'ı doğru şekilde oluşturmadığını gösterebilir.

    tokenizer = TokenizerFactory.create("morpheme")

    assert tokenizer.vocab_size == 0 # Eğitim öncesi vocab'ın boş olduğunu doğrular.