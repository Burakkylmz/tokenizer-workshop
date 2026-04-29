from __future__ import annotations

import pytest

from tokenizer_workshop.api.services.tokenizer_factory import TokenizerFactory


# ---------------------------------------------------------
# INIT TESTS
# ---------------------------------------------------------

def test_subword_tokenizer_init_invalid_subword_size_raises_error() -> None:
    """
    Guardrail testi: subword_size en az 1 olmalıdır.

    Gerekçe:
        SubwordTokenizer kelimeleri sabit uzunluklu parçalara böler.
        subword_size = 0 olursa range(..., step=0) gibi geçersiz bir durum oluşur.

    Bu test şunu garanti eder:
        - Geçersiz konfigürasyon erken aşamada reddedilir.
        - Tokenizer anlamsız bir state ile oluşturulmaz.

    Örnek:
        TokenizerFactory.create("subword", subword_size=0)  # ValueError

    Önlenen bug sınıfı:
        - subword_size = 0 veya negatif değerler tokenizer'ın çalışmasını bozabilir.
        - Bu durum encode/decode sırasında sonsuz döngülere veya hatalara yol açabilir. 
    """
    # subword_size = 0 veya negatif değerler geçersizdir.
    # Bu test, bu tür hatalı konfigürasyonların erken aşamada yakalanmasını sağlar.

    with pytest.raises(ValueError, match="subword_size must be at least 1"):
        TokenizerFactory.create("subword", subword_size=0)


# ---------------------------------------------------------
# TRAIN TESTS
# ---------------------------------------------------------

def test_subword_tokenizer_train_with_empty_text_raises_error() -> None:
    """
    Boş metinle train() çağrıldığında ValueError beklenmelidir.

    Gerekçe:
        Vocabulary oluşturmak için en az bir gerçek token gerekir.
        Boş string herhangi bir subword veya punctuation tokenı üretmez.

    Bu test şunu garanti eder:
        - Eğitim için anlamlı bir metin sağlanmalıdır.
        - Tokenizer'ın anlamsız bir vocabulary ile eğitilmesi engellenir.

    Örnek:
        tokenizer.train("")  # ValueError

    Önlenen bug sınıfı:
        - Boş metinle train() çağrıldığında tokenizer'ın boş bir vocabulary oluşturması ve encode/decode işlemlerinin hatalı sonuçlar üretmesi.
        - Bu durum downstream işlemlerde sessizce hatalara yol açabilir.
    """
    # testin amacı, boş metinle train() çağrıldığında ValueError beklenmesini sağlamaktır.
    # Bu test, tokenizer'ın anlamsız bir vocabulary oluşturmasını ve encode/decode işlemlerinin hatalı sonuçlar üretmesini engeller.

    tokenizer = TokenizerFactory.create("subword")

    with pytest.raises(ValueError, match="Training text cannot be empty"):
        tokenizer.train("")


def test_subword_tokenizer_train_with_whitespace_text_raises_error() -> None:
    """
    Sadece whitespace içeren metinle train() yapılmamalıdır.

    Gerekçe:
        Whitespace karakterleri token olarak saklanmaz.
        Bu yüzden "   " gibi inputlar anlamlı vocabulary üretemez.

    Bu test şunu garanti eder:
        - Eğitim metni en az bir gerçek token içermelidir.
        - Tokenizer'ın anlamsız bir vocabulary ile eğitilmesi engellenir.

    Örnek:
        tokenizer.train("   ")  # ValueError
    
    Önlenen bug sınıfı:
        - Sadece whitespace içeren metinle train() çağrıldığında tokenizer'ın boş bir vocabulary oluşturması ve encode/decode işlemlerinin hatalı sonuçlar üretmesi.
        - Bu durum downstream işlemlerde sessizce hatalara yol açabilir.
    """
    # testin amacı, sadece whitespace içeren metinle train() çağrıldığında ValueError beklenmesini sağlamaktır.
    # Bu test, tokenizer'ın anlamsız bir vocabulary oluşturmasını ve encode/decode işlemlerinin hatalı sonuçlar üretmesini engeller.

    tokenizer = TokenizerFactory.create("subword")

    with pytest.raises(ValueError, match="Training text cannot be empty"):
        tokenizer.train("   ")


def test_subword_tokenizer_train_builds_vocab() -> None:
    """
    train() sonrası vocabulary oluşturulmalıdır.

    Gerekçe:
        train() metni tokenize eder, unique tokenları belirler ve token -> id mapping oluşturur.
        Bu mapping encode/decode işlemleri için gereklidir.

    Input:
        "token"

    subword_size = 3

    Tokenlar:
        ["tok", "en"]

    Beklenen vocab_size: 2

    Bu test şunu garanti eder:  
        - train() metni işleyerek vocabulary oluşturur.
        - Vocabulary, tokenize edilen subword tokenlarının unique setini yansıtmalıdır.

    Önlenen bug sınıfı:
        - train() çağrıldığında tokenizer'ın vocabulary oluşturamaması veya yanlış tokenları içermesi.
        - Bu durum encode/decode işlemlerinin hatalı sonuçlar üretmesine yol açabilir.
    """
    # testin amacı, train() çağrıldığında tokenizer'ın metni işleyerek vocabulary oluşturduğunu ve bu vocabulary'nin tokenize edilen subword tokenlarının unique setini yansıttığını doğrulamaktır.
    # Bu test, tokenizer'ın train() fonksiyonunun temel işlevselliğini doğrular ve encode/decode işlemlerinin doğru çalışması için gerekli olan token -> id mapping'in oluşturulmasını sağlar.

    tokenizer = TokenizerFactory.create("subword", subword_size=3)

    tokenizer.train("token")

    assert tokenizer.vocab_size == 2 # "tok" ve "en" tokenları  


def test_subword_tokenizer_vocab_reflects_unique_subwords() -> None:
    """
    vocab_size unique subword token sayısını yansıtmalıdır.

    Gerekçe:
        Vocabulary, tokenize edilen subword tokenlarının unique setini yansıtmalıdır.

    Input:
        "token tokenization"

    subword_size = 3

    Tokenlar:
        ["tok", "en", "tok", "eni", "zat", "ion"]

    Unique tokenlar:
        ["tok", "en", "eni", "zat", "ion"]

    Beklenen vocab_size: 5

    Bu test şunu garanti eder:
        - Vocabulary, tokenize edilen subword tokenlarının unique setini yansıtmalıdır.
        - Aynı subword tokenı birden fazla kez görünse bile vocabulary'de tek bir entry olarak tutulmalıdır.

    Önlenen bug sınıfı:
        - train() çağrıldığında tokenizer'ın vocabulary oluştururken duplicate subword tokenlarını temizlememesi ve bu tokenları birden fazla kez kaydetmesi.
        - Bu durum encode/decode işlemlerinin hatalı sonuçlar üretmesine ve gereksiz yere büyük bir vocabulary oluşmasına yol açabilir.
    """
    # testin amacı, train() çağrıldığında tokenizer'ın vocabulary oluştururken duplicate subword tokenlarını temizlediğini ve aynı subword tokenının birden fazla kez görünse bile vocabulary'de tek bir entry olarak tutulduğunu doğrulamaktır.
    # Bu test, tokenizer'ın train() fonksiyonunun vocabulary oluştururken duplicate tokenları temizleme işlevselliğini doğrular ve encode/decode işlemlerinin doğru çalışması için gerekli olan token -> id mapping'in oluşturulmasını sağlar.

    tokenizer = TokenizerFactory.create("subword", subword_size=3)

    tokenizer.train("token tokenization")

    assert tokenizer.vocab_size == 5 # "tok", "en", "eni", "zat", "ion" tokenları (duplicate "tok" tek bir entry olarak sayılır)    


def test_subword_tokenizer_vocab_is_deterministic_for_same_input() -> None:
    """
    Aynı input ile eğitilen iki tokenizer aynı mapping'i üretmelidir.

    Gerekçe:
        Deterministik vocabulary davranışı testlerin, raporların ve compare çıktılarının stabil kalmasını sağlar.

    Input:
        "token tokenization"

    subword_size = 3

    Tokenlar:
        ["tok", "en", "tok", "eni", "zat", "ion"]

    Unique tokenlar:
        ["tok", "en", "eni", "zat", "ion"]

    Bu test şunu garanti eder:
        - Aynı eğitim metni ve aynı subword_size ile eğitilen tokenizer'lar aynı token -> id ve id -> token mapping'lerini oluşturmalıdır.
        - Bu durum, tokenizer'ların deterministic olduğunu ve aynı input için aynı vocabulary'yi ürettiğini doğrular.   

    Önlenen bug sınıfı:
        - train() çağrıldığında tokenizer'ın vocabulary oluştururken rastgelelik içermesi ve aynı input için farklı token -> id mapping'leri üretmesi.
        - Bu durum testlerin, raporların ve compare çıktılarının stabil kalmamasına ve güvenilir olmamasına yol açabilir.   
    """
    # testin amacı, aynı eğitim metni ve aynı subword_size ile eğitilen iki tokenizer'ın aynı token -> id ve id -> token mapping'lerini oluşturduğunu doğrulamaktır.
    # Bu test, tokenizer'ların deterministic olduğunu ve aynı input için aynı vocabulary'yi ürettiğini doğrular.

    tokenizer_a = TokenizerFactory.create("subword", subword_size=3)
    tokenizer_b = TokenizerFactory.create("subword", subword_size=3)

    text = "token tokenization tokenizer"

    tokenizer_a.train(text)
    tokenizer_b.train(text)

    assert tokenizer_a._token_to_id == tokenizer_b._token_to_id # token -> id mapping'leri aynı olmalıdır
    assert tokenizer_a._id_to_token == tokenizer_b._id_to_token # id -> token mapping'leri aynı olmalıdır


# ---------------------------------------------------------
# TOKENIZE TESTS
# ---------------------------------------------------------

def test_subword_tokenizer_tokenize_splits_word_into_fixed_size_chunks() -> None:
    """
    tokenize() kelimeyi sabit uzunluklu subword parçalarına bölmelidir.

    Gerekçe:
        SubwordTokenizer, kelimeleri subword_size parametresine göre sabit uzunluklu parçalara böler.   

    Input:
        "tokenization"

    subword_size = 3

    Beklenen:
        ["tok", "eni", "zat", "ion"]

    Bu test şunu garanti eder:
        - tokenize() kelimeyi subword_size parametresine göre sabit uzunluklu parçalara bölmelidir.
        - Eğer kelime uzunluğu subword_size'a tam bölünmezse son parça daha kısa kalmalıdır.    

    Önlenen bug sınıfı:
        - tokenize() kelimeyi subword_size parametresine göre bölmemesi veya yanlış bölmesi.
        - Bu durum encode/decode işlemlerinin hatalı sonuçlar üretmesine yol açabilir.    
    """
    # testin amacı, tokenize() çağrıldığında kelimenin subword_size parametresine göre sabit uzunluklu parçalara bölündüğünü doğrulamaktır.
    # Bu test, tokenize() fonksiyonunun temel işlevselliğini doğrular ve encode/decode işlemlerinin doğru çalışması için gerekli olan tokenization sürecinin doğru olduğunu sağlar.

    tokenizer = TokenizerFactory.create("subword", subword_size=3)

    tokens = tokenizer.tokenize("tokenization")

    assert tokens == ["tok", "eni", "zat", "ion"] # "tokenization" kelimesi "tok", "eni", "zat", "ion" olarak bölünmelidir (subword_size=3) 
    

def test_subword_tokenizer_tokenize_keeps_short_final_chunk() -> None:
    """
    Kelime uzunluğu subword_size'a tam bölünmezse son parça daha kısa kalmalıdır.

    Gerekçe:
        SubwordTokenizer, kelimeleri subword_size parametresine göre sabit uzunluklu parçalara böler.   
        Eğer kelime uzunluğu subword_size'a tam bölünmezse son parça daha kısa kalmalıdır.  

    Input:
        "hello"

    subword_size = 3

    Beklenen:
        ["hel", "lo"]

    Bu test şunu garanti eder:
        - tokenize() kelimeyi subword_size parametresine göre bölmelidir.
        - Eğer kelime uzunluğu subword_size'a tam bölünmezse son parça daha kısa kalmalıdır.    

    Önlenen bug sınıfı:
        - tokenize() kelimeyi subword_size parametresine göre bölmemesi veya yanlış bölmesi, özellikle son parçanın olması gerektiği gibi kısa kalmaması.   
        - Bu durum encode/decode işlemlerinin hatalı sonuçlar üretmesine yol açabilir.
    """
    # testin amacı, tokenize() çağrıldığında kelimenin subword_size parametresine göre bölündüğünü ve eğer kelime uzunluğu subword_size'a tam bölünmezse son parçanın daha kısa kaldığını doğrulamaktır.
    # Bu test, tokenize() fonksiyonunun subword_size parametresine göre bölme işlemini doğru yaptığını ve son parçanın gerektiği gibi kısa kaldığını doğrular.

    tokenizer = TokenizerFactory.create("subword", subword_size=3)

    tokens = tokenizer.tokenize("hello")

    assert tokens == ["hel", "lo"] # "hello" kelimesi "hel" ve "lo" olarak bölünmelidir (subword_size=3, son parça daha kısa kalmalıdır)    
    

def test_subword_tokenizer_tokenize_lowercases_text() -> None:
    """
    tokenize() metni lowercase normalize etmelidir.

    Gerekçe:
        "Token" ve "token" farklı vocabulary entry'leri üretmemelidir.
        Lowercase normalization, tokenizer'ın case-insensitive davranmasını sağlar.

    Input:
        "Token" 

    subword_size = 3

    Beklenen:
        ["tok", "en"]

    Bu test şunu garanti eder:
        - tokenize() metni lowercase normalize etmelidir.
        - "Token" ve "token" aynı tokenları üretmelidir.

    Önlenen bug sınıfı:
        - tokenize() metni lowercase normalize etmezse, aynı kelime farklı tokenlar üretebilir.
        - Bu durum encode/decode işlemlerinin hatalı sonuçlar üretmesine yol açabilir.  
    """
    # testin amacı, tokenize() çağrıldığında metnin lowercase normalize edildiğini doğrulamaktır.
    # Bu test, tokenize() fonksiyonunun case-insensitive olduğunu ve aynı kelimenin farklı case'lerde aynı tokenları ürettiğini doğrular.

    tokenizer = TokenizerFactory.create("subword", subword_size=3)

    tokens = tokenizer.tokenize("TOKEN")

    assert tokens == ["tok", "en"] # "TOKEN" kelimesi lowercase normalize edilerek "token" olarak işlenmeli ve "tok", "en" olarak bölünmelidir (subword_size=3) 
    

def test_subword_tokenizer_tokenize_empty_string_returns_empty_list() -> None:
    """
    Boş string tokenize edildiğinde boş liste dönmelidir.

    Gerekçe:
        tokenize() boş veya sadece whitespace içeren string'ler için token üretmemelidir.   
        Bu durum, encode/decode işlemlerinin hatalı sonuçlar üretmesini engeller.

    Input:
        ""

    subword_size = 3

    Beklenen: []

    Bu test şunu garanti eder:
        - tokenize() boş string için boş liste döndürmelidir.

    Önlenen bug sınıfı:
        - tokenize() boş string için token üretirse, encode/decode işlemleri hatalı sonuçlar üretebilir.
    """
    # testin amacı, tokenize() çağrıldığında boş string için boş liste döndürüldüğünü doğrulamaktır.
    # Bu test, tokenize() fonksiyonunun boş string'ler için token üretmediğini ve encode/decode işlemlerinin hatalı sonuçlar üretmesini engellediğini doğrular.

    tokenizer = TokenizerFactory.create("subword")

    assert tokenizer.tokenize("") == [] # boş string tokenize edildiğinde boş liste dönmelidir


def test_subword_tokenizer_tokenize_whitespace_only_returns_empty_list() -> None:
    """
    Sadece whitespace içeren input tokenize edildiğinde boş liste dönmelidir.

    Gerekçe:
        tokenize() boş veya sadece whitespace içeren string'ler için token üretmemelidir.
        Bu durum, encode/decode işlemlerinin hatalı sonuçlar üretmesini engeller.

    Input:
        "   \n\t"
    
    subword_size = 3
    
    Beklenen: []
    
    Bu test şunu garanti eder:
        - tokenize() sadece whitespace içeren string'ler için boş liste döndürmelidir.
    
    Önlenen bug sınıfı:
        - tokenize() sadece whitespace içeren string'ler için token üretirse, encode/decode işlemleri hatalı sonuçlar üretebilir.   
        - Bu durum downstream işlemlerde sessizce hatalara yol açabilir.
    """
    # testin amacı, tokenize() çağrıldığında sadece whitespace içeren string'ler için boş liste döndürüldüğünü doğrulamaktır.
    # Bu test, tokenize() fonksiyonunun sadece whitespace içeren string'ler için token üretmediğini ve encode/decode işlemlerinin hatalı sonuçlar üretmesini engellediğini doğrular.

    tokenizer = TokenizerFactory.create("subword")

    assert tokenizer.tokenize("   \n\t") == [] # sadece whitespace içeren input tokenize edildiğinde boş liste dönmelidir


def test_subword_tokenizer_preserves_punctuation_as_separate_tokens() -> None:
    """
    Noktalama işaretleri subword parçalama işlemine dahil edilmemelidir.

    Gerekçe:
        SubwordTokenizer, kelimeleri subword_size parametresine göre sabit uzunluklu parçalara böler.
        Noktalama işaretleri kelime parçalama işlemine dahil edilmemelidir, çünkü bu işaretler genellikle kelime parçalama kurallarına uymayan tek karakter tokenları olarak kalmalıdır.    

    Input:
        "Hello, world!"

    subword_size = 3

    Beklenen:
        ["hel", "lo", ",", "wor", "ld", "!"]

    Bu test şunu garanti eder:
        - tokenize() noktalama işaretlerini subword parçalama işlemine dahil etmemelidir.
        - Noktalama işaretleri tek karakter tokenları olarak kalmalıdır.
        - Bu durum, encode/decode işlemlerinin hatalı sonuçlar üretmesini engeller.

    Önlenen bug sınıfı:
        - tokenize() noktalama işaretlerini subword parçalama işlemine dahil ederse, bu işaretler kelime parçalama kurallarına uymayan tek karakter tokenları olarak kalmaz ve encode/decode işlemlerinin hatalı sonuçlar üretmesine yol açabilir.  
        - Bu durum downstream işlemlerde sessizce hatalara yol açabilir.
    """
    # testin amacı, tokenize() çağrıldığında noktalama işaretlerinin subword parçalama işlemine dahil edilmediğini ve tek karakter tokenları olarak kaldığını doğrulamaktır.
    # Bu test, tokenize() fonksiyonunun noktalama işaretlerini doğru şekilde işlediğini ve encode/decode işlemlerinin hatalı sonuçlar üretmesini engellediğini doğrular.

    tokenizer = TokenizerFactory.create("subword", subword_size=3)

    tokens = tokenizer.tokenize("Hello, world!")

    assert tokens == ["hel", "lo", ",", "wor", "ld", "!"] # "Hello, world!" kelimesi "hel", "lo", ",", "wor", "ld", "!" olarak bölünmelidir (subword_size=3, noktalama işaretleri ayrı tokenlar olarak kalmalıdır)


def test_subword_tokenizer_handles_tabs_newlines_and_spaces() -> None:
    """
    Whitespace türleri token olarak saklanmamalıdır.

    Gerekçe:
        SubwordTokenizer, kelimeleri subword_size parametresine göre sabit uzunluklu parçalara böler.
        Whitespace karakterleri token olarak saklanmamalıdır, çünkü bu karakterler genellikle kelime parçalama kurallarına uymayan tek karakter tokenları olarak kalmalıdır.    

    Input:
        "Hello\\tworld\\nagain"

    subword_size = 3

    Beklenen:
        ["hel", "lo", "wor", "ld", "aga", "in"]

    Bu test şunu garanti eder:
        - tokenize() whitespace karakterlerini token olarak saklamamalıdır.
        - Whitespace karakterleri tek karakter tokenları olarak kalmalıdır.
        - Bu durum, encode/decode işlemlerinin hatalı sonuçlar üretmesini engeller.

    Önlenen bug sınıfı:
        - tokenize() whitespace karakterlerini token olarak saklarsa, bu karakterler kelime parçalama kurallarına uymayan tek karakter tokenları olarak kalmaz ve encode/decode işlemlerinin hatalı sonuçlar üretmesine yol açabilir.  
        - Bu durum downstream işlemlerde sessizce hatalara yol açabilir.
    """
    # testin amacı, tokenize() çağrıldığında whitespace karakterlerinin token olarak saklanmadığını ve tek karakter tokenları olarak kaldığını doğrulamaktır.
    # Bu test, tokenize() fonksiyonunun whitespace karakterlerini doğru şekilde işlediğini ve encode/decode işlemlerinin hatalı sonuçlar üretmesini engellediğini doğrular.

    tokenizer = TokenizerFactory.create("subword", subword_size=3)

    tokens = tokenizer.tokenize("Hello\tworld\nagain")

    assert tokens == ["hel", "lo", "wor", "ld", "aga", "in"] # "Hello\tworld\nagain" kelimesi "hel", "lo", "wor", "ld", "aga", "in" olarak bölünmelidir (subword_size=3, whitespace karakterleri token olarak saklanmamalıdır)  
    

def test_subword_tokenizer_handles_turkish_characters() -> None:
    """
    Türkçe karakterler kelime tokenı içinde korunmalıdır.

    Gerekçe:
        Tokenizer karakterleri ASCII'ye çevirmemeli veya kaybetmemelidir.
        Yalnızca lowercase ve sabit uzunluklu parçalama yapmalıdır.

    Input:
        "Çalışma"

    subword_size = 3

    Beklenen:
        ["çal", "ışm", "a"]

    Bu test şunu garanti eder:
        - tokenize() Türkçe karakterleri kelime tokenı içinde korumalıdır.
        - tokenize() yalnızca lowercase ve sabit uzunluklu parçalama yapmalıdır.
        - Bu durum, encode/decode işlemlerinin hatalı sonuçlar üretmesini engeller.

    Önlenen bug sınıfı:
        - tokenize() Türkçe karakterleri ASCII'ye çevirmeye çalışırsa, bu karakterler kaybolabilir veya yanlış temsil edilebilir ve encode/decode işlemlerinin hatalı sonuçlar üretmesine yol açabilir.  
        - Bu durum downstream işlemlerde sessizce hatalara yol açabilir.
    """
    # testin amacı, tokenize() çağrıldığında Türkçe karakterlerin kelime tokenı içinde korunduğunu doğrulamaktır.
    # Bu test, tokenize() fonksiyonunun Türkçe karakterleri doğru şekilde işlediğini ve encode/decode işlemlerinin hatalı sonuçlar üretmesini engellediğini doğrular.

    tokenizer = TokenizerFactory.create("subword", subword_size=3)

    tokens = tokenizer.tokenize("Çalışma")

    assert tokens == ["çal", "ışm", "a"] # "Çalışma" kelimesi lowercase normalize edilerek "çalışma" olarak işlenmeli ve "çal", "ışm", "a" olarak bölünmelidir (subword_size=3, Türkçe karakterler korunmalıdır)


def test_subword_tokenizer_supports_custom_subword_size() -> None:
    """
    subword_size parametresi parçalama uzunluğunu değiştirmelidir.

    Gerekçe:
        SubwordTokenizer, kelimeleri subword_size parametresine göre sabit uzunluklu parçalara böler.
        Bu test, farklı subword_size değerlerinin tokenize() çıktısını etkilediğini doğrular.

    Input:
        "tokenizer"

    subword_size = 4

    Beklenen:
        ["toke", "nize", "r"]

    Bu test şunu garanti eder:
        - subword_size parametresi parçalama uzunluğunu değiştirmelidir.
        - Eğer kelime uzunluğu subword_size'a tam bölünmezse son parça daha kısa kalmalıdır.

    Önlenen bug sınıfı:
        - subword_size parametresi tokenize() çıktısını etkilemezse, bu durum encode/decode işlemlerinin hatalı sonuçlar üretmesine yol açabilir.
         - Bu durum downstream işlemlerde sessizce hatalara yol açabilir.
    """
    # testin amacı, subword_size parametresinin tokenize() çıktısını etkilediğini doğrulamaktır.
    # Bu test, subword_size parametresinin parçalama uzunluğunu değiştirdiğini ve eğer kelime uzunluğu subword_size'a tam bölünmezse son parçanın daha kısa kaldığını doğrular.

    tokenizer = TokenizerFactory.create("subword", subword_size=4)

    tokens = tokenizer.tokenize("tokenizer")

    assert tokens == ["toke", "nize", "r"] # "tokenizer" kelimesi "toke", "nize", "r" olarak bölünmelidir (subword_size=4, son parça daha kısa kalmalıdır)  
    

# ---------------------------------------------------------
# ENCODE TESTS
# ---------------------------------------------------------

def test_subword_tokenizer_encode_before_training_raises_error() -> None:
    """
    train() çağrılmadan encode() çalışmamalıdır.

    Gerekçe:
        Encode işlemi token -> id mapping'e ihtiyaç duyar.
        Bu mapping train() sırasında oluşturulur.

    Bu test şunu garanti eder:
        - train() çağrılmadan encode() çalışmamalıdır.
        - Bu durum, tokenizer'ın anlamsız bir state ile encode() işlemi yapmasını engeller.
    
    Önlenen bug sınıfı:
        - train() çağrılmadan encode() çalışırsa, tokenizer'ın token -> id mapping'ine sahip olmaması nedeniyle hatalı sonuçlar üretmesine veya beklenmedik hatalar vermesine yol açabilir.
        - Bu durum downstream işlemlerde sessizce hatalara yol açabilir.
    """
    # testin amacı, train() çağrılmadan encode() çalıştırıldığında ValueError beklenmesini sağlamaktır.
    # Bu test, tokenizer'ın anlamsız bir state ile encode() işlemi yapmasını engeller ve encode() fonksiyonunun token -> id mapping'e ihtiyaç duyduğunu doğrular.

    tokenizer = TokenizerFactory.create("subword", subword_size=3)

    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.encode("tokenization")


def test_subword_tokenizer_encode_returns_integer_ids() -> None:
    """
    encode() integer token id listesi döndürmelidir.

    Gerekçe:
        encode() tokenları integer id'lere dönüştürmelidir.

    Train text:
        "tokenization"

    Tokenlar:
        ["tok", "eni", "zat", "ion"]

    Beklenen encode:
        [0, 1, 2, 3]

    Bu test şunu garanti eder:
        - encode() tokenları integer id'lere dönüştürmelidir.
        - encode() çıktısı bir liste olmalıdır.
        - encode() çıktısındaki her token id bir integer olmalıdır.
        - encode() çıktısındaki token id'ler, train() sırasında oluşturulan token -> id mapping'ine göre doğru id'leri döndürmelidir.

    Önlenen bug sınıfı:
        - encode() tokenları integer id'lere dönüştürmezse, bu durum downstream işlemlerde hatalı sonuçlar üretmesine veya beklenmedik hatalar vermesine yol açabilir.
        - Bu durum downstream işlemlerde sessizce hatalara yol açabilir.       
    """
    # testin amacı, encode() çağrıldığında tokenların integer id'lere dönüştürüldüğünü doğrulamaktır.
    # Bu test, encode() fonksiyonunun tokenları integer id'lere dönüştürdüğünü, çıktısının bir liste olduğunu, her token id'nin bir integer olduğunu ve token id'lerin train() sırasında oluşturulan token -> id mapping'ine göre doğru id'leri döndürdüğünü doğrular.

    tokenizer = TokenizerFactory.create("subword", subword_size=3)

    tokenizer.train("tokenization")

    encoded = tokenizer.encode("tokenization")

    assert isinstance(encoded, list) # encode() çıktısı bir liste olmalıdır
    assert encoded == [0, 1, 2, 3] # "tok", "eni", "zat", "ion" tokenları sırasıyla 0, 1, 2, 3 id'lerine dönüştürülmelidir
    assert all(isinstance(token_id, int) for token_id in encoded) # encode() çıktısındaki her token id bir integer olmalıdır    
    

def test_subword_tokenizer_encode_unknown_token_raises_error() -> None:
    """
    Eğitim sırasında görülmeyen subword token encode edilmeye çalışılırsa hata vermelidir.

    Gerekçe:
        encode() yalnızca train() sırasında oluşturulan token -> id mapping'ine göre tokenları id'lere dönüştürmelidir.
        Eğitim sırasında görülmeyen tokenlar mapping'de olmayacağı için encode edilmeye çalışıldığında hata vermelidir. 

    Train text:
        "token"

    Encode text:
        "python"

    "python" subword_size=3 ile:
        ["pyt", "hon"]

    Bu parçalar vocabulary içinde olmadığı için ValueError beklenir.

    Bu test şunu garanti eder:
        - encode() yalnızca train() sırasında oluşturulan token -> id mapping'ine göre tokenları id'lere dönüştürmelidir.
        - Eğitim sırasında görülmeyen tokenlar encode edilmeye çalışıldığında hata vermelidir.
        - Bu durum, tokenizer'ın anlamsız tokenları sessizce encode etmesini engeller ve downstream işlemlerde hatalı sonuçlar üretmesini engeller.

    Önlenen bug sınıfı:
        - encode() eğitim sırasında görülmeyen tokenları sessizce encode etmeye çalışırsa, bu tokenlar mapping'de olmadığı için hatalı id'lere dönüştürülmesine veya beklenmedik hatalar vermesine yol açabilir.
        - Bu durum downstream işlemlerde sessizce hatalara yol açabilir.
    """
    # testin amacı, encode() çağrıldığında eğitim sırasında görülmeyen tokenların encode edilmeye çalışıldığında ValueError beklenmesini sağlamaktır.
    # Bu test, encode() fonksiyonunun yalnızca train() sırasında oluşturulan token -> id mapping'ine göre tokenları id'lere dönüştürdüğünü ve eğitim sırasında görülmeyen tokenların encode edilmeye çalışıldığında hata verdiğini doğrular.

    tokenizer = TokenizerFactory.create("subword", subword_size=3)

    tokenizer.train("token")

    with pytest.raises(ValueError, match="Unknown token"):
        tokenizer.encode("python")


def test_subword_tokenizer_encode_empty_text_returns_empty_list() -> None:
    """
    Eğitim sonrası boş metin encode edildiğinde boş liste dönmelidir.

    Gerekçe:
        encode() boş veya sadece whitespace içeren string'ler için token üretmemelidir.
        Bu durum, encode/decode işlemlerinin hatalı sonuçlar üretmesini engeller.

    Train text:
        "tokenization"

    Encode text:
        ""

    Beklenen:
        []  

    Bu test şunu garanti eder:
        - encode() boş string için boş liste döndürmelidir.
        - encode() sadece whitespace içeren string'ler için boş liste döndürmelidir.
        - Bu durum, tokenizer'ın anlamsız tokenları sessizce encode etmesini engeller ve downstream işlemlerde hatalı sonuçlar üretmesini engeller.
    
    Önlenen bug sınıfı:
        - encode() boş veya sadece whitespace içeren string'ler için token üretirse, bu durum downstream işlemlerde hatalı sonuçlar üretmesine veya beklenmedik hatalar vermesine yol açabilir.
        - Bu durum downstream işlemlerde sessizce hatalara yol açabilir.
    """
    # testin amacı, encode() çağrıldığında boş string için boş liste döndürüldüğünü doğrulamaktır.
    # Bu test, encode() fonksiyonunun boş veya sadece whitespace içeren string'ler için token üretmediğini ve downstream işlemlerde hatalı sonuçlar üretmesini engellediğini doğrular.

    tokenizer = TokenizerFactory.create("subword", subword_size=3)

    tokenizer.train("tokenization")

    assert tokenizer.encode("") == [] # boş string encode edildiğinde boş liste dönmelidir  
    assert tokenizer.encode("   ") == [] # sadece whitespace içeren string encode edildiğinde boş liste dönmelidir  

def test_subword_tokenizer_encode_is_case_insensitive_due_to_lowercase_normalization() -> None:
    """
    encode() lowercase normalization nedeniyle case-insensitive davranmalıdır.

    Ggerekçe:
        tokenize() metni lowercase normalize eder, bu yüzden encode() da case-insensitive davranmalıdır.

    Train text:
        "Token"

    Encode text:
        "TOKEN"

    Beklenti:
        Aynı subword parçaları üretildiği için encode başarılı olur.

    Bu test şunu garanti eder:
        - encode() lowercase normalization nedeniyle case-insensitive davranmalıdır.
        - "Token" ve "TOKEN" aynı tokenları üretmelidir.
        - Bu durum, tokenizer'ın case-sensitive hatalar yapmasını engeller ve downstream işlemlerde hatalı sonuçlar üretmesini engeller.

    Önlenen bug sınıfı:
        - encode() case-sensitive davranırsa, aynı kelimenin farklı case'lerde farklı tokenlar üretmesine ve hatalı sonuçlar üretmesine yol açabilir.
        - Bu durum downstream işlemlerde sessizce hatalara yol açabilir.
    """
    # testin amacı, encode() çağrıldığında lowercase normalization nedeniyle case-insensitive davrandığını doğrulamaktır.
    # Bu test, encode() fonksiyonunun tokenize() tarafından yapılan lowercase normalization nedeniyle case-insensitive davrandığını doğrular.

    tokenizer = TokenizerFactory.create("subword", subword_size=3)

    tokenizer.train("Token")

    assert tokenizer.encode("TOKEN") == tokenizer.encode("token") # "TOKEN" ve "token" aynı tokenları üretmelidir (case-insensitive davranmalıdır)
    

# ---------------------------------------------------------
# DECODE TESTS
# ---------------------------------------------------------

def test_subword_tokenizer_decode_before_training_raises_error() -> None:
    """
    train() çağrılmadan decode() çalışmamalıdır.

    Çünkü decode işlemi id -> token mapping'e ihtiyaç duyar.

    Gerekçe:
        Decode işlemi id -> token mapping'e ihtiyaç duyar.
        Bu mapping train() sırasında oluşturulur.

    Bu test şunu garanti eder:
        - train() çağrılmadan decode() çalışmamalıdır.  
        - Bu durum, tokenizer'ın anlamsız bir state ile decode() işlemi yapmasını engeller ve decode() fonksiyonunun id -> token mapping'e ihtiyaç duyduğunu doğrular.  

    Önlenen bug sınıfı:
        - train() çağrılmadan decode() çalışırsa, tokenizer'ın id -> token mapping'ine sahip olmaması nedeniyle hatalı sonuçlar üretmesine veya beklenmedik hatalar vermesine yol açabilir.
        - Bu durum downstream işlemlerde sessizce hatalara yol açabilir.
    """
    # testin amacı, train() çağrılmadan decode() çalıştırıldığında ValueError beklenmesini sağlamaktır.
    # Bu test, tokenizer'ın anlamsız bir state ile decode() işlemi yapmasını engeller ve decode() fonksiyonunun id -> token mapping'e ihtiyaç duyduğunu doğrular.

    tokenizer = TokenizerFactory.create("subword", subword_size=3)

    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.decode([0, 1])


def test_subword_tokenizer_decode_returns_joined_text() -> None:
    """
    decode() token id listesini string'e dönüştürmelidir.

    Gerekçe:
        decode() token id listesini string'e dönüştürmelidir.
        Bu, model çıktılarının insan tarafından okunabilir bir formata dönüştürülmesini sağlar. 

    Input ids:
        [0, 1, 2, 3]

    Tokenlar:
        ["tok", "eni", "zat", "ion"]

    Beklenen:
        "tokenization"

    Bu test şunu garanti eder:
        - decode() token id listesini string'e dönüştürmelidir.    
        - decode() token id listesindeki her id'yi token'a dönüştürmelidir.
        - decode() tokenları doğrudan birleştirerek tek bir string döndürmelidir.
        - decode() çıktısı, train() sırasında oluşturulan id -> token mapping'ine göre token id'leri doğru tokenlara dönüştürmelidir.
        - Bu durum, model çıktılarının insan tarafından okunabilir bir formata dönüştürülmesini sağlar ve downstream işlemlerde hatalı sonuçlar üretmesini engeller.
    
    Önlenen bug sınıfı:
        - decode() token id listesini string'e dönüştürmezse, bu durum model çıktılarının insan tarafından okunabilir bir formata dönüştürülmemesine ve hatalı sonuçlar üretmesine yol açabilir.
        - decode() token id listesindeki her id'yi token'a dönüştürmezse, bu durum hatalı tokenlara ve hatalı sonuçlara yol açabilir.
        - decode() tokenları doğrudan birleştirmezse, bu durum hatalı formatta sonuçlara yol açabilir.
        - decode() çıktısı, train() sırasında oluşturulan id -> token mapping'ine göre token id'leri doğru tokenlara dönüştürmezse, bu durum hatalı sonuçlara yol açabilir.
        - Bu durum downstream işlemlerde sessizce hatalara yol açabilir.
    """
    # testin amacı, decode() çağrıldığında token id listesinin string'e dönüştürüldüğünü doğrulamaktır.
    # Bu test, decode() fonksiyonunun token id listesini string'e dönüştürdüğünü, her id'yi token'a dönüştürdüğünü, tokenları doğrudan birleştirerek tek bir string döndürdüğünü ve decode() çıktısının train() sırasında oluşturulan id -> token mapping'ine göre token id'leri doğru tokenlara dönüştürdüğünü doğrular.

    tokenizer = TokenizerFactory.create("subword", subword_size=3)

    tokenizer.train("tokenization")

    decoded = tokenizer.decode([0, 1, 2, 3])

    assert decoded == "tokenization"


def test_subword_tokenizer_decode_unknown_token_id_raises_error() -> None:
    """
    Vocabulary içinde olmayan token id decode edilmeye çalışılırsa hata vermelidir.

    Bu strict davranış hatalı model çıktılarının sessizce kabul edilmesini engeller.

    Gerekçe:
        decode() yalnızca train() sırasında oluşturulan id -> token mapping'ine göre token id'leri tokenlara dönüştürmelidir.
        Vocabulary içinde olmayan token id'ler mapping'de olmayacağı için decode edilmeye çalışıldığında hata vermelidir.

    Train text:
        "tokenization"

    Encode text:
        [999] # 999 id'si vocabulary içinde olmayan bir token id'sidir. 

    Bu test şunu garanti eder:  
        - decode() yalnızca train() sırasında oluşturulan id -> token mapping'ine göre token id'leri tokenlara dönüştürmelidir.
        - Vocabulary içinde olmayan token id'ler decode edilmeye çalışıldığında hata vermelidir.
        - Bu durum, tokenizer'ın anlamsız token id'leri sessizce decode etmesini engeller ve downstream işlemlerde hatalı sonuçlar üretmesini engeller.

    Önlenen bug sınıfı:
        - decode() vocabulary içinde olmayan token id'leri sessizce decode etmeye çalışırsa, bu token id'ler mapping'de olmadığı için hatalı tokenlara dönüştürülmesine veya beklenmedik hatalar vermesine yol açabilir.
        - Bu durum downstream işlemlerde sessizce hatalara yol açabilir.
    """
    # testin amacı, decode() çağrıldığında vocabulary içinde olmayan token id'lerin decode edilmeye çalışıldığında ValueError beklenmesini sağlamaktır.
    # Bu test, decode() fonksiyonunun yalnızca train() sırasında oluşturulan id -> token mapping'ine göre token id'leri tokenlara dönüştürdüğünü ve vocabulary içinde olmayan token id'lerin decode edilmeye çalışıldığında hata verdiğini doğrular.

    tokenizer = TokenizerFactory.create("subword", subword_size=3)

    tokenizer.train("tokenization")

    with pytest.raises(ValueError, match="Unknown token id"):
        tokenizer.decode([999])


def test_subword_tokenizer_decode_empty_list_returns_empty_string() -> None:
    """
    Boş token id listesi decode edildiğinde boş string dönmelidir.

    Train text:
        "tokenization"

    Encode text:
        [] # boş token id listesi

    Beklenen:
        "" # boş string

    Bu test şunu garanti eder:
        - decode() boş token id listesi için boş string döndürmelidir.
        - Bu durum, decode() fonksiyonunun boş inputları doğru şekilde işlediğini doğrular ve downstream işlemlerde hatalı sonuçlar üretmesini engeller.

    Önlenen bug sınıfı:
        - decode() boş token id listesi için boş string döndürmezse, bu durum model çıktılarının hatalı formatta dönmesine ve downstream işlemlerde hatalı sonuçlar üretmesine yol açabilir.
        - Bu durum downstream işlemlerde sessizce hatalara yol açabilir.
    """
    # testin amacı, decode() çağrıldığında boş token id listesi için boş string döndürüldüğünü doğrulamaktır.
    # Bu test, decode() fonksiyonunun boş token id listesi için boş string döndürdüğünü doğrular ve downstream işlemlerde hatalı sonuçlar üretmesini engeller.

    tokenizer = TokenizerFactory.create("subword", subword_size=3)

    tokenizer.train("tokenization")

    assert tokenizer.decode([]) == "" # boş token id listesi decode edildiğinde boş string dönmelidir
    assert tokenizer.decode([0, 1, 2, 3]) == "tokenization" # boş token id listesi decode edildiğinde boş string dönmelidir, ayrıca bilindiği gibi [0, 1, 2, 3] token id'leri "tokenization" kelmesine karşılık gelmektedir, bu da decode() fonksiyonunun doğru çalıştığını doğrular    


# ---------------------------------------------------------
# ROUNDTRIP / BEHAVIOR TESTS
# ---------------------------------------------------------

def test_subword_tokenizer_encode_decode_roundtrip_for_single_word() -> None:
    """
    Tek kelimelik inputlarda encode -> decode roundtrip korunmalıdır.

    Gerekçe:
        encode() ve decode() birbirini tamamlamalıdır.

    Input:
        "tokenization"

    Beklenen:
        decode(encode(input)) == input

    Bu test şunu garanti eder:
        - Tek kelimelik inputlarda encode -> decode roundtrip korunmalıdır.
        - Bu durum, tokenizer'ın temel işlevselliğinin doğru olduğunu doğrular ve downstream işlemlerde hatalı sonuçlar üretmesini engeller.

    Önlenen bug sınıfı:
        - encode() ve decode() birbirini tamamlamazsa, bu durum hatalı sonuçlar üretmesine veya beklenmedik hatalar vermesine yol açabilir.
        - Bu durum downstream işlemlerde sessizce hatalara yol açabilir.
    """
    # testin amacı, tek kelimelik inputlarda encode() ve decode() işlemlerinin birbirini tamamladığını doğrulamaktır.
    # Bu test, tokenizer'ın temel işlevselliğinin doğru olduğunu doğrular ve downstream işlemlerde hatalı sonuçlar üretmesini engeller.

    tokenizer = TokenizerFactory.create("subword", subword_size=3)

    text = "tokenization"

    tokenizer.train(text)

    decoded = tokenizer.decode(tokenizer.encode(text))

    assert decoded == text # tek kelimelik inputlarda encode -> decode roundtrip korunmalıdır, yani decode(encode(input)) == input olmalıdır


def test_subword_tokenizer_decode_does_not_preserve_whitespace() -> None:
    """
    SubwordTokenizer whitespace bilgisini korumaz.

    Çünkü:
        tokenize() whitespace karakterlerini saklamaz.
        decode() tokenları doğrudan birleştirir.

    Input:
        "hello world"

    Decode:
        "helloworld"

    Beklenti:
        SubwordTokenizer, kelimeleri subword_size parametresine göre sabit uzunluklu parçalara böler ve whitespace karakterlerini saklamaz.
        decode() tokenları doğrudan birleştirir, bu yüzden whitespace geri üretilmez.

    Bu test şunu garanti eder:
        - SubwordTokenizer whitespace bilgisini korumaz.
        - tokenize() whitespace karakterlerini saklamaz.
        - decode() tokenları doğrudan birleştirir, bu yüzden whitespace geri üretilmez.
        - Bu durum, tokenizer'ın whitespace karakterlerini doğru şekilde işlediğini doğrular ve downstream işlemlerde hatalı sonuçlar üretmesini engeller.

    Önlenen bug sınıfı:
        - SubwordTokenizer whitespace bilgisini korursa, bu durum encode/decode işlemlerinin hatalı sonuçlar üretmesine veya beklenmedik hatalar vermesine yol açabilir.
        - Bu durum downstream işlemlerde sessizce hatalara yol açabilir.
    """
    # testin amacı, SubwordTokenizer'ın whitespace bilgisini korumadığını doğrulamaktır.
    # Bu test, tokenize() fonksiyonunun whitespace karakterlerini saklamadığını ve decode() fonksiyonunun tokenları doğrudan birleştirerek whitespace geri üretmediğini doğrular.

    tokenizer = TokenizerFactory.create("subword", subword_size=3)

    text = "hello world"

    tokenizer.train(text)

    decoded = tokenizer.decode(tokenizer.encode(text))

    assert decoded == "helloworld" # SubwordTokenizer whitespace bilgisini korumaz, tokenize() whitespace karakterlerini saklamaz ve decode() tokenları doğrudan birleştirir, bu yüzden whitespace geri üretilmez, sonuç "helloworld" olmalıdır 
    assert decoded != text # decode edilen sonuç orijinal input ile aynı olmamalıdır, çünkü whitespace geri üretilmez   
    

def test_subword_tokenizer_roundtrip_with_punctuation_normalizes_spacing() -> None:
    """
    Punctuation tokenları korunur, fakat whitespace geri üretilmez.

    Gerekçe:
        SubwordTokenizer, kelimeleri subword_size parametresine göre sabit uzunluklu parçalara böler ve whitespace karakterlerini saklamaz.
        decode() tokenları doğrudan birleştirir, bu yüzden whitespace geri üretilmez.
        Noktalama işaretleri token olarak kalır, bu yüzden korunur.

    Input:
        "hello, world!"

    Decode:
        "hello,world!"

    Açıklama:
        "," ve "!" korunur.
        Fakat virgülden sonraki boşluk korunmaz.

    Bu test şunu garanti eder:
        - Punctuation tokenları korunur, fakat whitespace geri üretilmez.

    Önlenen bug sınıfı:
        - SubwordTokenizer whitespace bilgisini korursa, bu durum encode/decode işlemlerinin hatalı sonuçlar üretmesine veya beklenmedik hatalar vermesine yol açabilir.
        - Bu durum downstream işlemlerde sessizce hatalara yol açabilir.
    """
    # testin amacı, SubwordTokenizer'ın punctuation tokenlarını koruduğunu fakat whitespace bilgisini korumadığını doğrulamaktır.
    # Bu test, SubwordTokenizer'ın kelimeleri subword_size parametresine göre sabit uzunluklu parçalara böldüğünü, whitespace karakterlerini saklamadığını, decode() fonksiyonunun tokenları doğrudan birleştirerek whitespace geri üretmediğini ve noktalama işaretlerinin token olarak kalıp korunduğunu doğrular.

    tokenizer = TokenizerFactory.create("subword", subword_size=3)

    text = "hello, world!"

    tokenizer.train(text)

    decoded = tokenizer.decode(tokenizer.encode(text))

    assert decoded == "hello,world!" # SubwordTokenizer punctuation tokenlarını korur fakat whitespace bilgisini korumaz, sonuç "hello,world!" olmalıdır
    assert decoded != text # decode edilen sonuç orijinal input ile aynı olmamalıdır, çünkü whitespace geri üretilmez


def test_subword_tokenizer_vocab_size_is_zero_before_training() -> None:
    """
    Eğitim öncesinde vocabulary boş olmalıdır.

    Bu test tokenizer state'inin açık ve tahmin edilebilir olduğunu doğrular.

    Gerekçe:
        SubwordTokenizer'ın vocabulary'si train() sırasında oluşturulur.
        Eğitim öncesinde vocabulary boş olmalıdır, bu durum tokenizer state'inin açık ve tahmin edilebilir olduğunu doğrular ve train() çağrılmadan encode/decode işlemlerinin hatalı sonuçlar üretmesini engeller.

    Bu test şunu garanti eder:
        - Eğitim öncesinde vocabulary boş olmalıdır.
        - Bu durum tokenizer state'inin açık ve tahmin edilebilir olduğunu doğrular ve train() çağrılmadan encode/decode işlemlerinin hatalı sonuçlar üretmesini engeller.
        - Eğitim öncesinde encode() veya decode() çağrıldığında hata verilmelidir, çünkü vocabulary boş olduğu için bu işlemler anlamsız olur.  

    Önlenen bug sınıfı:
        - Eğitim öncesinde vocabulary boş olmazsa, bu durum tokenizer state'inin belirsiz olmasına ve train() çağrılmadan encode/decode işlemlerinin hatalı sonuçlar üretmesine yol açabilir.
        - Bu durum downstream işlemlerde sessizce hatalara yol açabilir.
    """
    # testin amacı, SubwordTokenizer'ın eğitim öncesinde vocabulary'sinin boş olduğunu doğrulamaktır.
    # Bu test, SubwordTokenizer'ın vocabulary'sinin train() sırasında oluşturulduğunu ve eğitim öncesinde boş olduğunu doğrular, bu durum tokenizer state'inin açık ve tahmin edilebilir olduğunu doğrular ve train() çağrılmadan encode/decode işlemlerinin hatalı sonuçlar üretmesini engeller.

    tokenizer = TokenizerFactory.create("subword", subword_size=3)

    assert tokenizer.vocab_size == 0 # eğitim öncesinde vocabulary boş olmalıdır, vocab_size 0 olmalıdır

    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.encode("tokenization") # eğitim öncesinde encode() çağrıldığında hata verilmelidir, çünkü vocabulary boş olduğu için bu işlem anlamsız olur
    
    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.decode([0, 1]) # eğitim öncesinde decode() çağrıldığında hata verilmelidir, çünkü vocabulary boş olduğu için bu işlem anlamsız olur