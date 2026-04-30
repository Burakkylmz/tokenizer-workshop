from __future__ import annotations

import pytest

from tokenizer_workshop.api.services.tokenizer_factory import TokenizerFactory


# ---------------------------------------------------------
# TRAIN TESTS
# ---------------------------------------------------------

def test_whitespace_tokenizer_train_with_empty_text_raises_error() -> None:
    """
    Boş metinle train() çağrıldığında ValueError beklenmelidir.

    Gerekçe:
        Vocabulary oluşturmak için en az bir gerçek token gerekir.
        Boş string hiçbir token üretmez.

    Bu test şunu garanti eder:
        tokenizer.train("") -> ValueError("Training text cannot be empty")

    Önlenen bug sınıfı:
        - Boş metinle train() yapıldığında tokenizer'ın boş bir vocab oluşturması
          ve encode/decode işlemlerinin anlamsız sonuçlar üretmesi.
        - Boş metinle train() yapıldığında tokenizer'ın beklenmedik şekilde davranması
          veya çökmesi.
    """
    # testin amacı, train() metodunun boş metinle çağrıldığında uygun şekilde hata vermesini sağlamaktır.

    tokenizer = TokenizerFactory.create("white_space")

    with pytest.raises(ValueError, match="Training text cannot be empty"):
        tokenizer.train("")


def test_whitespace_tokenizer_train_with_whitespace_text_raises_error() -> None:
    """
    Sadece whitespace içeren metinle train() yapılmamalıdır.

    Örnek:
        "   " -> tokenize edilebilir gerçek token yoktur.

    Gerekçe:
        Sadece whitespace içeren metin gerçek token üretmez, bu da anlamlı bir vocab oluşturmaz.

    Bu test şunu garanti eder:
        tokenizer.train("   ") -> ValueError("Training text cannot be empty")

    Önlenen bug sınıfı:
        - Sadece whitespace içeren metinle train() yapıldığında tokenizer'ın boş bir vocab oluşturması ve encode/decode işlemlerinin anlamsız sonuçlar üretmesi.
        - Sadece whitespace içeren metinle train() yapıldığında tokenizer'ın beklenmedik şekilde davranması veya çökmesi.
        - Sadece whitespace içeren metinle train() yapıldığında tokenizer'ın encode/decode işlemlerinde sonsuz döngüye girmesi veya bellek hatası vermesi.
    """
    # testin amacı, train() metodunun sadece whitespace içeren metinle çağrıldığında uygun şekilde hata vermesini sağlamaktır.

    tokenizer = TokenizerFactory.create("white_space")

    with pytest.raises(ValueError, match="Training text cannot be empty"):
        tokenizer.train("   ")


def test_whitespace_tokenizer_train_builds_vocab() -> None:
    """
    train() sonrası vocabulary oluşturulmalıdır.

    Gerekçe:
        Vocabulary oluşturulmazsa encode/decode işlemleri anlamsız olur.

    Input: "hello world hello"

    Unique tokenlar: ["hello", "world"]

    Beklenen vocab_size: 2

    Bu test şunu garanti eder:
        - train() sonrası tokenizer.vocab_size == 2
        - train() sonrası tokenizer._token_to_id == {"hello": 0, "world": 1}
        - train() sonrası tokenizer._id_to_token == {0: "hello", 1: "world"}
    
    Önlenen bug sınıfı:
        - train() sonrası vocabulary'nin oluşturulmaması veya yanlış oluşturulması. 
        - train() sonrası vocab_size'ın yanlış hesaplanması.
        - train() sonrası token_to_id ve id_to_token mapping'lerinin yanlış oluşturulması.
        - train() sonrası tokenizer'ın encode/decode işlemlerinde beklenmedik hatalar vermesi veya yanlış sonuçlar üretmesi.    
    """
    # testin amacı, train() metodunun verilen metinle doğru şekilde vocabulary oluşturduğunu ve vocab_size'ı doğru hesapladığını sağlamaktır.

    tokenizer = TokenizerFactory.create("white_space")

    tokenizer.train("hello world hello")

    assert tokenizer.vocab_size == 2 # "hello" ve "world" olmak üzere 2 unique token vardır.


def test_whitespace_tokenizer_vocab_is_deterministic_for_same_input() -> None:
    """
    Aynı eğitim metniyle eğitilen iki tokenizer aynı mapping'i üretmelidir.

    Bu test deterministik vocabulary davranışını doğrular.

    Gerekçe:
        Aynı metinle eğitilen tokenizer'ların aynı token_to_id ve id_to_token mapping'lerini üretmesi beklenir. 
    
    Bu test şunu garanti eder:
        - İki farklı tokenizer instance'ı aynı metinle eğitildiğinde aynı token_to_id ve id_to_token mapping'lerini üretir.
    
    Önlenen bug sınıfı:
        - Aynı metinle eğitilen tokenizer'ların farklı token_to_id ve id_to_token mapping'leri üretmesi.
        - Aynı metinle eğitilen tokenizer'ların farklı vocab_size'lara sahip olması.
        - Aynı metinle eğitilen tokenizer'ların encode/decode işlemlerinde farklı sonuçlar üretmesi.
    """
    # testin amacı, aynı eğitim metniyle eğitilen iki farklı tokenizer instance'ının aynı vocabulary'yi oluşturduğunu doğrulamaktır.

    tokenizer_a = TokenizerFactory.create("white_space")
    tokenizer_b = TokenizerFactory.create("white_space")

    text = "elma armut kiraz elma"

    tokenizer_a.train(text)
    tokenizer_b.train(text)

    assert tokenizer_a._token_to_id == tokenizer_b._token_to_id # "elma": 0, "armut": 1, "kiraz": 2 gibi aynı mapping'i üretmelidir.
    assert tokenizer_a._id_to_token == tokenizer_b._id_to_token # 0: "elma", 1: "armut", 2: "kiraz" gibi aynı mapping'i üretmelidir.


# ---------------------------------------------------------
# TOKENIZE TESTS
# ---------------------------------------------------------

def test_whitespace_tokenizer_tokenize_splits_on_spaces() -> None:
    """
    tokenize() metni boşluklara göre ayırmalıdır.

    Gerekçe:
        WhitespaceTokenizer'ın temel işlevi metni whitespace'e göre bölmektir.
    
    Input: "hello world tokenizer"

    Beklenen: ["hello", "world", "tokenizer"]

    Bu test şunu garanti eder:
        - tokenize() metni doğru şekilde boşluklara göre ayırır.
        - tokenize() metni karakterlere ayırmaz veya noktalama işaretlerini ayrı token yapmaz.
    
    Önlenen bug sınıfı:
        - tokenize() metni boşluklara göre ayırmaması veya yanlış şekilde bölmesi.
        - tokenize() metni karakterlere ayırması veya noktalama işaretlerini ayrı token yapması.
        - tokenize() metni beklenmedik şekilde bölmesi veya boş tokenlar üretmesi.
    """
    # testin amacı, tokenize() metodunun verilen metni doğru şekilde boşluklara göre ayırdığını doğrulamaktır.

    tokenizer = TokenizerFactory.create("white_space")

    tokens = tokenizer.tokenize("hello world tokenizer")

    assert tokens == ["hello", "world", "tokenizer"] # tokenize() metni boşluklara göre doğru şekilde ayırmalıdır.


def test_whitespace_tokenizer_tokenize_handles_multiple_spaces() -> None:
    """
    Ardışık boşluklar tek ayırıcı gibi davranmalıdır.

    Gerekçe:
        WhitespaceTokenizer'ın temel işlevi metni whitespace'e göre bölmektir.
        Ardışık boşluklar tek bir ayırıcı gibi davranmalıdır.

    Input: "hello   world"

    Beklenen: ["hello", "world"]

    Python str.split() default davranışı:
        "hello   world" -> ["hello", "world"]
        " hello world " -> ["hello", "world"]
        "hello\\nworld" -> ["hello", "world"]
        "hello\tworld" -> ["hello", "world"]

    Bu test şunu garanti eder:
        - Ardışık boşlukların tek bir ayırıcı gibi davranması.
        - tokenize() metni doğru şekilde boşluklara göre ayırır.
    
    Önlenen bug sınıfı:
        - Ardışık boşlukların yanlış şekilde işlenmesi.
        - tokenize() metni beklenmedik şekilde bölmesi veya boş tokenlar üretmesi.
        - tokenize() metni karakterlere ayırması veya noktalama işaretlerini ayrı token yapması.
    """
    # testin amacı, tokenize() metodunun ardışık boşlukları tek bir ayırıcı gibi işlediğini doğrulamaktır.

    tokenizer = TokenizerFactory.create("white_space")

    tokens = tokenizer.tokenize("hello   world")

    assert tokens == ["hello", "world"] # tokenize() metni boşluklara göre doğru şekilde ayırmalıdır.


def test_whitespace_tokenizer_tokenize_handles_tabs_and_newlines() -> None:
    """
    split() yalnızca normal boşluğu değil, tab ve newline gibi whitespace
    karakterlerini de ayırıcı kabul eder.

    Gerekçe:
        WhitespaceTokenizer'ın temel işlevi metni whitespace'e göre bölmektir.
        Tab ve newline gibi diğer whitespace karakterleri de ayırıcı kabul edilmelidir.

    Input: "hello\tworld\npython"

    Beklenen: ["hello", "world", "python"]

    Bu test şunu garanti eder:
        - tokenize() metni tab ve newline gibi diğer whitespace karakterlerine göre de doğru şekilde ayırır.
        - tokenize() metni karakterlere ayırmaz veya noktalama işaretlerini ayrı token yapmaz.
        - tokenize() metni beklenmedik şekilde bölmez veya boş tokenlar üretmez.

    Önlenen bug sınıfı:
        - tokenize() metni tab ve newline gibi diğer whitespace karakterlerine göre ayırmaması veya yanlış şekilde bölmesi.
        - tokenize() metni karakterlere ayırması veya noktalama işaretlerini ayrı token yapması.
        - tokenize() metni beklenmedik şekilde bölmesi veya boş tokenlar üretmesi.
    """
    # testin amacı, tokenize() metodunun metni tab ve newline gibi diğer whitespace karakterlerine göre de doğru şekilde ayırdığını doğrulamaktır.

    tokenizer = TokenizerFactory.create("white_space")

    tokens = tokenizer.tokenize("hello\tworld\npython")

    assert tokens == ["hello", "world", "python"] # tokenize() metni tab ve newline gibi diğer whitespace karakterlerine göre de doğru şekilde ayırmalıdır.


def test_whitespace_tokenizer_tokenize_empty_string_returns_empty_list() -> None:
    """
    Boş string tokenize edildiğinde boş liste dönmelidir.

    Gerekçe:
        Boş string tokenize edildiğinde token yoktur, bu nedenle boş liste dönmelidir.

    Bu test şunu garanti eder:
        - tokenize("") -> []
        - tokenize("   ") -> [] gibi sadece whitespace içeren string'ler de boş liste dönmelidir.

    Önlenen bug sınıfı:
        - tokenize("") -> None veya tokenize("") -> [""] gibi beklenmedik sonuçlar üretmesi.
        - tokenize("") -> [""] gibi boş token içeren liste dönmesi.
        - tokenize("") -> [] yerine tokenize("") -> None gibi yanlış türde sonuç dönmesi.
    """
    # testin amacı, tokenize() metodunun boş string veya sadece whitespace içeren string'leri doğru şekilde işleyerek boş liste döndürdüğünü doğrulamaktır.

    tokenizer = TokenizerFactory.create("white_space")

    assert tokenizer.tokenize("") == [] # tokenize("") -> [] gibi boş liste dönmelidir.


def test_whitespace_tokenizer_tokenize_whitespace_only_returns_empty_list() -> None:
    """
    Sadece whitespace içeren input tokenize edildiğinde boş liste dönmelidir.

    Gerekçe:
        Sadece whitespace içeren string tokenize edildiğinde token yoktur, bu nedenle boş liste dönmelidir.
    
    Bu test şunu garanti eder:
        - tokenize("   ") -> [] gibi sadece whitespace içeren string'ler boş liste dönmelidir.
        - tokenize("   \n\t") -> [] gibi sadece whitespace içeren string'ler boş liste dönmelidir.
        - tokenize(" \t\n ") -> [] gibi sadece whitespace içeren string'ler boş liste dönmelidir.

    Önlenen bug sınıfı:
        - tokenize("   ") -> None veya tokenize("   ") -> [""] gibi beklenmedik sonuçlar üretmesi.
        - tokenize("   ") -> [""] gibi boş token içeren liste dönmesi.
        - tokenize("   ") -> [] yerine tokenize("   ") -> None gibi yanlış türde sonuç dönmesi.
    """
    # testin amacı, tokenize() metodunun sadece whitespace içeren string'leri doğru şekilde işleyerek boş liste döndürdüğünü doğrulamaktır.

    tokenizer = TokenizerFactory.create("white_space")

    assert tokenizer.tokenize("   \n\t") == [] # tokenize("   \n\t") -> [] gibi sadece whitespace içeren string'ler boş liste dönmelidir.


def test_whitespace_tokenizer_does_not_split_punctuation() -> None:
    """
    WhitespaceTokenizer noktalama işaretlerini ayrı token yapmaz.

    Gerekçe:
        WhitespaceTokenizer'ın temel işlevi metni whitespace'e göre bölmektir.

    Input: "hello, world!"

    Beklenen: ["hello,", "world!"]

    Bu test şunu garanti eder:
        - tokenize() metni sadece whitespace'e göre ayırır, noktalama işaretlerini ayrı token yapmaz.
        - tokenize() metni karakterlere ayırmaz veya beklenmedik şekilde bölmez.
        - tokenize() metni beklenmedik şekilde bölmesi veya boş tokenlar üretmesi.
        - tokenize() metni boşluklara göre doğru şekilde ayırır.

    Önlenen bug sınıfı:
        - tokenize() metni karakterlere ayırması veya noktalama işaretlerini ayrı token yapması.
        - tokenize() metni beklenmedik şekilde bölmesi veya boş tokenlar üretmesi.
        - tokenize() metni boşluklara göre ayırmaması veya yanlış şekilde bölmesi.
    """
    # testin amacı, tokenize() metodunun metni sadece whitespace'e göre ayırdığını ve noktalama işaretlerini ayrı token yapmadığını doğrulamaktır.

    tokenizer = TokenizerFactory.create("white_space")

    tokens = tokenizer.tokenize("hello, world!")

    assert tokens == ["hello,", "world!"] # tokenize() metni sadece whitespace'e göre ayırmalı, noktalama işaretlerini ayrı token yapmamalıdır.


# ---------------------------------------------------------
# ENCODE TESTS
# ---------------------------------------------------------

def test_whitespace_tokenizer_encode_before_training_raises_error() -> None:
    """
    train() çağrılmadan encode() çalışmamalıdır.

    Çünkü encode işlemi token -> id mapping'e ihtiyaç duyar.

    Gerekçe:
        encode() işlemi token -> id mapping'e ihtiyaç duyar, bu mapping train() sırasında oluşturulur. 
        train() çağrılmadan encode() çalıştırılırsa mapping olmadığı için hata vermelidir.

    Bu test şunu garanti eder:
        - encode() train() çağrılmadan çalıştırıldığında ValueError fırlatır.
        - encode() train() çağrılmadan çalıştırıldığında hata mesajı "Tokenizer has not been trained yet" içerir.

    Önlenen bug sınıfı:
        - train() çağrılmadan encode() çalıştırıldığında beklenmedik şekilde davranması veya anlamsız sonuçlar üretmesi.
        - train() çağrılmadan encode() çalıştırıldığında tokenizer'ın çökmesi veya bellek hatası vermesi.
        - train() çağrılmadan encode() çalıştırıldığında tokenizer'ın sonsuz döngüye girmesi veya performans sorunları yaşaması.
    """
    # testin amacı, encode() metodunun train() çağrılmadan çalıştırıldığında uygun şekilde hata verdiğini doğrulamaktır.

    tokenizer = TokenizerFactory.create("white_space")

    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.encode("hello world")


def test_whitespace_tokenizer_encode_returns_integer_ids() -> None:
    """
    encode() integer token id listesi döndürmelidir.
    
    Gerekçe:
        encode() işlemi tokenları integer id'lere dönüştürmelidir, bu nedenle dönen sonuç integer id listesi olmalıdır.
    
    Input:
        Train text: "hello world"
        Encode text: "hello world"
    
    Beklenen:
        [0, 1]
    
    Bu test şunu garanti eder:
        - encode() doğru şekilde integer id listesi döndürür.
        - encode() dönen id'lerin tipinin integer olduğunu doğrular.
        - encode() dönen id'lerin sırasının token sırasına uygun olduğunu doğrular.

    Önlenen bug sınıfı:
        - encode() işleminin integer id listesi yerine başka türde sonuç döndürmesi (örneğin, string listesi veya None).
        - encode() işleminin token sırasını korumaması (örneğin, "world hello" -> [1, 0] gibi yanlış sıralama).
        - encode() işleminin beklenmedik şekilde davranması veya anlamsız sonuçlar üretmesi.
    """
    # testin amacı, encode() metodunun doğru şekilde integer token id listesi döndürdüğünü doğrulamaktır.

    tokenizer = TokenizerFactory.create("white_space")

    tokenizer.train("hello world")
    encoded = tokenizer.encode("hello world")

    assert isinstance(encoded, list) # encode() bir liste döndürmelidir.
    assert encoded == [0, 1] # encode() "hello" token'ını 0, "world" token'ını 1 olarak encode etmelidir.
    assert all(isinstance(token_id, int) for token_id in encoded) # encode() dönen id'lerin tümünün integer olduğunu doğrular.


def test_whitespace_tokenizer_encode_unknown_token_raises_error() -> None:
    """
    Eğitim sırasında görülmeyen token encode edilmeye çalışılırsa hata vermelidir.

    Gerekçe:
        WhitespaceTokenizer strict vocabulary davranışı kullanır. 
        Eğitimde görülmeyen token encode edilmeye çalışılırsa açık hata verir.    

    Train text: "hello world"

    Encode text: "hello python"

    "python" vocabulary içinde olmadığı için ValueError beklenir.

    Bu test şunu garanti eder:
        - Eğitim sırasında görülmeyen token encode edilmeye çalışıldığında ValueError fırlatır.
        - Hata mesajı "Unknown token: python" içerir.

    Önlenen bug sınıfı:
        - Eğitim sırasında görülmeyen token encode edilmeye çalışıldığında beklenmedik şekilde davranması veya anlamsız sonuçlar üretmesi.
        - Eğitim sırasında görülmeyen token encode edilmeye çalışıldığında tokenizer'ın çökmesi veya bellek hatası vermesi.
        - Eğitim sırasında görülmeyen token encode edilmeye çalışıldığında tokenizer'ın sonsuz döngüye girmesi veya performans sorunları yaşaması.  
    """
    # testin amacı, encode() metodunun eğitim sırasında görülmeyen token'ları encode etmeye çalışıldığında uygun şekilde hata verdiğini doğrulamaktır.

    tokenizer = TokenizerFactory.create("white_space")

    tokenizer.train("hello world")

    with pytest.raises(ValueError, match="Unknown token"):
        tokenizer.encode("hello python")


def test_whitespace_tokenizer_encode_empty_text_returns_empty_list() -> None:
    """
    Eğitim sonrası boş metin encode edildiğinde boş liste dönmelidir.

    Bu edge case API/report tarafında güvenli davranış sağlar.

    Gerekçe:
        Eğitim sonrası boş metin encode edildiğinde token yoktur, bu nedenle boş liste dönmelidir.

    Train text: "hello world"
    
    Encode text: ""
    
    Beklenen: []

    Bu test şunu garanti eder:
        - Eğitim sonrası encode("") -> [] gibi boş liste dönmelidir.
        - Eğitim sonrası encode("   ") -> [] gibi sadece whitespace içeren string'ler de boş liste dönmelidir.
        - Eğitim sonrası encode() metni doğru şekilde boşluklara göre ayırır.

    Önlenen bug sınıfı:
        - Eğitim sonrası encode("") -> None veya encode("") -> [""] gibi beklenmedik sonuçlar üretmesi.
        - Eğitim sonrası encode("") -> [""] gibi boş token içeren liste dönmesi.
        - Eğitim sonrası encode("") -> [] yerine encode("") -> None gibi yanlış türde sonuç dönmesi.
    """
    # testin amacı, encode() metodunun eğitim sonrası boş metin veya sadece whitespace içeren string'leri doğru şekilde işleyerek boş liste döndürdüğünü doğrulamaktır.

    tokenizer = TokenizerFactory.create("white_space")

    tokenizer.train("hello world")

    assert tokenizer.encode("") == [] # eğitim sonrası encode("") -> [] gibi boş liste dönmelidir.
    assert tokenizer.encode("   ") == [] # eğitim sonrası encode("   ") -> [] gibi sadece whitespace içeren string'ler de boş liste dönmelidir. 


# ---------------------------------------------------------
# DECODE TESTS
# ---------------------------------------------------------

def test_whitespace_tokenizer_decode_before_training_raises_error() -> None:
    """
    train() çağrılmadan decode() çalışmamalıdır.

    Çünkü decode işlemi id -> token mapping'e ihtiyaç duyar.

    Gerekçe:
        decode() işlemi id -> token mapping'e ihtiyaç duyar, bu mapping train() sırasında oluşturulur.
        train() çağrılmadan decode() çalıştırılırsa mapping olmadığı için hata vermelidir.

    Bu test şunu garanti eder:
        - decode() train() çağrılmadan çalıştırıldığında ValueError fırlatır.
        - decode() train() çağrılmadan çalıştırıldığında hata mesajı "Tokenizer has not been trained yet" içerir.   

    Önlenen bug sınıfı:
        - train() çağrılmadan decode() çalıştırıldığında beklenmedik şekilde davranması veya anlamsız sonuçlar üretmesi.
        - train() çağrılmadan decode() çalıştırıldığında tokenizer'ın çökmesi veya bellek hatası vermesi.
        - train() çağrılmadan decode() çalıştırıldığında tokenizer'ın sonsuz döngüye girmesi veya performans sorunları yaşaması.
    """
    # testin amacı, decode() metodunun train() çağrılmadan çalıştırıldığında uygun şekilde hata verdiğini doğrulamaktır.

    tokenizer = TokenizerFactory.create("white_space")

    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.decode([0, 1])


def test_whitespace_tokenizer_decode_returns_text() -> None:
    """
    decode() token id listesini tekrar string'e dönüştürmelidir.

    Gerekçe:
        decode() işlemi token id listesini tekrar string'e dönüştürmelidir, bu nedenle dönen sonuç string olmalıdır.
    
    Train text: "hello world"
    
    Decode text: [0, 1]
    
    Beklenen: "hello world"
    
    Bu test şunu garanti eder:
        - decode() doğru şekilde string döndürür.
        - decode() dönen string'in tipinin str olduğunu doğrular.
        - decode() dönen string'in encode() ile verilen token id'lerin karşılığı olan token'ları içerdiğini doğrular.
    
    Önlenen bug sınıfı:
        - decode() işleminin string yerine başka türde sonuç döndürmesi (örneğin, list veya None).
        - decode() işleminin token id'lerin karşılığı olan token'ları içermemesi (örneğin, [0, 1] -> "unknown unknown" gibi yanlış sonuç).
        - decode() işleminin beklenmedik şekilde davranması veya anlamsız sonuçlar üretmesi.
    """
    # testin amacı, decode() metodunun doğru şekilde string döndürdüğünü ve dönen string'in encode() ile verilen token id'lerin karşılığı olan token'ları içerdiğini doğrulamaktır.

    tokenizer = TokenizerFactory.create("white_space")

    tokenizer.train("hello world")

    decoded = tokenizer.decode([0, 1])

    assert decoded == "hello world"


def test_whitespace_tokenizer_decode_unknown_token_id_raises_error() -> None:
    """
    Vocabulary içinde olmayan token id decode edilmeye çalışılırsa hata vermelidir.

    Gerekçe:
        WhitespaceTokenizer strict vocabulary davranışı kullanır.
        Vocabulary'de olmayan token id decode edilmeye çalışılırsa açık hata verir.

    Train text: "hello world"

    Decode token ids: [999]
    
    "999" token id'si vocabulary içinde olmadığı için ValueError beklenir.
    
    Bu test şunu garanti eder:
        - Vocabulary içinde olmayan token id decode edilmeye çalışıldığında ValueError fırlatır.
        - Hata mesajı "Unknown token id: 999" içerir.   

    Önlenen bug sınıfı:
        - Vocabulary içinde olmayan token id decode edilmeye çalışıldığında beklenmedik şekilde davranması veya anlamsız sonuçlar üretmesi.
        - Vocabulary içinde olmayan token id decode edilmeye çalışıldığında tokenizer'ın çökmesi veya bellek hatası vermesi.
        - Vocabulary içinde olmayan token id decode edilmeye çalışıldığında tokenizer'ın sonsuz döngüye girmesi veya performans sorunları yaşaması.
    """
    # testin amacı, decode() metodunun vocabulary içinde olmayan token id'leri decode etmeye çalışıldığında uygun şekilde hata verdiğini doğrulamaktır.

    tokenizer = TokenizerFactory.create("white_space")

    tokenizer.train("hello world")

    with pytest.raises(ValueError, match="Unknown token id"):
        tokenizer.decode([999])


def test_whitespace_tokenizer_decode_empty_list_returns_empty_string() -> None:
    """
    Boş token id listesi decode edildiğinde boş string dönmelidir.

    Çünkü decode edilecek token yoktur.

    Gerekçe:
        Boş token id listesi decode edildiğinde token yoktur, bu nedenle boş string dönmelidir.

    Train text: "hello world"

    Decode token ids: []
    
    Beklenen: ""

    Bu test şunu garanti eder:
        - decode([]) -> "" gibi boş string dönmelidir.
        - decode() metni doğru şekilde boşluklara göre birleştirir.

    Önlenen bug sınıfı:
        - decode([]) -> None veya decode([]) -> [""] gibi beklenmedik sonuçlar üretmesi.
        - decode([]) -> [""] gibi boş token içeren string dönmesi.
        - decode([]) -> "" yerine decode([]) -> None gibi yanlış türde sonuç dönmesi.
    """
    # testin amacı, decode() metodunun boş token id listesi decode edildiğinde uygun şekilde boş string döndürdüğünü doğrulamaktır.

    tokenizer = TokenizerFactory.create("white_space")

    tokenizer.train("hello world")

    assert tokenizer.decode([]) == "" # decode([]) -> "" gibi boş string dönmelidir.


# ---------------------------------------------------------
# ROUNDTRIP / BEHAVIOR TESTS
# ---------------------------------------------------------

def test_whitespace_tokenizer_encode_decode_roundtrip() -> None:
    """
    Basit whitespace metinlerde encode -> decode roundtrip korunmalıdır.

    Gerekçe:
        Basit whitespace metinlerde encode -> decode roundtrip korunmalıdır, 
        yani decode(encode(input)) == input olmalıdır.

    Input: "the cat sat"

    Beklenen: decode(encode(input)) == input

    Bu test şunu garanti eder:
        - Basit whitespace metinlerde encode -> decode roundtrip korunur.
        - encode() ve decode() birlikte tutarlı şekilde çalışır.
        - encode() ve decode() birlikte beklenmedik şekilde davranmaz veya anlamsız sonuçlar üretmez.

    Önlenen bug sınıfı:
        - Basit whitespace metinlerde encode -> decode roundtrip korunmaması (örneğin, "the cat sat" -> [0, 1, 2] -> "the cat sat" gibi beklenmedik sonuçlar).
        - encode() ve decode() birlikte tutarsız şekilde çalışması (örneğin, encode() doğru şekilde token id listesi döndürürken decode() yanlış şekilde string döndürmesi).
        - encode() ve decode() birlikte beklenmedik şekilde davranması veya anlamsız sonuçlar üretmesi (örneğin, encode() ve decode() birlikte çökmesi veya bellek hatası vermesi).
        - encode() ve decode() birlikte sonsuz döngüye girmesi veya performans sorunları yaşaması.
    """
    # testin amacı, basit whitespace metinlerde encode -> decode roundtrip'un korunup korunmadığını doğrulamaktır.

    tokenizer = TokenizerFactory.create("white_space")

    text = "the cat sat"

    tokenizer.train(text)

    decoded = tokenizer.decode(tokenizer.encode(text))

    assert decoded == text


def test_whitespace_tokenizer_decode_normalizes_multiple_spaces() -> None:
    """
    WhitespaceTokenizer orijinal whitespace biçimini birebir korumaz.

    Gerekçe:
        WhitespaceTokenizer'ın temel işlevi metni whitespace'e göre bölmektir.
        tokenize() multiple-space bilgisini kaybeder, decode() tokenları tek boşluk ile birleştirir. 
        Bu nedenle decode(encode(input)) == input olmayabilir, ancak decode(encode(input)) == "hello world" gibi normalize edilmiş bir string dönebilir.

    Input: "hello   world"

    Decode: "hello world"

    Bu test şunu garanti eder:
        - WhitespaceTokenizer orijinal whitespace biçimini birebir korumaz, ancak tokenları tek boşluk ile birleştirerek normalize edilmiş bir string döndürür.
        - encode() ve decode() birlikte tutarlı şekilde çalışır, ancak whitespace bilgisi kaybolur.
        - encode() ve decode() birlikte beklenmedik şekilde davranmaz veya anlamsız sonuçlar üretmez.
        - encode() ve decode() birlikte çökmez veya bellek hatası vermez.
        - encode() ve decode() birlikte sonsuz döngüye girmez veya performans sorunları yaşamaz.

    Önlenen bug sınıfı:
        - WhitespaceTokenizer'ın orijinal whitespace biçimini birebir koruması beklenirken, encode() ve decode() birlikte beklenmedik şekilde davranması veya anlamsız sonuçlar üretmesi (örneğin, "hello   world" -> [0, 1] -> "hello world" gibi beklenmedik sonuçlar).
        - encode() ve decode() birlikte çökmesi veya bellek hatası vermesi.
        - encode() ve decode() birlikte sonsuz döngüye girmesi veya performans sorunları yaşaması.
    """
    # testin amacı, WhitespaceTokenizer'ın orijinal whitespace biçimini birebir korumadığını, ancak tokenları tek boşluk ile birleştirerek normalize edilmiş bir string döndürdüğünü doğrulamaktır.

    tokenizer = TokenizerFactory.create("white_space")

    text = "hello   world"

    tokenizer.train(text)

    decoded = tokenizer.decode(tokenizer.encode(text))

    assert decoded == "hello world"
    assert decoded != text


def test_whitespace_tokenizer_handles_turkish_characters() -> None:
    """
    Türkçe karakterler whitespace tokenizer tarafından olduğu gibi korunmalıdır.

    WhitespaceTokenizer karakter seviyesinde işlem yapmadığı için token içeriğini değiştirmez; yalnızca whitespace'e göre böler.

    Gerekçe:
        WhitespaceTokenizer karakter seviyesinde işlem yapmaz, bu nedenle Türkçe karakterler gibi özel karakterler tokenize edilirken veya decode edilirken değiştirilmemelidir.
        Bu test, tokenizer'ın Türkçe karakterler gibi özel karakterleri doğru şekilde işlediğini doğrular.

    Input: "çalışma öğrenme dünya"

    Beklenen: decode(encode(input)) == input

    Bu test şunu garanti eder:
        - WhitespaceTokenizer Türkçe karakterler gibi özel karakterleri değiştirmez, tokenize ederken veya decode ederken olduğu gibi korur.
        - encode() ve decode() birlikte tutarlı şekilde çalışır, Türkçe karakterler gibi özel karakterleri doğru şekilde işler.
        - encode() ve decode() birlikte beklenmedik şekilde davranmaz veya anlamsız sonuçlar üretmez, Türkçe karakterler gibi özel karakterleri doğru şekilde işler.
        - encode() ve decode() birlikte çökmez veya bellek hatası vermez, Türkçe karakterler gibi özel karakterleri doğru şekilde işler.
        - encode() ve decode() birlikte sonsuz döngüye girmez veya performans sorunları yaşamaz, Türkçe karakterler gibi özel karakterleri doğru şekilde işler.

    Önlenen bug sınıfı:
        - WhitespaceTokenizer'ın Türkçe karakterler gibi özel karakterleri değiştirmesi beklenirken, encode() ve decode() birlikte beklenmedik şekilde davranması veya anlamsız sonuçlar üretmesi (örneğin, "çalışma öğrenme dünya" -> [0, 1, 2] -> "calisma ogrenme dunya" gibi beklenmedik sonuçlar).
        - encode() ve decode() birlikte çökmesi veya bellek hatası vermesi.
        - encode() ve decode() birlikte sonsuz döngüye girmesi veya performans sorunları yaşaması.
    """
    # testin amacı, WhitespaceTokenizer'ın Türkçe karakterler gibi özel karakterleri değiştirmediğini, tokenize ederken veya decode ederken olduğu gibi koruduğunu doğrulamaktır.

    tokenizer = TokenizerFactory.create("white_space")

    text = "çalışma öğrenme dünya"

    tokenizer.train(text)

    decoded = tokenizer.decode(tokenizer.encode(text))

    assert decoded == text


def test_whitespace_tokenizer_vocab_size_is_zero_before_training() -> None:
    """
    Eğitim öncesinde vocabulary boş olmalıdır.

    Gerekçe:
        Vocabulary oluşturmak için train() çağrılmalıdır. 
        Eğitim öncesinde token -> id mapping yoktur, bu nedenle vocab_size 0 olmalıdır.   
    
    Bu test şunu garanti eder:
        - Eğitim öncesinde tokenizer.vocab_size == 0 olmalıdır.
        - Eğitim öncesinde tokenizer._token_to_id == {} olmalıdır.
        - Eğitim öncesinde tokenizer._id_to_token == {} olmalıdır.
        - Eğitim öncesinde tokenizer state'inin açık ve tahmin edilebilir olduğunu doğrular.

    Önlenen bug sınıfı:
        - Eğitim öncesinde vocabulary'nin yanlış şekilde oluşturulması veya boş olmaması.
        - Eğitim öncesinde vocab_size'ın yanlış hesaplanması (örneğin, 0 yerine başka bir değer).
        - Eğitim öncesinde token_to_id ve id_to_token mapping'lerinin yanlış şekilde oluşturulması (örneğin, boş olmaması veya beklenmedik token/id'ler içermesi).
        - Eğitim öncesinde tokenizer'ın encode/decode işlemlerinde beklenmedik şekilde davranması veya anlamsız sonuçlar üretmesi (örneğin, encode() veya decode() çağrıldığında beklenmedik hatalar vermesi veya anlamsız sonuçlar üretmesi).
        - Eğitim öncesinde tokenizer'ın çökmesi veya bellek hatası vermesi.
        - Eğitim öncesinde tokenizer'ın sonsuz döngüye girmesi veya performans sorunları yaşaması.
    """
    # testin amacı, eğitim öncesinde tokenizer'ın vocabulary'sinin boş olduğunu, vocab_size'ın 0 olduğunu ve token_to_id ile id_to_token mapping'lerinin boş olduğunu doğrulamaktır. Bu, tokenizer'ın eğitim öncesinde açık ve tahmin edilebilir bir state'e sahip olduğunu garanti eder.

    tokenizer = TokenizerFactory.create("white_space")

    assert tokenizer.vocab_size == 0 # eğitim öncesinde vocab_size 0 olmalıdır.
    assert tokenizer._token_to_id == {} # eğitim öncesinde token_to_id mapping boş olmalıdır.
    assert tokenizer._id_to_token == {} # eğitim öncesinde id_to_token mapping boş olmalıdır.