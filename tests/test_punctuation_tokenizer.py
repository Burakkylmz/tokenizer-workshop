from __future__ import annotations

import pytest

from tokenizer_workshop.api.services.tokenizer_factory import TokenizerFactory


# ---------------------------------------------------------
# TRAIN TESTS
# ---------------------------------------------------------

def test_punctuation_tokenizer_train_with_empty_text_raises_error() -> None:
    """
    Boş metinle train() çağrıldığında ValueError beklenmelidir.

    Gerekçe:
        Vocabulary oluşturmak için en az bir gerçek token gerekir.
        Boş string, kelime ya da noktalama tokenı üretmez.

    Bu test şunu garanti eder:
        - Geçersiz eğitim input'u erken aşamada reddedilir.
        - Tokenizer boş vocabulary ile trained state'e geçmez.

    Önlenen bug sınıfı:
        - Boş metinle train() çağrıldığında tokenizer'ın boş bir vocab ile trained state'e geçmesi.
        - Sonrasında encode/decode çağrıldığında beklenmeyen davranışlara veya hatalara yol açması. 
    """
    # testin amacı, PunctuationTokenizer'ın train() metodunun boş metinle çağrıldığında uygun şekilde hata vermesini sağlamaktır.

    tokenizer = TokenizerFactory.create("punctuation")

    with pytest.raises(ValueError, match="Training text cannot be empty"):
        tokenizer.train("")


def test_punctuation_tokenizer_train_with_whitespace_text_raises_error() -> None:
    """
    Sadece whitespace içeren metinle eğitim yapılmamalıdır.

    Gerekçe:
        Whitespace karakterleri bu tokenizer tarafından token olarak saklanmaz.
        Bu yüzden "   " gibi inputlar anlamlı vocabulary üretemez.

    Bu test şunu garanti eder:
        - Sadece whitespace içeren eğitim input'u reddedilir.
        - Tokenizer boş vocabulary ile trained state'e geçmez.
        - Sonrasında encode/decode çağrıldığında beklenmeyen davranışlara veya hatalara yol açması önlenir.

    Önlenen bug sınıfı:
        - Sadece whitespace içeren metinle train() çağrıldığında tokenizer'ın boş bir vocab ile trained state'e geçmesi.
        - Sonrasında encode/decode çağrıldığında beklenmeyen davranışlara veya hatalara yol açması.
    """
    # testin amacı, PunctuationTokenizer'ın train() metodunun sadece whitespace içeren metinle çağrıldığında uygun şekilde hata vermesini sağlamaktır.

    tokenizer = TokenizerFactory.create("punctuation")

    with pytest.raises(ValueError, match="Training text cannot be empty"):
        tokenizer.train("   ")


def test_punctuation_tokenizer_train_builds_vocab() -> None:
    """
    train() sonrası vocabulary oluşturulmalıdır.

    Gerekçe:
        Vocabulary, encode/decode işlemleri için gereklidir.

    Input:
        "Hello, world!"

    Tokenlar:
        ["hello", ",", "world", "!"]

    Beklenen vocab_size: 4

    Bu test şunu garanti eder:
        - train() çağrıldığında tokenizer'ın tokenları doğru şekilde ayrıştırarak vocabulary oluşturması.
        - Vocabulary'nin token sayısını doğru şekilde yansıtması.   

    Önlenen bug sınıfı:
        - train() çağrıldığında tokenizer'ın tokenları doğru şekilde ayrıştıramaması ve dolayısıyla eksik veya hatalı bir vocabulary oluşturması.
        - Vocabulary'nin token sayısını yanlış şekilde yansıtması (örneğin, sadece kelimeleri sayması ve noktalama işaretlerini göz ardı etmesi).   
        - Sonrasında encode/decode işlemlerinde beklenmeyen davranışlara veya hatalara yol açması.  
    """
    # testin amacı, PunctuationTokenizer'ın train() metodunun verilen metni doğru şekilde tokenlara ayırarak vocabulary oluşturmasını ve vocab_size'ın beklenen değeri yansıtmasını sağlamaktır.

    tokenizer = TokenizerFactory.create("punctuation")

    tokenizer.train("Hello, world!")

    assert tokenizer.vocab_size == 4 # "hello", ",", "world", "!"


def test_punctuation_tokenizer_vocab_reflects_unique_tokens() -> None:
    """
    vocab_size unique kelime + noktalama token sayısını yansıtmalıdır.

    Gerekçe:
        Vocabulary oluşturulurken duplicate tokenlar temizlenir, ancak ilk görülme sırası korunur.
        Bu test, tokenizer'ın duplicate tokenları doğru şekilde temizleyip temizlemediğini ve vocab_size'ın unique token sayısını doğru şekilde yansıtıp yansıtmadığını doğrular.   

    Input:
        "Hello, hello!"

    Tokenlar:
        ["hello", ",", "hello", "!"]

    Unique tokenlar:
        ["hello", ",", "!"]

    Beklenen vocab_size: 3

    Bu test şunu garanti eder:
        - train() çağrıldığında tokenizer'ın duplicate tokenları doğru şekilde temizlemesi.
        - Vocabulary'nin unique token sayısını doğru şekilde yansıtması.    

    Önlenen bug sınıfı:
        - train() çağrıldığında tokenizer'ın duplicate tokenları temizlememesi ve dolayısıyla vocabulary'de gereksiz tekrarlar olması.  
        - Vocabulary'nin unique token sayısını yanlış şekilde yansıtması (örneğin, duplicate tokenları da sayması).  
        - Sonrasında encode/decode işlemlerinde beklenmeyen davranışlara veya hatalara yol açması.
    """
    # testin amacı, PunctuationTokenizer'ın train() metodunun verilen metni doğru şekilde tokenlara ayırarak vocabulary oluşturmasını, duplicate tokenları temizlemesini ve vocab_size'ın unique token sayısını doğru şekilde yansıtmasını sağlamaktır.

    tokenizer = TokenizerFactory.create("punctuation")

    tokenizer.train("Hello, hello!")

    assert tokenizer.vocab_size == 3 # "hello", ",", "!"


def test_punctuation_tokenizer_vocab_is_deterministic_for_same_input() -> None:
    """
    Aynı input ile eğitilen iki tokenizer aynı mapping'i üretmelidir.

    Gerekçe:
        Deterministik vocabulary davranışı testlerin, raporların ve
        karşılaştırma çıktılarının stabil kalmasını sağlar.

    Input:
        "Merhaba, dünya! Merhaba."

    Tokenlar:
        ["merhaba", ",", "dünya", "!", "merhaba"]

    Unique tokenlar:
        ["merhaba", ",", "dünya", "!"]

    Beklenen mapping:
        merhaba -> 0
        , -> 1
        dünya -> 2
        ! -> 3  

    Bu test şunu garanti eder:
        - Aynı eğitim metniyle eğitilen iki tokenizer'ın aynı token -> id ve id -> token mapping'lerini üretmesi.
        - Bu da tokenizer'ın eğitim sürecinin deterministik olduğunu ve aynı input için aynı vocabulary oluşturduğunu doğrular. 
    
    Önlenen bug sınıfı:
        - Aynı eğitim metniyle eğitilen iki tokenizer'ın farklı token -> id veya id -> token mapping'leri üretmesi, bu da eğitim sürecinin deterministik olmaması ve aynı input için farklı vocabulary'ler oluşturması anlamına gelir.
        - Sonrasında encode/decode işlemlerinde beklenmeyen davranışlara veya hatalara yol açması.  
    """
    # testin amacı, PunctuationTokenizer'ın train() metodunun aynı eğitim metniyle çağrıldığında aynı token -> id ve id -> token mapping'lerini üretmesini sağlamaktır.

    tokenizer_a = TokenizerFactory.create("punctuation")
    tokenizer_b = TokenizerFactory.create("punctuation")

    text = "Merhaba, dünya! Merhaba."

    tokenizer_a.train(text)
    tokenizer_b.train(text)

    assert tokenizer_a._token_to_id == tokenizer_b._token_to_id # token -> id mapping'lerinin aynı olduğunu doğrular
    assert tokenizer_a._id_to_token == tokenizer_b._id_to_token # id -> token mapping'lerinin aynı olduğunu doğrular


# ---------------------------------------------------------
# TOKENIZE TESTS
# ---------------------------------------------------------

def test_punctuation_tokenizer_tokenize_splits_words_and_punctuation() -> None:
    """
    tokenize() kelimeleri ve noktalama işaretlerini ayrı tokenlara bölmelidir.

    Gerekçe:
        Bu tokenizer'ın temel amacı, kelimeleri ve noktalama işaretlerini ayrı tokenlara ayırmaktır. 
        Bu test, tokenize() metodunun bu temel işlevi doğru şekilde yerine getirip getirmediğini doğrular.

    Input:
        "Hello, world!"

    Beklenen:
        ["hello", ",", "world", "!"]

    Bu test şunu garanti eder:
        - tokenize() metodunun kelimeleri ve noktalama işaretlerini doğru şekilde ayırması.
        - Kelimeleri lowercase normalize etmesi.

    Önlenen bug sınıfı:
        - tokenize() metodunun kelimeleri ve noktalama işaretlerini ayrı tokenlara ayırmaması (örneğin, "hello," gibi tek bir token üretmesi).
        - tokenize() metodunun kelimeleri lowercase normalize etmemesi (örneğin, "Hello" ve "hello" gibi farklı tokenlar üretmesi). 
        - Sonrasında encode/decode işlemlerinde beklenmeyen davranışlara veya hatalara yol açması.
    """
    # testin amacı, PunctuationTokenizer'ın tokenize() metodunun verilen metni doğru şekilde kelimelere ve noktalama işaretlerine ayırarak token listesi üretmesini sağlamaktır.

    tokenizer = TokenizerFactory.create("punctuation")

    tokens = tokenizer.tokenize("Hello, world!")

    assert tokens == ["hello", ",", "world", "!"]


def test_punctuation_tokenizer_tokenize_lowercases_text() -> None:
    """
    tokenize() metni lowercase normalize etmelidir.

    Gerekçe:
        "Hello" ve "hello" farklı tokenlar olarak görülmemelidir.
        Bu, vocabulary fragmentation riskini azaltır.

    Input:  
        "Hello HELLO hello"

    Beklenen:
        ["hello", "hello", "hello"]

    Bu test şunu garanti eder:
        - tokenize() metodunun metni lowercase normalize etmesi.
        - Aynı kelimenin farklı case'lerdeki görünümlerini tek bir token olarak temsil etmesi.

    Önlenen bug sınıfı:
        - tokenize() metodunun metni lowercase normalize etmemesi, bu da aynı kelimenin farklı case'lerdeki görünümlerini farklı tokenlar olarak temsil etmesine yol açar (örneğin, "Hello", "HELLO" ve "hello" gibi üç farklı token üretmesi).
        - Sonrasında encode/decode işlemlerinde beklenmeyen davranışlara veya hatalara yol açması.  
    """
    # testin amacı, PunctuationTokenizer'ın tokenize() metodunun verilen metni lowercase normalize ederek token listesi üretmesini sağlamaktır.
    
    tokenizer = TokenizerFactory.create("punctuation")

    tokens = tokenizer.tokenize("Hello HELLO hello")

    assert tokens == ["hello", "hello", "hello"]


def test_punctuation_tokenizer_tokenize_empty_string_returns_empty_list() -> None:
    """
    Boş string tokenize edildiğinde boş liste dönmelidir.

    Gerekçe:
        Boş string tokenize edildiğinde herhangi bir token üretilmemelidir.
        
    Bu test şunu garanti eder:
        - tokenize() metodunun boş string input'u doğru şekilde işlemesi ve boş liste döndürmesi.

    Önlenen bug sınıfı:
        - tokenize() metodunun boş string input'u yanlış şekilde işlemesi ve None veya hata gibi beklenmeyen bir değer döndürmesi.
        - Sonrasında encode/decode işlemlerinde beklenmeyen davranışlara veya hatalara yol açması.
    """
    # testin amacı, PunctuationTokenizer'ın tokenize() metodunun boş string input'u doğru şekilde işlemesini ve boş liste döndürmesini sağlamaktır.

    tokenizer = TokenizerFactory.create("punctuation")

    assert tokenizer.tokenize("") == [] # tokenize() metodunun boş string input'u doğru şekilde işlemesini ve boş liste döndürmesini doğrular


def test_punctuation_tokenizer_tokenize_whitespace_only_returns_empty_list() -> None:
    """
    Sadece whitespace içeren input tokenize edildiğinde boş liste dönmelidir.

    Gerekçe:
        Whitespace karakterleri bu tokenizer tarafından token olarak saklanmaz.

    Bu test şunu garanti eder:
        - tokenize() metodunun sadece whitespace içeren input'u doğru şekilde işlemesi ve boş liste döndürmesi. 
    
    Önlenen bug sınıfı:
        - tokenize() metodunun sadece whitespace içeren input'u yanlış şekilde işlemesi ve None veya hata gibi beklenmeyen bir değer döndürmesi.
        - Sonrasında encode/decode işlemlerinde beklenmeyen davranışlara veya hatalara yol açması.
    """
    # testin amacı, PunctuationTokenizer'ın tokenize() metodunun sadece whitespace içeren input'u doğru şekilde işlemesini ve boş liste döndürmesini sağlamaktır.

    tokenizer = TokenizerFactory.create("punctuation")

    assert tokenizer.tokenize("   \n\t") == [] # tokenize() metodunun sadece whitespace içeren input'u doğru şekilde işlemesini ve boş liste döndürmesini doğrular  


def test_punctuation_tokenizer_handles_multiple_punctuation_marks() -> None:
    """
    Ardışık noktalama işaretleri ayrı ayrı token olmalıdır.

    Gerekçe:
        "Wait!!!" gibi bir input, ["wait", "!", "!", "!"] şeklinde tokenize edilmelidir.
        Bu, noktalama işaretlerinin kelimelerden ayrı tokenlar olarak temsil edilmesini sağlar.

    Input:
        "Wait!!!"

    Beklenen:
        ["wait", "!", "!", "!"]

    Bu test şunu garanti eder:
        - tokenize() metodunun ardışık noktalama işaretlerini ayrı tokenlar olarak ayırması.
        - Kelimeleri lowercase normalize etmesi.

    Önlenen bug sınıfı:
        - tokenize() metodunun ardışık noktalama işaretlerini ayrı tokenlar olarak ayırmaması (örneğin, "Wait!!!" gibi tek bir token üretmesi).
        - tokenize() metodunun kelimeleri lowercase normalize etmemesi (örneğin, "Wait" ve "wait" gibi farklı tokenlar üretmesi).
        - Sonrasında encode/decode işlemlerinde beklenmeyen davranışlara veya hatalara yol açması.
    """
    # testin amacı, PunctuationTokenizer'ın tokenize() metodunun ardışık noktalama işaretlerini ayrı tokenlar olarak ayırmasını ve kelimeleri lowercase normalize etmesini sağlamaktır.

    tokenizer = TokenizerFactory.create("punctuation")

    tokens = tokenizer.tokenize("Wait!!!")

    assert tokens == ["wait", "!", "!", "!"]


def test_punctuation_tokenizer_handles_tabs_newlines_and_spaces() -> None:
    """
    Whitespace türleri token olarak saklanmamalıdır.

    Gerekçe:
        "Hello\\tworld\\nagain!" gibi bir input, ["hello", "world", "again", "!"] şeklinde tokenize edilmelidir.
        Bu, farklı whitespace türlerinin token olarak saklanmamasını ve kelimelerin doğru şekilde ayrıştırılmasını sağlar.  

    Input:
        "Hello\\tworld\\nagain!"

    Beklenen:
        ["hello", "world", "again", "!"]

    Bu test şunu garanti eder:
        - tokenize() metodunun farklı whitespace türlerini token olarak saklamaması.
        - Kelimeleri lowercase normalize etmesi.

    Önlenen bug sınıfı:
        - tokenize() metodunun farklı whitespace türlerini token olarak saklaması (örneğin, "Hello\\tworld\\nagain!" gibi tek bir token üretmesi).
        - tokenize() metodunun kelimeleri lowercase normalize etmemesi (örneğin, "Hello" ve "hello" gibi farklı tokenlar üretmesi).
        - Sonrasında encode/decode işlemlerinde beklenmeyen davranışlara veya hatalara yol açması.
    """
    # testin amacı, PunctuationTokenizer'ın tokenize() metodunun verilen metni doğru şekilde kelimelere ve noktalama işaretlerine ayırarak token listesi üretmesini sağlamaktır. Özellikle, farklı whitespace türlerini token olarak saklamaması ve kelimeleri lowercase normalize etmesi sağlanmalıdır.

    tokenizer = TokenizerFactory.create("punctuation")

    tokens = tokenizer.tokenize("Hello\tworld\nagain!")

    assert tokens == ["hello", "world", "again", "!"] # tokenize() metodunun verilen metni doğru şekilde kelimelere ve noktalama işaretlerine ayırarak token listesi üretmesini doğrular. Özellikle, farklı whitespace türlerini token olarak saklamaması ve kelimeleri lowercase normalize etmesi doğrulanır.


def test_punctuation_tokenizer_preserves_turkish_characters() -> None:
    """
    Türkçe karakterler kelime tokenı içinde korunmalıdır.

    Gerekçe:
        Tokenizer karakterleri dönüştürmemeli; yalnızca case normalize edip kelime/noktalama ayrımı yapmalıdır.

    Input:
        "Çalışma, öğrenme!"

    Beklenen:
        ["çalışma", ",", "öğrenme", "!"]

    Bu test şunu garanti eder:
        - tokenize() metodunun Türkçe karakterleri kelime tokenları içinde koruması.
        - Kelimeleri lowercase normalize etmesi.

    Önlenen bug sınıfı:
        - tokenize() metodunun Türkçe karakterleri kelime tokenları içinde korumaması (örneğin, "Çalışma" yerine "Calisma" gibi bir token üretmesi).
        - tokenize() metodunun kelimeleri lowercase normalize etmemesi (örneğin, "Çalışma" ve "çalışma" gibi farklı tokenlar üretmesi).
        - Sonrasında encode/decode işlemlerinde beklenmeyen davranışlara veya hatalara yol açması.
    """
    # testin amacı, PunctuationTokenizer'ın tokenize() metodunun Türkçe karakterleri kelime tokenları içinde korumasını ve kelimeleri lowercase normalize etmesini sağlamaktır.

    tokenizer = TokenizerFactory.create("punctuation")

    tokens = tokenizer.tokenize("Çalışma, öğrenme!")

    assert tokens == ["çalışma", ",", "öğrenme", "!"]


def test_punctuation_tokenizer_differs_from_whitespace_behavior() -> None:
    """
    PunctuationTokenizer noktalama işaretlerini kelimeden ayırmalıdır.

    Bu test, tokenizer'ın WhitespaceTokenizer gibi davranmadığını özellikle dokümante eder.

    Gerekçe:
        WhitespaceTokenizer'da "hello, world!" -> ["hello,", "world!"] iken, PunctuationTokenizer'da "hello, world!" -> ["hello", ",", "world", "!"] olmalıdır. 
        Bu test, PunctuationTokenizer'ın tokenize() metodunun kelimeleri ve noktalama işaretlerini ayrı tokenlara ayırarak WhitespaceTokenizer'dan farklı davrandığını doğrular.    

    Input:
        "hello, world!"   

    WhitespaceTokenizer:
        "hello, world!" -> ["hello,", "world!"]

    PunctuationTokenizer:
        "hello, world!" -> ["hello", ",", "world", "!"]

    Bu test şunu garanti eder:
        - PunctuationTokenizer'ın tokenize() metodunun kelimeleri ve noktalama işaretlerini ayrı tokenlara ayırarak WhitespaceTokenizer'dan farklı davrandığını doğrular.   
    
    Önlenen bug sınıfı:
        - PunctuationTokenizer'ın tokenize() metodunun WhitespaceTokenizer gibi davranması, yani noktalama işaretlerini kelimelerle birlikte tek token olarak üretmesi (örneğin, "hello, world!" gibi tek bir token üretmesi).
        - Sonrasında encode/decode işlemlerinde beklenmeyen davranışlara veya hatalara yol açması.
    """
    # testin amacı, PunctuationTokenizer'ın tokenize() metodunun kelimeleri ve noktalama işaretlerini ayrı tokenlara ayırarak WhitespaceTokenizer'dan farklı davrandığını doğrulamaktır.

    tokenizer = TokenizerFactory.create("punctuation")

    tokens = tokenizer.tokenize("hello, world!")

    assert tokens != ["hello,", "world!"] # PunctuationTokenizer'ın WhitespaceTokenizer gibi davranmadığını doğrular
    assert tokens == ["hello", ",", "world", "!"] # PunctuationTokenizer'ın tokenize() metodunun kelimeleri ve noktalama işaretlerini ayrı tokenlara ayırarak WhitespaceTokenizer'dan farklı davrandığını doğrular  


# ---------------------------------------------------------
# ENCODE TESTS
# ---------------------------------------------------------

def test_punctuation_tokenizer_encode_before_training_raises_error() -> None:
    """
    train() çağrılmadan encode() çalışmamalıdır.

    Gerekçe:
        Encode işlemi token -> id mapping'e ihtiyaç duyar.
        Bu mapping train() sırasında oluşturulur.

    Bu test şunu garanti eder:
        - train() çağrılmadan encode() çağrıldığında ValueError beklenir.
        - Bu, encode() işleminin tokenizer'ın eğitim durumuna bağlı olduğunu doğrular.

    Önlenen bug sınıfı:
        - train() çağrılmadan encode() çağrıldığında hata vermemesi, bu da encode() işleminin tokenizer'ın eğitim durumuna bağlı olmaması ve beklenmeyen davranışlara yol açması anlamına gelir.
        - Sonrasında encode() işlemi beklenmedik sonuçlar üretmesi veya hatalara yol açması.
    """
    # testin amacı, PunctuationTokenizer'ın encode() metodunun train() çağrılmadan önce çağrıldığında uygun şekilde hata vermesini sağlamaktır.

    tokenizer = TokenizerFactory.create("punctuation")

    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.encode("Hello, world!")


def test_punctuation_tokenizer_encode_returns_integer_ids() -> None:
    """
    encode() integer token id listesi döndürmelidir.

    Gerekçe:
        Tokenizer'ın encode() metodunun çıktısı, token -> id mapping'e göre integer id listesi olmalıdır.
        Bu test, encode() metodunun doğru türde ve beklenen formatta çıktı ürettiğini doğrular.

    Input:
        "Hello, world!"

    Eğitim sonrası beklenen mapping:
        hello -> 0
        ,     -> 1
        world -> 2
        !     -> 3

    Beklenen encode:
        [0, 1, 2, 3]

    Bu test şunu garanti eder:
        - encode() metodunun token -> id mapping'e göre integer token id listesi döndürmesi.
        - Döndürülen token id'lerin integer türünde olması.

    Önlenen bug sınıfı:
        - encode() metodunun token -> id mapping'e göre integer token id listesi döndürmemesi, bu da encode() çıktısının beklenen formatta olmaması ve beklenmeyen davranışlara yol açması anlamına gelir.
        - encode() metodunun token id'lerini integer türünde döndürmemesi, bu da encode() çıktısının beklenen türde olmaması ve beklenmeyen davranışlara yol açması anlamına gelir. 
        - Sonrasında encode() işlemi beklenmedik sonuçlar üretmesi veya hatalara yol açması.
    """
    # testin amacı, PunctuationTokenizer'ın encode() metodunun token -> id mapping'e göre integer token id listesi döndürmesini ve döndürülen token id'lerin integer türünde olmasını sağlamaktır.

    tokenizer = TokenizerFactory.create("punctuation")

    tokenizer.train("Hello, world!")

    encoded = tokenizer.encode("Hello, world!")

    assert isinstance(encoded, list) # encode() metodunun bir liste döndürdüğünü doğrular
    assert encoded == [0, 1, 2, 3] # encode() metodunun token -> id mapping'e göre beklenen integer token id listesi döndürdüğünü doğrular
    assert all(isinstance(token_id, int) for token_id in encoded) # encode() metodunun döndürdüğü token id'lerin integer türünde olduğunu doğrular


def test_punctuation_tokenizer_encode_unknown_token_raises_error() -> None:
    """
    Eğitim sırasında görülmeyen token encode edilmeye çalışılırsa hata vermelidir.

    Gerekçe:
        Encode işlemi token -> id mapping'e ihtiyaç duyar.
        Eğitim sırasında "python" tokenı görülmemişse, encode("python") çağrıldığında ValueError beklenir.

    Train text:
        "hello, world!"

    Encode text:
        "hello, python!"

    "python" vocabulary içinde olmadığı için ValueError beklenir.

    Bu test şunu garanti eder:
        - Eğitim sırasında görülmeyen tokenların encode edilmeye çalışıldığında uygun şekilde hata vermesini sağlar.
        - Bu, encode() işleminin tokenizer'ın eğitim durumuna ve oluşturulan vocabulary'e bağlı olduğunu doğrular.

    Önlenen bug sınıfı:
        - Eğitim sırasında görülmeyen tokenların encode edilmeye çalışıldığında hata vermemesi, bu da encode() işleminin tokenizer'ın eğitim durumuna ve oluşturulan vocabulary'e bağlı olmaması ve beklenmeyen davranışlara yol açması anlamına gelir.
        - Sonrasında encode() işlemi beklenmedik sonuçlar üretmesi veya hatalara yol açması.
    """
    # testin amacı, PunctuationTokenizer'ın encode() metodunun eğitim sırasında görülmeyen tokenların encode edilmeye çalışıldığında uygun şekilde hata vermesini sağlamaktır.

    tokenizer = TokenizerFactory.create("punctuation")

    tokenizer.train("hello, world!")

    with pytest.raises(ValueError, match="Unknown token"):
        tokenizer.encode("hello, python!")


def test_punctuation_tokenizer_encode_empty_text_returns_empty_list() -> None:
    """
    Eğitim sonrası boş metin encode edildiğinde boş liste dönmelidir.

    Bu edge case API/report tarafında güvenli davranış sağlar.

    Gerekçe:
        Encode işlemi token -> id mapping'e ihtiyaç duyar, ancak boş metin tokenize edildiğinde herhangi bir token üretilmemelidir.
        Bu durumda encode() metodunun boş liste döndürmesi beklenir.

    Train text:
        "hello, world!"

    Encode text:
        ""

    Beklenen encode:
        []

    Bu test şunu garanti eder:
        - Eğitim sonrası boş metin encode edildiğinde encode() metodunun boş liste döndürmesini sağlar.
        - Bu, encode() metodunun boş metin edge case'ini doğru şekilde ele aldığını doğrular.

    Önlenen bug sınıfı:
        - Eğitim sonrası boş metin encode edildiğinde encode() metodunun boş liste döndürmemesi, bu da encode() metodunun boş metin edge case'ini yanlış şekilde ele alması ve beklenmeyen davranışlara yol açması anlamına gelir.
        - Sonrasında encode() işlemi beklenmedik sonuçlar üretmesi veya hatalara yol açması.
    """
    # testin amacı, PunctuationTokenizer'ın encode() metodunun eğitim sonrası boş metin encode edildiğinde boş liste döndürmesini sağlamaktır.

    tokenizer = TokenizerFactory.create("punctuation")

    tokenizer.train("hello, world!")

    assert tokenizer.encode("") == [] # encode() metodunun eğitim sonrası boş metin encode edildiğinde boş liste döndürmesini doğrular


def test_punctuation_tokenizer_encode_is_case_insensitive_due_to_lowercase_normalization() -> None:
    """
    encode() lowercase normalization nedeniyle case-insensitive davranmalıdır.

    Gerekçe:
        Tokenizer'ın tokenize() metodunun metni lowercase normalize etmesi nedeniyle, encode() metodunun da case-insensitive davranması beklenir.
        Bu test, encode() metodunun tokenize() metodunun lowercase normalization davranışını doğru şekilde yansıttığını doğrular.

    Train text:
        "Hello, World!"

    Encode text:
        "hello, world!"

    Beklenti:
        Aynı token id dizisi üretilir.

    Bu test şunu garanti eder:
        - encode() metodunun tokenize() metodunun lowercase normalization davranışını doğru şekilde yansıtması, yani case-insensitive davranması.
        - Bu, encode() metodunun tokenize() metodunun davranışına bağlı olduğunu doğrular.

    Önlenen bug sınıfı:
        - encode() metodunun tokenize() metodunun lowercase normalization davranışını doğru şekilde yansıtmaması, bu da encode() metodunun case-sensitive davranmasına ve aynı kelimenin farklı case'lerdeki görünümlerini farklı token id'leri olarak temsil etmesine yol açar (örneğin, "Hello" ve "hello" gibi farklı token id'leri üretmesi).
        - Sonrasında encode() işlemi beklenmedik sonuçlar üretmesi veya hatalara yol açması.
    """
    # testin amacı, PunctuationTokenizer'ın encode() metodunun tokenize() metodunun lowercase normalization davranışını doğru şekilde yansıtarak case-insensitive davranmasını sağlamaktır.

    tokenizer = TokenizerFactory.create("punctuation")

    tokenizer.train("Hello, World!")

    encoded_upper = tokenizer.encode("Hello, World!")
    encoded_lower = tokenizer.encode("hello, world!")

    assert encoded_upper == encoded_lower # encode() metodunun tokenize() metodunun lowercase normalization davranışını doğru şekilde yansıtarak case-insensitive davrandığını doğrular
    assert encoded_upper == [0, 1, 2, 3] # encode() metodunun tokenize() metodunun lowercase normalization davranışını doğru şekilde yansıtarak case-insensitive davrandığını doğrular
    assert encoded_lower == [0, 1, 2, 3] # encode() metodunun tokenize() metodunun lowercase normalization davranışını doğru şekilde yansıtarak case-insensitive davrandığını doğrular


# ---------------------------------------------------------
# DECODE TESTS
# ---------------------------------------------------------

def test_punctuation_tokenizer_decode_before_training_raises_error() -> None:
    """
    train() çağrılmadan decode() çalışmamalıdır.

    Çünkü decode işlemi id -> token mapping'e ihtiyaç duyar.

    Gerekçe:
        Decode işlemi id -> token mapping'e ihtiyaç duyar.
        Bu mapping train() sırasında oluşturulur.

    Bu test şunu garanti eder:
        - train() çağrılmadan decode() çağrıldığında ValueError beklenir
        - decode() metodunun train() çağrılmadan önce çağrıldığında uygun şekilde hata vermesini sağlar ve decode() işleminin tokenizer'ın eğitim durumuna bağlı olduğunu doğrular.

    Önlenen bug sınıfı:
        - decode() metodunun train() çağrılmadan önce çağrıldığında beklenmedik davranış sergilemesi veya sessizce hatalı sonuçlar üretmesi.
        - Sonrasında decode() işlemi beklenmedik sonuçlar üretmesi veya hatalara yol açması.
    """
    # testin amacı, PunctuationTokenizer'ın decode() metodunun train() çağrılmadan önce çağrıldığında uygun şekilde hata vermesini sağlamaktır.

    tokenizer = TokenizerFactory.create("punctuation")

    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.decode([0, 1])


def test_punctuation_tokenizer_decode_returns_text_with_space_join() -> None:
    """
    decode() token id listesini string'e dönüştürmelidir.

    Gerekçe:
        Decode işlemi, token id listesini string'e dönüştürmek için kullanılır.
        Bu, model çıktılarının insan tarafından okunabilir hale getirilmesini sağlar.   

    Not:
        Decode basit " ".join(...) kullandığı için punctuation çevresinde boşluk üretir.

    Input ids:
        [0, 1, 2, 3]

    Beklenen:
        "hello , world !"

    Bu test şunu garanti eder:
        - decode() metodunun token id listesini string'e dönüştürmesi.
        - Decode işleminin token id'lerini doğru şekilde tokenlara dönüştürmesi ve bunları space ile birleştirmesi.
    
    Önlenen bug sınıfı:
        - decode() metodunun token id listesini string'e dönüştürmemesi veya beklenmeyen formatta dönüştürmesi (örneğin, tokenları doğru şekilde birleştirmemesi veya yanlış tokenlara dönüştürmesi).
        - Sonrasında decode() işlemi beklenmedik sonuçlar üretmesi veya hatalara yol açması.
    """
    # testin amacı, PunctuationTokenizer'ın decode() metodunun token id listesini string'e dönüştürmesini ve decode işleminin token id'lerini doğru şekilde tokenlara dönüştürerek bunları space ile birleştirmesini sağlamaktır.

    tokenizer = TokenizerFactory.create("punctuation")

    tokenizer.train("hello, world!")

    decoded = tokenizer.decode([0, 1, 2, 3])

    assert decoded == "hello , world !"


def test_punctuation_tokenizer_decode_unknown_token_id_raises_error() -> None:
    """
    Vocabulary içinde olmayan token id decode edilmeye çalışılırsa hata vermelidir.

    Bu strict davranış, hatalı model çıktılarının sessizce kabul edilmesini engeller.

    Gerekçe:
        Decode işlemi id -> token mapping'e ihtiyaç duyar.
        Eğitim sırasında oluşturulan vocabulary'e göre token id'leri tokenlara dönüştürür.
        Eğer decode edilmeye çalışılan token id'si vocabulary içinde yoksa, bu durum hatalı model çıktılarının sessizce kabul edilmesine yol açabilir.
        Bu test, decode() metodunun vocabulary içinde olmayan token id'lerin decode edilmeye çalışıldığında uygun şekilde hata vermesini sağlayarak bu durumu önler.

    Train text:
        "hello, world!"

    Decode ids:
        [999]

    Beklenen:
        ValueError: "Unknown token id: 999"

    Bu test şunu garanti eder:
        - Vocabulary içinde olmayan token id'lerin decode edilmeye çalışıldığında uygun şekilde hata vermesini sağlar.
        - Bu, decode() işleminin tokenizer'ın oluşturulan vocabulary'e bağlı olduğunu doğrular.

    Önlenen bug sınıfı:
        - Vocabulary içinde olmayan token id'lerin decode edilmeye çalışıldığında hata vermemesi, bu da decode() işleminin tokenizer'ın oluşturulan vocabulary'e bağlı olmaması ve beklenmeyen davranışlara yol açması anlamına gelir.
        - Sonrasında decode() işlemi beklenmedik sonuçlar üretmesi veya hatalara yol açması.
    """
    # testin amacı, PunctuationTokenizer'ın decode() metodunun vocabulary içinde olmayan token id'lerin decode edilmeye çalışıldığında uygun şekilde hata vermesini sağlamaktır.

    tokenizer = TokenizerFactory.create("punctuation")

    tokenizer.train("hello, world!")

    with pytest.raises(ValueError, match="Unknown token id"):
        tokenizer.decode([999])


def test_punctuation_tokenizer_decode_empty_list_returns_empty_string() -> None:
    """
    Boş token id listesi decode edildiğinde boş string dönmelidir.

    Bu edge case API/report tarafında güvenli davranış sağlar.

    Gerekçe:
        Decode işlemi boş token id listesiyle çağrıldığında herhangi bir token üretilmemelidir.
        Bu durumda decode() metodunun boş string döndürmesi beklenir.

    Train text:
        "hello, world!"

    Decode ids:
        []

    Beklenen decode:
        ""

    Bu test şunu garanti eder:
        - Decode işlemi boş token id listesiyle çağrıldığında decode() metodunun boş string döndürmesini sağlar.
        - Bu, decode() metodunun boş token id listesi edge case'ini doğru şekilde ele aldığını doğrular.

    Önlenen bug sınıfı:
        - Decode işlemi boş token id listesiyle çağrıldığında decode() metodunun boş string döndürmemesi, bu da decode() metodunun boş token id listesi edge case'ini yanlış şekilde ele alması ve beklenmeyen davranışlara yol açması anlamına gelir.
        - Sonrasında decode() işlemi beklenmedik sonuçlar üretmesi veya hatalara yol açması.
    """
    # testin amacı, PunctuationTokenizer'ın decode() metodunun boş token id listesiyle çağrıldığında boş string döndürmesini sağlamaktır.

    tokenizer = TokenizerFactory.create("punctuation")

    tokenizer.train("hello, world!")

    assert tokenizer.decode([]) == "" # decode() metodunun boş token id listesiyle çağrıldığında boş string döndürmesini doğrular   


# ---------------------------------------------------------
# ROUNDTRIP / BEHAVIOR TESTS
# ---------------------------------------------------------

def test_punctuation_tokenizer_encode_decode_roundtrip_normalizes_spacing() -> None:
    """
    PunctuationTokenizer roundtrip sırasında spacing'i normalize eder.

    Gerekçe:
        PunctuationTokenizer'ın decode() metodunun tokenları space ile birleştirdiği bilindiği için, encode/decode roundtrip'inin spacing'i normalize etmesi beklenir.
        Bu test, encode/decode roundtrip'inin spacing'i normalize ettiğini doğrular.

    Input:
        "hello, world!"

    Tokenlar:
        ["hello", ",", "world", "!"]

    Decode:
        "hello , world !"

    Bu davranış bilinçlidir çünkü decoder basit space-join kullanır.

    Bu test şunu garanti eder:
        - PunctuationTokenizer'ın encode/decode roundtrip'inin spacing'i normalize ettiğini doğrular.
        - Bu, PunctuationTokenizer'ın decode() metodunun tokenları space ile birleştirdiği bilindiği için beklenen bir davranıştır.

    Önlenen bug sınıfı:
        - PunctuationTokenizer'ın encode/decode roundtrip'inin spacing'i normalize etmemesi, bu da encode/decode roundtrip'inin beklenmeyen şekilde davranmasına ve beklenmeyen sonuçlara yol açması anlamına gelir.
        - Sonrasında encode/decode işlemi beklenmedik sonuçlar üretmesi veya hatalara yol açması.
    """
    # testin amacı, PunctuationTokenizer'ın encode/decode roundtrip'inin spacing'i normalize ettiğini doğrulamaktır. Özellikle, decode() metodunun tokenları space ile birleştirdiği bilindiği için, encode/decode roundtrip'inin spacing'i normalize etmesi beklenir ve bu test bu davranışı doğrular.

    tokenizer = TokenizerFactory.create("punctuation")

    text = "hello, world!"

    tokenizer.train(text)

    decoded = tokenizer.decode(tokenizer.encode(text))

    assert decoded == "hello , world !" # PunctuationTokenizer'ın encode/decode roundtrip'inin spacing'i normalize ettiğini doğrular. Özellikle, decode() metodunun tokenları space ile birleştirdiği bilindiği için, encode/decode roundtrip'inin spacing'i normalize etmesi beklenir ve bu test bu davranışı doğrular.
    assert decoded != text # PunctuationTokenizer'ın encode/decode roundtrip'inin spacing'i normalize ettiğini doğrular. Özellikle, decode() metodunun tokenları space ile birleştirdiği bilindiği için, encode/decode roundtrip'inin spacing'i normalize etmesi beklenir ve bu test bu davranışı doğrular. 


def test_punctuation_tokenizer_handles_turkish_roundtrip_with_normalized_spacing() -> None:
    """
    Türkçe karakterler encode/decode sürecinde korunmalıdır.

    Decode spacing normalize eder, fakat token içerikleri bozulmamalıdır.

    Gerekçe:
        PunctuationTokenizer'ın encode/decode roundtrip'inin Türkçe karakterleri korumasını ve decode işleminin spacing'i normalize etmesini sağlamaktır. Özellikle, decode() metodunun tokenları space ile birleştirdiği bilindiği için, encode/decode roundtrip'inin spacing'i normalize etmesi beklenir, ancak Türkçe karakterlerin korunması da önemlidir.

    Input:
        "Çalışma, öğrenme!"

    Tokenlar:
        ["çalışma", ",", "öğrenme", "!"]

    Decode:
        "çalışma , öğrenme !"

    Bu test şunu garanti eder:
        - PunctuationTokenizer'ın encode/decode roundtrip'inin Türkçe karakterleri koruduğunu doğrular.
        - PunctuationTokenizer'ın encode/decode roundtrip'inin spacing'i normalize ettiğini doğrular. Özellikle, decode() metodunun tokenları space ile birleştirdiği bilindiği için, encode/decode roundtrip'inin spacing'i normalize etmesi beklenir, ancak Türkçe karakterlerin korunması da önemlidir.

    Önlenen bug sınıfı:
        - PunctuationTokenizer'ın encode/decode roundtrip'inin Türkçe karakterleri korumaması, bu da encode/decode roundtrip'inin beklenmeyen şekilde davranmasına ve beklenmeyen sonuçlara yol açması anlamına gelir.
        - PunctuationTokenizer'ın encode/decode roundtrip'inin spacing'i normalize etmemesi, bu da encode/decode roundtrip'inin beklenmeyen şekilde davranmasına ve beklenmeyen sonuçlara yol açması anlamına gelir.
        - Sonrasında encode/decode işlemi beklenmedik sonuçlar üretmesi veya hat
    """
    # testin amacı, PunctuationTokenizer'ın encode/decode roundtrip'inin Türkçe karakterleri korumasını ve decode işleminin spacing'i normalize etmesini sağlamaktır. Özellikle, decode() metodunun tokenları space ile birleştirdiği bilindiği için, encode/decode roundtrip'inin spacing'i normalize etmesi beklenir, ancak Türkçe karakterlerin korunması da önemlidir.
    tokenizer = TokenizerFactory.create("punctuation")

    text = "Çalışma, öğrenme!"

    tokenizer.train(text)

    decoded = tokenizer.decode(tokenizer.encode(text))

    assert decoded == "çalışma , öğrenme !" # Türkçe karakterlerin encode/decode sürecinde korunduğunu doğrular. Decode spacing normalize eder, fakat token içerikleri bozulmamalıdır.
    assert decoded != text # Türkçe karakterlerin encode/decode sürecinde korunduğunu doğrular. Decode spacing normalize eder, fakat token içerikleri bozulmamalıdır.   

def test_punctuation_tokenizer_vocab_size_is_zero_before_training() -> None:
    """
    Eğitim öncesinde vocabulary boş olmalıdır.

    Gerekçe:
        Tokenizer'ın eğitim öncesinde herhangi bir token -> id mapping'e sahip olmaması beklenir.

    Bu test şunu garanti eder:
        - Eğitim öncesinde tokenizer'ın vocab_size'ının 0 olduğunu doğrular.
        - Eğitim öncesinde tokenizer'ın herhangi bir token -> id mapping'e sahip olmadığını doğrular.
        - Eğitim öncesinde tokenizer state'inin açık ve tahmin edilebilir olduğunu doğrular.
        - Eğitim öncesinde encode/decode işleminin uygun şekilde hata vermesini sağlar.

    Önlenen bug sınıfı:
        - Eğitim öncesinde tokenizer'ın vocab_size'ının 0 olmaması, bu da tokenizer'ın eğitim öncesinde beklenmeyen token -> id mapping'lerine sahip olması ve beklenmeyen davranışlara yol açması anlamına gelir.
        - Eğitim öncesinde tokenizer state'inin belirsiz veya hatalı olması, bu da eğitim sürecinde beklenmeyen sonuçlara veya hatalara yol açması anlamına gelir.
        - Sonrasında encode/decode işlemi beklenmedik sonuçlar üretmesi veya hatalara yol açması.
    """
    # testin amacı, PunctuationTokenizer'ın eğitim öncesinde vocab_size'ının 0 olduğunu doğrulamaktır. 
    # Bu, tokenizer'ın eğitim öncesinde herhangi bir token -> id mapping'e sahip olmadığını ve tokenizer state'inin açık ve tahmin edilebilir olduğunu doğrular. Ayrıca, eğitim öncesinde encode/decode işleminin uygun şekilde hata vermesini sağlar.

    tokenizer = TokenizerFactory.create("punctuation")

    assert tokenizer.vocab_size == 0 # Eğitim öncesinde vocabulary'nin boş olduğunu doğrular 

    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.encode("Hello, world!") # Eğitim öncesinde encode() metodunun uygun şekilde hata vermesini sağlar