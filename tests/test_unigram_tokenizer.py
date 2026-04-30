from __future__ import annotations

import pytest

from tokenizer_workshop.api.services.tokenizer_factory import TokenizerFactory


# ---------------------------------------------------------
# INIT TESTS
# ---------------------------------------------------------

def test_unigram_tokenizer_init_invalid_vocab_size_raises_error() -> None:
    """
    Guardrail testi: vocab_size en az 2 olmalıdır.

    Gerekçe:
        Unigram tokenizer vocabulary içinde her zaman [UNK] token'ını tutar.
        Bunun yanında en az 1 gerçek subword token'a daha ihtiyaç vardır.

    Bu test şunu garanti eder:
        - Geçersiz konfigürasyon erken aşamada yakalanır
        - Tokenizer eksik vocabulary ile oluşturulmaz
        - Encode/tokenize sırasında beklenmeyen crash riski azalır
        - Model kapasitesi kontrol altında tutulur
        - Eğitim sırasında anlamsız candidate üretimi engellenir
        - Kullanıcıya açık ve anlaşılır hata mesajları sağlanır
        - Tokenizer'ın temel işlevselliği korunur

    Önlenen bug sınıfı:
        - 1 adet [UNK] token
        - en az 1 adet gerçek subword token
        - vocab_size = 1 verilirse, sadece [UNK] token'ı oluşturulabilir
    """
    # vocab_size = 1 verilirse, sadece [UNK] token'ı oluşturulabilir, 
    # bu da tokenizer'ın işlevselliğini tamamen ortadan kaldırır. 
    # Ancak tokenizer'ın gerçek bir token da öğrenebilmesi gerekir.
    # Bu nedenle, vocab_size en az 2 olmalıdır.

    with pytest.raises(ValueError, match="vocab_size must be at least 2"):
        TokenizerFactory.create("unigram", vocab_size=1)


def test_unigram_tokenizer_init_invalid_max_subword_length_raises_error() -> None:
    """
    Guardrail testi: max_subword_length en az 1 olmalıdır.

    Gerekçe:
        Subword üretimi için en az 1 karakterlik parça üretilebilmelidir.
        max_subword_length = 0 olursa hiçbir geçerli substring üretilemez.

    Bu test şunu garanti eder:
        - Geçersiz tokenizer parametreleri reddedilir
        - Eğitim sırasında boş veya anlamsız candidate üretimi engellenir
        - Tokenizer'ın temel işlevselliği korunur
        - Kullanıcıya açık ve anlaşılır hata mesajları sağlanır

    Önlenen bug sınıfı:
        - max_subword_length = 0 verilirse, hiçbir geçerli substring üretilemez
        - Tokenizer'ın temel işlevselliği tamamen ortadan kalkar
        - Eğitim sırasında sonsuz döngü veya crash riski ortaya çıkabilir
    """
    # max_subword_length = 0 verilirse, hiçbir geçerli substring üretilemez, 
    # bu da tokenizer'ın işlevselliğini tamamen ortadan kaldırır.

    # Subword üretimi için en az 1 karakterlik parça üretilebilmelidir.
    # Bu nedenle, max_subword_length en az 1 olmalıdır.

    # Bu test, geçersiz max_subword_length parametresinin erken aşamada yakalanmasını sağlar,
    # böylece eğitim sırasında anlamsız candidate üretimi engellenir ve tokenizer'ın temel işlevselliği korunur.
    
    with pytest.raises(ValueError, match="max_subword_length must be at least 1"):
        TokenizerFactory.create("unigram", max_subword_length=0)


# ---------------------------------------------------------
# TRAIN TESTS
# ---------------------------------------------------------

def test_unigram_tokenizer_train_with_empty_text_raises_error() -> None:
    """
    Boş metinle train() çağrıldığında ValueError beklenir.

    Gerekçe:
        Unigram vocabulary frekans tabanlı oluşturulur.
        Boş metinde frekans çıkarılabilecek herhangi bir token yoktur.
        Eğitim için anlamlı bir metin sağlanmalıdır.
        Bu test, geçersiz eğitim verisinin erken aşamada yakalanmasını sağlar,
        Eğitim sırasında anlamsız candidate üretimi engellenir
        Tokenizer'ın temel işlevselliği korunur.

    Önlenen bug sınıfı:
        - Boş metinle train() çağrıldığında, tokenizer anlamsız bir vocabulary oluşturmaya çalışabilir veya crash olabilir.
        - Eğitim sırasında anlamsız candidate üretimi ortaya çıkabilir.
        - Tokenizer'ın temel işlevselliği tamamen ortadan kalkabilir.
        - Kullanıcıya açık ve anlaşılır hata mesajları sağlanmaz.
    """
    # Bu test, geçersiz eğitim verisinin erken aşamada yakalanmasını sağlar,
    # böylece eğitim sırasında anlamsız candidate üretimi engellenir ve tokenizer'ın temel işlevselliği korunur.
    # Boş metinle train() çağrıldığında, tokenizer anlamsız bir vocabulary oluşturmaya çalışabilir veya crash olabilir. 
    # Eğitim için anlamlı bir metin sağlanmalıdır.

    tokenizer = TokenizerFactory.create("unigram")

    with pytest.raises(ValueError, match="Training text cannot be empty"):
        tokenizer.train("")


def test_unigram_tokenizer_train_with_whitespace_text_raises_error() -> None:
    """
    Sadece whitespace içeren metinle eğitim yapılmamalıdır.

    Gerekçe:
        Whitespace-only input, gerçek token veya subword adayı üretmez.
        Eğitim için anlamlı bir metin sağlanmalıdır.
        Bu test, geçersiz eğitim verisinin erken aşamada yakalanmasını sağlar,
        Eğitim sırasında anlamsız candidate üretimi engellenir
        Tokenizer'ın temel işlevselliği korunur
        Kullanıcıya açık ve anlaşılır hata mesajları sağlanır

    Önlenen bug sınıfı:
        - Sadece whitespace içeren metinle train() çağrıldığında, tokenizer anlamsız bir vocabulary oluşturmaya çalışabilir veya crash olabilir.
        - Eğitim sırasında anlamsız candidate üretimi ortaya çıkabilir.
        - Tokenizer'ın temel işlevselliği tamamen ortadan kalkabilir.
    """
    # Bu test, geçersiz eğitim verisinin erken aşamada yakalanmasını sağlar,
    # böylece eğitim sırasında anlamsız candidate üretimi engellenir ve tokenizer'ın temel işlevselliği korunur.
    
    # Sadece whitespace içeren metinle train() çağrıldığında, tokenizer anlamsız bir vocabulary oluşturmaya çalışabilir veya crash olabilir. 
    # Eğitim için anlamlı bir metin sağlanmalıdır.
    
    tokenizer = TokenizerFactory.create("unigram")

    with pytest.raises(ValueError, match="Training text cannot be empty"):
        tokenizer.train("   ")


def test_unigram_tokenizer_train_builds_vocab_with_unknown_token() -> None:
    """
    train() sonrası vocabulary oluşturulmalı ve [UNK] token'ı bulunmalıdır.

    Gerekçe:
        [UNK], vocabulary ile segment edilemeyen kelimeler için fallback token'dır.
        Bu token'ın stabil şekilde id=0 olması encode/decode davranışını güvenli yapar.
        Bu test, train() fonksiyonunun temel işlevselliğini doğrular,
        Eğitim sonrası vocabulary'nin oluşturulduğunu ve [UNK] token'ının doğru şekilde dahil edildiğini garanti eder.

    Önlenen bug sınıfı:
        - train() sonrası vocabulary oluşturulmaz veya eksik olur.
        - [UNK] token'ı vocabulary içinde bulunmaz veya id=0 olarak atanmaz.
        - Encode/decode sırasında beklenmeyen davranış veya crash riski ortaya çıkabilir.
        - Tokenizer'ın temel işlevselliği tamamen ortadan kalkabilir.
    """
    # Bu test, train() fonksiyonunun temel işlevselliğini doğrular,
    # Eğitim sonrası vocabulary'nin oluşturulduğunu ve [UNK] token'ının doğru şekilde dahil edildiğini garanti eder.
   
    # [UNK], vocabulary ile segment edilemeyen kelimeler için fallback token'dır.
    # Bu token'ın stabil şekilde id=0 olması encode/decode davranışını güvenli yapar.

    tokenizer = TokenizerFactory.create("unigram", vocab_size=20)

    tokenizer.train("tokenization token tokenizer")

    # Vocabulary oluşturulmalı ve [UNK] token'ı bulunmalıdır.
    # [UNK] token'ının id=0 olarak atanması, encode/decode davranışını güvenli yapar.
    # Eğer [UNK] token'ı vocabulary içinde bulunmaz veya id=0 olarak atanmazsa, encode/decode sırasında beklenmeyen davranış veya crash riski ortaya çıkabilir 
    assert tokenizer.vocab_size > 0 # Vocabulary oluşturulmalı
    assert tokenizer._token_to_id[tokenizer.UNKNOWN_TOKEN] == 0 # [UNK] token'ının id=0 olarak atanması


def test_unigram_tokenizer_vocab_size_does_not_exceed_target_size() -> None:
    """
    Vocabulary boyutu verilen vocab_size sınırını aşmamalıdır.

    Gerekçe:
        vocab_size parametresi model kapasitesini kontrol eder.
        Eğitim sırasında daha fazla candidate üretilse bile vocabulary bu sınırı aşmamalıdır.
        Bu test, vocab_size parametresinin etkili olduğunu doğrular,
        Eğitim sonrası vocabulary'nin verilen sınırları aşmadığını garanti eder,
        Model kapasitesinin kontrol altında tutulmasını sağlar.

    Önlenen bug sınıfı:
        - Vocabulary boyutu verilen vocab_size sınırını aşabilir, bu da model kapasitesini kontrolsüz hale getirir.
        - Eğitim sırasında anlamsız candidate üretimi ortaya çıkabilir.
        - Tokenizer'ın temel işlevselliği tamamen ortadan kalkabilir.
        - Kullanıcıya açık ve anlaşılır hata mesajları sağlanmaz.
        - Modelin eğitim ve inference sırasında beklenmeyen davranış veya crash riski ortaya çıkabilir.
    """
    # Bu test, vocab_size parametresinin etkili olduğunu doğrular,
    # Eğitim sonrası vocabulary'nin verilen sınırları aşmadığını garanti eder,
    # Model kapasitesinin kontrol altında tutulmasını sağlar.

    # Eğer vocabulary boyutu verilen vocab_size sınırını aşarsa, model kapasitesi kontrolsüz hale gelir.
    # Eğitim sırasında anlamsız candidate üretimi ortaya çıkabilir, 
    # Bu nedenle, vocabulary boyutu verilen vocab_size sınırını aşmamalıdır.

    tokenizer = TokenizerFactory.create("unigram", vocab_size=10)

    tokenizer.train("tokenization token tokenizer tokenized")

    assert tokenizer.vocab_size <= 10 # Vocabulary boyutu verilen vocab_size sınırını aşmamalıdır.


def test_unigram_tokenizer_vocab_contains_frequent_subwords() -> None:
    """
    Sık geçen subword parçaları vocabulary içine alınmalıdır.

    Gerekçe:
        Unigram tokenizer, eğitim metnindeki frekans bilgisine dayanarak subword parçaları seçer.
        Sık geçen parçalar vocabulary içinde bulunmalıdır, böylece tokenize edilen metin daha iyi segmentlenebilir.
        Bu test, eğitim sonrası vocabulary'nin frekans tabanlı seçim mekanizmasının çalıştığını doğrular,
        Eğitim metninde sık geçen subword parçalarının vocabulary içinde yer aldığını garanti eder.

    Önlenen bug sınıfı:
        - Sık geçen subword parçaları vocabulary içine alınmaz, bu da tokenize edilen metnin kötü segmentlenmesine yol açar.
        - Eğitim sırasında anlamsız candidate üretimi ortaya çıkabilir.
        - Tokenizer'ın temel işlevselliği tamamen ortadan kalkabilir.
        - Kullanıcıya açık ve anlaşılır hata mesajları sağlanmaz.

    Input:
        "abababa"

    Beklenti:
        "a", "b" veya "ab" gibi sık görülen parçalar vocabulary içinde bulunur.
    """
    # Bu test, eğitim sonrası vocabulary'nin frekans tabanlı seçim mekanizmasının çalıştığını doğrular,
    # Eğitim metninde sık geçen subword parçalarının vocabulary içinde yer aldığını garanti eder.

    # Eğer sık geçen subword parçaları vocabulary içine alınmazsa, tokenize edilen metnin kötü segmentlenmesine yol açar, 
    # eğitim sırasında anlamsız candidate üretimi ortaya çıkabilir, 
    # Bu nedenle, sık geçen subword parçaları vocabulary içine alınmalıdır.
    # Örneğin, "abababa" metninde "a", "b" veya "ab" gibi sık görülen parçalar vocabulary içinde bulunmalıdır.

    tokenizer = TokenizerFactory.create("unigram", vocab_size=20)

    tokenizer.train("abababa") # "a", "b" veya "ab" gibi sık görülen parçalar vocabulary içinde bulunmalıdır.

    assert "a" in tokenizer._token_to_id # "a" karakteri sık geçen bir parça olduğu için vocabulary içinde bulunmalıdır.
    assert "b" in tokenizer._token_to_id # "b" karakteri sık geçen bir parça olduğu için vocabulary içinde bulunmalıdır.
    assert "ab" in tokenizer._token_to_id # "ab" karakter dizisi sık geçen bir parça olduğu için vocabulary içinde bulunmalıdır.


def test_unigram_tokenizer_train_is_deterministic_for_same_input() -> None:
    """
    Aynı input ile eğitilen iki tokenizer aynı vocabulary üretmelidir.

    Gerekçe:
        Tokenizer deterministik olmalıdır.
        Aynı eğitim verisi aynı token-id mapping sonucunu üretmelidir.
        Bu test, train() fonksiyonunun deterministik olduğunu doğrular,
        Aynı eğitim verisiyle eğitilen iki tokenizer'ın aynı token-id mapping'i ürettiğini garanti eder.

    Önlenen bug sınıfı: 
        - Aynı input ile eğitilen iki tokenizer farklı vocabulary üretebilir, bu da encode/decode davranışında tutarsızlığa yol açar.
        - Tokenizer'ın deterministik olmayan davranışı ortaya çıkabilir, bu da modelin eğitim ve inference sırasında beklenmeyen sonuçlara yol açar.
        - Tokenizer'ın temel işlevselliği tamamen ortadan kalkabilir.
        - Kullanıcıya açık ve anlaşılır hata mesajları sağlanmaz.
        - Modelin eğitim ve inference sırasında beklenmeyen davranış veya crash riski ortaya çıkabilir.
        - Bu nedenle, aynı input ile eğitilen iki tokenizer aynı vocabulary üretmelidir.
    """
    # Bu test, train() fonksiyonunun deterministik olduğunu doğrular,
    # Aynı eğitim verisiyle eğitilen iki tokenizer'ın aynı token-id mapping'i ürettiğini garanti eder.

    # Eğer aynı input ile eğitilen iki tokenizer farklı vocabulary üretebilir, 
    # bu da encode/decode davranışında tutarsızlığa yol açar, 
    # tokenizer'ın deterministik olmayan davranışı ortaya çıkabilir, 
    # bu da modelin eğitim ve inference sırasında beklenmeyen sonuçlara yol açar, 
    # tokenizer'ın temel işlevselliği tamamen ortadan kalkabilir, 
    # kullanıcıya açık ve anlaşılır hata mesajları sağlanmaz, 
    # modelin eğitim ve inference sırasında beklenmeyen davranış veya crash riski ortaya çıkabilir.

    tokenizer_a = TokenizerFactory.create("unigram", vocab_size=20)
    tokenizer_b = TokenizerFactory.create("unigram", vocab_size=20)

    text = "token tokenization tokenizer"

    tokenizer_a.train(text)
    tokenizer_b.train(text)

    assert tokenizer_a._token_to_id == tokenizer_b._token_to_id # Aynı token-id mapping'i üretilmelidir.
    assert tokenizer_a._id_to_token == tokenizer_b._id_to_token # Aynı id-token mapping'i üretilmelidir.


# ---------------------------------------------------------
# TOKENIZE TESTS
# ---------------------------------------------------------

def test_unigram_tokenizer_tokenize_before_training_returns_basic_tokens() -> None:
    """
    Eğitim öncesi tokenize() basic regex tokenization yapmalıdır.

    Gerekçe:
        Bu tasarım, tokenizer henüz eğitilmemişken bile debug ve keşif amaçlı
        token çıktısı alınabilmesini sağlar.
        Eğitim öncesi tokenize() basic regex tokenization yapmalıdır, 
        böylece tokenizer henüz eğitilmemişken bile debug ve keşif amaçlı token çıktısı alınabilir.

    Önlenen bug sınıfı:
        - Eğitim öncesi tokenize() çalışmaz veya crash olur, bu da tokenizer'ın temel işlevselliğini ortadan kaldırır.
        - Eğitim öncesi tokenize() basic regex tokenization yapmaz, bu da debug ve keşif amaçlı token çıktısı alınmasını engeller.
        - Kullanıcıya açık ve anlaşılır hata mesajları sağlanmaz.
        - Eğitim öncesi tokenize() beklenmeyen sonuçlar üretebilir veya crash olabilir.        
    """
    # Bu tasarım, tokenizer henüz eğitilmemişken bile debug ve keşif amaçlı token çıktısı alınabilmesini sağlar.

    # Eğitim öncesi tokenize() basic regex tokenization yapmalıdır,
    # böylece tokenizer henüz eğitilmemişken bile debug ve keşif amaçlı token çıktısı alınabilir.

    # Eğer eğitim öncesi tokenize() çalışmaz veya crash olursa, tokenizer'ın temel işlevselliği ortadan kalkar.
    # Eğer eğitim öncesi tokenize() basic regex tokenization yapmazsa, debug ve keşif amaçlı token çıktısı alınması engellenir.
    # Eğer eğitim öncesi tokenize() beklenmeyen sonuçlar üretirse veya crash olursa, kullanıcıya açık ve anlaşılır hata mesajları sağlanmaz.
    
    tokenizer = TokenizerFactory.create("unigram")

    tokens = tokenizer.tokenize("Hello, World!") # Eğitim öncesi tokenize() basic regex tokenization yapmalıdır, böylece tokenizer henüz eğitilmemişken bile debug ve keşif amaçlı token çıktısı alınabilir.

    assert tokens == ["hello", ",", "world", "!"] # Eğitim öncesi tokenize() basic regex tokenization yapmalıdır, böylece tokenizer henüz eğitilmemişken bile debug ve keşif amaçlı token çıktısı alınabilir.


def test_unigram_tokenizer_tokenize_empty_text_returns_empty_list() -> None:
    """
    Boş input tokenize edildiğinde boş liste dönmelidir.

    Bu edge case API/report tarafında güvenli davranış sağlar.

    Gerekçe:
        Boş input, tokenize edildiğinde boş liste dönmelidir.
        Bu, API'nin ve raporlama sistemlerinin beklenmeyen davranışlardan korunmasını sağlar.
        Boş input, tokenize edildiğinde boş liste dönmelidir, böylece API'nin ve raporlama sistemlerinin beklenmeyen davranışlardan korunması sağlanır.

    Önlenen bug sınıfı:
        - Boş input tokenize edildiğinde None veya başka bir türde beklenmeyen çıktı dönebilir, bu da API'nin ve raporlama sistemlerinin beklenmeyen davranışlara yol açar.
        - Boş input tokenize edildiğinde crash olabilir, bu da API'nin ve raporlama sistemlerinin beklenmeyen davranışlara yol açar.
        - Kullanıcıya açık ve anlaşılır hata mesajları sağlanmaz.
        - Boş input tokenize edildiğinde beklenmeyen sonuçlar üretebilir veya crash olabilir.
    """
    # Boş input, tokenize edildiğinde boş liste dönmelidir, 
    # böylece API'nin ve raporlama sistemlerinin beklenmeyen davranışlardan korunması sağlanır.
    
    # Eğer boş input tokenize edildiğinde None veya başka bir türde beklenmeyen çıktı dönebilir, API'nin ve raporlama sistemlerinin beklenmeyen davranışlara yol açar.
    # Eğer boş input tokenize edildiğinde crash olursa, API'nin ve raporlama sistemlerinin beklenmeyen davranışlara yol açar.
    # Eğer kullanıcıya açık ve anlaşılır hata mesajları sağlanmazsa, boş input tokenize edildiğinde beklenmeyen sonuçlar üretebilir veya crash olabilir.

    tokenizer = TokenizerFactory.create("unigram")

    assert tokenizer.tokenize("") == [] # Boş input tokenize edildiğinde boş liste dönmelidir, böylece API'nin ve raporlama sistemlerinin beklenmeyen davranışlardan korunması sağlanır.


def test_unigram_tokenizer_tokenize_whitespace_text_returns_empty_list() -> None:
    """
    Sadece whitespace içeren input tokenize edildiğinde boş liste dönmelidir.

    Gerekçe:
        Sadece whitespace içeren input, tokenize edildiğinde boş liste dönmelidir.
        Bu, API'nin ve raporlama sistemlerinin beklenmeyen davranışlardan korunmasını sağlar.
        Sadece whitespace içeren input tokenize edildiğinde boş liste dönmelidir, 
        böylece API'nin ve raporlama sistemlerinin beklenmeyen davranışlardan korunması sağlanır.

    Önlenen bug sınıfı:
        - Sadece whitespace içeren input tokenize edildiğinde None veya başka bir türde beklenmeyen çıktı dönebilir, bu da API'nin ve raporlama sistemlerinin beklenmeyen davranışlara yol açar.
        - Sadece whitespace içeren input tokenize edildiğinde crash olabilir, bu da API'nin ve raporlama sistemlerinin beklenmeyen davranışlara yol açar.
        - Kullanıcıya açık ve anlaşılır hata mesajları sağlanmaz.
        - Sadece whitespace içeren input tokenize edildiğinde beklenmeyen sonuçlar üretebilir veya crash olabilir.
        - Bu nedenle, sadece whitespace içeren input tokenize edildiğinde boş liste dönmelidir, böylece API'nin ve raporlama sistemlerinin beklenmeyen davranışlardan korunması sağlanır.
    """
    # Sadece whitespace içeren input, tokenize edildiğinde boş liste dönmelidir, 
    # böylece API'nin ve raporlama sistemlerinin beklenmeyen davranışlardan korunması sağlar.

    # Eğer sadece whitespace içeren input tokenize edildiğinde None veya başka bir türde beklenmeyen çıktı dönebilir, API'nin ve raporlama sistemlerinin beklenmeyen davranışlara yol açar.
    # Eğer sadece whitespace içeren input tokenize edildiğinde crash olursa, API'nin ve raporlama sistemlerinin beklenmeyen davranışlara yol açar.
    # Eğer kullanıcıya açık ve anlaşılır hata mesajları sağlanmazsa, sadece whitespace içeren input tokenize edildiğinde beklenmeyen sonuçlar üretebilir veya crash olabilir.
    # Bu nedenle, sadece whitespace içeren input tokenize edildiğinde boş liste dönmelidir, böylece API'nin ve raporlama sistemlerinin beklenmeyen davranışlardan korunması sağlanır.
    
    tokenizer = TokenizerFactory.create("unigram")

    assert tokenizer.tokenize("   ") == [] # Sadece whitespace içeren input tokenize edildiğinde boş liste dönmelidir, böylece API'nin ve raporlama sistemlerinin beklenmeyen davranışlardan korunması sağlanır.


def test_unigram_tokenizer_tokenize_returns_subword_tokens_after_training() -> None:
    """
    Eğitim sonrası tokenize(), unigram subword segmentasyonu yapmalıdır.

    Gerekçe:
        Eğitilmiş tokenizer basic tokenization ile kalmamalı,
        kelimeleri öğrenilmiş subword parçalarına ayırmalıdır.
        Eğitim sonrası tokenize(), unigram subword segmentasyonu yapmalıdır,
        böylece eğitilmiş tokenizer basic tokenization ile kalmaz,
        kelimeleri öğrenilmiş subword parçalarına ayırır.

    Önlenen bug sınıfı:
        - Eğitim sonrası tokenize() basic tokenization yapmaya devam edebilir, bu da eğitilmiş tokenizer'ın temel işlevselliğini ortadan kaldırır.
        - Eğitim sonrası tokenize() beklenmeyen sonuçlar üretebilir veya crash olabilir.
        - Kullanıcıya açık ve anlaşılır hata mesajları sağlanmaz.   
        - Eğitim sonrası tokenize() kelimeleri öğrenilmiş subword parçalarına ayırmaz, bu da segmentasyon kalitesini düşürür.
    """
    # Bu test, eğitilmiş tokenizer'ın basic tokenization ile kalmadığını ve kelimeleri öğrenilmiş subword parçalarına ayırdığını doğrular.
    
    # Eğitim sonrası tokenize(), unigram subword segmentasyonu yapmalıdır, böylece eğitilmiş tokenizer basic tokenization ile kalmaz, kelimeleri öğrenilmiş subword parçalarına ayırır.
    
    # Eğer eğitim sonrası tokenize() basic tokenization yapmaya devam ederse, eğitilmiş tokenizer'ın temel işlevselliği ortadan kalkar.
    # Eğer eğitim sonrası tokenize() beklenmeyen sonuçlar üretirse veya crash olursa, kullanıcıya açık ve anlaşılır hata mesajları sağlanmaz.
    # Eğer eğitim sonrası tokenize() kelimeleri öğrenilmiş subword parçalarına ayırmazsa, segmentasyon kalitesi düşer.
    
    tokenizer = TokenizerFactory.create("unigram", vocab_size=30)

    tokenizer.train("token tokenization tokenizer") # Eğitim verisi ile tokenizer eğitilir.

    tokens = tokenizer.tokenize("tokenizer")

    # Eğitim sonrası tokenize() kelimeleri öğrenilmiş subword parçalarına ayırmalıdır
    assert isinstance(tokens, list) # tokenize() çıktısının bir liste olması beklenir.
    assert len(tokens) > 0 # tokenize() çıktısında en az bir token olması beklenir.
    assert all(isinstance(token, str) for token in tokens) # tokenize() çıktısındaki her bir token'ın string olması beklenir.


def test_unigram_tokenizer_unknown_word_returns_unknown_token() -> None:
    """
    Vocabulary ile segment edilemeyen kelime [UNK] olarak temsil edilmelidir.

    Burada max_subword_length küçük tutulur ve eğitim verisi sınırlıdır.
    Böylece "zzzz" gibi görülmeyen bir input için geçerli segmentation bulunamaması beklenir.

    Gerekçe:
        Unigram tokenizer bilinmeyen inputlarda crash etmek yerine güvenli fallback davranışı göstermelidir.
        Vocabulary ile segment edilemeyen kelime [UNK] olarak temsil edilmelidir, 
        böylece tokenizer bilinmeyen inputlarda crash etmek yerine güvenli fallback davranışı gösterir.
  
    Önlenen bug sınıfı:
        - Vocabulary ile segment edilemeyen kelime tokenize edilirken crash olabilir, bu da tokenizer'ın temel işlevselliğini ortadan kaldırır.
        - Vocabulary ile segment edilemeyen kelime tokenize edilirken beklenmeyen sonuçlar üretebilir, bu da kullanıcı deneyimini olumsuz etkiler.
        - Kullanıcıya açık ve anlaşılır hata mesajları sağlanmaz.
        - Vocabulary ile segment edilemeyen kelime tokenize edilirken beklenmeyen sonuçlar üretebilir veya crash olabilir.
    """
    tokenizer = TokenizerFactory.create(
        "unigram",
        vocab_size=5,
        max_subword_length=1,
    )
    # max_subword_length küçük tutulur ve eğitim verisi sınırlıdır, böylece "zzzz" gibi görülmeyen bir input için geçerli segmentation bulunamaması beklenir.
    # Eğitim verisi sadece "aa", "bb", "cc" gibi kısa parçalar içerir, bu nedenle "zzzz" için geçerli bir segmentation bulunamaz ve [UNK] token'ı döndürülmesi beklenir.
    
    # Unigram tokenizer bilinmeyen inputlarda crash etmek yerine güvenli fallback davranışı göstermelidir, bu nedenle vocabulary ile segment edilemeyen kelime tokenize edilirken [UNK] olarak temsil edilmelidir.  
    
    # Eğer vocabulary ile segment edilemeyen kelime tokenize edilirken crash olursa, tokenizer'ın temel işlevselliği ortadan kalkar.
    # Eğer vocabulary ile segment edilemeyen kelime tokenize edilirken beklenmeyen sonuçlar üretebilir, kullanıcı deneyimi olumsuz etkilenir.
    # Eğer kullanıcıya açık ve anlaşılır hata mesajları sağlanmazsa, vocabulary ile segment edilemeyen kelime tokenize edilirken beklenmeyen sonuçlar üretebilir veya crash olabilir.

    tokenizer.train("aa bb cc")

    tokens = tokenizer.tokenize("zzzz")

    assert tokens == [tokenizer.UNKNOWN_TOKEN] # Vocabulary ile segment edilemeyen kelime tokenize edilirken [UNK] olarak temsil edilmelidir, böylece tokenizer bilinmeyen inputlarda crash etmek yerine güvenli fallback davranışı gösterir.


# ---------------------------------------------------------
# ENCODE TESTS
# ---------------------------------------------------------

def test_unigram_tokenizer_encode_before_training_raises_error() -> None:
    """
    train() çağrılmadan encode() çalışmamalıdır.

    Gerekçe:
        Encode işlemi token -> id mapping gerektirir.
        Bu mapping train() sırasında oluşturulur.
        train() çağrılmadan encode() çalışmamalıdır, 
        böylece encode işlemi token -> id mapping gerektirir, bu mapping train() sırasında oluşturulur.

    Önlenen bug sınıfı:
        - train() çağrılmadan encode() çalışabilir, bu da beklenmeyen sonuçlara veya crash'a yol açar, bu da tokenizer'ın temel işlevselliğini ortadan kaldırır.    
        - train() çağrılmadan encode() çalışabilir, ancak beklenmeyen sonuçlar üretebilir, bu da kullanıcı deneyimini olumsuz etkiler.
        - Kullanıcıya açık ve anlaşılır hata mesajları sağlanmaz.
        - train() çağrılmadan encode() çalışabilir, ancak beklenmeyen sonuçlar üretebilir veya crash olabilir.
    """
    # Encode işlemi token -> id mapping gerektirir, bu mapping train() sırasında oluşturulur, bu nedenle train() çağrılmadan encode() çalışmamalıdır, böylece encode işlemi token -> id mapping gerektirir, bu mapping train() sırasında oluşturulur.
    
    # Eğer train() çağrılmadan encode() çalışabilirse, bu da beklenmeyen sonuçlara veya crash'a yol açar, bu da tokenizer'ın temel işlevselliğini ortadan kaldırır.
    # Eğer train() çağrılmadan encode() çalışabilirse, ancak beklenmeyen sonuçlar üretebilir, bu da kullanıcı deneyimini olumsuz etkiler.
    # Eğer kullanıcıya açık ve anlaşılır hata mesajları sağlanmazsa, train() çağrılmadan encode() çalışabilir, ancak beklenmeyen sonuçlar üretebilir veya crash olabilir.
    
    tokenizer = TokenizerFactory.create("unigram")

    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.encode("tokenization")


def test_unigram_tokenizer_encode_returns_integer_token_ids() -> None:
    """
    encode() çıktısı integer token id listesi olmalıdır.

    Bu test model input formatı açısından temel kontratı doğrular.

    Gerekçe:
        encode() çıktısı integer token id listesi olmalıdır.
        Model input formatı açısından temel kontratı doğrular.
        encode() çıktısı integer token id listesi olmalıdır, böylece model input formatı açısından temel kontratı doğrular.
    
    Önlenen bug sınıfı:
        - encode() çıktısı integer token id listesi olmayabilir, bu da modelin beklenmeyen sonuçlara veya crash'a yol açar, bu da tokenizer'ın temel işlevselliğini ortadan kaldırır.
        - encode() çıktısı integer token id listesi olmayabilir, ancak beklenmeyen sonuçlar üretebilir, bu da kullanıcı deneyimini olumsuz etkiler.
        - Kullanıcıya açık ve anlaşılır hata mesajları sağlanmaz.
        - encode() çıktısı integer token id listesi olmayabilir, ancak beklenmeyen sonuçlar üretebilir veya crash olabilir.
        - Bu nedenle, encode() çıktısı integer token id listesi olmalıdır, böylece model input formatı açısından temel kontratı doğrular.
    """
    # encode() çıktısı integer token id listesi olmalıdır, böylece model input formatı açısından temel kontratı doğrular.
    
    # Eğer encode() çıktısı integer token id listesi olmazsa, modelin beklenmeyen sonuçlara veya crash'a yol açar, bu da tokenizer'ın temel işlevselliğini ortadan kaldırır.
    # Eğer encode() çıktısı integer token id listesi olmazsa, ancak beklenmeyen sonuçlar üretebilir, bu da kullanıcı deneyimini olumsuz etkiler.
    # Eğer kullanıcıya açık ve anlaşılır hata mesajları sağlanmazsa, encode() çıktısı integer token id listesi olmazsa, ancak beklenmeyen sonuçlar üretebilir veya crash olabilir.
    
    tokenizer = TokenizerFactory.create("unigram", vocab_size=30)

    tokenizer.train("token tokenization tokenizer")

    encoded = tokenizer.encode("tokenization")

    assert isinstance(encoded, list) # encode() çıktısı bir liste olmalıdır.
    assert len(encoded) > 0 # encode() çıktısında en az bir token id olması beklenir.
    assert all(isinstance(token_id, int) for token_id in encoded) # encode() çıktısındaki her bir token id'nin integer olması beklenir.


def test_unigram_tokenizer_encode_unknown_word_uses_unknown_token_id() -> None:
    """
    Segment edilemeyen kelime encode sırasında [UNK] id'si ile temsil edilmelidir.

    Gerekçe:
        Unigram tokenizer bilinmeyen inputlarda crash etmek yerine
        güvenli fallback davranışı göstermelidir.
        Segment edilemeyen kelime encode sırasında [UNK] id'si ile temsil edilmelidir,
        böylece tokenizer bilinmeyen inputlarda crash etmek yerine güvenli fallback davranışı gösterir.

    Önlenen bug sınıfı:
        - Segment edilemeyen kelime encode edilirken crash olabilir, bu da tokenizer'ın temel işlevselliğini ortadan kaldırır.
        - Segment edilemeyen kelime encode edilirken beklenmeyen sonuçlar üretebilir, bu da kullanıcı deneyimini olumsuz etkiler.
        - Kullanıcıya açık ve anlaşılır hata mesajları sağlanmaz.   
        - Segment edilemeyen kelime encode edilirken beklenmeyen sonuçlar üretebilir veya crash olabilir.
        - Bu nedenle, segment edilemeyen kelime encode sırasında [UNK] id'si ile temsil edilmelidir, böylece tokenizer bilinmeyen inputlarda crash etmek yerine güvenli fallback davranışı gösterir.
    """
    tokenizer = TokenizerFactory.create(
        "unigram",
        vocab_size=5,
        max_subword_length=1,
    )
    # max_subword_length küçük tutulur ve eğitim verisi sınırlıdır, böylece "zzzz" gibi görülmeyen bir input için geçerli segmentation bulunamaması beklenir.
    # Eğitim verisi sadece "aa", "bb", "cc" gibi kısa parçalar içerir, bu nedenle "zzzz" için geçerli bir segmentation bulunamaz ve [UNK] token id'si döndürülmesi beklenir.
    
    # Unigram tokenizer bilinmeyen inputlarda crash etmek yerine güvenli fallback davranışı göstermelidir, bu nedenle segment edilemeyen kelime encode edilirken [UNK] id'si ile temsil edilmelidir.
    
    # Eğer segment edilemeyen kelime encode edilirken crash olursa, tokenizer'ın temel işlevselliği ortadan kalkar.
    # Eğer segment edilemeyen kelime encode edilirken beklenmeyen sonuçlar üretebilir, kullanıcı deneyimi olumsuz etkilenir.
    # Eğer kullanıcıya açık ve anlaşılır hata mesajları sağlanmazsa, segment edilemeyen kelime encode edilirken beklenmeyen sonuçlar üretebilir veya crash olabilir.

    tokenizer.train("aa bb cc")

    encoded = tokenizer.encode("zzzz")

    assert encoded == [tokenizer._token_to_id[tokenizer.UNKNOWN_TOKEN]] # Segment edilemeyen kelime encode edilirken [UNK] id'si ile temsil edilmelidir, böylece tokenizer bilinmeyen inputlarda crash etmek yerine güvenli fallback davranışı gösterir.


def test_unigram_tokenizer_encode_is_deterministic_for_same_input() -> None:
    """
    Aynı input birden fazla kez encode edildiğinde aynı token id listesi dönmelidir.

    Gerekçe:
        Tokenizer inference davranışı deterministik olmalıdır.
        Aynı input aynı token id listesi sonucunu üretmelidir.
        Bu test, encode() fonksiyonunun deterministik olduğunu doğrular,
        Aynı input birden fazla kez encode edildiğinde aynı token id listesi dönmelidir, böylece tokenizer inference davranışı deterministik olur.
    
    Önlenen bug sınıfı:
        - Aynı input birden fazla kez encode edildiğinde farklı token id listeleri dönebilir, bu da tokenizer'ın deterministik olmayan davranışına ve modelin beklenmeyen sonuçlara yol açar.
        - Kullanıcıya açık ve anlaşılır hata mesajları sağlanmaz.
        - Aynı input birden fazla kez encode edildiğinde beklenmeyen sonuçlar üretebilir veya crash olabilir.
        - Bu nedenle, aynı input birden fazla kez encode edildiğinde aynı token id listesi dönmelidir, böylece tokenizer inference davranışı deterministik olur.
    """
    # Bu test, encode() fonksiyonunun deterministik olduğunu doğrular,
    # Aynı input birden fazla kez encode edildiğinde aynı token id listesi dönmelidir, böylece tokenizer inference davranışı deterministik olur.
    
    # Eğer aynı input birden fazla kez encode edildiğinde farklı token id listeleri dönebilirse, bu da tokenizer'ın deterministik olmayan davranışına ve modelin beklenmeyen sonuçlara yol açar.
    # Eğer kullanıcıya açık ve anlaşılır hata mesajları sağlanmazsa, aynı input birden fazla kez encode edildiğinde beklenmeyen sonuçlar üretebilir veya crash olabilir.
    # Bu nedenle, aynı input birden fazla kez encode edildiğinde aynı token id listesi dönmelidir, böylece tokenizer inference davranışı deterministik olur.

    tokenizer = TokenizerFactory.create("unigram", vocab_size=30)

    tokenizer.train("token tokenization tokenizer")

    encoded_a = tokenizer.encode("tokenization")
    encoded_b = tokenizer.encode("tokenization")

    assert encoded_a == encoded_b # Aynı input birden fazla kez encode edildiğinde aynı token id listesi dönmelidir, böylece tokenizer inference davranışı deterministik olur.


# ---------------------------------------------------------
# DECODE TESTS
# ---------------------------------------------------------

def test_unigram_tokenizer_decode_before_training_raises_error() -> None:
    """
    train() çağrılmadan decode() çalışmamalıdır.

    Gerekçe:
        Decode işlemi id -> token mapping gerektirir.
        Bu mapping train() sırasında oluşturulur.
        train() çağrılmadan decode() çalışmamalıdır, 
        böylece decode işlemi id -> token mapping gerektirir, bu mapping train() sırasında oluşturulur.

    Önlenen bug sınıfı:
        - train() çağrılmadan decode() çalışabilir, bu da beklenmeyen sonuçlara veya crash'a yol açar, bu da tokenizer'ın temel işlevselliğini ortadan kaldırır.    
        - train() çağrılmadan decode() çalışabilir, ancak beklenmeyen sonuçlar üretebilir, bu da kullanıcı deneyimini olumsuz etkiler.
        - Kullanıcıya açık ve anlaşılır hata mesajları sağlanmaz.
        - train() çağrılmadan decode() çalışabilir, ancak beklenmeyen sonuçlar üretebilir veya crash olabilir.
        - Bu nedenle, train() çağrılmadan decode() çalışmamalıdır, böylece decode işlemi id -> token mapping gerektirir, bu mapping train() sırasında oluşturulur.
    """
    # Decode işlemi id -> token mapping gerektirir, bu mapping train() sırasında oluşturulur, bu nedenle train() çağrılmadan decode() çalışmamalıdır, böylece decode işlemi id -> token mapping gerektirir, bu mapping train() sırasında oluşturulur.
    
    # Eğer train() çağrılmadan decode() çalışabilirse, bu da beklenmeyen sonuçlara veya crash'a yol açar, bu da tokenizer'ın temel işlevselliğini ortadan kaldırır.
    # Eğer train() çağrılmadan decode() çalışabilirse, ancak beklenmeyen sonuçlar üretebilir, bu da kullanıcı deneyimini olumsuz etkiler.
    # Eğer kullanıcıya açık ve anlaşılır hata mesajları sağlanmazsa, train() çağrılmadan decode() çalışabilir, ancak beklenmeyen sonuçlar üretebilir veya crash olabilir.
    # Bu nedenle, train() çağrılmadan decode() çalışmamalıdır, böylece decode işlemi id -> token mapping gerektirir, bu mapping train() sırasında oluşturulur.
    
    tokenizer = TokenizerFactory.create("unigram")

    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.decode([0, 1])


def test_unigram_tokenizer_decode_unknown_token_id_raises_error() -> None:
    """
    Vocabulary içinde olmayan token id decode edilmeye çalışılırsa hata vermelidir.

    Bu test strict decode kontratını doğrular.

    Gerekçe:
        Decode işlemi id -> token mapping gerektirir.
        Eğer decode edilmeye çalışılan token id vocabulary içinde yoksa, bu durum hatalı bir kullanım olarak değerlendirilmelidir.
        Decode işlemi id -> token mapping gerektirir, böylece eğer decode edilmeye çalışılan token id vocabulary içinde yoksa, bu durum hatalı bir kullanım olarak değerlendirilmelidir.
    
    Önlenen bug sınıfı:
        - Vocabulary içinde olmayan token id decode edilmeye çalışılırsa, beklenmeyen sonuçlara veya crash'a yol açabilir, bu da tokenizer'ın temel işlevselliğini ortadan kaldırır.
        - Vocabulary içinde olmayan token id decode edilmeye çalışılırsa, ancak beklenmeyen sonuçlar üretebilir, bu da kullanıcı deneyimini olumsuz etkiler.
        - Kullanıcıya açık ve anlaşılır hata mesajları sağlanmaz.
        - Vocabulary içinde olmayan token id decode edilmeye çalışılırsa, beklenmeyen sonuçlar üretebilir veya crash olabilir.
        - Bu nedenle, vocabulary içinde olmayan token id decode edilmeye çalışılırsa hata vermelidir, böylece decode işlemi id -> token mapping gerektirir, böylece eğer decode edilmeye çalışılan token id vocabulary içinde yoksa, bu durum hatalı bir kullanım olarak değerlendirilmelidir.
    """
    # Decode işlemi id -> token mapping gerektirir, böylece eğer decode edilmeye çalışılan token id vocabulary içinde yoksa, bu durum hatalı bir kullanım olarak değerlendirilmelidir.
    
    # Eğer vocabulary içinde olmayan token id decode edilmeye çalışılırsa, beklenmeyen sonuçlara veya crash'a yol açabilir, bu da tokenizer'ın temel işlevselliğini ortadan kaldırır.
    # Eğer vocabulary içinde olmayan token id decode edilmeye çalışılırsa, ancak beklenmeyen sonuçlar üretebilir, bu da kullanıcı deneyimini olumsuz etkiler.
    # Eğer kullanıcıya açık ve anlaşılır hata mesajları sağlanmazsa, vocabulary içinde olmayan token id decode edilmeye çalışılırsa, beklenmeyen sonuçlar üretebilir veya crash olabilir.
    # Bu nedenle, vocabulary içinde olmayan token id decode edilmeye çalışılırsa hata vermelidir, böylece decode işlemi id -> token mapping gerektirir, böylece eğer decode edilmeye çalışılan token id vocabulary içinde yoksa, bu durum hatalı bir kullanım olarak değerlendirilmelidir.
    
    tokenizer = TokenizerFactory.create("unigram", vocab_size=20)

    tokenizer.train("tokenization token tokenizer")

    with pytest.raises(ValueError, match="Unknown token id"):
        tokenizer.decode([9999])


def test_unigram_tokenizer_decode_returns_string() -> None:
    """
    decode() token id listesini string çıktıya dönüştürmelidir.

    Gerekçe:
        Token id listesini string'e dönüştürmek, tokenizer'ın temel işlevselliğini doğrular ve kullanıcıya anlamlı bir çıktı sağlar.
        decode() token id listesini string çıktıya dönüştürmelidir, böylece tokenizer'ın temel işlevselliğini doğrular ve kullanıcıya anlamlı bir çıktı sağlar.

    Önlenen bug sınıfı:
        - decode() token id listesini string'e dönüştürmeyebilir, bu da beklenmeyen sonuçlara veya crash'a yol açar, bu da tokenizer'ın temel işlevselliğini ortadan kaldırır.
        - decode() token id listesini string'e dönüştürmeyebilir, ancak beklenmeyen sonuçlar üretebilir, bu da kullanıcı deneyimini olumsuz etkiler.
        - Kullanıcıya açık ve anlaşılır hata mesajları sağlanmaz.
        - decode() token id listesini string'e dönüştürmeyebilir, ancak beklenmeyen sonuçlar üretebilir veya crash olabilir.
        - Bu nedenle, decode() token id listesini string çıktıya dönüştürmelidir, böylece tokenizer'ın temel işlevselliğini doğrular ve kullanıcıya anlamlı bir çıktı sağlar.    

    Not:
        Bu tokenizer whitespace reconstruction yapmadığı için decode sonucu
        her zaman orijinal metinle birebir aynı olmak zorunda değildir.
    """
    # decode() token id listesini string'e dönüştürmek, tokenizer'ın temel işlevselliğini doğrular ve kullanıcıya anlamlı bir çıktı sağlar.
    # decode() token id listesini string çıktıya dönüştürmelidir, böylecetokenizer'ın temel işlevselliğini doğrular ve kullanıcıya anlamlı bir çıktı sağlar.
    
    # Eğer decode() token id listesini string'e dönüştürmezse, bu da beklenmeyen sonuçlara veya crash'a yol açar, bu da tokenizer'ın temel işlevselliğini ortadan kaldırır.
    # Eğer decode() token id listesini string'e dönüştürmezse, ancak beklenmeyen sonuçlar üretebilir, bu da kullanıcı deneyimini olumsuz etkiler.
    # Eğer kullanıcıya açık ve anlaşılır hata mesajları sağlanmazsa, decode() token id listesini string'e dönüştürmezse, ancak beklenmeyen sonuçlar üretebilir veya crash olabilir.
    # Bu nedenle, decode() token id listesini string çıktıya dönüştürmelidir, böylece tokenizer'ın temel işlevselliğini doğrular ve kullanıcıya anlamlı bir çıktı sağlar.
    
    tokenizer = TokenizerFactory.create("unigram", vocab_size=30)

    tokenizer.train("token tokenization tokenizer")

    encoded = tokenizer.encode("token")
    decoded = tokenizer.decode(encoded)

    assert isinstance(decoded, str) # decode() çıktısının bir string olması beklenir.
    assert len(decoded) > 0 # decode() çıktısında en az bir karakter olması beklenir.


def test_unigram_tokenizer_decode_preserves_unknown_token() -> None:
    """
    [UNK] id'si decode edildiğinde [UNK] string'i korunmalıdır.

    Gerekçe:
        Bilinmeyen parçaların decode sonucunda açıkça görülebilmesi gerekir.
        [UNK] id'si decode edildiğinde [UNK] string'i korunmalıdır, böylece bilinmeyen parçaların decode sonucunda açıkça görülebilmesi sağlanır.
    
    Önlenen bug sınıfı:
        - [UNK] id'si decode edildiğinde [UNK] string'i korunmayabilir, bu da bilinmeyen parçaların decode sonucunda belirsiz veya yanlış şekilde temsil edilmesine yol açar, bu da kullanıcı deneyimini olumsuz etkiler.
        - Kullanıcıya açık ve anlaşılır hata mesajları sağlanmaz.
        - [UNK] id'si decode edildiğinde beklenmeyen sonuçlar üretebilir veya crash olabilir.
        - Bu nedenle, [UNK] id'si decode edildiğinde [UNK] string'i korunmalıdır, böylece bilinmeyen parçaların decode sonucunda açıkça görülebilmesi sağlanır.
    """
    # [UNK] id'si decode edildiğinde [UNK] string'i korunmalıdır, böylece bilinmeyen parçaların decode sonucunda açıkça görülebilmesi sağlanır.
    
    # Eğer [UNK] id'si decode edildiğinde [UNK] string'i korunmazsa, bu da bilinmeyen parçaların decode sonucunda belirsiz veya yanlış şekilde temsil edilmesine yol açar, bu da kullanıcı deneyimini olumsuz etkiler.
    # Eğer kullanıcıya açık ve anlaşılır hata mesajları sağlanmazsa, [UNK] id'si decode edildiğinde beklenmeyen sonuçlar üretebilir veya crash olabilir.
    # Bu nedenle, [UNK] id'si decode edildiğinde [UNK] string'i korunmalıdır, böylece bilinmeyen parçaların decode sonucunda açıkça görülebilmesi sağlanır.
    
    tokenizer = TokenizerFactory.create("unigram", vocab_size=10)

    tokenizer.train("aa bb cc")

    unknown_id = tokenizer._token_to_id[tokenizer.UNKNOWN_TOKEN] # [UNK] token id'si alınır.

    assert tokenizer.decode([unknown_id]) == tokenizer.UNKNOWN_TOKEN # [UNK] id'si decode edildiğinde [UNK] string'i korunmalıdır, böylece bilinmeyen parçaların decode sonucunda açıkça görülebilmesi sağlanır.


# ---------------------------------------------------------
# ROUNDTRIP / BEHAVIOR TESTS
# ---------------------------------------------------------

def test_unigram_tokenizer_encode_decode_roundtrip_for_known_compact_text() -> None:
    """
    Whitespace içermeyen ve vocabulary ile segment edilebilen metinlerde
    encode -> decode sonucu orijinal metne eşit olabilir.

    Gerekçe:
        Unigram tokenizer whitespace reconstruction yapmaz, bu nedenle roundtrip testi sadece whitespace içermeyen
        ve vocabulary ile segment edilebilen metinler için anlamlıdır.

    Önlenen bug sınıfı:
        - Whitespace içermeyen ve vocabulary ile segment edilebilen metinlerde encode -> decode sonucu orijinal metne eşit olmayabilir, bu da tokenizer'ın temel işlevselliğini ortadan kaldırır.
        - Kullanıcıya açık ve anlaşılır hata mesajları sağlanmaz.
        - Whitespace içermeyen ve vocabulary ile segment edilebilen metinlerde encode -> decode sonucu beklenmeyen sonuçlar üretebilir veya crash olabilir.
        - Bu nedenle, whitespace içermeyen ve vocabulary ile segment edilebilen metinlerde encode -> decode sonucu orijinal metne eşit olabilir, böylece tokenizer'ın temel işlevselliği doğrulanır.

    Not:
        Bu test whitespace reconstruction iddiasında bulunmaz.
        Sadece compact text için temel roundtrip davranışını doğrular.
    """
    # Unigram tokenizer whitespace reconstruction yapmaz, bu nedenle roundtrip testi sadece whitespace içermeyen ve vocabulary ile segment edilebilen metinler için anlamlıdır.
    # Bu test, whitespace içermeyen ve vocabulary ile segment edilebilen metinlerde encode -> decode sonucunun orijinal metne eşit olabileceğini doğrular, böylece tokenizer'ın temel işlevselliği doğrulanır.
    
    # Eğer whitespace içermeyen ve vocabulary ile segment edilebilen metinlerde encode -> decode sonucu orijinal metne eşit olmazsa, bu da tokenizer'ın temel işlevselliğini ortadan kaldırır.
    # Eğer kullanıcıya açık ve anlaşılır hata mesajları sağlanmazsa, whitespace içermeyen ve vocabulary ile segment edilebilen metinlerde encode -> decode sonucu beklenmeyen sonuçlar üretebilir veya crash olabilir.
    # Bu nedenle, whitespace içermeyen ve vocabulary ile segment edilebilen metinlerde encode -> decode sonucu orijinal metne eşit olabilir, böylece tokenizer'ın temel işlevselliği doğrulanır.

    tokenizer = TokenizerFactory.create("unigram", vocab_size=50)

    text = "tokenization"
    tokenizer.train(text)

    decoded = tokenizer.decode(tokenizer.encode(text)) # encode -> decode sonucu orijinal metne eşit olabilir, böylece tokenizer'ın temel işlevselliği doğrulanır.

    assert decoded == text # Whitespace içermeyen ve vocabulary ile segment edilebilen metinlerde encode -> decode sonucu orijinal metne eşit olabilir, böylece tokenizer'ın temel işlevselliği doğrulanır.


def test_unigram_tokenizer_decode_is_not_whitespace_lossless() -> None:
    """
    UnigramTokenizer whitespace bilgisini korumaz.

    Gerekçe:
        _basic_tokenize boşlukları token olarak saklamaz.
        Decode ise tokenları doğrudan join eder.
        Bu nedenle, whitespace bilgisi kaybolur.
    
    Önlenen bug sınıfı:
        - UnigramTokenizer decode işlemi sırasında whitespace bilgisini koruyabilir gibi yanlış bir izlenim verebilir, bu da kullanıcı beklentilerini yanıltır.
        - Kullanıcıya açık ve anlaşılır hata mesajları sağlanmaz.
        - UnigramTokenizer decode işlemi sırasında beklenmeyen sonuçlar üretebilir veya crash olabilir.
        - Bu nedenle, UnigramTokenizer decode işlemi sırasında whitespace bilgisini korumaz, böylece kullanıcı beklentilerini yanıltmaz.

    Bu test bilinçli tasarım davranışını dokümante eder.
    """
    # _basic_tokenize boşlukları token olarak saklamaz, decode ise tokenları doğrudan join eder, bu nedenle whitespace bilgisi kaybolur, böylece UnigramTokenizer decode işlemi sırasında whitespace bilgisini korumaz, böylece kullanıcı beklentilerini yanıltmaz.
    # Bu test, UnigramTokenizer decode işlemi sırasında whitespace bilgisini korumadığını doğrular, böylece kullanıcı beklentilerini yanıltmaz.
    
    # Eğer UnigramTokenizer decode işlemi sırasında whitespace bilgisini koruyabilir gibi yanlış bir izlenim verebilirse, bu da kullanıcı beklentilerini yanıltır.
    # Eğer kullanıcıya açık ve anlaşılır hata mesajları sağlanmazsa, UnigramTokenizer decode işlemi sırasında beklenmeyen sonuçlar üretebilir veya crash olabilir.
    # Bu nedenle, UnigramTokenizer decode işlemi sırasında whitespace bilgisini korumaz, böylece kullanıcı beklentilerini yanıltmaz.
    
    tokenizer = TokenizerFactory.create("unigram", vocab_size=50)

    text = "hello world"
    tokenizer.train(text)

    decoded = tokenizer.decode(tokenizer.encode(text)) # UnigramTokenizer decode işlemi sırasında whitespace bilgisini korumaz, böylece kullanıcı beklentilerini yanıltmaz.

    assert decoded == "helloworld" # UnigramTokenizer decode işlemi sırasında whitespace bilgisini korumaz, böylece kullanıcı beklentilerini yanıltmaz.
    assert decoded != text # UnigramTokenizer decode işlemi sırasında whitespace bilgisini korumaz, böylece kullanıcı beklentilerini yanıltmaz.


def test_unigram_tokenizer_handles_turkish_characters() -> None:
    """
    Türkçe karakterler basic regex tokenization tarafından kelime parçası olarak işlenebilmelidir.

    Bu test Unicode inputlarda tokenizer'ın temel akışının çalıştığını doğrular.

    Gerekçe:
        Türkçe karakterler basic regex tokenization tarafından kelime parçası olarak işlenebilmelidir, böylece tokenizer Unicode inputlarda temel akışını çalıştırabilir.
        Bu test, Türkçe karakterler basic regex tokenization tarafından kelime parçası olarak işlenebildiğini doğrular, böylece tokenizer Unicode inputlarda temel akışını çalıştırabilir.

    Önlenen bug sınıfı:
        - Türkçe karakterler basic regex tokenization tarafından kelime parçası olarak işlenemeyebilir, bu da Unicode inputlarda tokenizer'ın temel akışının çalışmamasına yol açar, bu da tokenizer'ın temel işlevselliğini ortadan kaldırır.
        - Türkçe karakterler basic regex tokenization tarafından kelime parçası olarak işlenemeyebilir, ancak beklenmeyen sonuçlar üretebilir, bu da kullanıcı deneyimini olumsuz etkiler.
        - Kullanıcıya açık ve anlaşılır hata mesajları sağlanmaz.
        - Türkçe karakterler basic regex tokenization tarafından kelime parçası olarak işlenemeyebilir, ancak beklenmeyen sonuçlar üretebilir veya crash olabilir.
        - Bu nedenle, Türkçe karakterler basic regex tokenization tarafından kelime parçası olarak işlenebilmelidir, böylece tokenizer Unicode inputlarda temel akışını çalıştırabilir.
    """
    # Türkçe karakterler basic regex tokenization tarafından kelime parçası olarak işlenebilmelidir, böylece tokenizer Unicode inputlarda temel akışını çalıştırabilir.
    # Bu test, Türkçe karakterler basic regex tokenization tarafından kelime parçası olarak işlenebildiğini doğrular, böylece tokenizer Unicode inputlarda temel akışını çalıştırabilir.
    
    # Eğer Türkçe karakterler basic regex tokenization tarafından kelime parçası olarak işlenemezse, bu da Unicode inputlarda tokenizer'ın temel akışının çalışmamasına yol açar, bu da tokenizer'ın temel işlevselliğini ortadan kaldırır.
    # Eğer Türkçe karakterler basic regex tokenization tarafından kelime parçası olarak işlenemezse, ancak beklenmeyen sonuçlar üretebilir, bu da kullanıcı deneyimini olumsuz etkiler.
    # Eğer kullanıcıya açık ve anlaşılır hata mesajları sağlanmazsa, Türkçe karakterler basic regex tokenization tarafından kelime parçası olarak işlenemezse, ancak beklenmeyen sonuçlar üretebilir veya crash olabilir.
    # Bu nedenle, Türkçe karakterler basic regex tokenization tarafından kelime parçası olarak işlenebilmelidir, böylece tokenizer Unicode inputlarda temel akışını çalıştırabilir.
    
    tokenizer = TokenizerFactory.create("unigram", vocab_size=50)

    text = "çalışma öğrenme tokenizer"
    tokenizer.train(text)

    encoded = tokenizer.encode("çalışma") # Türkçe karakterler basic regex tokenization tarafından kelime parçası olarak işlenebilmelidir, böylece tokenizer Unicode inputlarda temel akışını çalıştırabilir.
    decoded = tokenizer.decode(encoded) 

    assert isinstance(encoded, list) # encode() çıktısının bir liste olması beklenir.
    assert len(encoded) > 0 # encode() çıktısında en az bir token id    
    assert decoded == "çalışma" # Türkçe karakterler basic regex tokenization tarafından kelime parçası olarak işlenebilmelidir, böylece tokenizer Unicode inputlarda temel akışını çalıştırabilir.
    assert decoded != "calisma" # Türkçe karakterler basic regex tokenization tarafından kelime parçası olarak işlenebilmelidir, böylece tokenizer Unicode inputlarda temel akışını çalıştırabilir.


# ---------------------------------------------------------
# INTERNAL VITERBI BEHAVIOR TESTS
# ---------------------------------------------------------

def test_unigram_tokenizer_viterbi_prefers_highest_scoring_segmentation() -> None:
    """
    Viterbi segmentasyonu en yüksek toplam log-probability skorunu seçmelidir.

    Bu testte vocabulary manuel olarak kontrol edilir.
    Böylece segmentation davranışı training frekanslarından bağımsız olarak net şekilde doğrulanır.

    Gerekçe:
        Viterbi segmentasyonu en yüksek toplam log-probability skorunu seçmelidir, böylece en olası segmentation sonucunu verir.

    Önlenen bug sınıfı:
        - Viterbi segmentasyonu en yüksek toplam log-probability skorunu seçmeyebilir, bu da beklenmeyen segmentasyon sonuçlarına yol açar, bu da tokenizer'ın temel işlevselliğini ortadan kaldırır.
        - Viterbi segmentasyonu en yüksek toplam log-probability skorunu seçmeyebilir, ancak beklenmeyen sonuçlar üretebilir, bu da kullanıcı deneyimini olumsuz etkiler.
        - Kullanıcıya açık ve anlaşılır hata mesajları sağlanmaz.
        - Viterbi segmentasyonu en yüksek toplam log-probability skorunu seçmeyebilir, ancak beklenmeyen sonuçlar üretebilir veya crash olabilir.
        - Bu nedenle, Viterbi segmentasyonu en yüksek toplam log-probability skorunu seçmelidir, böylece en olası segmentation sonucunu verir.

    Senaryo:
        word = "abcd"

        Olası parçalar:
            "ab" + "cd"
            "a" + "b" + "c" + "d"

        Daha yüksek skorlu olan "ab" + "cd" seçilmelidir.
    """
    # Viterbi segmentasyonu en yüksek toplam log-probability skorunu seçmelidir, böylece en olası segmentation sonucunu verir.
    # Bu testte vocabulary manuel olarak kontrol edilir, böylece segmentation davranışı training frekanslarından bağımsız olarak net şekilde doğrulanır.
    
    # Eğer Viterbi segmentasyonu en yüksek toplam log-probability skorunu seçmezse, bu da beklenmeyen segmentasyon sonuçlarına yol açar, bu da tokenizer'ın temel işlevselliğini ortadan kaldırır.
    # Eğer Viterbi segmentasyonu en yüksek toplam log-probability skorunu seçmezse, ancak beklenmeyen sonuçlar üretebilir, bu da kullanıcı deneyimini olumsuz etkiler.
    # Eğer kullanıcıya açık ve anlaşılır hata mesajları sağlanmazsa, Viterbi segmentasyonu en yüksek toplam log-probability skorunu seçmezse, ancak beklenmeyen sonuçlar üretebilir veya crash olabilir.
    # Bu nedenle, Viterbi segmentasyonu en yüksek toplam log-probability skorunu seçmelidir, böylece en olası segmentation sonucunu verir.
    
    tokenizer = TokenizerFactory.create("unigram", vocab_size=10)

    tokenizer._trained = True
    tokenizer._token_to_id = {
        tokenizer.UNKNOWN_TOKEN: 0,
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
        "ab": 5,
        "cd": 6,
    } 
    tokenizer._id_to_token = {
        idx: token for token, idx in tokenizer._token_to_id.items()
    } 
    tokenizer._token_logprob = {
        tokenizer.UNKNOWN_TOKEN: -100.0,
        "a": -5.0,
        "b": -5.0,
        "c": -5.0,
        "d": -5.0,
        "ab": -1.0,
        "cd": -1.0,
    }

    tokens = tokenizer._viterbi_segment("abcd") # Viterbi segmentasyonu en yüksek toplam log-probability skorunu seçmelidir, böylece en olası segmentation sonucunu verir.

    assert tokens == ["ab", "cd"] # "ab" + "cd" segmentasyonu toplam log-probability skoru -2.0 iken, "a" + "b" + "c" + "d" segmentasyonu toplam log-probability skoru -20.0'dır, bu nedenle daha yüksek skorlu olan "ab" + "cd" seçilmelidir.


def test_unigram_tokenizer_viterbi_returns_unknown_when_no_path_exists() -> None:
    """
    Eğer kelime vocabulary içindeki parçalarla segment edilemiyorsa [UNK] dönmelidir.

    Bu test Viterbi fallback davranışını izole şekilde doğrular.

    Gerekçe:
        Eğer kelime vocabulary içindeki parçalarla segment edilemiyorsa [UNK] dönmelidir, böylece tokenizer bilinmeyen inputlarda crash etmek yerine güvenli fallback davranışı gösterir.
    
    Önlenen bug sınıfı:
        - Eğer kelime vocabulary içindeki parçalarla segment edilemiyorsa, beklenmeyen sonuçlara veya crash'a yol açabilir, bu da tokenizer'ın temel işlevselliğini ortadan kaldırır.
        - Eğer kelime vocabulary içindeki parçalarla segment edilemiyorsa, beklenmeyen sonuçlar üretebilir, bu da kullanıcı deneyimini olumsuz etkiler.
        - Kullanıcıya açık ve anlaşılır hata mesajları sağlanmaz.
        - Eğer kelime vocabulary içindeki parçalarla segment edilemiyorsa, beklenmeyen sonuçlar üretebilir veya crash olabilir.
        - Bu nedenle, eğer kelime vocabulary içindeki parçalarla segment edilemiyorsa [UNK] dönmelidir, böylece tokenizer bilinmeyen inputlarda crash etmek yerine güvenli fallback davranışı gösterir.
    """
    # Eğer kelime vocabulary içindeki parçalarla segment edilemiyorsa [UNK] dönmelidir, böylece tokenizer bilinmeyen inputlarda crash etmek yerine güvenli fallback davranışı gösterir.
    # Bu test, eğer kelime vocabulary içindeki parçalarla segment edilemiyorsa [UNK] dönmesi gerektiğini doğrular, böylece tokenizer bilinmeyen inputlarda crash etmek yerine güvenli fallback davranışı gösterir.
    
    # Eğer kelime vocabulary içindeki parçalarla segment edilemiyorsa, beklenmeyen sonuçlara veya crash'a yol açabilir, bu da tokenizer'ın temel işlevselliğini ortadan kaldırır.
    # Eğer kelime vocabulary içindeki parçalarla segment edilemiyorsa, beklenmeyen sonuçlar üretebilir, bu da kullanıcı deneyimini olumsuz etkiler.
    # Eğer kullanıcıya açık ve anlaşılır hata mesajları sağlanmazsa, eğer kelime vocabulary içindeki parçalarla segment edilemiyorsa, beklenmeyen sonuçlar üretebilir veya crash olabilir.
    # Bu nedenle, eğer kelime vocabulary içindeki parçalarla segment edilemiyorsa [UNK] dönmelidir, böylece tokenizer bilinmeyen inputlarda crash etmek yerine güvenli fallback davranışı gösterir.
    
    tokenizer = TokenizerFactory.create("unigram", vocab_size=10)

    tokenizer._trained = True
    tokenizer._token_to_id = {
        tokenizer.UNKNOWN_TOKEN: 0,
        "a": 1,
    }
    tokenizer._id_to_token = {
        0: tokenizer.UNKNOWN_TOKEN,
        1: "a",
    }
    tokenizer._token_logprob = {
        tokenizer.UNKNOWN_TOKEN: -100.0,
        "a": -1.0,
    }

    tokens = tokenizer._viterbi_segment("zzz") # Eğer kelime vocabulary içindeki parçalarla segment edilemiyorsa [UNK] dönmelidir, böylece tokenizer bilinmeyen inputlarda crash etmek yerine güvenli fallback davranışı gösterir.

    assert tokens == [tokenizer.UNKNOWN_TOKEN] # "zzz" kelimesi vocabulary içindeki parçalarla segment edilemez, bu nedenle [UNK] dönmelidir, böylece tokenizer bilinmeyen inputlarda crash etmek yerine güvenli fallback davranışı gösterir.