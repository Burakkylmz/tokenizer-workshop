from __future__ import annotations

import pytest

from tokenizer_workshop.api.services.tokenizer_factory import TokenizerFactory


# ---------------------------------------------------------
# INIT TESTS
# ---------------------------------------------------------

def test_wordpiece_tokenizer_init_invalid_vocab_size_raises_error() -> None:
    """
    Guardrail testi: vocab_size >= 2 olmalıdır.

    Gerekçe:
        WordPiece vocabulary her zaman [UNK] token'ını içerir.
        vocab_size < 2 olması durumunda hem [UNK] hem de en az bir geçerli token temsil edilemez.

    Bu test şunu garanti eder:
        - Hatalı konfigürasyonlar erken aşamada engellenir
        - Tokenizer geçersiz bir state ile çalışmaz

    Önlenen bug sınıfı:
        - Sessiz hatalı konfigürasyon
        - Encode sırasında crash (UNK yokluğu)
    """
    # vocab_size 1 sadece [UNK] token'ını içereceğinden geçerli değildir.
    # Bu durum, tokenizer'ın herhangi bir gerçek token'ı temsil edememesine yol açar.
    # Bu test, bu tür yanlış yapılandırmaların önlenmesini sağlar.
    # Eğer vocab_size 1 olarak ayarlanırsa, bu durum sessiz hatalı konfigürasyona ve encode sırasında [UNK] token'ının eksikliği nedeniyle crash'e yol açabilir.

    with pytest.raises(ValueError, match="vocab_size must be at least 2"):
        TokenizerFactory.create("wordpiece", vocab_size=1)


def test_wordpiece_tokenizer_init_invalid_max_subword_length_raises_error() -> None:
    """
    Guardrail testi: max_subword_length >= 1 olmalıdır.

    Gerekçe:
        Subword üretimi en az 1 karakter gerektirir.
        0 uzunluklu parça üretimi algoritmayı bozar.

    Bu test şunu garanti eder:
        - Geçersiz parametreler kabul edilmez
        - Candidate üretimi düzgün çalışır

    Önlenen bug sınıfı:
        - Sonsuz loop
        - Boş token üretimi
    """
    # max_subword_length 0, kelimelerden herhangi bir subword parça üretilemeyeceği anlamına gelir.
    # Bu durum, tokenizer'ın kelimeleri parçalayamamasına ve dolayısıyla etkisiz hale gelmesine yol açar.
    # Bu test, bu tür yanlış yapılandırmaların önlenmesini sağlar.
    # Eğer max_subword_length 0 olarak ayarlanırsa, bu durum sonsuz döngüye veya boş token üretimine yol açabilir, çünkü tokenizer kelimeleri parçalayamaz ve sürekli aynı pozisyonda kalabilir.

    with pytest.raises(ValueError, match="max_subword_length must be at least 1"):
        TokenizerFactory.create("wordpiece", max_subword_length=0)


# ---------------------------------------------------------
# TRAIN TESTS
# ---------------------------------------------------------

def test_wordpiece_tokenizer_train_with_empty_text_raises_error() -> None:
    """
    Training invariant: boş metinle eğitim yapılmamalıdır.

    Gerekçe:
        WordPiece vocabulary istatistiksel frekanslara dayanır.
        Boş input anlamlı bir vocab üretemez.

    Bu test şunu garanti eder:
        - Geçersiz input açık şekilde reddedilir
        - Vocabulary her zaman anlamlı veri üzerinden oluşur

    Önlenen bug sınıfı:
        - Boş vocabulary
        - Frekans hesaplama hataları
    """
    # Boş metinle eğitim yapmak geçerli değildir 
    # Tokenizer'ın kelime istatistiklerine dayalı bir vocabulary oluşturması beklenir.
    # Bu test, bu tür yanlış eğitim girişlerinin önlenmesini sağlar.
    # Eğer tokenizer boş metinle eğitilirse, oluşturulan vocabulary anlamsız olur ve tokenization işlemi etkisiz hale gelir.
    
    tokenizer = TokenizerFactory.create("wordpiece")

    with pytest.raises(ValueError, match="Training text cannot be empty"):
        tokenizer.train("")


def test_wordpiece_tokenizer_train_with_whitespace_text_raises_error() -> None:
    """
    Training invariant: sadece whitespace içeren metinle eğitim yapılmamalıdır.

    Gerekçe:
        Sadece whitespace içeren metin, kelime frekansları açısından anlamsızdır.
        Bu durum, tokenizer'ın etkisiz bir vocabulary oluşturmasına yol açar.

    Bu test şunu garanti eder:
        - Geçersiz input açık şekilde reddedilir
        - Vocabulary her zaman anlamlı veri üzerinden oluşur

    Önlenen bug sınıfı:
        - Boş vocabulary
        - Frekans hesaplama hataları
    """
    # Sadece whitespace içeren metinle eğitim yapmak geçerli değildir
    # Tokenizer'ın kelime istatistiklerine dayalı bir vocabulary oluşturması beklenir.
    # Bu test, bu tür yanlış eğitim girişlerinin önlenmesini sağlar.
    # Eğer tokenizer sadece whitespace içeren metinle eğitilirse, oluşturulan vocabulary anlamsız olur ve tokenization işlemi etkisiz hale gelir.

    tokenizer = TokenizerFactory.create("wordpiece")

    with pytest.raises(ValueError, match="Training text cannot be empty"):
        tokenizer.train("   ")


def test_wordpiece_tokenizer_train_builds_vocab_with_unknown_token() -> None:
    """
    Vocabulary invariant: [UNK] her zaman mevcut olmalıdır.

    Gerekçe:
        WordPiece parçalayamadığı kelimeler için fallback olarak [UNK] kullanır.

    Bu test şunu garanti eder:
        - UNK token her zaman vardır
        - UNK mapping stabil (genelde index 0)

    Önlenen bug sınıfı:
        - OOV tokenlarda encode crash
        - Tutarsız unknown handling
    """
    # [UNK] token'ı her zaman vocabulary içinde bulunmalıdır.
    # Bu token, tokenizer'ın vocabulary ile parçalayamadığı kelimeleri temsil etmek için kullanılır.
    # Eğer [UNK] token'ı eksikse, tokenizer bilinmeyen kelimeleri encode etmeye çalışırken hata verecektir.
    # Bu test, [UNK] token'ının her zaman mevcut olduğunu ve genellikle index 0 olarak atandığını doğrular.
    # Bu, tokenizer'ın bilinmeyen kelimelerle karşılaştığında tutarlı bir şekilde davranmasını sağlar.
    # Eğer [UNK] token'ı eksikse, bu durum OOV (Out-Of-Vocabulary) token'larda encode sırasında crash'e ve tutarsız unknown handling'e yol açar.

    tokenizer = TokenizerFactory.create("wordpiece", vocab_size=20)

    tokenizer.train("tokenization token tokenizer")

    assert tokenizer.vocab_size > 0
    assert tokenizer._token_to_id[tokenizer.UNKNOWN_TOKEN] == 0


def test_wordpiece_tokenizer_vocab_size_does_not_exceed_target_size() -> None:
    """
    Kapasite kısıtı testi: vocabulary hedef boyutu aşmamalıdır.

    Gerekçe:
        Vocabulary büyümesi kontrol altında tutulmalıdır:
            - Memory kontrolü
            - Performans stabilitesi

    Bu test şunu garanti eder:
        - Frekans sıralaması doğru çalışır
        - Fazla token eklenmez

    Önlenen bug sınıfı:
        - Kontrolsüz vocab büyümesi
        - Performans düşüşü
    """
    # Vocabulary hedef boyutu aşmamalıdır. 
    # Bu, tokenizer'ın eğitim sırasında en sık görülen tokenları seçerek vocab büyümesini kontrol altında tutmasını sağlar.
    # Bu test, tokenizer'ın frekans sıralamasını doğru şekilde uyguladığını ve vocab_size parametresine uygun şekilde davranarak fazla token eklemediğini doğrular.
    # Eğer tokenizer vocab_size'ı aşarsa, bu durum bellek kullanımını artırabilir ve tokenization performansını düşürebilir.
    
    tokenizer = TokenizerFactory.create("wordpiece", vocab_size=10)

    tokenizer.train("tokenization token tokenizer")

    assert tokenizer.vocab_size <= 10


# ---------------------------------------------------------
# TOKENIZE TESTS
# ---------------------------------------------------------

def test_wordpiece_tokenizer_tokenize_before_training_returns_basic_tokens() -> None:
    """
    Pre-training fallback davranışı.

    Gerekçe:
        Tokenizer train edilmeden de basic tokenize edebilmelidir.

    Bu test şunu garanti eder:
        - API kullanıcı dostu davranır
        - Debug / exploration mümkün olur

    Tasarım notu:
        Bu bilinçli bir tasarım tercihidir.

    Önlenen bug sınıfı:
        - Gereksiz runtime hataları
    """
    # Tokenizer train edilmeden tokenize işlemi yapılmaya çalışıldığında, tokenizer'ın basic tokenization yaparak çalışması beklenir.
    # Bu test, tokenizer'ın train edilmeden de basic tokenization yapabildiğini doğrularak API'nin kullanıcı dostu ve esnek olduğunu gösterir.
    # Bu tasarım tercihi, kullanıcıların tokenizer'ı keşfetmesine ve debug yapmasına olanak tanır, aynı zamanda gereksiz runtime hatalarının önüne geçer.
    # Eğer tokenizer train edilmeden tokenize işlemi yapmaya çalışıldığında hata verirse, bu durum kullanıcı deneyimini olumsuz etkileyebilir ve gereksiz runtime hatalarına yol açabilir.

    tokenizer = TokenizerFactory.create("wordpiece")

    tokens = tokenizer.tokenize("Hello, World!")

    assert tokens == ["hello", ",", "world", "!"]


def test_wordpiece_tokenizer_tokenize_empty_text_returns_empty_list() -> None:
    """
    Boş input tokenize edildiğinde boş liste dönmelidir.

    Gerekçe:
        Boş input, tokenize edilebilecek herhangi bir token içermediğinden, tokenizer'ın boş liste döndürmesi beklenir.

    Bu test şunu garanti eder:
        - Edge case'ler doğru şekilde işlenir
        - Hatalar önlenir

    Önlenen bug sınıfı:
        - Hatalı tokenization sonuçları
        - Encode/decode hataları (boş token listesi beklenirken null veya hata)
        - Boş inputların yanlış işlenmesi
    """
    # Boş input, tokenize edilebilecek herhangi bir token içermediğinden, tokenizer'ın boş liste döndürmesi beklenir.
    # Bu test, tokenizer'ın boş inputları doğru şekilde işleyerek boş liste döndürdüğünü doğrular.
    # Eğer tokenizer boş inputu yanlış işleyerek null döndürürse veya hata verirse, bu durum encode/decode işlemlerinde hatalara yol açabilir, çünkü encode/decode işlemleri genellikle token listesi bekler.

    tokenizer = TokenizerFactory.create("wordpiece")

    assert tokenizer.tokenize("") == []


def test_wordpiece_tokenizer_tokenize_whitespace_text_returns_empty_list() -> None:
    """
    Sadece whitespace içeren input tokenize edildiğinde boş liste dönmelidir.

    Gerekçe:
        Sadece whitespace içeren input, tokenize edilebilecek herhangi bir token içermediğinden, tokenizer'ın boş liste döndürmesi beklenir.

    Bu test şunu garanti eder:
        - Edge case'ler doğru şekilde işlenir
        - Hatalar önlenir

    Önlenen bug sınıfı:
        - Hatalı tokenization sonuçları
        - Encode/decode hataları (boş token listesi beklenirken null veya hata)
        - Boş inputların yanlış işlenmesi
    """
    # Sadece whitespace içeren input, tokenize edilebilecek herhangi bir token içermediğinden, tokenizer'ın boş liste döndürmesi beklenir.
    # Bu test, tokenizer'ın bu tür inputları doğru şekilde işleyerek boş liste döndürdüğünü doğrular.
    # Eğer tokenizer sadece whitespace içeren inputu yanlış işleyerek null döndürürse veya hata verirse, bu durum encode/decode işlemlerinde hatalara yol açabilir, çünkü encode/decode işlemleri genellikle token listesi bekler.

    tokenizer = TokenizerFactory.create("wordpiece")

    assert tokenizer.tokenize("   ") == []


def test_wordpiece_tokenizer_uses_full_word_when_available_in_vocab() -> None:
    """
    Eğer kelime vocabulary içinde tam olarak mevcutsa, tokenize işlemi sırasında o kelimenin tamamı tek bir token olarak dönmelidir.

    Gerekçe:
        WordPiece tokenizer, mümkün olan en uzun eşleşmeyi tercih eder. 
        Eğer kelimenin tamamı vocabulary içinde mevcutsa, onu tek bir token olarak kullanmak en uzun eşleşmeyi sağlar.
        Bu test, tokenizer'ın mümkün olan en uzun eşleşmeyi tercih ettiğini doğrular.
        Eğer kelimenin tamamı vocabulary içinde mevcutsa, tokenizer'ın onu tek bir token olarak kullanması beklenir.
        Bu durum, tokenizer'ın doğru şekilde çalıştığını gösterir.

    Bu test şunu garanti eder:
        - Mümkün olan en uzun eşleşme tercih edilir
        - Kelime tam olarak mevcutsa tek token olarak kullanılır
        - Tokenizer doğru şekilde çalışır

    Önlenen bug sınıfı:
        - Subword parçalama hataları (tam kelime mevcutken parçalama)
        - Performans düşüşü (gereksiz parçalama)
        - Hatalı tokenization sonuçları (beklenenden farklı tokenlar)
    """
    # Eğer kelime vocabulary içinde tam olarak mevcutsa, tokenize işlemi sırasında o kelimenin tamamı tek bir token olarak dönmelidir.
    # Bu test, tokenizer'ın mümkün olan en uzun eşleşmeyi tercih ettiğini doğrular. Eğer kelimenin tamamı vocabulary içinde mevcutsa, onu tek bir token olarak kullanmak en uzun eşleşmeyi sağlar ve tokenizer'ın doğru şekilde çalıştığını gösterir.
    # Eğer kelimenin tamamı vocabulary içinde mevcutsa, tokenizer'ın onu tek bir token olarak kullanması beklenir. Bu durum, tokenizer'ın doğru şekilde çalıştığını gösterir.

    tokenizer = TokenizerFactory.create("wordpiece", vocab_size=50)

    tokenizer.train("token tokenization tokenizer")

    tokens = tokenizer.tokenize("token")

    assert tokens == ["token"]


def test_wordpiece_tokenizer_can_generate_subword_tokens() -> None:
    """
    Core davranış testi: WordPiece tokenizer continuation subword üretebilmelidir.

    Gerekçe:
        WordPiece'in temel farkı, kelimeleri yalnızca full-word token olarak değil,
        kelime içi devam parçalarıyla da temsil edebilmesidir. Bu devam parçaları
        '##' prefix'i ile gösterilir.

    Bu test şunu garanti eder:
        - Public tokenize() akışı '##' continuation token üretebilir
        - Kelime başı token ile devam tokenı birlikte çalışır
        - Tokenizer word-level davranışa düşmez

    Önlenen bug sınıfı:
        - Continuation prefix mantığının bozulması
        - Subword segmentation'ın devre dışı kalması
        - WordPiece'in yalnızca basic word tokenizer gibi davranması
    """
    # WordPiece tokenizer'ın temel farkı, kelimeleri yalnızca full-word token olarak değil, kelime içi devam parçalarıyla da temsil edebilmesidir. 
    # Bu devam parçaları '##' prefix'i ile gösterilir. 
    # Bu test, tokenizer'ın public tokenize() akışında '##' continuation token üretebildiğini doğrular.
    # Eğer kelimenin tamamı vocabulary içinde mevcut değilse, tokenizer'ın kelimeyi parçalayarak '##' prefix'li continuation tokenları üretebilmesi beklenir.
    
    tokenizer = TokenizerFactory.create("wordpiece", vocab_size=10)

    tokenizer.train("tokenization")

    tokenizer._token_to_id = {
        tokenizer.UNKNOWN_TOKEN: 0,
        "token": 1,
        "##ization": 2,
    }
    tokenizer._id_to_token = {
        0: tokenizer.UNKNOWN_TOKEN,
        1: "token",
        2: "##ization",
    }

    tokens = tokenizer.tokenize("tokenization")

    assert tokens == ["token", "##ization"]
    assert any(token.startswith("##") for token in tokens)


def test_wordpiece_tokenizer_unknown_word_returns_unknown_token() -> None:
    """
    Vocabulary ile parçalanamayan kelime [UNK] olarak temsil edilmelidir.

    Gerekçe:
        WordPiece tokenizer, vocabulary ile parçalayamadığı kelimeleri temsil etmek için [UNK] token'ını kullanır. 
        Bu test, tokenizer'ın bilinmeyen kelimeleri doğru şekilde [UNK] token'ı ile temsil ettiğini doğrular. 
        Eğer tokenizer bilinmeyen kelimeleri [UNK] olarak temsil edemezse, bu durum tokenizer'ın beklenmedik inputlarla karşılaştığında hatalı davranmasına yol açar.

    Bu test şunu garanti eder:
        - Bilinmeyen kelimeler [UNK] token'ı ile temsil edilir
        - Tokenizer beklenmedik inputlarla karşılaştığında hatalı davranmaz
        - OOV handling tutarlı olur

    Önlenen bug sınıfı:
        - OOV kelimelerde encode crash
        - Hatalı tokenization sonuçları (beklenmeyen tokenlar)
        - Tutarsız unknown handling
    """
    # Vocabulary ile parçalanamayan kelime [UNK] olarak temsil edilmelidir.
    # Bu test, tokenizer'ın bilinmeyen kelimeleri doğru şekilde [UNK] token'ı ile temsil ettiğini doğrular. 
    # Eğer tokenizer bilinmeyen kelimeleri [UNK] olarak temsil edemezse, bu durum tokenizer'ın beklenmedik inputlarla karşılaştığında hatalı davranmasına yol açar.
    
    tokenizer = TokenizerFactory.create(
        "wordpiece",
        vocab_size=5,
        max_subword_length=2,
    )

    tokenizer.train("aa bb cc")

    tokens = tokenizer.tokenize("zzzz")

    assert tokens == [tokenizer.UNKNOWN_TOKEN]


# ---------------------------------------------------------
# ENCODE TESTS
# ---------------------------------------------------------

def test_wordpiece_tokenizer_encode_before_training_raises_error() -> None:
    """
    train() çağrılmadan encode() çalışmamalıdır.

    Gerekçe:
        Encode işlemi token -> id mapping'ine ihtiyaç duyar.
        Bu mapping, train() sırasında oluşturulur. 
        train() çağrılmadan encode() yapılmaya çalışılırsa, mapping mevcut olmadığından hata verilmelidir.

    Bu test şunu garanti eder:
        - API yanlış kullanımına karşı koruma sağlar
        - Hatalar erken aşamada yakalanır
    
    Önlenen bug sınıfı:
        - Sessiz hatalı kullanım
        - Encode sırasında crash (mapping yokluğu)
    """
    # Encode işlemi token -> id mapping'ine ihtiyaç duyar. Bu mapping, train() sırasında oluşturulur.
    # train() çağrılmadan encode() yapılmaya çalışılırsa, mapping mevcut olmadığından hata verilmelidir. 
    # Bu test, tokenizer'ın train edilmeden encode edilmeye çalışıldığında uygun şekilde hata verdiğini doğrular, 
    # böylece API'nin yanlış kullanımına karşı koruma sağlanır ve hatalar erken aşamada yakalanır.
    # Eğer tokenizer train edilmeden encode edilmeye çalışılırsa, bu durum sessiz hatalı kullanıma ve encode sırasında mapping yokluğu nedeniyle crash'e yol açabilir. 

    tokenizer = TokenizerFactory.create("wordpiece")

    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.encode("tokenization")


def test_wordpiece_tokenizer_encode_returns_integer_token_ids() -> None:
    """
    encode() string tokenları integer id listesine dönüştürmelidir.

    Gerekçe:
        Tokenizer'ın encode() metodu, tokenları modelin anlayabileceği sayısal formata dönüştürmelidir. 
        Bu test, encode() metodunun string tokenları integer id listesine doğru şekilde dönüştürdüğünü doğrular.
        Eğer encode() string tokenları integer id'lere dönüştüremezse, bu durum modelin beklediği formatta input alamamasına ve downstream işlemlerde hatalara yol açar.

    Bu test şunu garanti eder:
        - Encode işlemi doğru şekilde integer id listesi döndürür
        - Modelin beklediği formatta input sağlanır
        - Hatalar önlenir (string tokenlar yerine id'ler beklenirken hatalar)
    
    Önlenen bug sınıfı:
        - Encode sırasında crash (string tokenlar yerine id'ler beklenirken hatalar)
        - Hatalı token id'ler (string tokenlar doğru şekilde id'lere dönüştürülmezse)
    """
    # Encode() string tokenları integer id listesine dönüştürmelidir. 
    # Bu test, encode() metodunun string tokenları integer id listesine doğru şekilde dönüştürdüğünü doğrular.
    # Eğer encode() string tokenları integer id'lere dönüştüremezse, bu durum modelin beklediği formatta input alamamasına ve downstream işlemlerde hatalara yol açar.
    
    tokenizer = TokenizerFactory.create("wordpiece", vocab_size=50)
    tokenizer.train("tokenization token tokenizer")

    encoded = tokenizer.encode("tokenization")

    assert isinstance(encoded, list)
    assert len(encoded) > 0
    assert all(isinstance(token_id, int) for token_id in encoded)


def test_wordpiece_tokenizer_encode_unknown_word_uses_unknown_token_id() -> None:
    """
    Vocabulary ile parçalanamayan kelime encode sırasında [UNK] id'si ile temsil edilmelidir.

    Gerekçe:
        WordPiece tokenizer, vocabulary ile parçalayamadığı kelimeleri temsil etmek için [UNK] token'ını kullanır. 
        Encode işlemi sırasında, tokenizer'ın bilinmeyen kelimeleri doğru şekilde [UNK] token'ının id'si ile temsil ettiğini doğrular.
        Eğer tokenizer bilinmeyen kelimeleri [UNK] id'si ile temsil edemezse, bu durum tokenizer'ın beklenmedik inputlarla karşılaştığında hatalı davranmasına yol açar.

    Bu test şunu garanti eder:
        - Bilinmeyen kelimeler [UNK] id'si ile temsil edilir
        - Tokenizer beklenmedik inputlarla karşılaştığında hatalı davranmaz
        - OOV handling tutarlı olur 

    Önlenen bug sınıfı:
        - OOV kelimelerde encode crash
        - Hatalı tokenization sonuçları (beklenmeyen token id'leri)
        - Tutarsız unknown handling
    """
    # Vocabulary ile parçalanamayan kelime encode sırasında [UNK] id'si ile temsil edilmelidir. 
    # Bu test, tokenizer'ın bilinmeyen kelimeleri doğru şekilde [UNK] token'ının id'si ile temsil ettiğini doğrular.    
    # Eğer tokenizer bilinmeyen kelimeleri [UNK] id'si ile temsil edemezse, bu durum tokenizer'ın beklenmedik inputlarla karşılaştığında hatalı davranmasına yol açar.
    
    tokenizer = TokenizerFactory.create(
        "wordpiece",
        vocab_size=5,
        max_subword_length=2,
    )

    tokenizer.train("aa bb cc")

    encoded = tokenizer.encode("zzzz")

    assert encoded == [tokenizer._token_to_id[tokenizer.UNKNOWN_TOKEN]]


# ---------------------------------------------------------
# DECODE TESTS
# ---------------------------------------------------------

def test_wordpiece_tokenizer_decode_before_training_raises_error() -> None:
    """
    train() çağrılmadan decode() yapılmaya çalışılırsa, mapping mevcut olmadığından hata verilmelidir.

    Bu test şunu garanti eder:  
        - API yanlış kullanımına karşı koruma sağlar
        - Hatalar erken aşamada yakalanır

    Önlenen bug sınıfı:
        - Sessiz hatalı kullanım
        - Decode sırasında crash (mapping yokluğu)
    """
    # Decode işlemi id -> token mapping'ine ihtiyaç duyar. Bu mapping, train() sırasında oluşturulur.
    # train() çağrılmadan decode() yapılmaya çalışılırsa, mapping mevcut olmadığından hata verilmelidir.
    # Bu test, tokenizer'ın train edilmeden decode edilmeye çalışıldığında uygun şekilde hata verdiğini doğrular, böylece API'nin yanlış kullanımına karşı koruma sağlanır ve hatalar erken aşamada yakalanır.
    # Eğer tokenizer train edilmeden decode edilmeye çalışılırsa, bu durum sessiz hatalı kullanıma ve decode sırasında mapping yokluğu nedeniyle crash'e yol açabilir. 

    tokenizer = TokenizerFactory.create("wordpiece")

    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.decode([0, 1])


def test_wordpiece_tokenizer_decode_unknown_token_id_raises_error() -> None:
    """
    Vocabulary içinde olmayan token id decode edilmeye çalışılırsa hata vermelidir.

    Gerekçe:
        Decode işlemi sırasında, tokenizer'ın id -> token mapping'inde olmayan token id'leri doğru şekilde tespit edip hata vermesi beklenir.
        Eğer tokenizer bilinmeyen token id'leri tespit edip hata veremezse, bu durum tokenizer'ın beklenmedik inputlarla karşılaştığında hatalı davranmasına yol açar.
        Bu test, tokenizer'ın bilinmeyen token id'leri doğru şekilde tespit edip hata verdiğini doğrular.

    Bu test şunu garanti eder:
        - Bilinmeyen token id'ler tespit edilir
        - Hatalar önlenir (string tokenlar yerine id'ler beklenirken hatalar)
        - Tokenizer beklenmedik inputlarla karşılaştığında hatalı davranmaz
        - Decode işlemi güvenilir olur

    Önlenen bug sınıfı:
        - Decode sırasında crash (bilinmeyen token id'leri)
        - Hatalı decode sonuçları (beklenmeyen token id'leri decode edilmeye çalışıldığında hatalar)
        - Tokenizer beklenmedik inputlarla karşılaştığında hatalı davranmaz
        - Decode işlemi güvenilir olur
    """
    # Decode işlemi sırasında, tokenizer'ın id -> token mapping'inde olmayan token id'leri doğru şekilde tespit edip hata vermesi beklenir.
    # Eğer tokenizer bilinmeyen token id'leri tespit edip hata veremezse, bu durum tokenizer'ın beklenmedik inputlarla karşılaştığında hatalı davranmasına yol açar.
    # Bu test, tokenizer'ın bilinmeyen token id'leri doğru şekilde tespit edip hata verdiğini doğrular.

    tokenizer = TokenizerFactory.create("wordpiece", vocab_size=20)
    tokenizer.train("tokenization token tokenizer")

    with pytest.raises(ValueError, match="Unknown token id encountered"):
        tokenizer.decode([9999])


def test_wordpiece_tokenizer_decode_merges_continuation_pieces() -> None:
    """
    '##' prefix'li parçalar decode sırasında önceki token ile birleştirilmelidir.

    Gerekçe:
        WordPiece tokenizer, continuation parçaları '##' prefix'i ile işaretler.
        Decode işlemi sırasında, tokenizer'ın bu '##' prefix'li parçaları önceki token ile doğru şekilde birleştirmesi beklenir.
        Eğer tokenizer '##' prefix'li parçaları doğru şekilde birleştiremezse, bu durum decode sonucunun beklenenden farklı ve hatalı olmasına yol açar.
        Bu test, tokenizer'ın '##' prefix'li parçaları doğru şekilde birleştirdiğini doğrular.

    Bu test şunu garanti eder:
        - '##' prefix'li parçalar doğru şekilde birleştirilir
        - Decode sonucunda beklenen string elde edilir
        - Tokenizer doğru şekilde çalışır

    Önlenen bug sınıfı:
        - '##' prefix'li parçaların hatalı birleştirilmesi
        - Hatalı decode sonuçları (beklenmeyen token birleşimleri)
        - Tokenizer beklenmedik inputlarla karşılaştığında hatalı davranmaz
        - Decode işlemi güvenilir olur
    """
    # '##' prefix'li parçalar decode sırasında önceki token ile birleştirilmelidir. 
    # WordPiece tokenizer, continuation parçaları '##' prefix'i ile işaretler.
    # Decode işlemi sırasında, tokenizer'ın bu '##' prefix'li parçaları önceki token ile doğru şekilde birleştirmesi beklenir.
    # Eğer tokenizer '##' prefix'li parçaları doğru şekilde birleştiremezse, bu durum decode sonucunun beklenenden farklı ve hatalı olmasına yol açar. 
    # Bu test, tokenizer'ın '##' prefix'li parçaları doğru şekilde birleştirdiğini doğrular.
    
    tokenizer = TokenizerFactory.create("wordpiece", vocab_size=10)
    tokenizer.train("token ization")

    tokenizer._token_to_id = {
        tokenizer.UNKNOWN_TOKEN: 0,
        "token": 1,
        "##ization": 2,
    }
    tokenizer._id_to_token = {
        0: tokenizer.UNKNOWN_TOKEN,
        1: "token",
        2: "##ization",
    }

    decoded = tokenizer.decode([1, 2])

    assert decoded == "tokenization"


def test_wordpiece_tokenizer_decode_preserves_unknown_token() -> None:
    """
    [UNK] token decode sırasında korunmalıdır.

    Gerekçe:
        [UNK] token'ı, tokenizer'ın vocabulary ile parçalayamadığı kelimeleri temsil etmek için kullanılır. 
        Decode işlemi sırasında, tokenizer'ın [UNK] token'ını doğru şekilde tanıyıp koruması beklenir.
        Eğer tokenizer [UNK] token'ını tanıyıp koruyamazsa, bu durum decode sonucunun beklenenden farklı ve hatalı olmasına yol açar.
        Bu test, tokenizer'ın [UNK] token'ını doğru şekilde tanıyıp koruduğunu doğrular.

    Bu test şunu garanti eder:
        - [UNK] token decode sırasında korunur
        - Decode sonucunda beklenen string elde edilir
        - Tokenizer doğru şekilde çalışır
        - Decode işlemi güvenilir olur

    Önlenen bug sınıfı:
        - [UNK] token'ının hatalı işlenmesi
        - Hatalı decode sonuçları (beklenmeyen token birleşimleri)
        - Tokenizer beklenmedik inputlarla karşılaştığında hatalı davranmaz
        - Decode işlemi güvenilir olur
    """
    # [UNK] token decode sırasında korunmalıdır. 
    # [UNK] token'ı, tokenizer'ın vocabulary ile parçalayamadığı kelimeleri temsil etmek için kullanılır. 
    # Decode işlemi sırasında, tokenizer'ın [UNK] token'ını doğru şekilde tanıyıp koruması beklenir.
    # Eğer tokenizer [UNK] token'ını tanıyıp koruyamazsa, bu durum decode sonucunun beklenenden farklı ve hatalı olmasına yol açar. 
    # Bu test, tokenizer'ın [UNK] token'ını doğru şekilde tanıyıp koruduğunu doğrular.
    
    tokenizer = TokenizerFactory.create("wordpiece", vocab_size=10)
    tokenizer.train("token")

    decoded = tokenizer.decode([tokenizer._token_to_id[tokenizer.UNKNOWN_TOKEN]])

    assert decoded == tokenizer.UNKNOWN_TOKEN


def test_wordpiece_tokenizer_encode_decode_returns_readable_text() -> None:
    """
    encode -> decode akışı okunabilir bir string üretmelidir.   

    Gerekçe: 
        WordPiece tokenizer, encode ve decode işlemlerinin birlikte çalışarak anlamlı ve okunabilir bir string üretmesini sağlamalıdır. 
        Bu test, tokenizer'ın encode ve decode işlemlerinin birlikte çalışarak okunabilir bir string ürettiğini doğrular. 
        Eğer encode ve decode işlemleri doğru şekilde çalışmazsa, bu durum decode sonucunun beklenenden farklı ve hatalı olmasına yol açar.

    Bu test şunu garanti eder:
        - Encode ve decode işlemleri birlikte çalışarak anlamlı ve okunabilir bir string üretir
        - Decode sonucunda beklenen string elde edilir
        - Tokenizer doğru şekilde çalışır
        - Decode işlemi güvenilir olur

    Önlenen bug sınıfı:
        - Hatalı decode sonuçları (beklenmeyen token birleşimleri)
        - Tokenizer beklenmedik inputlarla karşılaştığında hatalı davranmaz
        - Decode işlemi güvenilir olur

    Not:
        WordPiece lowercase ve spacing bilgisini birebir korumadığı için
        her durumda orijinal metinle birebir eşitlik beklenmez.
    """
    # encode -> decode akışı okunabilir bir string üretmelidir. 
    # WordPiece lowercase ve spacing bilgisini birebir korumadığı için her durumda orijinal metinle birebir eşitlik beklenmez, ancak sonuç okunabilir ve anlamlı bir string olmalıdır.
    # Bu test, tokenizer'ın encode ve decode işlemlerinin birlikte çalışarak okunabilir bir string ürettiğini doğrular. 
    # Eğer encode ve decode işlemleri doğru şekilde çalışmazsa, bu durum decode sonucunun beklenenden farklı ve hatalı olmasına yol açar.
    
    tokenizer = TokenizerFactory.create("wordpiece", vocab_size=50)
    tokenizer.train("tokenization token tokenizer")

    text = "tokenization"
    decoded = tokenizer.decode(tokenizer.encode(text))

    assert isinstance(decoded, str)
    assert len(decoded) > 0


# ---------------------------------------------------------
# BASIC TOKENIZATION TESTS
# ---------------------------------------------------------

def test_wordpiece_tokenizer_basic_tokenize_lowercases_text() -> None:
    """
    _basic_tokenize() metni lowercase hale getirmelidir.

    Gerekçe:
        WordPiece tokenizer genellikle lowercase tokenization yapar.
        Bu test, _basic_tokenize() metodunun metni doğru şekilde lowercase hale getirdiğini doğrular.
        Eğer _basic_tokenize() metni lowercase hale getiremezse, bu durum tokenizer'ın beklenmedik inputlarla karşılaştığında hatalı davranmasına yol açar.

    Bu test şunu garanti eder:
        - Metin doğru şekilde lowercase hale getirilir
        - Tokenizer doğru şekilde çalışır
        - Decode işlemi güvenilir olur

    Önlenen bug sınıfı:
        - Hatalı lowercase işlemleri
        - Tokenizer beklenmedik inputlarla karşılaştığında hatalı davranmaz
        - Decode işlemi güvenilir olur
    """
    # _basic_tokenize() metni lowercase hale getirmelidir. 
    # WordPiece tokenizer genellikle lowercase tokenization yapar.
    # Bu test, _basic_tokenize() metodunun metni doğru şekilde lowercase hale getirdiğini doğrular. 
    # Eğer _basic_tokenize() metni lowercase hale getiremezse, bu durum tokenizer'ın beklenmedik inputlarla karşılaştığında hatalı davranmasına yol açar.
    
    tokenizer = TokenizerFactory.create("wordpiece")

    tokens = tokenizer._basic_tokenize("Hello WORLD")

    assert tokens == ["hello", "world"]


def test_wordpiece_tokenizer_basic_tokenize_splits_punctuation() -> None:
    """
    Noktalama işaretleri ayrı token olarak ayrılmalıdır.

    Gerekçe:
        WordPiece tokenizer, metni tokenize ederken noktalama işaretlerini ayrı tokenlar olarak ayırmalıdır.
        Bu test, _basic_tokenize() metodunun noktalama işaretlerini doğru şekilde ayırdığını doğrular.
        Eğer noktalama işaretleri doğru şekilde ayrılmazsa, bu durum tokenizer'ın beklenmedik inputlarla karşılaştığında hatalı davranmasına yol açar.

    Bu test şunu garanti eder:
        - Noktalama işaretleri doğru şekilde ayrılır
        - Tokenizer doğru şekilde çalışır
        - Decode işlemi güvenilir olur

    Önlenen bug sınıfı:
        - Hatalı noktalama ayrımı
        - Tokenizer beklenmedik inputlarla karşılaştığında hatalı davranmaz
        - Decode işlemi güvenilir olur
    """
    # Noktalama işaretleri ayrı token olarak ayrılmalıdır. 
    # WordPiece tokenizer, metni tokenize ederken noktalama işaretlerini ayrı tokenlar olarak ayırmalıdır.
    # Bu test, _basic_tokenize() metodunun noktalama işaretlerini doğru şekilde ayırdığını doğrular. 
    # Eğer noktalama işaretleri doğru şekilde ayrılmazsa, bu durum tokenizer'ın beklenmedik inputlarla karşılaştığında hatalı davranmasına yol açar.
    
    tokenizer = TokenizerFactory.create("wordpiece")

    tokens = tokenizer._basic_tokenize("Merhaba, dünya!")

    assert tokens == ["merhaba", ",", "dünya", "!"]


# ---------------------------------------------------------
# CANDIDATE GENERATION TESTS
# ---------------------------------------------------------

def test_wordpiece_tokenizer_generate_candidates_for_word() -> None:
    """
    Bir kelimeden hem başlangıç hem continuation WordPiece adayları üretilmelidir.

    Gerekçe:
        WordPiece tokenizer, kelimeleri parçalarken hem başlangıç (örneğin "to", "tok") hem de continuation (örneğin "##o", "##k") parçalar üretebilmelidir.
        Bu test, _generate_wordpiece_candidates() metodunun bir kelimeden doğru şekilde başlangıç ve continuation WordPiece adayları ürettiğini doğrular.
        Eğer _generate_wordpiece_candidates() doğru şekilde adaylar üretemezse, bu durum tokenizer'ın kelimeleri parçalarken beklenmedik ve hatalı davranmasına yol açar.
    
    Bu test şunu garanti eder:
        - Bir kelimeden başlangıç ve continuation adayları üretilir
        - WordPiece tokenizer'ın temel davranışı sağlanır
        - Tokenizer doğru şekilde çalışır

    Önlenen bug sınıfı:
        - Hatalı aday üretimi (başlangıç veya continuation parçaların eksik olması)
        - Tokenizer beklenmedik inputlarla karşılaştığında hatalı davranmaz
        - Decode işlemi güvenilir olur
    """
    # Bir kelimeden hem başlangıç hem continuation WordPiece adayları üretilmelidir. 
    # WordPiece tokenizer, kelimeleri parçalarken hem başlangıç (örneğin "to", "tok") hem de continuation (örneğin "##o", "##k") parçalar üretebilmelidir.
    # Bu test, _generate_wordpiece_candidates() metodunun bir kelimeden doğru şekilde başlangıç ve continuation WordPiece adayları ürettiğini doğrular. 
    # Eğer _generate_wordpiece_candidates() doğru şekilde adaylar üretemezse, bu durum tokenizer'ın kelimeleri parçalarken beklenmedik ve hatalı davranmasına yol açar.
    
    tokenizer = TokenizerFactory.create("wordpiece", max_subword_length=3)

    candidates = tokenizer._generate_wordpiece_candidates("token")

    assert "t" in candidates
    assert "to" in candidates
    assert "tok" in candidates
    assert "##o" in candidates
    assert "##ok" in candidates
    assert "##k" in candidates


def test_wordpiece_tokenizer_generate_candidates_respects_max_subword_length() -> None:
    """
    Üretilen adayların gerçek parça uzunluğu max_subword_length değerini aşmamalıdır.
    
    '##' prefix'i uzunluk hesabına dahil edilmez.

    Gerekçe:
        WordPiece tokenizer, ürettiği adayların gerçek parça uzunluğunun max_subword_length parametresini aşmamasını sağlamalıdır.
        Bu test, _generate_wordpiece_candidates() metodunun ürettiği adayların gerçek parça uzunluğunun max_subword_length değerini aşmadığını doğrular.
        Eğer _generate_wordpiece_candidates() max_subword_length kısıtını doğru şekilde uygulayamazsa, bu durum tokenizer'ın beklenmedik ve hatalı adaylar üretmesine yol açar.

    Bu test şunu garanti eder:
        - Üretilen adayların gerçek parça uzunluğu max_subword_length değerini aşmaz
        - '##' prefix'i uzunluk hesabına dahil edilmez
        - Tokenizer doğru şekilde çalışır
        - Decode işlemi güvenilir olur

    Önlenen bug sınıfı:
        - Hatalı aday üretimi (max_subword_length kısıtının ihlal edilmesi)
        - Tokenizer beklenmedik inputlarla karşılaştığında hatalı davranmaz
        - Decode işlemi güvenilir olur
    """
    # Üretilen adayların gerçek parça uzunluğu max_subword_length değerini aşmamalıdır. 
    # '##' prefix'i uzunluk hesabına dahil edilmez.
    # Bu test, _generate_wordpiece_candidates() metodunun ürettiği adayların gerçek parça uzunluğunun max_subword_length değerini aşmadığını doğrular. 
    # Eğer _generate_wordpiece_candidates() max_subword_length kısıtını doğru şekilde uygulayamazsa, bu durum tokenizer'ın beklenmedik ve hatalı adaylar üretmesine yol açar.

    tokenizer = TokenizerFactory.create("wordpiece", max_subword_length=2)

    candidates = tokenizer._generate_wordpiece_candidates("token")

    for candidate in candidates:
        raw_piece = candidate[2:] if candidate.startswith("##") else candidate
        assert len(raw_piece) <= 2


# ---------------------------------------------------------
# GREEDY TOKENIZATION TESTS
# ---------------------------------------------------------

def test_wordpiece_tokenizer_greedy_prefers_longest_match() -> None:
    """
    Algoritmik invariant: longest-match greedy uygulanmalıdır.

    Gerekçe:
        WordPiece en uzun eşleşmeyi seçerek:
            - token sayısını azaltır
            - daha verimli temsil sağlar

    Bu test şunu garanti eder:
        - Uzun parça tercih edilir
        - Kısa parçalar yanlış seçilmez

    Örnek:
        "tokenization" → ["token", "##ization"]
        yanlış: ["to", "##ken", ...]

    Önlenen bug sınıfı:
        - Suboptimal tokenization
        - Gereksiz uzun sequence
    """
    # Longest-match greedy algoritması, WordPiece tokenizer'ın temel bir özelliğidir. 
    # Bu test, tokenizer'ın longest-match greedy algoritmasını doğru şekilde uyguladığını doğrular. 
    # Eğer tokenizer longest-match greedy algoritmasını doğru şekilde uygulayamazsa, bu durum suboptimal tokenization'a ve gereksiz uzun token sequence'lerine yol açar.

    tokenizer = TokenizerFactory.create("wordpiece", vocab_size=10)
    tokenizer.train("token ization")

    tokenizer._token_to_id = {
        tokenizer.UNKNOWN_TOKEN: 0,
        "to": 1,
        "token": 2,
        "##ization": 3,
        "##i": 4,
    }
    tokenizer._id_to_token = {
        index: token
        for token, index in tokenizer._token_to_id.items()
    }

    tokens = tokenizer._greedy_wordpiece_tokenize("tokenization")

    assert tokens == ["token", "##ization"]


def test_wordpiece_tokenizer_greedy_returns_unknown_when_no_piece_matches() -> None:
    """
    Fallback invariant: parçalanamayan kelime [UNK] olmalıdır.

    Gerekçe:
        Hiçbir subword eşleşmesi yoksa tokenizer partial üretmemelidir.

    Bu test şunu garanti eder:
        - Strict fallback davranışı
        - Hatalı parçalama yok

    Önlenen bug sınıfı:
        - Bozuk token dizileri
    """
    # Hiçbir subword eşleşmesi yoksa tokenizer partial üretmemelidir. 
    # Bu test, tokenizer'ın hiçbir subword eşleşmesi olmayan bir kelimeyle karşılaştığında strict fallback davranışı sergileyerek [UNK] token'ını döndürdüğünü doğrular. 
    # Eğer tokenizer hiçbir subword eşleşmesi olmayan bir kelimeyle karşılaştığında strict fallback davranışı sergileyemezse, bu durum bozuk token dizilerine ve hatalı parçalamaya yol açar.

    tokenizer = TokenizerFactory.create("wordpiece", vocab_size=10)
    tokenizer.train("token")

    tokenizer._token_to_id = {
        tokenizer.UNKNOWN_TOKEN: 0,
        "token": 1,
    }
    tokenizer._id_to_token = {
        0: tokenizer.UNKNOWN_TOKEN,
        1: "token",
    }

    tokens = tokenizer._greedy_wordpiece_tokenize("xyz")

    assert tokens == [tokenizer.UNKNOWN_TOKEN]


# ---------------------------------------------------------
# DETERMINISM TESTS
# ---------------------------------------------------------

def test_wordpiece_tokenizer_is_deterministic_for_same_input() -> None:
    """
    Aynı config ve aynı training text ile eğitilen iki WordPiece tokenizer
    aynı tokenization ve encode çıktısını üretmelidir.

    Gerekçe:
        Tokenizer'ın deterministik olması beklenir. 
        Aynı eğitim verisi ve aynı yapılandırma ile eğitilen tokenizer'ların aynı çıktıyı üretmesi beklenir.

    Bu test şunu garanti eder:
        - Tokenizer deterministikdir
        - Aynı eğitim verisi ve yapılandırma aynı çıktıyı üretir
        - Hatalar önlenir (aynı eğitim verisi ve yapılandırma ile farklı çıktılar hatalara yol açar)
        - Decode işlemi güvenilir olur
    
    Önlenen bug sınıfı:
        - Non-deterministic tokenization
        - Hatalı tokenization sonuçları (aynı eğitim verisi ve yapılandırma ile farklı çıktılar)
        - Tokenizer beklenmedik inputlarla karşılaştığında hatalı davranmaz
        - Decode işlemi güvenilir olur
    """
    # Aynı config ve aynı training text ile eğitilen iki WordPiece tokenizer aynı tokenization ve encode çıktısını üretmelidir. 
    # Bu test, tokenizer'ın deterministik olduğunu doğrular.
    # Eğer aynı eğitim verisi ve aynı yapılandırma ile eğitilen tokenizer'lar farklı çıktılar üretirse, bu durum tokenizer'ın beklenmedik ve hatalı davranmasına yol açar.

    text = "tokenization token tokenizer tokenization"

    tokenizer_a = TokenizerFactory.create("wordpiece", vocab_size=30)
    tokenizer_b = TokenizerFactory.create("wordpiece", vocab_size=30)

    tokenizer_a.train(text)
    tokenizer_b.train(text)

    assert tokenizer_a.tokenize(text) == tokenizer_b.tokenize(text)
    assert tokenizer_a.encode(text) == tokenizer_b.encode(text)