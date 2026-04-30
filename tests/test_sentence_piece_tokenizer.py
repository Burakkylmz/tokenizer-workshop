from __future__ import annotations

import pytest

sentencepiece = pytest.importorskip("sentencepiece")

from tokenizer_workshop.api.services.tokenizer_factory import TokenizerFactory


# ---------------------------------------------------------
# INIT TESTS
# ---------------------------------------------------------

def test_sentencepiece_tokenizer_init_invalid_vocab_size_raises_error() -> None:
    """
    Guardrail: vocab_size >= 2 olmalıdır.

    Neden?
        SentencePiece modelinde:
            - 1 adet UNK token (zorunlu)
            - en az 1 adet gerçek token
        bulunmalıdır.

    Bu test şunu garanti eder:
        - yanlış konfigürasyonlar erken fail olur
        - model unusable state’e girmez
        - kullanıcıya anlamlı hata mesajları sağlanır

    Önlenen bug:
        - tek tokenlı vocab (sadece UNK)
        - encode sırasında anlamsız çıktı
        - runtime crash (trainer içinde)
        - silent failure (model eğitilir ama unusable olur)
    """
    # vocab_size 1 geçersizdir çünkü sadece UNK token'ı olurdu, gerçek token olmazdı.
    # vocab_size 0 geçersizdir çünkü hiçbir token olmazdı, ne UNK ne gerçek token.
    # vocab_size 2 geçerlidir çünkü 1 UNK + 1 gerçek token olurdu, minimum viable vocab olurdu.
    
    # Bu test, TokenizerFactory.create()'un SentencePiece tokenizer'ı oluştururken geçersiz vocab_size değerlerine karşı koruma sağladığını doğrular.
    
    with pytest.raises(ValueError, match="vocab_size must be at least 2"):
        TokenizerFactory.create("sentencepiece", vocab_size=1)


def test_sentencepiece_tokenizer_init_invalid_model_type_raises_error() -> None:
    """
    model_type yalnızca SentencePiece tarafından desteklenen değerlerden biri olmalıdır.

    Bu test şunu garanti eder:
        - API misuse engellenir
        - yanlış algoritma seçimi erken fail olur
        - kullanıcıya anlamlı hata mesajları sağlanır

    Önlenen bug:
        - silent fallback
        - runtime crash (trainer içinde)
        - anlamsız tokenization sonuçları
        - unusable model state
        - silent failure (model eğitilir ama beklenmedik şekilde davranır)
        - yanlış model türüyle eğitim yapılır ve sonuçlar beklenmedik olur
    """
    # SentencePiece tarafından desteklenen model_type değerleri:
    # - unigram
    # - bpe
    # - char
    # - word

    # Bu test, TokenizerFactory.create()'un SentencePiece tokenizer'ı oluştururken geçersiz model_type değerlerine karşı koruma sağladığını doğrular.
    
    with pytest.raises(ValueError, match="model_type must be one of"):
        TokenizerFactory.create("sentencepiece", model_type="invalid")


def test_sentencepiece_tokenizer_init_invalid_character_coverage_raises_error() -> None:
    """
    character_coverage 0 ile 1 arasında olmalıdır.

    0 geçersizdir -> hiçbir karakter kapsanmaz.
    1.0 geçerlidir -> tüm karakterlerin kapsanması hedeflenir.

    Neden?
        coverage → modelin hangi karakterleri kapsayacağını belirler

    Bu test şunu garanti eder:
        - geçersiz coverage değerleri reddedilir
        - model training stabil kalır
        - kullanıcıya anlamlı hata mesajları sağlanır

    Önlenen bug:
        - coverage 0 ile 1 arasında değilse model unusable olurdu
        - silent failure (model eğitilir ama unusable olurdu)
        - anlamsız tokenization sonuçları (hiçbir karakter kapsanmazsa)
        - runtime crash (trainer içinde)
        - kullanıcı yanlış coverage değeriyle eğitim yapar ve sonuçlar beklenmedik olur
        - model eğitilir ama beklenmedik şekilde davranır (örneğin, hiç token üretmez veya çok fazla token üretir)
    """
    # Bu test, TokenizerFactory.create()'un SentencePiece tokenizer'ı oluştururken geçersiz character_coverage değerlerine karşı koruma sağladığını doğrular.

    with pytest.raises(ValueError, match="character_coverage must be between 0 and 1"):
        TokenizerFactory.create("sentencepiece", character_coverage=0)


# ---------------------------------------------------------
# TRAIN TESTS
# ---------------------------------------------------------

def test_sentencepiece_tokenizer_train_with_empty_text_raises_error() -> None:
    """
    Training invariant: boş metinle model eğitilmemelidir.

    Neden?
        SentencePiece frekans tabanlı çalışır
        boş input → anlamlı istatistik yok

    Bu test şunu garanti eder:
        - geçersiz training input reddedilir
        - model unusable state’e girmez
        - kullanıcıya anlamlı hata mesajları sağlanır
        - tokenizer'ın eğitim için geçerli olmayan inputlara karşı koruma sağladığını doğrular

    Önlenen bug:
        - boş model
        - undefined behavior
        - runtime crash (trainer içinde)
        - silent failure (model eğitilir ama unusable olur)
        - kullanıcı boş metinle eğitim yapar ve sonuçlar beklenmedik olur
        - model eğitilir ama beklenmedik şekilde davranır (örneğin, hiç token üretmez veya anlamsız tokenlar üretir)
    """
    # Bu test, SentencePieceTokenizer'ın train() metodunun boş metinle çağrıldığında ValueError ile hata verdiğini doğrular.

    tokenizer = TokenizerFactory.create("sentencepiece", vocab_size=30)

    with pytest.raises(ValueError, match="Training text cannot be empty"):
        tokenizer.train("")


def test_sentencepiece_tokenizer_train_with_whitespace_text_raises_error() -> None:
    """
    Sadece whitespace içeren metin eğitim için geçerli değildir.
    
    Neden?
        - whitespace-only text → meaningful tokens yok
        - SentencePiece frekans tabanlı çalışır
    
    Bu test şunu garanti eder:
        - geçersiz training input reddedilir
        - model unusable state’e girmez
        - kullanıcıya anlamlı hata mesajları sağlanır
    
    Önlenen bug:
        - boş model
        - undefined behavior
        - runtime crash (trainer içinde)
        - silent failure (model eğitilir ama unusable olur)
        - kullanıcı sadece whitespace içeren metinle eğitim yapar ve sonuçlar beklenmedik olur
        - model eğitilir ama beklenmedik şekilde davranır (örneğin, hiç token üretmez veya anlamsız tokenlar üretir)
    """
    # Bu test, SentencePieceTokenizer'ın train() metodunun sadece whitespace içeren metinle çağrıldığında ValueError ile hata verdiğini doğrular.

    tokenizer = TokenizerFactory.create("sentencepiece", vocab_size=30)

    with pytest.raises(ValueError, match="Training text cannot be empty"):
        tokenizer.train("   ")


def test_sentencepiece_tokenizer_train_builds_vocab() -> None:
    """
    train() sonrası vocabulary oluşmalıdır.

    Neden?
        - SentencePiece frekans tabanlı çalışır
        - eğitim sonrası meaningful tokens oluşur

    Bu test şunu garanti eder:
        - SentencePieceTrainer başarılı çalışır
        - model load edilir
        - processor usable hale gelir
        - tokenizer'ın eğitim sonrası geçerli bir vocabulary oluşturduğunu doğrular

    Önlenen bug:
        - boş model
        - undefined behavior
        - runtime crash (trainer içinde)
        - silent failure (model eğitilir ama unusable olur)
        - kullanıcı eğitim yapar ama model unusable olur
        - model eğitilir ama beklenmedik şekilde davranır (örneğin, hiç token üretmez veya anlamsız tokenlar üretir)
        - vocab_size beklenenden farklı olur (örneğin, eğitim başarısız olur ama trainer içinde hata yakalanmaz ve model boş kalır)
    """
    # Bu test, SentencePieceTokenizer'ın train() metodunun başarılı bir şekilde çalıştığında vocab_size'ın 0'dan büyük olduğunu doğrular.

    tokenizer = TokenizerFactory.create("sentencepiece", vocab_size=30)

    tokenizer.train("merhaba dünya tokenizer sentencepiece test")

    # vocab_size > 0 → model loaded
    assert tokenizer.vocab_size > 0

    # üst sınır kontrolü
    assert tokenizer.vocab_size <= 30


def test_sentencepiece_tokenizer_vocab_size_zero_before_training() -> None:
    """
    Eğitim öncesi vocab_size = 0 olmalıdır.

    Neden?
        - SentencePiece eğitilmeden önce meaningful tokens olmaz
        - vocab_size 0 → model eğitilmemiş ve processor yüklenmemiş demektir
        - eğitim öncesi state açık ve anlaşılır olur

    Bu test şunu garanti eder:
        - tokenizer state açık ve anlaşılır
        - "trained vs untrained" ayrımı net
        - kullanıcı eğitim yapmadan önce vocab_size'ı kontrol edebilir ve modelin eğitilmemiş olduğunu anlayabilir
        - tokenizer'ın eğitim öncesi state'inin doğru şekilde yansıtıldığını doğrular

    Önlenen bug:
        - yanlış state algısı
        - kullanıcı eğitim yapmadan tokenize/encode/decode yapmaya çalışır ve beklenmedik sonuçlarla karşılaşır
        - model unusable state’e girer (örneğin, encode/decode çalışır ama anlamsız sonuçlar üretir)
        - silent failure (model eğitilmemiş ama encode/decode çalışır ve anlamsız sonuçlar üretir)
        - kullanıcı eğitim yapmadan önce vocab_size'ı kontrol eder ve yanlış state algısı nedeniyle eğitim yapmadan tokenize/encode/decode yapmaya çalışır ve beklenmedik sonuçlarla karşılaşır
    """
    # Bu test, SentencePieceTokenizer'ın train() metodunun çağrılmadan önce vocab_size'ın 0 olduğunu doğrular.

    tokenizer = TokenizerFactory.create("sentencepiece", vocab_size=30)

    assert tokenizer.vocab_size == 0 # eğitim öncesi vocab_size 0 olmalı


# ---------------------------------------------------------
# TOKENIZE TESTS
# ---------------------------------------------------------

def test_sentencepiece_tokenizer_tokenize_before_training_raises_error() -> None:
    """
    SentencePiece tokenize işlemi eğitimden önce çalışmamalıdır.

    Çünkü SentencePieceProcessor encode işlemi için yüklenmiş model ister.

    Neden?
        SentencePieceProcessor encode için model ister
        eğitim öncesi model yok → tokenize işlemi yapılamaz

    Bu test şunu garanti eder:
        - API misuse engellenir
        - kullanıcı eğitim yapmadan tokenize yapmaya çalışırsa erken fail olur
        - kullanıcıya anlamlı hata mesajları sağlanır
        - model unusable state’e girmez (örneğin, tokenize çalışmaz ama encode/decode çalışır gibi tutarsız durumlar olmaz)
        - tokenizer'ın eğitim yapılmadan tokenize() metodunun çağrıldığında ValueError ile hata verdiğini doğrular

    Önlenen bug:
        - kullanıcı eğitim yapmadan tokenize yapmaya çalışır ve beklenmedik sonuçlarla karşılaşır
        - model unusable state’e girer (örneğin, tokenize çalışmaz ama encode/decode çalışır gibi tutarsız durumlar olur)
        - silent failure (model eğitilmemiş ama tokenize çalışır ve anlamsız sonuçlar üretir)
        - kullanıcı eğitim yapmadan tokenize yapmaya çalışır ve beklenmedik sonuçlarla karşılaşır
        - model eğitilmemiş ama tokenize çalışır ve anlamsız sonuçlar üretir
    """
    # Bu test, SentencePieceTokenizer'ın tokenize() metodunun train() çağrılmadan önce çağrıldığında ValueError ile hata verdiğini doğrular.

    tokenizer = TokenizerFactory.create("sentencepiece", vocab_size=30)

    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.tokenize("merhaba dünya")


def test_sentencepiece_tokenizer_tokenize_empty_text_returns_empty_list() -> None:
    """
    Boş input tokenize edildiğinde boş liste dönmelidir.

    Neden?
        - boş metin → meaningful tokens yok
        - tokenize("") → []

    Bu test şunu garanti eder:
        - tokenize() boş metinle çağrıldığında boş liste döner
        - kullanıcı boş metinle tokenize yapmaya çalışırsa beklenen sonuçla karşılaşır
        - model unusable state’e girmez (örneğin, tokenize("") çalışır ve boş liste döner, encode/decode çalışır gibi tutarsız durumlar olmaz)
        - tokenizer'ın tokenize() metodunun boş metinle çağrıldığında boş liste döndürdüğünü doğrular

    Onlenen bug:
        - tokenize("") anlamsız sonuçlar üretir (örneğin, [UNK] token'ı üretir)
        - model unusable state’e girer (örneğin, tokenize("") çalışmaz ama encode/decode çalışır gibi tutarsız durumlar olur)
        - silent failure (model eğitilmiş ama tokenize("") anlamsız sonuçlar üretir)
        - kullanıcı tokenize("") yapar ve anlamsız sonuçlarla karşılaşır
        - model eğitilmiş ama tokenize("") anlamsız sonuçlar üretir    
    """
    # Bu test, SentencePieceTokenizer'ın tokenize() metodunun boş metinle çağrıldığında boş liste döndürdüğünü doğrular.

    tokenizer = TokenizerFactory.create("sentencepiece", vocab_size=30)
    tokenizer.train("merhaba dünya tokenizer test")

    assert tokenizer.tokenize("") == [] # tokenize("") → []


def test_sentencepiece_tokenizer_tokenize_returns_string_tokens() -> None:
    """
    tokenize() string token listesi döndürmelidir.

    Neden?
        - tokenize() → string tokenlar
        - model unusable state’e girmez (örneğin, tokenize() çalışır ve string tokenlar döner, encode/decode çalışır gibi tutarsız durumlar olmaz)
        - tokenizer'ın tokenize() metodunun string tokenlar döndürdüğünü doğrular

    Bu test şunu garanti eder:
        - API contract doğru
        - downstream işlemler (UI/report) düzgün çalışır
        - kullanıcı tokenize()'un string tokenlar döndürmesini bekler ve bu test bunu doğrular

    Önlenen bug:
        - tokenize() integer token id'leri döndürür (örneğin, encode()un token id'lerini döndürür) → API contract ihlali
        - model unusable state’e girer (örneğin, tokenize() çalışmaz ama encode/decode çalışır gibi tutarsız durumlar olur)
        - silent failure (model eğitilmiş ama tokenize() yanlış türde sonuçlar üretir)
        - kullanıcı tokenize() yapar ve beklenmedik türde sonuçlarla karşılaşır
        - model eğitilmiş ama tokenize() yanlış türde sonuçlar üretir
    """
    # Bu test, SentencePieceTokenizer'ın tokenize() metodunun string tokenlar döndürdüğünü doğrular.

    tokenizer = TokenizerFactory.create("sentencepiece", vocab_size=30)

    tokenizer.train("merhaba dünya tokenizer test sentencepiece")
    tokens = tokenizer.tokenize("merhaba dünya")

    assert isinstance(tokens, list) # tokenize() → list
    assert len(tokens) > 0 # tokenize() meaningful tokens üretmeli
    assert all(isinstance(token, str) for token in tokens) # tokenize() string tokenlar döndürmeli


def test_sentencepiece_tokenizer_uses_whitespace_marker() -> None:
    """
    SentencePiece whitespace bilgisini genellikle ▁ marker ile temsil eder.

    Bu test, tokenların içinde SentencePiece boundary marker üretilebildiğini doğrular.

    SentencePiece whitespace bilgisini özel token ile tutar.

    Neden?
        - whitespace kaybı olmaz
        - tokenization sonrası metin reconstruct edilebilir
        - model unusable state’e girmez (örneğin, tokenize() çalışır ve whitespace marker'ları içerir, encode/decode çalışır gibi tutarsız durumlar olmaz)

    Bu test şunu garanti eder:
        - gerçek SentencePiece davranışı kullanılıyor
        - kullanıcı tokenize()'un SentencePiece whitespace marker'larını içerdiğini bekler ve bu test bunu doğrular
        - tokenizer'ın tokenize() metodunun SentencePiece whitespace marker'larını içerdiğini doğrular

    Önlenen bug:
        - tokenize() whitespace marker'larını içermez → whitespace kaybı olur, metin reconstruct edilemez
        - silent failure (model eğitilmiş ama tokenize() whitespace marker'larını içermez)  
        - kullanıcı tokenize() yapar ve whitespace marker'larının eksik olduğunu fark eder
        - model eğitilmiş ama tokenize() whitespace marker'larını içermez

    Kritik nokta:
        "▁" → kelime başlangıcı marker
    """
    # Bu test, SentencePieceTokenizer'ın tokenize() metodunun SentencePiece whitespace marker'larını içerdiğini doğrular.

    tokenizer = TokenizerFactory.create("sentencepiece", vocab_size=30)

    tokenizer.train("merhaba dünya merhaba dünya tokenizer test")
    tokens = tokenizer.tokenize("merhaba dünya")

    assert any("▁" in token for token in tokens) # SentencePiece whitespace marker'ı içeriyor mu?


# ---------------------------------------------------------
# ENCODE TESTS
# ---------------------------------------------------------

def test_sentencepiece_tokenizer_encode_before_training_raises_error() -> None:
    """
    train() çağrılmadan encode() yapılmamalıdır.

    Çünkü SentencePieceProcessor encode işlemi için yüklenmiş model ister.

    Neden?
        SentencePieceProcessor encode için model ister
        eğitim öncesi model yok → encode işlemi yapılamaz

    Bu test şunu garanti eder:
        - API misuse engellenir
        - kullanıcı eğitim yapmadan encode yapmaya çalışırsa erken fail olur
        - kullanıcıya anlamlı hata mesajları sağlanır
        - model unusable state’e girmez (örneğin, encode çalışmaz ama tokenize/decode çalışır gibi tutarsız durumlar olmaz)
        - tokenizer'ın eğitim yapılmadan encode() metodunun çağrıldığında ValueError ile hata verdiğini doğrular
    
    Önlenen bug:
        - silent failure (model eğitilmemiş ama encode() çalışır ve anlamsız sonuçlar üretir)
        - kullanıcı eğitim yapmadan encode yapmaya çalışır ve beklenmedik sonuçlarla karşılaşır
        - model eğitilmemiş ama encode() çalışır ve anlamsız sonuçlar üretir
        - kullanıcı eğitim yapmadan önce vocab_size'ı kontrol eder ve yanlış state algısı nedeniyle eğitim yapmadan encode yapmaya çalışır ve beklenmedik sonuçlarla karşılaşır
        - model eğitilmemiş ama encode() çalışır ve anlamsız sonuçlar üretir
    """
    # Bu test, SentencePieceTokenizer'ın encode() metodunun train() çağrılmadan önce çağrıldığında ValueError ile hata verdiğini doğrular.

    tokenizer = TokenizerFactory.create("sentencepiece", vocab_size=30)

    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.encode("merhaba")


def test_sentencepiece_tokenizer_encode_empty_text_returns_empty_list() -> None:
    """
    Boş input encode edildiğinde boş liste dönmelidir.

    Neden?
        - boş metin → meaningful tokens yok
        - encode("") → []

    Bu test şunu garanti eder:
        - encode() boş metinle çağrıldığında boş liste döner
        - kullanıcı boş metinle encode yapmaya çalışırsa beklenen sonuçla karşılaşır
        - model unusable state’e girmez (örneğin, encode("") çalışır ve boş liste döner, tokenize/decode çalışır gibi tutarsız durumlar olmaz)
        - tokenizer'ın encode() metodunun boş metinle çağrıldığında boş liste döndürdüğünü doğrular
    
    Önlenen bug:
        - encode("") anlamsız sonuçlar üretir (örneğin, [UNK] token'ı üretir)
        - silent failure (model eğitilmiş ama encode("") anlamsız sonuçlar üretir)
        - kullanıcı encode("") yapar ve anlamsız sonuçlarla karşılaşır
        - model eğitilmiş ama encode("") anlamsız sonuçlar üretir
        - kullanıcı eğitim yapmadan önce vocab_size'ı kontrol eder ve yanlış state algısı nedeniyle eğitim yapmadan encode yapmaya çalışır ve beklenmedik sonuçlarla karşılaşır
        - model eğitilmemiş ama encode() çalışır ve anlamsız sonuçlar üretir
    """
    # Bu test, SentencePieceTokenizer'ın encode() metodunun boş metinle çağrıldığında boş liste döndürdüğünü doğrular.

    tokenizer = TokenizerFactory.create("sentencepiece", vocab_size=30)
    tokenizer.train("merhaba dünya tokenizer test")

    assert tokenizer.encode("") == [] # encode("") → []


def test_sentencepiece_tokenizer_encode_returns_integer_ids() -> None:
    """
    encode() integer ID listesi döndürmelidir.

    Neden?
        - encode() → token ID'leri
        - kullanıcı encode()'un token ID'leri döndürmesini bekler

    Bu test şunu garanti eder:
        - model input formatı doğru
        - inference pipeline uyumlu
        - kullanıcı encode()'un token ID'leri döndürmesini bekler ve bu test bunu doğrular
        - model unusable state’e girmez (örneğin, encode() çalışır ve token ID'leri döner, tokenize/decode çalışır gibi tutarsız durumlar olmaz)
        - tokenizer'ın encode() metodunun token ID'leri döndürdüğünü doğrular
        - gerçek SentencePiece davranışı kullanılıyor
        - SentencePiece string → token → id mapping'i internal olarak yapar, bu test encode()'un token ID'leri döndürdüğünü doğrular
        - kullanıcı encode()'un token ID'leri döndürmesini bekler ve bu test bunu doğrular
    
     Önlenen bug:
        - encode() string tokenlar döndürür (örneğin, tokenize()un tokenlarını döndürür) → API contract ihlali
        - silent failure (model eğitilmiş ama encode() yanlış türde sonuçlar üretir)
        - model eğitilmiş ama encode() yanlış türde sonuçlar üretir
        - kullanıcı eğitim yapmadan önce vocab_size'ı kontrol eder ve yanlış state algısı nedeniyle eğitim yapmadan encode yapmaya çalışır ve beklenmedik sonuçlarla karşılaşır
        - model eğitilmemiş ama encode() çalışır ve anlamsız sonuçlar üretir
    """
    # Bu test, SentencePieceTokenizer'ın encode() metodunun token ID'leri döndürdüğünü doğrular.

    tokenizer = TokenizerFactory.create("sentencepiece", vocab_size=30)

    tokenizer.train("merhaba dünya tokenizer test sentencepiece")
    encoded = tokenizer.encode("merhaba dünya")

    assert isinstance(encoded, list) # encode() → list
    assert len(encoded) > 0 # encode() meaningful token ID'leri üretmeli
    assert all(isinstance(token_id, int) for token_id in encoded) # encode() integer token ID'leri döndürmeli


def test_sentencepiece_tokenizer_encode_is_deterministic_for_same_input() -> None:
    """
    encode() deterministik olmalıdır.

    Aynı input → aynı output

    Neden?
        - reproducibility
        - caching / testing stabilitesi

    Bu test şunu garanti eder:
        - reproducibility
        - caching / testing stabilitesi
        - kullanıcı encode()'un deterministik olduğunu bekler ve bu test bunu doğrular
        - model unusable state’e girmez (örneğin, encode() çalışır ve aynı input için aynı output döner, tokenize/decode çalışır gibi tutarsız durumlar olmaz)
        - tokenizer'ın encode() metodunun deterministik olduğunu doğrular
        - gerçek SentencePiece davranışı kullanılıyor
        - SentencePiece aynı input için aynı token ID'lerini üretir, bu test encode()'un deterministik olduğunu doğrular
        - kullanıcı encode()'un deterministik olduğunu bekler ve bu test bunu doğrular

    Önlenen bug:
        - encode() non-deterministik sonuçlar üretir (örneğin, aynı input için farklı token ID'leri döndürür) → reproducibility ve caching sorunları
        - silent failure (model eğitilmiş ama encode() non-deterministik sonuçlar üretir)
        - kullanıcı encode() yapar ve aynı input için farklı sonuçlarla karşılaşır
        - model eğitilmiş ama encode() non-deterministik sonuçlar üretir
        - kullanıcı eğitim yapmadan önce vocab_size'ı kontrol eder ve yanlış state algısı nedeniyle eğitim yapmadan encode yapmaya çalışır ve beklenmedik sonuçlarla karşılaşır
        - model eğitilmemiş ama encode() çalışır ve anlamsız sonuçlar üretir
    """
    # Bu test, SentencePieceTokenizer'ın encode() metodunun aynı input için aynı output'u döndürdüğünü doğrular.

    tokenizer = TokenizerFactory.create("sentencepiece", vocab_size=30)

    tokenizer.train("merhaba dünya tokenizer test sentencepiece")
    encoded_a = tokenizer.encode("merhaba dünya")
    encoded_b = tokenizer.encode("merhaba dünya")

    assert encoded_a == encoded_b


# ---------------------------------------------------------
# DECODE TESTS
# ---------------------------------------------------------

def test_sentencepiece_tokenizer_decode_before_training_raises_error() -> None:
    """
    decode() string döndürmelidir.

    Neden?
        - decode() → string
        - kullanıcı decode()'un string döndürmesini bekler
        - tokenizer'ın decode() metodunun string döndürdüğünü doğrular

    Bu test şunu garanti eder:
        - encode/decode pipeline çalışıyor
        - output tipi doğru
    
    onlenen bug:
        - decode() string yerine token ID'leri döndürür (örneğin, encode()un token ID'lerini döndürür) → API contract ihlali
        - model unusable state’e girer (örneğin, decode() çalışmaz ama tokenize/encode çalışır gibi tutarsız durumlar olur)
        - silent failure (model eğitilmiş ama decode() yanlış türde sonuçlar üretir)
        - kullanıcı decode() yapar ve beklenmedik türde sonuçlarla karşılaşır
        - model eğitilmiş ama decode() yanlış türde sonuçlar üretir
        - kullanıcı eğitim yapmadan önce vocab_size'ı kontrol eder ve yanlış state algısı nedeniyle eğitim yapmadan decode yapmaya çalışır ve beklenmedik sonuçlarla karşılaşır
        - model eğitilmemiş ama decode() çalışır ve anlamsız sonuçlar üretir
    """
    tokenizer = TokenizerFactory.create("sentencepiece", vocab_size=30)

    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.decode([0, 1])


def test_sentencepiece_tokenizer_decode_returns_string() -> None:
    """
    decode() id listesini string çıktıya dönüştürmelidir.

    Neden?
        - decode() → string
        - kullanıcı decode()'un string döndürmesini bekler
        - model unusable state’e girmez (örneğin, decode() çalışır ve string döner, tokenize/encode çalışır gibi tutarsız durumlar olmaz)
        - tokenizer'ın decode() metodunun string döndürdüğünü doğrular
        - gerçek SentencePiece davranışı kullanılıyor
        - SentencePiece token ID'lerini string'e dönüştürür, bu test decode()un string döndürdüğünü doğrular
    
    Bu test şunu garanti eder:
        - encode/decode pipeline çalışıyor
        - output tipi doğru
        - kullanıcı decode()'un string döndürmesini bekler ve bu test bunu doğrular
        - model unusable state’e girmez (örneğin, decode() çalışır ve string döner, tokenize/encode çalışır gibi tutarsız durumlar olmaz)

    Önlenen bug:
        - decode() string yerine token ID'leri döndürür (örneğin, encode()un token ID'lerini döndürür) → API contract ihlali
        - silent failure (model eğitilmiş ama decode() yanlış türde sonuçlar üretir)
        - kullanıcı decode() yapar ve beklenmedik türde sonuçlarla karşılaşır
        - model eğitilmiş ama decode() yanlış türde sonuçlar üretir
        - kullanıcı eğitim yapmadan önce vocab_size'ı kontrol eder ve yanlış state algısı nedeniyle eğitim yapmadan decode yapmaya çalışır ve beklenmedik sonuçlarla karşılaşır
        - model eğitilmemiş ama decode() çalışır ve anlamsız sonuçlar üretir
    """
    # Bu test, SentencePieceTokenizer'ın decode() metodunun token ID'lerini string'e dönüştürdüğünü doğrular.

    tokenizer = TokenizerFactory.create("sentencepiece", vocab_size=30)

    tokenizer.train("merhaba dünya tokenizer test sentencepiece")

    encoded = tokenizer.encode("merhaba dünya")
    decoded = tokenizer.decode(encoded)

    assert isinstance(decoded, str) # decode() → string
    assert len(decoded) > 0 # decode() meaningful string döndürmeli


def test_sentencepiece_tokenizer_encode_decode_roundtrip_preserves_text() -> None:
    """
    SentencePiece en önemli avantaj:
        encode → decode → aynı text

    Bu test şunu garanti eder:
        - whitespace korunur
        - gerçek SentencePiece davranışı var
        - tokenizer'ın encode-decode roundtrip'inin input text'i koruduğunu doğrular
        - kullanıcı encode-decode roundtrip'inin input text'i korumasını bekler ve bu test bunu doğrular
        - model unusable state’e girmez (örneğin, encode() çalışır ve decode() çalışır ve aynı text'i döner, tokenize() çalışır gibi tutarsız durumlar olmaz)
        - gerçek SentencePiece davranışı kullanılıyor
        - SentencePiece encode-decode roundtrip'inde input text'i korur, bu test tokenizer'ın encode-decode roundtrip'inin input text'i koruduğunu doğrular
        - kullanıcı encode-decode roundtrip'inin input text'i korumasını bekler ve bu test bunu doğrular
    
    Önlenen bug:
        - encode-decode roundtrip input text'i değiştirebilir → kullanıcı beklenmedik sonuçlarla karşılaşır
        - silent failure (model eğitilmiş ama encode-decode roundtrip input text'i değiştirir)
        - kullanıcı encode-decode roundtrip yapar ve input text'in değiştiğini fark eder
        - model eğitilmiş ama encode-decode roundtrip input text'i değiştirir
        - kullanıcı eğitim yapmadan önce vocab_size'ı kontrol eder ve yanlış state algısı nedeniyle eğitim yapmadan encode-decode roundtrip yapmaya çalışır ve beklenmedik sonuçlarla karşılaşır
        - model eğitilmemiş ama encode() çalışır ve anlamsız sonuçlar üretir
    """
    tokenizer = TokenizerFactory.create("sentencepiece", vocab_size=40)

    text = "merhaba dünya"
    tokenizer.train("merhaba dünya merhaba dünya tokenizer sentencepiece test")

    decoded = tokenizer.decode(tokenizer.encode(text))

    assert decoded == text # encode-decode roundtrip input text'i korumalı


def test_sentencepiece_tokenizer_handles_turkish_characters() -> None:
    """
    Türkçe karakterler encode/decode akışından geçebilmelidir.

    Neden?
        - SentencePiece Unicode karakterleri destekler
        - Türkçe karakterler Unicode içinde yer alır
        - Türkçe karakterler kaybolmaz veya bozulmaz
        - model unusable state’e girmez (örneğin, encode() çalışır ve decode() çalışır ve Türkçe karakterler doğru şekilde korunur, tokenize() çalışır gibi tutarsız durumlar olmaz)
        - gerçek SentencePiece davranışı kullanılıyor
        - SentencePiece Unicode karakterleri destekler, bu test Türkçe karakterlerin encode-decode roundtrip'inde korunup korunmadığını doğrular
        
    Bu test şunu garanti eder:
        - kullanıcı encode-decode roundtrip'inin Türkçe karakterleri korumasını bekler ve bu test bunu doğrular
        
    Önlenen bug:
        - encode-decode roundtrip Türkçe karakterleri değiştirebilir → kullanıcı beklenmedik sonuçlarla karşılaşır
        - silent failure (model eğitilmiş ama encode-decode roundtrip Türkçe karakterleri değiştirir)
        - kullanıcı encode-decode roundtrip yapar ve Türkçe karakterlerin değiştiğini fark eder
        - model eğitilmiş ama encode-decode roundtrip Türkçe karakterleri değiştirir
        - kullanıcı eğitim yapmadan önce vocab_size'ı kontrol eder ve yanlış state algısı nedeniyle eğitim yapmadan encode-decode roundtrip yapmaya çalışır ve beklenmedik sonuçlarla karşılaşır
        - model eğitilmemiş ama encode() çalışır ve anlamsız sonuçlar üretir
    """
    tokenizer = TokenizerFactory.create("sentencepiece", vocab_size=40)

    text = "çalışma öğrenme dünya"
    tokenizer.train("çalışma öğrenme dünya tokenizer sentencepiece testi")

    decoded = tokenizer.decode(tokenizer.encode(text))

    assert decoded == text # encode-decode roundtrip Türkçe karakterleri korumalı


# ---------------------------------------------------------
# MODEL TYPE TESTS
# ---------------------------------------------------------

@pytest.mark.parametrize("model_type", ["unigram", "bpe", "char", "word"])
def test_sentencepiece_tokenizer_supports_valid_model_types(model_type: str) -> None:
    """
    Tüm model tipleri çalışmalıdır.

    Bu test şunu garanti eder:
        - API flexibility
        - farklı tokenizer türleri destekleniyor

    Önlenen bug:
        - sadece default model çalışması
        - kullanıcı farklı model tipiyle eğitim yapmaya çalışır ve beklenmedik sonuçlarla karşılaşır
        - model unusable state’e girer (örneğin, eğitim çalışır ama encode/decode çalışmaz gibi tutarsız durumlar olur)
        - silent failure (model eğitilmiş ama farklı model tipi beklenmedik sonuçlar üretir)
        - kullanıcı farklı model tipiyle eğitim yapar ve beklenmedik sonuçlarla karşılaşır
        - model eğitilmiş ama farklı model tipi beklenmedik sonuçlar üretir
    """
    tokenizer = TokenizerFactory.create(
        "sentencepiece",
        vocab_size=40,
        model_type=model_type,
    )

    tokenizer.train("merhaba dünya tokenizer sentencepiece test metni")

    encoded = tokenizer.encode("merhaba dünya")
    decoded = tokenizer.decode(encoded)

    assert isinstance(encoded, list) # encode() → list
    assert len(encoded) > 0 # encode() meaningful token ID'leri üretmeli
    assert isinstance(decoded, str) # decode() → string
    assert len(decoded) > 0 # decode() meaningful string döndürmeli