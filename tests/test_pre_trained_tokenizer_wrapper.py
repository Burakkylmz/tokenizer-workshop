from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from tokenizer_workshop.api.services.tokenizer_factory import TokenizerFactory
from tokenizer_workshop.tokenizers.pre_trained_tokenizer_wrapper import (
    PreTrainedTokenizerWrapper,
)


# =========================================================
# MOCK BASED TESTS
# =========================================================
#
# Bu testler gerçek bir Hugging Face modeli indirmez.
# Bunun yerine AutoTokenizer.from_pretrained çağrısı patch'lenir
# ve davranışı kontrollü bir MagicMock döndürür.
#
# Bu yaklaşım tercih edildi çünkü:
#     - Network bağımlılığı yoktur (CI'da flaky olmaz)
#     - Test hızlıdır (model indirmesi yok)
#     - Wrapper'ın kendi mantığı izole olarak doğrulanır
#
# Gerçek pretrained tokenizer davranışı `INTEGRATION TESTS` bölümünde
# `pytest.mark.integration` ile işaretlenmiş ayrı testlerde test edilir.
# =========================================================


# ---------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------

@pytest.fixture
def mock_hf_tokenizer() -> MagicMock:
    """
    Hugging Face tokenizer davranışını simüle eden bir MagicMock üretir.

    Bu mock şu metod ve property'leri sağlar:
        - tokenize(text)           -> list[str]
        - encode(text, **kwargs)   -> list[int]
        - decode(ids, **kwargs)    -> str
        - convert_ids_to_tokens()  -> list[str]
        - convert_tokens_to_ids()  -> list[int]
        - vocab_size               -> int
        - special_tokens_map       -> dict

    Mock üzerinde basit ama tutarlı bir tokenizer simülasyonu yapılır:
    her boşlukla ayrılmış kelime bir token olarak kabul edilir.

    Returns:
        Hugging Face tokenizer benzeri davranan MagicMock.
    """
    mock = MagicMock()

    # tokenize() -> string token listesi
    mock.tokenize.side_effect = lambda text: text.split()

    # encode() -> integer token id listesi (basit bir hash benzeri)
    mock.encode.side_effect = lambda text, **kwargs: [
        hash(token) % 30000 for token in text.split()
    ]

    # decode() -> id listesinden string'e dönüş; mock olduğu için sabit string döner
    mock.decode.side_effect = lambda ids, **kwargs: " ".join(
        f"tok{i}" for i in ids
    )

    # convert_ids_to_tokens / convert_tokens_to_ids
    mock.convert_ids_to_tokens.side_effect = lambda ids: [f"tok{i}" for i in ids]
    mock.convert_tokens_to_ids.side_effect = lambda tokens: [
        hash(t) % 30000 for t in tokens
    ]

    # vocab_size ve special_tokens_map
    mock.vocab_size = 30522  # BERT-base-uncased varsayılan değeri
    mock.special_tokens_map = {
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
    }

    return mock


@pytest.fixture
def wrapper(mock_hf_tokenizer: MagicMock) -> PreTrainedTokenizerWrapper:
    """
    AutoTokenizer.from_pretrained patch'lenmiş bir wrapper instance üretir.

    Wrapper, mock_hf_tokenizer fixture'ı ile kurulur. Böylece her test
    aynı kontrollü tokenizer davranışına sahip olur ve testler birbirinden
    izole çalışır.
    """
    with patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=mock_hf_tokenizer,
    ):
        return PreTrainedTokenizerWrapper(model_name="bert-base-uncased")


# ---------------------------------------------------------
# INIT TESTS
# ---------------------------------------------------------

def test_pre_trained_tokenizer_init_raises_error_for_empty_model_name() -> None:
    """
    model_name boş string verildiğinde ValueError fırlatıldığını test eder.

    Çünkü:
        Boş bir model adıyla Hugging Face Hub'dan tokenizer yüklenemez.
        Bu, kullanıcının yanlış konfigürasyon yaptığının işaretidir.

    Tokenizer bu durumu erken yakalar; AutoTokenizer çağrısına dahi gitmez.
    Böylece kullanıcı anlamlı bir hata mesajı alır:
        - "model_name cannot be empty"
    yerine Hugging Face'in ham hata mesajını görmek zorunda kalmaz.

    Beklenen:
        ValueError("model_name cannot be empty")
    """
    with pytest.raises(ValueError, match="model_name cannot be empty"):
        PreTrainedTokenizerWrapper(model_name="")


def test_pre_trained_tokenizer_init_raises_error_for_whitespace_only_model_name() -> None:
    """
    Sadece whitespace içeren model_name'in de boş kabul edildiğini test eder.

    Input örnekleri:
        "   "       -> sadece boşluklar
        "\\t\\n"      -> tab + newline

    Çünkü:
        Whitespace-only bir model adı pratik olarak boş bir adla aynıdır.
        empty kontrolünün sadece string uzunluğuna değil, anlamlı içeriğe
        bakması daha defansif bir yaklaşımdır.

    Beklenen:
        ValueError("model_name cannot be empty")
    """
    with pytest.raises(ValueError, match="model_name cannot be empty"):
        PreTrainedTokenizerWrapper(model_name="   \t\n  ")


def test_pre_trained_tokenizer_init_raises_import_error_when_transformers_missing() -> None:
    """
    transformers paketi kurulu değilse anlamlı bir ImportError fırlatıldığını
    test eder.

    Çünkü:
        Wrapper transformers paketine dinamik olarak (lazy import) bağımlıdır.
        Paket yoksa kullanıcıya teknik bir ModuleNotFoundError yerine
        açıklayıcı bir mesaj gösterilmelidir:
            "PreTrainedTokenizerWrapper requires the 'transformers' package."

    Test stratejisi:
        sys.modules üzerinde 'transformers' modülünü None'a çevirip Python'un
        import sistemini ImportError fırlatmaya zorlarız. Bu yaklaşım gerçek
        "paket kurulu değil" senaryosunu birebir simüle eder.
    """
    # transformers'ı geçici olarak import edilemez hale getir
    with patch.dict(sys.modules, {"transformers": None}):
        with pytest.raises(ImportError, match="requires the 'transformers' package"):
            PreTrainedTokenizerWrapper(model_name="bert-base-uncased")


def test_pre_trained_tokenizer_init_wraps_hf_load_failure_in_runtime_error(
    mock_hf_tokenizer: MagicMock,
) -> None:
    """
    Hugging Face tokenizer yüklenmesi başarısız olduğunda RuntimeError
    fırlatıldığını test eder.

    Senaryo:
        AutoTokenizer.from_pretrained beklenmedik bir Exception fırlatır.
        Wrapper bunu yakalayıp "Failed to load pretrained tokenizer: ..."
        mesajıyla RuntimeError'a sarmalı.

    Çünkü:
        Hugging Face çok çeşitli exception tipleri fırlatabilir:
            - OSError (model bulunamadı)
            - HTTPError (Hub bağlantı hatası)
            - JSONDecodeError (config bozuk)
            - vb.

    Bunların hepsini tek bir RuntimeError altında toplamak:
        - Public API'yi sade tutar
        - Kullanıcıya tutarlı bir hata yüzeyi sunar
        - Üst katmanlardaki hata yakalama kodunu basitleştirir
    """
    with patch(
        "transformers.AutoTokenizer.from_pretrained",
        side_effect=Exception("Hub connection failed"),
    ):
        with pytest.raises(RuntimeError, match="Failed to load pretrained tokenizer"):
            PreTrainedTokenizerWrapper(model_name="bert-base-uncased")


def test_pre_trained_tokenizer_init_passes_use_fast_to_hf(
    mock_hf_tokenizer: MagicMock,
) -> None:
    """
    use_fast parametresinin AutoTokenizer.from_pretrained'e doğru iletildiğini
    test eder.

    Çünkü:
        use_fast Hugging Face tarafında Rust-based tokenizer'ın kullanılıp
        kullanılmayacağını belirler. Wrapper bu parametreyi kendi kontratının
        bir parçası olarak alır ve aşağıya geçirmelidir.

    Eğer parametre kaybolursa:
        - Wrapper API'si vaatte bulunur ama uygulamaz
        - Kullanıcı use_fast=True der, slow tokenizer çalışır
        - Sessiz bir bug oluşur

    Test stratejisi:
        Mock'un from_pretrained çağrısı izlenir ve `call_args` üzerinden
        gönderilen kwargs doğrulanır.
    """
    with patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=mock_hf_tokenizer,
    ) as mock_from_pretrained:
        PreTrainedTokenizerWrapper(model_name="bert-base-uncased", use_fast=False)

    # Çağrı tek seferlik olmalı ve use_fast=False parametresi geçirilmiş olmalı
    mock_from_pretrained.assert_called_once()
    _, kwargs = mock_from_pretrained.call_args
    assert kwargs.get("use_fast") is False


def test_pre_trained_tokenizer_init_forwards_extra_kwargs_to_hf(
    mock_hf_tokenizer: MagicMock,
) -> None:
    """
    **tokenizer_kwargs ile verilen ek parametrelerin AutoTokenizer'a
    geçirildiğini test eder.

    Çünkü:
        Hugging Face from_pretrained() pek çok parametre alır:
            - cache_dir
            - revision
            - trust_remote_code
            - vb.

    Wrapper bu parametreleri kendisinin bilmesine gerek kalmadan,
    **tokenizer_kwargs ile şeffaf biçimde aşağıya iletmelidir.

    Bu şeffaflık önemli çünkü:
        - Wrapper'ı her yeni HF parametresi için güncellemek gerekmez
        - Kullanıcı HF'in tüm zenginliğine erişebilir
        - Wrapper soyutlama yerine adaptör olarak kalır
    """
    with patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=mock_hf_tokenizer,
    ) as mock_from_pretrained:
        PreTrainedTokenizerWrapper(
            model_name="bert-base-uncased",
            cache_dir="/tmp/cache",
            revision="main",
        )

    _, kwargs = mock_from_pretrained.call_args
    assert kwargs.get("cache_dir") == "/tmp/cache"
    assert kwargs.get("revision") == "main"


def test_pre_trained_tokenizer_init_stores_constructor_arguments(
    wrapper: PreTrainedTokenizerWrapper,
) -> None:
    """
    Constructor parametrelerinin instance üzerinde saklandığını test eder.

    Çünkü:
        Bu attribute'lar daha sonra:
            - encode() içinde add_special_tokens kararı için
            - backend_tokenizer_name property'si için
            - debug ve raporlama amacıyla
        kullanılır.

    Eğer kaydedilmezlerse downstream davranış kırılır.
    """
    assert wrapper.model_name == "bert-base-uncased"
    assert wrapper.use_fast is True
    assert wrapper.add_special_tokens is False


# ---------------------------------------------------------
# TRAIN TESTS
# ---------------------------------------------------------

def test_pre_trained_tokenizer_train_is_a_noop(
    wrapper: PreTrainedTokenizerWrapper,
) -> None:
    """
    train() metodunun no-op olduğunu test eder.

    Çünkü:
        Pretrained tokenizer'lar ZATEN eğitilmiş gelir. Vocabulary ve
        normalization kuralları model dosyasında saklıdır.

    Wrapper'ın train() metodu sadece BaseTokenizer kontratına uyum sağlamak
    için vardır. Hiçbir şey yapmamalı, hata fırlatmamalıdır.

    Beklenen davranış:
        - None döner
        - Hata fırlatmaz
        - Backend tokenizer üzerinde herhangi bir mutasyon yapmaz

    Bu test, wrapper'ın "training-free" doğasını dokümante eden canlı bir
    örnek olarak da işlev görür.
    """
    result = wrapper.train("any text doesn't matter")

    assert result is None


def test_pre_trained_tokenizer_train_does_not_call_backend(
    wrapper: PreTrainedTokenizerWrapper,
) -> None:
    """
    train() çağrısının backend tokenizer'ı tetiklemediğini test eder.

    Çünkü:
        train() bilinçli olarak no-op'tur. Eğer kazara backend'in encode/
        tokenize gibi bir metodunu çağırırsa beklenmedik yan etkiler oluşur.

    Test stratejisi:
        train() çağrılmadan önce mock'un çağrı sayısı sıfırlanır,
        train() sonrası backend'in HİÇBİR metodunun çağrılmadığı doğrulanır.
    """
    # Backend mock'unun şimdiye kadarki çağrılarını temizle
    wrapper._tokenizer.reset_mock()

    wrapper.train("some text")

    # train() backend'i hiç çağırmamış olmalı
    wrapper._tokenizer.assert_not_called()
    wrapper._tokenizer.tokenize.assert_not_called()
    wrapper._tokenizer.encode.assert_not_called()


# ---------------------------------------------------------
# TOKENIZE TESTS
# ---------------------------------------------------------

def test_pre_trained_tokenizer_tokenize_delegates_to_backend(
    wrapper: PreTrainedTokenizerWrapper,
) -> None:
    """
    tokenize()'ın çağrıyı backend tokenizer'a yönlendirdiğini test eder.

    Çünkü:
        Wrapper kendi tokenization mantığı yazmaz; sadece pretrained
        tokenizer'ın tokenize() metodunu çağırır ve sonucu standart liste
        formatına dönüştürür.

    Bu test:
        - Doğru metodun çağrıldığını
        - Argümanın aynen iletildiğini
        - Dönüş değerinin liste olduğunu
    bir arada doğrular.
    """
    result = wrapper.tokenize("hello world")

    wrapper._tokenizer.tokenize.assert_called_once_with("hello world")
    assert result == ["hello", "world"]
    assert isinstance(result, list)


def test_pre_trained_tokenizer_tokenize_raises_error_for_empty_text(
    wrapper: PreTrainedTokenizerWrapper,
) -> None:
    """
    Boş metin için tokenize()'ın ValueError fırlattığını test eder.

    Çünkü:
        Wrapper, _validate_text() helper'ı üzerinden ortak bir input
        validasyonu uygular. Boş veya whitespace-only input için tüm
        public input metodları (tokenize, encode) aynı davranışı sergiler.

    Bu tutarlılık önemlidir:
        - Kullanıcı her metod için farklı edge case ezberlemek zorunda kalmaz
        - API kontratı tahmin edilebilir kalır

    Beklenen:
        ValueError("Text cannot be empty")
    """
    with pytest.raises(ValueError, match="Text cannot be empty"):
        wrapper.tokenize("")


def test_pre_trained_tokenizer_tokenize_raises_error_for_whitespace_only_text(
    wrapper: PreTrainedTokenizerWrapper,
) -> None:
    """
    Sadece whitespace içeren metin için tokenize()'ın ValueError fırlattığını
    test eder.

    Çünkü:
        _validate_text() whitespace-only input'u boş kabul eder.
        Bu, "anlamlı içerik" tanımının sadece string uzunluğuna değil,
        gerçek karakter içeriğine dayandığı anlamına gelir.
    """
    with pytest.raises(ValueError, match="Text cannot be empty"):
        wrapper.tokenize("   \t\n  ")


# ---------------------------------------------------------
# ENCODE TESTS
# ---------------------------------------------------------

def test_pre_trained_tokenizer_encode_returns_list_of_integers(
    wrapper: PreTrainedTokenizerWrapper,
) -> None:
    """
    encode() çıktısının integer token id listesi olduğunu test eder.

    Çünkü:
        Hugging Face encode() bazı durumlarda farklı tipler dönebilir
        (örn. torch tensor `return_tensors="pt"` ile). Wrapper bunu
        defansif olarak `list(...)` ile sarar ki çıktı her zaman saf
        Python list[int] olsun.

    Tip kontrolleri:
        - Sonuç list olmalı
        - Tüm elemanlar int olmalı
    """
    result = wrapper.encode("hello world")

    assert isinstance(result, list)
    assert all(isinstance(token_id, int) for token_id in result)


def test_pre_trained_tokenizer_encode_passes_add_special_tokens_to_backend(
    wrapper: PreTrainedTokenizerWrapper,
) -> None:
    """
    encode()'un add_special_tokens parametresini backend'e doğru ilettiğini
    test eder.

    Çünkü:
        add_special_tokens BERT için [CLS]/[SEP] ekleme davranışını kontrol eder.
        Wrapper bu kararı constructor'da alıp her encode çağrısına geçirir.

    Test fixture'ı add_special_tokens=False (default) ile kurulduğu için
    bu değerin backend'e iletildiği doğrulanır.

    Eğer parametre iletilmezse:
        - HF'in kendi default'u (genellikle True) devreye girer
        - Beklenmedik tokenlar eklenir
        - Compare/report çıktıları model-specific tokenlarla bozulur
    """
    wrapper.encode("hello")

    _, kwargs = wrapper._tokenizer.encode.call_args
    assert kwargs.get("add_special_tokens") is False


def test_pre_trained_tokenizer_encode_uses_add_special_tokens_true_when_configured(
    mock_hf_tokenizer: MagicMock,
) -> None:
    """
    add_special_tokens=True ile kurulan wrapper'ın bunu backend'e ilettiğini
    test eder.

    Bu test, önceki testin negatif görüntüsünü tamamlar.
    Constructor parametresinin gerçekten encode davranışını etkilediğini
    iki yönlü doğrular.
    """
    with patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=mock_hf_tokenizer,
    ):
        wrapper = PreTrainedTokenizerWrapper(
            model_name="bert-base-uncased",
            add_special_tokens=True,
        )

    wrapper.encode("hello")

    _, kwargs = wrapper._tokenizer.encode.call_args
    assert kwargs.get("add_special_tokens") is True


def test_pre_trained_tokenizer_encode_raises_error_for_empty_text(
    wrapper: PreTrainedTokenizerWrapper,
) -> None:
    """
    Boş metin için encode()'un ValueError fırlattığını test eder.

    tokenize() ile aynı validasyon davranışı kullanılır.
    Bu tutarlılık _validate_text() helper'ı üzerinden sağlanır.
    """
    with pytest.raises(ValueError, match="Text cannot be empty"):
        wrapper.encode("")


def test_pre_trained_tokenizer_encode_raises_error_for_whitespace_only_text(
    wrapper: PreTrainedTokenizerWrapper,
) -> None:
    """
    Whitespace-only metin için encode()'un ValueError fırlattığını test eder.
    """
    with pytest.raises(ValueError, match="Text cannot be empty"):
        wrapper.encode("\n  \t")


# ---------------------------------------------------------
# DECODE TESTS
# ---------------------------------------------------------

def test_pre_trained_tokenizer_decode_returns_string(
    wrapper: PreTrainedTokenizerWrapper,
) -> None:
    """
    decode()'un string döndürdüğünü test eder.

    Çünkü:
        HF decode() bazı tokenizer'larda bytes veya tokenizer-specific
        bir tip dönebilir. Wrapper'ın `str(...)` ile sarmasının amacı
        her durumda saf Python string garantisi vermektir.
    """
    result = wrapper.decode([1, 2, 3])

    assert isinstance(result, str)


def test_pre_trained_tokenizer_decode_delegates_to_backend(
    wrapper: PreTrainedTokenizerWrapper,
) -> None:
    """
    decode()'un backend tokenizer'a doğru argümanlarla yetki devrettiğini
    test eder.

    Doğrulanan iki nokta:
        1. Token id listesi aynen iletilir
        2. skip_special_tokens parametresi add_special_tokens'ın TERSİ olur

    İkinci nokta neden önemli?
        add_special_tokens=False kullanan kullanıcı encode'da [CLS]/[SEP]
        eklemiyor. Decode'da da bu tokenları görmek istemez. Wrapper bunu
        otomatik yapar:
            skip_special_tokens = not self.add_special_tokens
    """
    wrapper.decode([1, 2, 3])

    args, kwargs = wrapper._tokenizer.decode.call_args
    assert args[0] == [1, 2, 3]
    # add_special_tokens=False -> skip_special_tokens=True
    assert kwargs.get("skip_special_tokens") is True


def test_pre_trained_tokenizer_decode_raises_error_for_empty_list(
    wrapper: PreTrainedTokenizerWrapper,
) -> None:
    """
    Boş id listesi için decode()'un ValueError fırlattığını test eder.

    Çünkü:
        Boş id listesi anlamlı bir decode operasyonu değildir.
        encode() boş input'u zaten reddettiği için decode()'a boş liste
        gelmesi muhtemelen bir bug işaretidir.

    Beklenen:
        ValueError("token_ids cannot be empty")
    """
    with pytest.raises(ValueError, match="token_ids cannot be empty"):
        wrapper.decode([])


def test_pre_trained_tokenizer_decode_raises_error_for_non_integer_ids(
    wrapper: PreTrainedTokenizerWrapper,
) -> None:
    """
    Liste integer dışı eleman içeriyorsa decode()'un ValueError fırlattığını
    test eder.

    Senaryolar:
        decode([1, "two", 3])    -> string karışmış
        decode([1.5, 2, 3])      -> float karışmış
        decode([None, 2])        -> None karışmış

    Çünkü:
        HF decode() integer olmayan id'lerle çalıştırılırsa anlaşılması zor
        TypeError veya beklenmedik davranışlar üretir. Wrapper'ın erken
        validasyonu kullanıcıya net hata mesajı sağlar.

    Beklenen:
        ValueError("token_ids must contain only integers")
    """
    with pytest.raises(ValueError, match="must contain only integers"):
        wrapper.decode([1, "two", 3])  # type: ignore[list-item]


# ---------------------------------------------------------
# CONVERT IDS <-> TOKENS TESTS
# ---------------------------------------------------------

def test_pre_trained_tokenizer_convert_ids_to_tokens_returns_list_of_strings(
    wrapper: PreTrainedTokenizerWrapper,
) -> None:
    """
    convert_ids_to_tokens()'un string token listesi döndürdüğünü test eder.

    Çünkü:
        Bu metod özellikle raporlama tarafında kullanılır.
        encode() integer döndürürken bu metod insan-okunabilir token
        string'lerini döndürür.

    Tip kontrolü:
        - Sonuç list olmalı
        - Tüm elemanlar str olmalı
    """
    result = wrapper.convert_ids_to_tokens([1, 2, 3])

    assert isinstance(result, list)
    assert all(isinstance(token, str) for token in result)


def test_pre_trained_tokenizer_convert_ids_to_tokens_returns_empty_list_for_empty_input(
    wrapper: PreTrainedTokenizerWrapper,
) -> None:
    """
    Boş id listesi için convert_ids_to_tokens()'un boş liste döndürdüğünü
    test eder.

    Çünkü:
        decode()'un aksine bu metod boş input'u hata olarak görmez.
        Boş input -> boş output mantıklı bir davranıştır ve raporlama
        tarafında özel kontrol gerektirmez.

    Bu, decode() ve convert_ids_to_tokens() arasındaki kasıtlı bir
    tasarım farkıdır:
        - decode([]):                   ValueError
        - convert_ids_to_tokens([]):    []

    Sebep: decode'a boş liste gelmek bir bug sinyalidir, ama
    convert_ids_to_tokens raporlama akışında doğal olarak boşla
    çağrılabilir.
    """
    result = wrapper.convert_ids_to_tokens([])

    assert result == []


def test_pre_trained_tokenizer_convert_ids_to_tokens_raises_error_for_non_integer(
    wrapper: PreTrainedTokenizerWrapper,
) -> None:
    """
    Liste integer dışı eleman içeriyorsa convert_ids_to_tokens()'un
    ValueError fırlattığını test eder.

    decode() ile aynı tip korumasını sağlar; iki metod tutarlı validation
    davranışı sergiler.
    """
    with pytest.raises(ValueError, match="must contain only integers"):
        wrapper.convert_ids_to_tokens([1, "abc", 3])  # type: ignore[list-item]


def test_pre_trained_tokenizer_convert_tokens_to_ids_returns_list_of_integers(
    wrapper: PreTrainedTokenizerWrapper,
) -> None:
    """
    convert_tokens_to_ids()'un integer id listesi döndürdüğünü test eder.

    Bu metod convert_ids_to_tokens()'un ters yönüdür:
        ids -> tokens   (convert_ids_to_tokens)
        tokens -> ids   (convert_tokens_to_ids)

    İki yönlü dönüşümün varlığı raporlama ve debug akışlarında esneklik sağlar.
    """
    result = wrapper.convert_tokens_to_ids(["hello", "world"])

    assert isinstance(result, list)
    assert all(isinstance(token_id, int) for token_id in result)


def test_pre_trained_tokenizer_convert_tokens_to_ids_returns_empty_list_for_empty_input(
    wrapper: PreTrainedTokenizerWrapper,
) -> None:
    """
    Boş token listesi için convert_tokens_to_ids()'un boş liste döndürdüğünü
    test eder.

    convert_ids_to_tokens() ile aynı boş-input davranışı.
    """
    assert wrapper.convert_tokens_to_ids([]) == []


def test_pre_trained_tokenizer_convert_tokens_to_ids_raises_error_for_non_string(
    wrapper: PreTrainedTokenizerWrapper,
) -> None:
    """
    Liste string dışı eleman içeriyorsa convert_tokens_to_ids()'un
    ValueError fırlattığını test eder.

    Senaryolar:
        convert_tokens_to_ids(["hello", 42])
        convert_tokens_to_ids([None, "world"])
    """
    with pytest.raises(ValueError, match="must contain only strings"):
        wrapper.convert_tokens_to_ids(["hello", 42])  # type: ignore[list-item]


# ---------------------------------------------------------
# VOCAB / METADATA TESTS
# ---------------------------------------------------------

def test_pre_trained_tokenizer_vocab_size_returns_integer(
    wrapper: PreTrainedTokenizerWrapper,
) -> None:
    """
    vocab_size property'sinin integer döndürdüğünü test eder.

    Çünkü:
        BaseTokenizer kontratı vocab_size'ın int olmasını gerektirir.
        Hugging Face'in bazı tokenizer'larında bu değer numpy int veya
        başka bir sayısal tip olabilir; wrapper `int(...)` ile bunu sarar.
    """
    result = wrapper.vocab_size

    assert isinstance(result, int)
    assert result > 0


def test_pre_trained_tokenizer_special_tokens_returns_dict(
    wrapper: PreTrainedTokenizerWrapper,
) -> None:
    """
    special_tokens property'sinin dict döndürdüğünü test eder.

    Çünkü:
        BERT tokenizer beş özel token tanımlar:
            [UNK], [SEP], [PAD], [CLS], [MASK]

    Wrapper bu bilgiyi raporlama amacıyla expose eder. Tip olarak dict
    olduğu garanti edilir; HF'in döndüğü `special_tokens_map` farklı
    iç tipler kullanabileceği için defansif `dict(...)` sarmalama uygulanır.

    Test mock'taki BERT-benzeri fixture'la doğrulama yapar:
        - Sonuç dict olmalı
        - "unk_token" gibi standard BERT key'leri içermeli
    """
    result = wrapper.special_tokens

    assert isinstance(result, dict)
    assert "unk_token" in result
    assert "cls_token" in result


def test_pre_trained_tokenizer_backend_tokenizer_name_returns_model_name(
    wrapper: PreTrainedTokenizerWrapper,
) -> None:
    """
    backend_tokenizer_name property'sinin constructor'da verilen model adını
    döndürdüğünü test eder.

    Çünkü:
        Bu property raporlama ve debug çıktılarında "hangi tokenizer
        kullanıldı?" sorusunun cevabını verir.

    Beklenen:
        wrapper.backend_tokenizer_name == "bert-base-uncased"
    """
    assert wrapper.backend_tokenizer_name == "bert-base-uncased"


# ---------------------------------------------------------
# REGISTRY / FACTORY TESTS
# ---------------------------------------------------------

def test_pre_trained_tokenizer_is_registered_under_correct_name(
    mock_hf_tokenizer: MagicMock,
) -> None:
    """
    TokenizerFactory.create("pre_trained", ...) çağrısının doğru sınıfı
    döndürdüğünü test eder.

    Çünkü:
        PreTrainedTokenizerWrapper @register_tokenizer("pretrained") decorator
        ile registry'ye kaydedilir. Bu kayıt yanlış olursa:
            - Factory yanlış sınıfı döndürür
            - "pretrained" ismi farklı bir tokenizer'a işaret eder
            - CompareManager pretrained tokenizer'ı bulamaz

    Doğrulanan iki property:
        - Üretilen instance PreTrainedTokenizerWrapper tipinde olmalı
        - tokenizer.name == "pretrained" olmalı (BaseTokenizer'dan)

    Network bağımlılığını engellemek için from_pretrained mock'lanır.
    """
    with patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=mock_hf_tokenizer,
    ):
        tokenizer = TokenizerFactory.create("pretrained", model_name="bert-base-uncased")

    assert isinstance(tokenizer, PreTrainedTokenizerWrapper)
    assert tokenizer.name == "pretrained"


# =========================================================
# INTEGRATION TESTS
# =========================================================
#
# Bu testler gerçek bir Hugging Face tokenizer'ı yükler.
# Network erişimi veya cache'lenmiş model gerektirirler ve mock
# testlerden çok daha yavaştır.
#
# Bu yüzden `pytest.mark.integration` ile işaretlenmiştir ve
# varsayılan olarak çalıştırılmazlar. Çalıştırmak için:
#
#     uv run pytest -m integration
#
# Mock testler "wrapper'ın kendi mantığını" doğrularken, integration
# testleri "gerçek HF tokenizer ile uçtan uca davranışı" doğrular.
# =========================================================


@pytest.fixture(scope="module")
def real_bert_wrapper() -> PreTrainedTokenizerWrapper:
    """
    Gerçek bir BERT-base-uncased tokenizer ile wrapper instance'ı üretir.

    scope="module":
        Aynı modül içindeki tüm integration testleri tek bir tokenizer
        instance'ını paylaşır. Tokenizer yüklemesi pahalı olduğu için
        her test için yeniden yapmak gereksiz yavaşlama yaratır.
    """
    pytest.importorskip("transformers")

    return PreTrainedTokenizerWrapper(model_name="bert-base-uncased")


@pytest.mark.integration
def test_integration_real_bert_tokenize_produces_subword_tokens(
    real_bert_wrapper: PreTrainedTokenizerWrapper,
) -> None:
    """
    Gerçek BERT tokenizer'ın WordPiece subword tokenization yaptığını test eder.

    Çünkü:
        BERT-base-uncased "tokenization" gibi nadir kelimeleri subword'lere
        ayırır. Bu davranış WordPiece algoritmasının imzasıdır:
            "tokenization" -> ["token", "##ization"]

    "##" prefix'i WordPiece'in subword belirteci olduğu için sonucun
    en az birinde bu prefix görülmelidir.

    Bu test wrapper'ın gerçek HF backend'i ile çalıştığını ve subword
    çıktısını koruduğunu uçtan uca doğrular.
    """
    tokens = real_bert_wrapper.tokenize("tokenization")

    assert len(tokens) >= 1
    # En az bir token "##" ile başlayan subword olmalı
    assert any(token.startswith("##") for token in tokens)


@pytest.mark.integration
def test_integration_real_bert_encode_decode_roundtrip(
    real_bert_wrapper: PreTrainedTokenizerWrapper,
) -> None:
    """
    Gerçek BERT tokenizer ile encode -> decode davranışını test eder.

    Önemli:
        BERT-base-uncased lowercase normalization uygular. Bu yüzden
        roundtrip BIREBIR korunmaz:
            "Hello World" -> "hello world"

    Bu yüzden test:
        - decode'un başarılı çalıştığını
        - Lowercase versiyonun decoded text'te olduğunu
    doğrular; birebir eşitlik istemez.

    Bu, wrapper'ın gerçek tokenizer'ın "lossy normalization" davranışını
    doğru şekilde delege ettiğini gösterir.
    """
    text = "Hello World"

    encoded = real_bert_wrapper.encode(text)
    decoded = real_bert_wrapper.decode(encoded)

    assert isinstance(decoded, str)
    assert "hello" in decoded.lower()
    assert "world" in decoded.lower()


@pytest.mark.integration
def test_integration_real_bert_vocab_size_is_around_30k(
    real_bert_wrapper: PreTrainedTokenizerWrapper,
) -> None:
    """
    Gerçek BERT-base-uncased'ın bilinen vocab_size'ını test eder.

    Çünkü:
        BERT-base-uncased'ın resmi vocab boyutu 30,522'dir. Bu, HF Hub'ından
        indirilen dosyanın doğru olduğunun ve wrapper'ın bu metadatayı
        doğru aktardığının kanıtıdır.

    Tolerance:
        Test 30,000-31,000 aralığını kabul eder. Birebir 30522 demek yerine
        aralık kullanmak, HF'in farklı versiyon tokenizer'larında küçük
        vocab değişikliklerine karşı dayanıklılık sağlar.
    """
    assert 30000 <= real_bert_wrapper.vocab_size <= 31000


@pytest.mark.integration
def test_integration_real_bert_tokenize_handles_turkish_characters(
    real_bert_wrapper: PreTrainedTokenizerWrapper,
) -> None:
    """
    BERT-base-uncased'ın Türkçe karakterleri tokenize ettiğini test eder.

    Not:
        BERT-base-uncased ASCII odaklı bir tokenizer'dır. Türkçe karakterler
        çok parçalı subword'lere ayrılabilir veya [UNK] olarak işaretlenebilir.

    Bu test BIREBIR doğru tokenization beklemez. Sadece:
        - tokenize() hata fırlatmadan çalışır
        - Çıktı boş değildir
    şeklindeki temel kontratı doğrular.

    Türkçeye uygun olmayan tokenizer kullanımının dokümante edilmiş bir
    sınırlama olduğunu hatırlatan canlı bir örnek görevi de görür.
    """
    tokens = real_bert_wrapper.tokenize("merhaba dünya")

    assert isinstance(tokens, list)
    assert len(tokens) > 0