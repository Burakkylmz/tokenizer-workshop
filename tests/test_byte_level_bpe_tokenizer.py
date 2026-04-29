from __future__ import annotations

from collections import Counter

import pytest

from tokenizer_workshop.api.services.tokenizer_factory import TokenizerFactory
from tokenizer_workshop.tokenizers.byte_level_bpe_tokenizer import (
    ByteLevelBPEMerge,
    ByteLevelBPETokenizer,
)


# ---------------------------------------------------------
# INIT TESTS
# ---------------------------------------------------------

def test_byte_level_bpe_tokenizer_init_raises_error_for_zero_num_merges() -> None:
    """
    num_merges=0 verildiğinde ValueError fırlatıldığını test eder.
 
    Çünkü:
        BPE algoritmasının özü merge kuralları öğrenmektir.
        Eğer num_merges=0 olursa hiç merge öğrenilmez ve tokenizer yalnızca
        base byte vocabulary üzerinde çalışır.
 
    Bu durum byte-level BPE'yi anlamsız kılar:
        - Hiçbir sıkıştırma yapılmaz
        - Tokenizer normal bir ByteTokenizer'a indirgenir
        - "BPE" ismi anlamsızlaşır
 
    Bu yüzden tokenizer yapıcı (constructor) seviyesinde bu durumu engeller
    ve açıkça hata fırlatır.
    """
    with pytest.raises(ValueError, match="num_merges must be at least 1"):
        TokenizerFactory.create("byte_level_bpe", num_merges=0)


def test_byte_level_bpe_tokenizer_init_raises_error_for_negative_num_merges() -> None:
    """
    num_merges negatif verildiğinde ValueError fırlatıldığını test eder.
 
    Çünkü:
        Negatif sayıda merge öğrenmek mantıksal olarak imkansızdır.
        Bu tür değerler genellikle bir hesaplama hatası veya yanlış
        konfigürasyondan kaynaklanır.
 
    Tokenizer bu durumu sessizce kabul edip 0 merge ile devam etmek yerine
    açıkça hata fırlatır. Bu sayede:
        - Hatalar erken aşamada yakalanır
        - Beklenmeyen davranışlar engellenir
        - API kullanıcısı yanlış konfigürasyon hakkında uyarılır
    """
    with pytest.raises(ValueError, match="num_merges must be at least 1"):
        TokenizerFactory.create("byte_level_bpe", num_merges=-5)


def test_byte_level_bpe_tokenizer_default_num_merges_is_accepted() -> None:
    """
    num_merges parametresi verilmediğinde varsayılan değerin kullanıldığını test eder.
 
    Constructor signature:
        def __init__(self, num_merges: int = 100) -> None
 
    Beklenen davranış:
        - Tokenizer hata fırlatmadan oluşturulmalı
        - Üretilen instance ByteLevelBPETokenizer tipinde olmalı
        - num_merges attribute'u 100 olmalı
 
    Neden bu test önemli?
        Default değerin değişmesi public API kontratının kırılması anlamına gelir.
        Bu test, ileride defaultu değiştirenlerin bunu bilinçli yaptığını garanti eder.
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe")
 
    assert isinstance(tokenizer, ByteLevelBPETokenizer) # Oluşturulan instance'ın doğru tipte olduğunu doğrular
    assert tokenizer.num_merges == 100 # Default değerin 100 olduğunu doğrular


def test_byte_level_bpe_tokenizer_initial_vocab_is_full_byte_range() -> None:
    """
    Eğitim öncesi vocabulary'nin tam olarak 256 byte tokenı içerdiğini test eder.
 
    Çünkü:
        Byte-Level BPE'de base vocabulary 0-255 aralığındaki tüm byte değerlerini
        kapsar. Bu, byte-level yaklaşımın en kritik özelliğidir.
 
    Neden 256 token zorunlu?
        Bir UTF-8 byte 0-255 arası bir değer alabilir.
        Eğer vocabulary tüm bu byte'ları kapsamazsa:
            - Bazı UTF-8 sequence'lar encode edilemez
            - OOV (out-of-vocabulary) sorunu yaşanır
            - Türkçe karakter, emoji, semboller sorun çıkarır
 
    Tokenizer henüz train edilmese bile bu 256 token mevcut olmalıdır.
    Çünkü base vocabulary __init__ aşamasında oluşturulur, train sırasında değil.
 
    Beklenen:
        vocab_size == 256
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
 
    assert tokenizer.vocab_size == 256 # Eğitim öncesi vocab_size 256 olmalıdır


def test_byte_level_bpe_tokenizer_initial_state_has_no_merge_steps() -> None:
    """
    Eğitim öncesi merge_steps listesinin boş olduğunu test eder.
 
    Çünkü:
        merge_steps, eğitim sırasında öğrenilen BPE merge kurallarını saklar.
        Henüz train() çağrılmadığı için hiçbir kural öğrenilmemiş olmalıdır.
 
    Eğer bu liste başlangıçta dolu olsaydı:
        - Tokenizer "kirli" bir state ile başlardı
        - Önceki eğitimlerden artık kalmış olurdu (instance cache problemi)
        - Determinism garantisi bozulurdu
 
    Bu test, tokenizer'ın temiz bir başlangıç state'i sağladığını doğrular.
 
    Beklenen:
        merge_steps == []
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
 
    assert tokenizer.merge_steps == [] # Eğitim öncesi merge_steps boş olmalıdır


# ---------------------------------------------------------
# TRAIN TESTS
# ---------------------------------------------------------

def test_byte_level_bpe_tokenizer_train_with_empty_text_raises_error() -> None:
    """
    Tamamen boş metinle train() çağrıldığında ValueError fırlatıldığını test eder.
 
    Çünkü:
        Boş metinde hiç byte yoktur, dolayısıyla hiçbir pair frekansı hesaplanamaz.
        Bu durumda merge öğrenmek imkansızdır.
 
    Tokenizer bu durumu erken yakalar ve açıkça hata fırlatır.
    Sessizce 0 merge öğrenmiş gibi davranmaz çünkü:
        - Kullanıcı yanlış input verdiğini bilmez
        - Eğitim "başarılı" görünür ama gerçekte boştur
        - Sonraki encode/decode çağrılarında beklenmedik davranışlar oluşur
 
    Beklenen:
        ValueError("Training text cannot be empty")
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
 
    with pytest.raises(ValueError, match="Training text cannot be empty"):
        tokenizer.train("")


def test_byte_level_bpe_tokenizer_train_with_whitespace_only_text_raises_error() -> None:
    """
    Sadece whitespace içeren metnin de boş kabul edildiğini test eder.
 
    Input örnekleri:
        "   "       -> sadece boşluklar
        "\\n\\t  "    -> newline, tab, boşluk
        " "         -> tek boşluk
 
    Çünkü:
        Whitespace-only input pratik olarak anlamlı bir eğitim verisi sağlamaz.
        Sadece space/newline/tab byte'ları içeren bir korpustan öğrenilebilecek
        merge'ler kullanışsızdır.
 
    Tokenizer bu durumu da boş kabul eder ve hata fırlatır.
    Bu, "boş input" tanımını sadece string uzunluğu üzerinden değil, anlamlı
    içerik üzerinden yapan defansif bir kontroldür.
 
    Beklenen:
        ValueError("Training text cannot be empty")
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
 
    with pytest.raises(ValueError, match="Training text cannot be empty"):
        tokenizer.train("   \n\t  ")


def test_byte_level_bpe_tokenizer_train_marks_tokenizer_as_trained() -> None:
    """
    train() başarıyla tamamlandığında tokenizer'ın eğitilmiş duruma geçtiğini test eder.
 
    Çünkü:
        Tokenizer'ın iki state'i vardır: "trained" ve "untrained".
        Bu state encode() ve decode() metodlarının çalışıp çalışmayacağını belirler.
 
    Bu test internal `_trained` flag'ine doğrudan bakmaz.
    Davranış üzerinden doğrular:
        - train() öncesi encode() hata fırlatır
        - train() sonrası encode() hata fırlatmaz
 
    Bu yaklaşım tercih edildi çünkü:
        - Internal attribute isimleri implementation detail'dir
        - Davranışsal test daha sağlam ve gelecekte kırılmaya daha dirençlidir
 
    Beklenen:
        tokenizer.encode("abababa") hata fırlatmadan çalışır
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
 
    tokenizer.train("abababa")
 
    # Encode çağrısı hata fırlatmadan tamamlanmalıdır.
    # Eğer tokenizer eğitilmiş olarak işaretlenmemişse buradan ValueError gelir.
    tokenizer.encode("abababa")


def test_byte_level_bpe_tokenizer_train_learns_at_least_one_merge() -> None:
    """
    Tekrarlayan pattern içeren metinde en az bir merge öğrenildiğini test eder.
 
    Input:
        "abababa"
 
    Initial byte sequence:
        [97, 98, 97, 98, 97, 98, 97]
 
    Pair frekansları:
        (97, 98) -> 3
        (98, 97) -> 3
 
    Burada en az bir pair sıklıkla tekrar ettiği için BPE algoritması
    en az bir merge kuralı üretmelidir.
 
    Eğer hiç merge öğrenilmezse:
        - num_merges=3 verilmesine rağmen sıfır kural çıkar
        - Bu, eğitim algoritmasında bir bug olduğunu gösterir
 
    Beklenen:
        len(merge_steps) >= 1
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
 
    tokenizer.train("abababa")
 
    assert len(tokenizer.merge_steps) >= 1 # En az bir merge öğrenilmelidir


def test_byte_level_bpe_tokenizer_train_first_merge_picks_pair_with_highest_frequency() -> None:
    """
    İlk merge edilen pair'in en yüksek frekanslı pair olduğunu test eder.
 
    Input:
        "abababa"
 
    Pair frekansları:
        (97, 98) -> 3   # ('a', 'b')
        (98, 97) -> 3   # ('b', 'a')
 
    Burada iki pair de aynı frekansa (3) sahiptir; bu bir "tie" durumudur.
 
    Tie-breaking kuralı (implementasyonda):
        key = lambda item: (item[1], item[0])    # (frequency, pair)
 
    max() bu key ile çalıştığı için:
        - Önce frekansa bakar (eşit)
        - Sonra pair'in kendisine bakar (leksikografik)
        - (98, 97) > (97, 98) olduğu için (98, 97) yani ('b', 'a') seçilir
 
    Bu test:
        - Frekansın doğru hesaplandığını (frequency == 3)
        - Tie-breaking'in deterministik çalıştığını ((98, 97) seçimi)
    aynı anda doğrular.
 
    Beklenen:
        first_step.pair == (ord("b"), ord("a"))
        first_step.frequency == 3
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
 
    tokenizer.train("abababa")
 
    first_step = tokenizer.merge_steps[0] # İlk öğrenilen merge adımını alır

    assert first_step.frequency == 3 # En yüksek frekansın 3 olduğunu doğrular
    assert first_step.pair == (ord("b"), ord("a")) # Tie-breaking kuralına göre ('b', 'a') pair'inin seçildiğini doğrular


def test_byte_level_bpe_tokenizer_train_assigns_new_token_id_starting_from_256() -> None:
    """
    İlk öğrenilen merge tokenının id'sinin 256 olduğunu test eder.
 
    Çünkü:
        Base byte vocabulary 0-255 aralığını kullanır.
        Bu yüzden yeni merge token id'leri 256'dan başlamak zorundadır.
 
    Eğer ilk merge id'si 256 değilse:
        - Mevcut byte token id'leri ile çakışır
        - Decode sırasında byte vs merge token karışıklığı oluşur
        - Vocabulary mantığı kırılır
 
    Beklenen:
        merge_steps[0].merged_token_id == 256
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
 
    tokenizer.train("abababa")
 
    assert tokenizer.merge_steps[0].merged_token_id == 256 # İlk merge token id'si 256 olmalıdır


def test_byte_level_bpe_tokenizer_train_assigns_sequential_token_ids() -> None:
    """
    Birden fazla merge öğrenildiğinde token id'lerin ardışık atandığını test eder.
 
    Çünkü:
        BPE algoritmasında her yeni merge bir önceki merged_token_id'den
        bir sonraki id'yi alır. Yani id'ler:
            256, 257, 258, 259, ...
        şeklinde olmalıdır.
 
    Eğer atlanan id'ler olsaydı (örn. 256, 258, 259):
        - Vocabulary'de "delik"ler oluşurdu
        - vocab_size hesaplamaları yanıltıcı olurdu
        - Olmayan id'lere dair lookup hataları çıkardı
 
    Test stratejisi:
        Beklenen id listesi range(256, 256 + len(merge_steps)) olarak hesaplanır.
        Sonra gerçek id'lerle karşılaştırılır.
 
    Bu yaklaşım, kaç merge öğrenildiğinden bağımsız olarak id ardışıklığını
    doğrular.
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=5)
 
    tokenizer.train("abababab")

    # Beklenen id'ler: 256, 257, 258, ... (kaç merge öğrenildiyse o kadar)
    expected_ids = list(range(256, 256 + len(tokenizer.merge_steps)))
    actual_ids = [step.merged_token_id for step in tokenizer.merge_steps]
 
    assert actual_ids == expected_ids # Merge token id'lerinin ardışık olduğunu doğrular


def test_byte_level_bpe_tokenizer_train_stops_when_no_more_pairs() -> None:
    """
    Metinde merge edilecek pair kalmadığında eğitimin erken durduğunu test eder.
 
    Input:
        "a"     -> tek karakter, hiç pair yok
 
    Beklenen davranış:
        num_merges=10 verilse bile öğrenilen merge sayısı 0 olmalıdır.
 
    Çünkü:
        BPE algoritması yan yana gelen byte çiftleri üzerinde çalışır.
        Tek byte'lık input'ta hiç çift yoktur, dolayısıyla:
            - Pair frequency Counter'ı boş döner
            - Algoritma erken break ile durur
            - merge_steps boş kalır
 
    Bu davranış neden önemli?
        Tokenizer küçük veya patolojik input'larla çağrılsa bile
        sonsuz döngüye girmez ya da hata fırlatmaz; sadece sessizce
        durur ve geçerli bir state ile çıkar.
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=10)
 
    tokenizer.train("a")
 
    assert tokenizer.merge_steps == [] # Hiç merge öğrenilmemelidir


def test_byte_level_bpe_tokenizer_train_respects_num_merges_upper_bound() -> None:
    """
    Öğrenilen merge sayısının num_merges'i aşmadığını test eder.
 
    Çünkü:
        num_merges parametresi öğrenilecek merge sayısı için ÜST sınırdır.
        Eğitim metni çok zenginse algoritma bu sınırda durmalıdır.
 
    Eğer num_merges aşılırsa:
        - vocab_size beklenenden büyük olur
        - Memory tüketimi artar
        - Public API kontratı kırılır
 
    Test stratejisi:
        num_merges=2 ile sınır düşük tutulur. "abcabcabcabc" gibi zengin
        bir metin verilse bile öğrenilen merge sayısı 2'yi aşmamalıdır.
 
    Beklenen:
        len(merge_steps) <= num_merges
    """
    num_merges = 2 # Çok düşük bir sınır belirlenir
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=num_merges)
 
    tokenizer.train("abcabcabcabc")
 
    assert len(tokenizer.merge_steps) <= num_merges # Öğrenilen merge sayısı num_merges'i aşmamalıdır


def test_byte_level_bpe_tokenizer_train_clears_previous_state_when_retrained() -> None:
    """
    Aynı tokenizer instance'ı tekrar train edildiğinde eski state'in
    sızmadığını test eder.
 
    Senaryo:
        1. tokenizer.train("abababa") -> bir merge seti öğrenilir
        2. tokenizer.train("xyxyxyx") -> tamamen farklı bir merge seti öğrenilmeli
 
    Çünkü:
        Tokenizer'lar genellikle uzun yaşar ve birden fazla deneyde yeniden
        eğitilebilir. Eğer eski state sızarsa:
            - İkinci eğitim "kirli" başlar
            - Determinism garantisi bozulur
            - Encode sonuçları tahmin edilemez hale gelir
 
    İmplementasyonda bu, _reset_training_state() metodu ile çözülür.
    Bu test o metodun davranışsal etkisini doğrular.
 
    Doğrulama stratejisi:
        - İlk eğitimde öğrenilen (b, a) pair'i ikinci eğitimde olmamalı
        - İkinci eğitimin ilk merge'ü 'x' ve 'y' byte'larından oluşmalı
          (tie-breaking nedeniyle leksikografik olarak büyük olan (y, x) seçilir)
        - İki eğitimin merge_steps listeleri birbirinden farklı olmalı
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)

    tokenizer.train("abababa")
    first_training_merges = list(tokenizer.merge_steps)
 
    tokenizer.train("xyxyxyx")
    second_training_merges = tokenizer.merge_steps
 
    # İkinci eğitimde ilk eğitimin (b, a) merge'ü olmamalıdır.
    # Eğer state sızmışsa eski merge'ler hala listede olur.
    assert (ord("b"), ord("a")) not in [m.pair for m in second_training_merges]
 
    # İkinci eğitimin ilk merge'ü 'x' ve 'y' byte'larından oluşmalı.
    # set() kullanılır çünkü tie-breaking yön bağımsızdır.
    first_pair = second_training_merges[0].pair
    assert set(first_pair) == {ord("x"), ord("y")}
 
    # İki eğitim seti birbirinden farklı olmalı (sanity check).
    assert second_training_merges != first_training_merges


def test_byte_level_bpe_tokenizer_train_resets_next_token_id_between_runs() -> None:
    """
    Yeniden eğitim yapıldığında merge token id'lerinin yeniden 256'dan
    başladığını test eder.
 
    Çünkü:
        İlk eğitimde id'ler 256, 257, 258 olarak atanır.
        Eğer bu sayaç sıfırlanmazsa ikinci eğitim 259, 260, 261'den başlardı.
        Bu, vocabulary'de gereksiz dağınıklık yaratır.
 
    Doğru davranış:
        Her train() çağrısı _reset_training_state() ile başlar ve
        _next_token_id sayacı 256'ya çekilir.
 
    Beklenen:
        İkinci eğitimden sonra merge_steps[0].merged_token_id == 256
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
 
    tokenizer.train("abababa")
    tokenizer.train("xyxyxyx")
 
    # İkinci eğitim sonrası ilk merge id'si yine 256'dan başlamalı.
    assert tokenizer.merge_steps[0].merged_token_id == 256 


# ---------------------------------------------------------
# VOCAB TESTS
# ---------------------------------------------------------

def test_byte_level_bpe_tokenizer_vocab_size_grows_after_training() -> None:
    """
    Eğitim sonrası vocab_size'ın "256 + öğrenilen merge sayısı" formülüne
    uyduğunu test eder.
 
    Vocabulary iki bölümden oluşur:
        1. Base byte vocabulary -> 256 token (0-255 byte değerleri)
        2. Learned merge vocabulary -> her merge için 1 yeni token
 
    Yani:
        vocab_size = 256 + len(merge_steps)
 
    Bu invariant'ı doğrulamak önemlidir çünkü:
        - vocab_size birçok downstream sistemde model boyutunu belirler
        - Embedding tablosunun boyutu vocab_size'a göre ayarlanır
        - Yanlış vocab_size memory hatalarına veya boyut uyumsuzluklarına yol açar
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
 
    tokenizer.train("abababa")
 
    assert tokenizer.vocab_size == 256 + len(tokenizer.merge_steps) # vocab_size'ın 256 + öğrenilen merge sayısı olduğunu doğrular


def test_byte_level_bpe_tokenizer_vocab_size_at_least_256_after_training() -> None:
    """
    Hiç merge öğrenilmese bile vocab_size'ın 256'nın altına inmediğini test eder.
 
    Çünkü:
        Base byte vocabulary HER ZAMAN mevcuttur. Eğitim sırasında merge
        öğrenilmemesi bu base vocabulary'i azaltmamalıdır.
 
    Senaryo:
        train("a") -> tek karakter, hiç pair yok, hiç merge öğrenilmez.
 
    Beklenen:
        vocab_size == 256 (256 + 0 merge)
 
    Bu test, edge case'te bile vocabulary'nin tutarlı kalmasını garanti eder.
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
 
    tokenizer.train("a")  # Hiç pair yok, hiç merge öğrenilmez
 
    assert tokenizer.vocab_size == 256 # vocab_size 256'dan az olmamalıdır (base vocabulary her zaman mevcut)


# ---------------------------------------------------------
# ENCODE TESTS
# ---------------------------------------------------------

def test_byte_level_bpe_tokenizer_encode_before_training_raises_error() -> None:
    """
    train() çağrılmadan encode() çağrılırsa ValueError fırlatıldığını test eder.
 
    Çünkü:
        encode() merge_steps listesini sırayla uygular.
        Eğitilmemiş tokenizer'da bu liste boştur ama daha önemlisi `_trained`
        flag'i False'tur ve tokenizer "kullanıma hazır değil" demektir.
 
    Eğer encode() bu durumda sessizce çalışsaydı:
        - Kullanıcı tokenizer'ı eğittiğini sanır ama eğitmemiştir
        - Beklenmedik (raw byte) çıktılar üretirdi
        - Hatalar üretim aşamasında geç fark edilirdi
 
    Açık hata fırlatmak "fail fast" prensibinin uygulamasıdır.
 
    Beklenen:
        ValueError("Tokenizer has not been trained yet")
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
 
    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.encode("abababa")


def test_byte_level_bpe_tokenizer_encode_returns_list_of_integers() -> None:
    """
    encode() metodunun integer token id listesi döndürdüğünü test eder.
 
    Beklenen tip kontrolleri:
        - Çıktı list olmalı
        - Listedeki tüm elemanlar int olmalı
        - Liste boş olmamalı (input boş değilse)
 
    Neden bu kadar detaylı tip kontrolü?
        Python dynamic typed olduğu için yanlışlıkla tuple, generator veya
        string döndürmek mümkündür. Bu testler API kontratının korunduğunu
        garanti eder.
 
    Downstream sistemler (ör. embedding lookup) genellikle list[int] bekler.
    Tip uyumsuzluğu üretim ortamında runtime hatalarına yol açabilir.
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
    tokenizer.train("abababa")
 
    encoded = tokenizer.encode("abababa")
 
    assert isinstance(encoded, list) # Çıktının list olduğunu doğrular
    assert all(isinstance(token_id, int) for token_id in encoded) # Listedeki tüm elemanların int olduğunu doğrular
    assert len(encoded) > 0 # Boş olmayan bir liste döndürülmelidir (input boş değilse)


def test_byte_level_bpe_tokenizer_encode_empty_string_returns_empty_list() -> None:
    """
    Boş metin encode edilirse boş liste döndürüldüğünü test eder.
 
    Çünkü:
        train()'in aksine encode() boş input'u hata olarak kabul ETMEZ.
        Boş input -> boş output mantıklı ve idempotent bir davranıştır.
 
    Bu davranış neden farklı?
        - train() boş input ile anlamlı bir model öğrenemez (hata)
        - encode() boş input için doğal sonuç boş listedir (hata değil)
 
    Pratik fayda:
        Pipeline'larda boş string'ler doğal olarak ortaya çıkabilir.
        Her seferinde boş kontrol yapmak yerine encode("") güvenle çağrılabilir.
 
    Beklenen:
        encode("") == []
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
    tokenizer.train("abababa")
 
    assert tokenizer.encode("") == [] # Boş string encode edilirse boş liste dönmelidir


def test_byte_level_bpe_tokenizer_encode_whitespace_only_returns_empty_list() -> None:
    """
    Sadece whitespace içeren input'un boş liste döndürdüğünü test eder.
 
    Input örnekleri:
        "   " -> üç boşluk
 
    Çünkü:
        encode() implementasyonunda boş kontrolü `not text or not text.strip()`
        ile yapılır. Bu yaklaşım whitespace-only input'u boş kabul eder.
 
    Bu davranış train() ile tutarlıdır:
        - train("   ") -> ValueError
        - encode("   ") -> []
 
    İki metod da whitespace-only input'u "anlamsız içerik" olarak işler.
    Sadece tepkileri farklıdır: train hata fırlatır, encode boş döner.
 
    Beklenen:
        encode("   ") == []
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
    tokenizer.train("abababa")
 
    assert tokenizer.encode("   ") == [] # Sadece whitespace içeren input encode edilirse boş liste döndürülmelidir


def test_byte_level_bpe_tokenizer_encode_uses_learned_merges() -> None:
    """
    Eğitim sonrası encode çıktısında öğrenilen merge token id'lerinden
    en az birinin yer aldığını test eder.
 
    Çünkü:
        Eğer encode() öğrenilen merge'leri uygulamıyor olsaydı çıktı yalnızca
        0-255 arası raw byte id'lerinden oluşurdu. Bu durumda BPE algoritması
        çalışıyor ama encode bunu kullanmıyor demektir.
 
    Test stratejisi:
        Spesifik bir id (örn. 256) beklemek yerine "256+ aralığında en az bir id"
        kontrolü yapılır. Bu yaklaşım daha sağlamdır çünkü:
            - Tie-breaking sonrası hangi spesifik id'nin çıkacağı değişebilir
            - Önemli olan "merge'ler kullanılıyor mu" sorusunun cevabıdır
 
    Bu test "merge'ler train'de öğrenildi ama encode'da uygulanmıyor" gibi
    bug'ları yakalamak için kritiktir.
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
    tokenizer.train("abababa")
 
    encoded = tokenizer.encode("abababa")
 
    # En az bir merge token id'si (>= 256) çıktıda olmalı.
    # Eğer çıktı sadece byte id'leri içeriyorsa merge'ler uygulanmamış demektir.
    assert any(token_id >= 256 for token_id in encoded)


def test_byte_level_bpe_tokenizer_encode_reduces_token_count_for_repetitive_text() -> None:
    """
    Tekrarlayan pattern içeren metinde encode çıktısının raw byte sayısından
    daha kısa olduğunu test eder.
 
    Çünkü:
        BPE'nin temel amacı sıkıştırmadır. Tekrarlayan pattern'lerin tek bir
        token altında birleşmesi token sayısını azaltır.
 
    Input:
        "abababa" -> 7 byte
 
    Eğitim sonrası:
        Pattern'lerin merge edilmesi sayesinde encode çıktısı 7'den az olmalı.
 
    Eğer bu test başarısız olursa:
        - Merge'ler etkili biçimde uygulanmıyor demektir
        - BPE algoritması byte sayısını koruyor (sıkıştırma yok)
        - Tokenizer ekonomik olarak kullanışsız hale gelir
 
    Beklenen:
        len(encoded) < len(text.encode("utf-8"))
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
    text = "abababa"
    tokenizer.train(text)
 
    encoded = tokenizer.encode(text)

    assert len(encoded) < len(text.encode("utf-8")) # Encode çıktısı raw byte sayısından daha kısa olmalıdır


def test_byte_level_bpe_tokenizer_encode_handles_unseen_characters() -> None:
    """
    Eğitimde görülmemiş karakterlerin de encode edilebildiğini test eder.
 
    Senaryo:
        Eğitim metni: "abababa"  -> sadece 'a' ve 'b'
        Encode metni: "xyz"      -> hiç görülmemiş karakterler
 
    Beklenen davranış:
        Encode hata fırlatmamalı. Çünkü base byte vocabulary tüm 256 byte'ı
        kapsar. 'x', 'y', 'z' karakterleri ASCII olduğu için zaten base
        vocabulary'de mevcuttur.
 
    Bu, byte-level BPE'nin EN GÜÇLÜ özelliğidir:
        - Word-level tokenizer: "xyz" görülmediyse OOV hatası verir
        - Char-level tokenizer: 'x' görülmediyse OOV hatası verir
        - Byte-level tokenizer: tüm UTF-8 karakterleri otomatik destekler
 
    Bu özellik özellikle şu durumlarda kritiktir:
        - Çok dilli korpuslar
        - Emoji ve özel sembol içeren input'lar
        - Eğitim setinde olmayan teknik terimler
 
    Beklenen:
        encode("xyz") == [ord("x"), ord("y"), ord("z")]
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
    tokenizer.train("abababa")  # x, y, z hiç görülmedi
 
    encoded = tokenizer.encode("xyz")
 
    # Hata fırlatmamalı ve her byte kendi id'sine map edilmiş olmalı.
    assert encoded == [ord("x"), ord("y"), ord("z")]


# ---------------------------------------------------------
# DECODE TESTS
# ---------------------------------------------------------

def test_byte_level_bpe_tokenizer_decode_before_training_raises_error() -> None:
    """
    train() çağrılmadan decode() çağrılırsa ValueError fırlatıldığını test eder.
 
    Çünkü:
        Decode işlemi token id -> byte mapping'ine ihtiyaç duyar. Bu mapping
        train() sonrası tam olarak hazırdır (base + learned merges).
 
    encode() ile aynı sebepten ötürü decode() de eğitilmemiş tokenizer'da
    açık hata fırlatır:
        - Kullanıcı yanlış kullanımı erken fark eder
        - "fail fast" prensibi
        - API kontratının tutarlılığı
 
    Beklenen:
        ValueError("Tokenizer has not been trained yet")
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
 
    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.decode([0, 1, 2])


def test_byte_level_bpe_tokenizer_decode_returns_string() -> None:
    """
    decode() metodunun string döndürdüğünü test eder.
 
    Çünkü:
        decode()'un sözleşmesi "id listesini metne dönüştürmektir".
        Çıktı tipi mutlaka str olmalıdır.
 
    Eğer yanlışlıkla bytes veya list[str] döndürülürse:
        - Downstream string operasyonları başarısız olur
        - Concatenation, regex, print gibi işlemler beklenmedik davranır
 
    Bu basit tip kontrolü API kontratının en temel garantisidir.
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
    tokenizer.train("abababa")
 
    decoded = tokenizer.decode(tokenizer.encode("abababa"))
 
    assert isinstance(decoded, str) # Decode çıktısının string olduğunu doğrular


def test_byte_level_bpe_tokenizer_decode_empty_list_returns_empty_string() -> None:
    """
    Boş id listesi decode edilirse boş string döndürdüğünü test eder.
 
    Çünkü:
        Boş id listesi -> boş byte sequence -> boş string.
        Bu doğal ve idempotent bir davranıştır.
 
    encode("") == [] testi ile birlikte düşünüldüğünde:
        decode(encode("")) == ""
    yani boş input için roundtrip lossless çalışır.
 
    Pratik fayda:
        Edge case'lerde özel kontrol gerektirmez. decode([]) güvenle çağrılır.
 
    Beklenen:
        decode([]) == ""
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
    tokenizer.train("abababa")
 
    assert tokenizer.decode([]) == "" # Boş id listesi decode edilirse boş string döndürülmelidir


def test_byte_level_bpe_tokenizer_decode_unknown_token_id_raises_error() -> None:
    """
    Vocabulary'de olmayan bir token id decode edilirse ValueError
    fırlatıldığını test eder.
 
    Senaryo:
        decode([99999]) -> 99999 ne base byte (0-255) ne de learned merge
 
    Çünkü:
        Bilinmeyen id'yi sessizce yok saymak veya rastgele byte döndürmek
        sessiz bug'lara yol açar. Açık hata fırlatmak gerekli.
 
    Bu durum genellikle şu hatalardan kaynaklanır:
        - Yanlış tokenizer ile eğitilmiş id'ler kullanılması
        - Tokenizer versiyon uyuşmazlığı
        - Manuel olarak yanlış id'lerin oluşturulması
 
    Beklenen:
        ValueError("Unknown token id")
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
    tokenizer.train("abababa")
 
    with pytest.raises(ValueError, match="Unknown token id"):
        tokenizer.decode([99999])


def test_byte_level_bpe_tokenizer_decode_invalid_utf8_sequence_raises_error() -> None:
    """
    Token id'lerin geçerli UTF-8 byte dizisi oluşturmadığı durumlarda
    anlaşılır bir ValueError fırlatıldığını test eder.
 
    Senaryo:
        decode([195]) -> 0xC3 byte'ı
 
    Çünkü:
        UTF-8'de 0xC3 (195) tek başına geçerli bir karakter değildir.
        2-byte sequence'in başlangıcıdır ve 0x80-0xBF aralığında bir
        ikinci byte ile takip edilmelidir.
 
    Eğer decode bu durumu yakalayıp UnicodeDecodeError'ı olduğu gibi
    fırlatsaydı kullanıcıya teknik bir hata mesajı verilirdi.
    Implementation bunu daha açıklayıcı bir ValueError'a sarar:
        "Token ids do not form a valid UTF-8 byte sequence"
 
    Bu yaklaşım:
        - Hata mesajını anlaşılır kılar
        - Tokenizer-level hatalar tutarlı tipte (ValueError) olur
        - Kullanıcı sorunun kaynağını daha iyi anlar
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
    tokenizer.train("abababa")
 
    with pytest.raises(ValueError, match="valid UTF-8 byte sequence"):
        tokenizer.decode([195]) # 0xC3 tek başına geçerli bir UTF-8 karakteri oluşturmaz


# ---------------------------------------------------------
# ROUNDTRIP TESTS
# ---------------------------------------------------------

def test_byte_level_bpe_tokenizer_roundtrip_simple_ascii() -> None:
    """
    Basit ASCII metin için encode -> decode'un orijinal metni geri verdiğini
    test eder.
 
    Roundtrip property:
        decode(encode(text)) == text
 
    Bu, tokenizer'ın "lossless" olduğunu ifade eden temel property'dir.
 
    Çünkü:
        Bir tokenizer'ın temel görevi metni id'lere çevirip geri döndürmektir.
        Bu işlem sırasında bilgi kaybı olursa tokenizer kullanışsız olur.
 
    Byte-level BPE bu property'i garanti eder çünkü:
        - Her byte tekrar geri dönülebilen bir id'ye map edilir
        - Merge'ler de byte sequence'leri saklar
        - Decode aşamasında byte parçaları join edilip UTF-8 decode edilir
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
    text = "abababa"
    tokenizer.train(text)
 
    assert tokenizer.decode(tokenizer.encode(text)) == text # Roundtrip encode -> decode orijinal metni geri vermelidir


def test_byte_level_bpe_tokenizer_roundtrip_with_turkish_characters() -> None:
    """
    Türkçe karakterler (multi-byte UTF-8) için roundtrip'in lossless
    çalıştığını test eder.
 
    Input:
        "çğüşöıİ"
 
    Türkçe karakterlerin UTF-8 temsili:
        ç -> 2 byte (0xC3 0xA7)
        ğ -> 2 byte (0xC4 0x9F)
        ü -> 2 byte (0xC3 0xBC)
        ...
 
    Çünkü:
        ASCII tokenizer'ları (ör. char-level basit implementasyonlar)
        Türkçe karakterleri sıklıkla bozar. Byte-level yaklaşım her byte'ı
        ayrı ayrı işlediği için multi-byte karakterler doğal olarak korunur.
 
    Bu test, byte-level mimarisinin gerçekten Unicode-safe olduğunu doğrular.
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=5)
    text = "çğüşöıİ"
    tokenizer.train(text)
 
    assert tokenizer.decode(tokenizer.encode(text)) == text # Roundtrip encode -> decode Türkçe karakterler için de orijinal metni geri vermelidir


def test_byte_level_bpe_tokenizer_roundtrip_with_emoji() -> None:
    """
    4-byte UTF-8 sequence olan emoji için roundtrip'in çalıştığını test eder.
 
    Input:
        "merhaba 😊 dünya"
 
    😊 emoji'sinin UTF-8 temsili:
        4 byte (0xF0 0x9F 0x98 0x8A)
 
    Çünkü:
        Emoji'ler ASCII'nin çok ötesinde 4-byte sequence'ler kullanır.
        Bu, tokenizer'ın multi-byte handling'ini en uç noktada test eder.
 
    Eğer tokenizer:
        - Sadece 1-byte (ASCII) destekliyorsa: emoji bozulur
        - Sadece 2-byte destekliyorsa: yine bozulur
        - Tam byte-level çalışıyorsa: emoji korunur
 
    Bu test byte-level yaklaşımın en güçlü yanını - dilsel sınırlardan
    bağımsız olma özelliğini - doğrular.
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=5)
    text = "merhaba 😊 dünya"
    tokenizer.train(text)
 
    assert tokenizer.decode(tokenizer.encode(text)) == text # Roundtrip encode -> decode emoji içeren metin için de orijinal metni geri vermelidir


def test_byte_level_bpe_tokenizer_roundtrip_with_mixed_content() -> None:
    """
    Sayı, harf ve noktalama içeren karma metin için roundtrip'in lossless
    çalıştığını test eder.
 
    Input:
        "Hello, world! 123 ABC."
 
    Bu metin içerir:
        - Büyük/küçük harfler
        - Virgül ve ünlem (noktalama)
        - Sayılar
        - Boşluklar ve nokta
 
    Çünkü:
        Gerçek dünya metinleri saf alfabetik değildir. Tokenizer karma
        içerikte de doğru çalışmalıdır.
 
    Eğer tokenizer noktalama veya whitespace bilgisini kaybediyorsa:
        - "Hello, world!" -> "Hello world" (virgül kaybı)
        - "1 2 3" -> "123" (boşluk kaybı)
 
    Byte-level yaklaşım bu sorunu yaşamaz çünkü her karakter (boşluk dahil)
    bir veya daha fazla byte olarak korunur.
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=5)
    text = "Hello, world! 123 ABC."
    tokenizer.train(text)
 
    assert tokenizer.decode(tokenizer.encode(text)) == text # Roundtrip encode -> decode karma metin için de orijinal metni geri vermelidir


def test_byte_level_bpe_tokenizer_roundtrip_for_unseen_text() -> None:
    """
    Eğitimde görülmemiş bir metin için de roundtrip'in lossless çalıştığını
    test eder.
 
    Senaryo:
        Eğitim metni: "abababa"
        Roundtrip metni: "completely different text 🚀"
 
    Çünkü:
        Byte-level BPE'nin temel garantisi: TÜM UTF-8 metinleri lossless
        encode/decode edebilir. Eğitim seti sınırlı olsa bile.
 
    Bu özellik şu sebepten önemli:
        - Üretim ortamında inference metni eğitim setinden farklıdır
        - Yeni terimler, isimler, emoji'ler sürekli ortaya çıkar
        - Tokenizer hiçbir input'u "anlamadığı için" reddedemez
 
    Test stratejisi:
        Eğitim metnini tamamen farklı tutarak (sadece 'a' ve 'b'),
        encode/decode'un base byte vocabulary üzerinden çalıştığını ve
        görülmemiş karakterlerle bile lossless olduğunu doğrular.
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
    tokenizer.train("abababa")
 
    text = "completely different text 🚀"
    assert tokenizer.decode(tokenizer.encode(text)) == text # Roundtrip encode -> decode eğitimde görülmemiş metin için de orijinal metni geri vermelidir


# ---------------------------------------------------------
# TOKENIZE TESTS
# ---------------------------------------------------------

def test_byte_level_bpe_tokenizer_tokenize_before_training_raises_error() -> None:
    """
    train() çağrılmadan tokenize() çağrılırsa ValueError fırlatıldığını test eder.
 
    Çünkü:
        tokenize() metodu içinde encode() çağırır ve sonra her id'yi
        byte karşılığına çevirir. Eğitilmemiş tokenizer'da bu zincir kırılır.
 
    encode() ve decode() ile aynı koruma mantığı:
        - Tokenizer hazır değilken tokenize çağrısı bilinçli bir hatadır
        - Sessiz başarı yerine açık hata fırlatılır
 
    Beklenen:
        ValueError("Tokenizer has not been trained yet")
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
 
    with pytest.raises(ValueError, match="Tokenizer has not been trained yet"):
        tokenizer.tokenize("abababa")


def test_byte_level_bpe_tokenizer_tokenize_returns_list_of_strings() -> None:
    """
    tokenize() metodunun string token listesi döndürdüğünü test eder.
 
    Tip kontrolleri:
        - Çıktı list olmalı
        - Listedeki tüm elemanlar str olmalı
        - Liste boş olmamalı (anlamlı input için)
 
    Çünkü:
        tokenize() insan-okunabilir tokenları döndürür. encode() integer id
        listesi döndürürken tokenize() string listesi döndürür.
 
    Bu ayrım:
        - Debug ve raporlama için tokenize() kullanılır
        - Model girdisi için encode() kullanılır
 
    Bu test iki API'nin de kendi kontratına uyduğunu doğrular.
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
    tokenizer.train("abababa")
 
    tokens = tokenizer.tokenize("abababa")
 
    assert isinstance(tokens, list) # Çıktının list olduğunu doğrular
    assert all(isinstance(token, str) for token in tokens) # Tüm elemanların string olduğunu doğrular
    assert len(tokens) > 0 # Anlamlı input için tokenize boş liste döndürmemelidir


def test_byte_level_bpe_tokenizer_tokenize_empty_string_returns_empty_list() -> None:
    """
    Boş input için tokenize'ın boş liste döndürdüğünü test eder.
 
    Çünkü:
        encode("") == [] olduğu için tokenize("") da [] olmalıdır.
        İki metod aynı boş input davranışını sergilemelidir.
 
    Bu tutarlılık önemlidir çünkü:
        - len(encode(text)) == len(tokenize(text)) invariant'ı korunur
        - Edge case handling kullanıcı için tahmin edilebilir kalır
 
    Beklenen:
        tokenize("") == []
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
    tokenizer.train("abababa")
 
    assert tokenizer.tokenize("") == [] # Boş string tokenize edilirse boş liste döndürülmelidir


def test_byte_level_bpe_tokenizer_tokenize_count_matches_encode_count() -> None:
    """
    tokenize() ve encode() çıktılarının aynı sayıda token ürettiğini test eder.
 
    Çünkü:
        Implementation'da tokenize() içeride encode() çağırır ve sonra her id'yi
        kendi byte temsiline çevirir. Bu yüzden iki listenin uzunluğu eşit olmalı.
 
    Eğer uzunluklar farklıysa:
        - tokenize() ek olarak split veya merge yapıyor demektir
        - İki API'nin "neyi sayıyor" anlamı farklılaşır
        - Token sayısı raporları yanıltıcı olur
 
    Bu invariant özellikle metric/raporlama kodları için kritiktir.
 
    Beklenen:
        len(tokenize(text)) == len(encode(text))
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
    text = "abababa"
    tokenizer.train(text)
 
    assert len(tokenizer.tokenize(text)) == len(tokenizer.encode(text)) # tokenize() ve encode() çıktılarının aynı sayıda token ürettiğini doğrular


def test_byte_level_bpe_tokenizer_tokenize_pieces_concatenate_to_original_text() -> None:
    """
    Tokenize çıktısındaki parçaların birleştirilmesinin orijinal metni
    verdiğini test eder.
 
    Property:
        "".join(tokenize(text)) == text
 
    Çünkü:
        Byte-level BPE tokenları orijinal byte sequence'lerini saklar.
        UTF-8 olarak decode edilebilir oldukları sürece string concatenation
        orijinal metni geri vermelidir.
 
    Bu davranış neden önemli?
        - Token sınırlarının görselleştirilmesinde kullanılır
        - "Bu kelime kaç token?" gibi soruların cevaplanmasını sağlar
        - tokenize() çıktısının lossless olduğunu garanti eder
 
    Senaryo örneği:
        Eğitim sonrası tokens = ['a', 'bababa']
        "".join(tokens) == "abababa" -> orijinal metin
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
    text = "abababa"
    tokenizer.train(text)
 
    tokens = tokenizer.tokenize(text)
 
    assert "".join(tokens) == text # Tokenize çıktısındaki parçaların birleştirilmesi orijinal metni vermelidir
 
 
def test_byte_level_bpe_tokenizer_tokenize_produces_multi_byte_tokens_after_training() -> None:
    """
    Eğitim sonrası tokenize çıktısında en az birinin birden fazla karakter
    içeren parçalardan oluştuğunu test eder.
 
    Çünkü:
        Eğitim sonrası BPE merge'ler oluşmuş demektir. Bu merge'lerin pratik
        karşılığı: bazı tokenlar artık tek byte değil, birleştirilmiş
        byte grupları olur.
 
    Eğer hiçbir token tek karakterden uzun değilse:
        - Merge'ler öğrenilmiş ama uygulanmıyor demektir
        - Encode/tokenize merge'leri kullanmıyor olabilir
        - BPE'nin sıkıştırma faydası kaybolmuş olur
 
    Senaryo:
        train("abababa") sonrası tokens = ['a', 'bababa']
        'bababa' tek karakterden uzun -> merge'ler doğru uygulanmış.
 
    Beklenen:
        tokens listesinde en az bir element için len(token) > 1
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
    tokenizer.train("abababa")
 
    tokens = tokenizer.tokenize("abababa")
 
    assert any(len(token) > 1 for token in tokens) # Tokenize çıktısında en az bir token tek karakterden uzun olmalıdır (merge'ler uygulanmış olmalı)
 
 
# ---------------------------------------------------------
# DETERMINISM TESTS
# ---------------------------------------------------------
 
def test_byte_level_bpe_tokenizer_is_deterministic_across_instances() -> None:
    """
    Aynı input ile eğitilen iki ayrı tokenizer instance'ının aynı sonuçları
    ürettiğini test eder.
 
    Determinism property:
        Aynı input + aynı algoritma -> aynı output
 
    Çünkü:
        BPE eğitimi rastgele bir süreç DEĞİLDİR. Her adımda en sık pair seçilir
        ve tie-breaking kuralı deterministiktir. Bu yüzden aynı input iki
        farklı instance'ta aynı merge sırasını üretmelidir.
 
    Eğer determinism bozulursa:
        - Test sonuçları reprodüksiyonu zorlaşır
        - Aynı kullanıcı iki kere train edip farklı sonuç alır
        - Production debug süreci içinden çıkılmaz hale gelir
 
    Test stratejisi:
        İki instance ayrı ayrı eğitilir, sonra:
            - merge_steps listeleri karşılaştırılır
            - encode çıktıları karşılaştırılır
 
    İki ayrı seviyede determinism doğrulanır: training ve inference.
    """
    tokenizer_a = TokenizerFactory.create("byte_level_bpe", num_merges=3)
    tokenizer_b = TokenizerFactory.create("byte_level_bpe", num_merges=3)
 
    tokenizer_a.train("abababa")
    tokenizer_b.train("abababa")
 
    assert tokenizer_a.merge_steps == tokenizer_b.merge_steps # Merge adımları aynı olmalıdır
    assert tokenizer_a.encode("abababa") == tokenizer_b.encode("abababa") # Encode çıktıları aynı olmalıdır
 
 
def test_byte_level_bpe_tokenizer_encode_is_deterministic_for_repeated_calls() -> None:
    """
    Aynı tokenizer üzerinde encode() defalarca çağrıldığında aynı sonucu
    döndürdüğünü test eder.
 
    Çünkü:
        encode() pure function gibi davranmalıdır:
            - Aynı input -> aynı output
            - Side effect yok
            - Internal state'i değiştirmiyor
 
    Eğer encode() çağrıları farklı sonuç dönerse:
        - Tokenizer'da gizli bir state mutation vardır
        - Caching mekanizması bug'lı olabilir
        - Concurrent kullanımlarda yarış (race) durumları oluşur
 
    Bu test "encode() idempotent ve referentially transparent" özelliğini doğrular.
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
    tokenizer.train("abababa")
 
    first_call = tokenizer.encode("abababa")
    second_call = tokenizer.encode("abababa")
 
    assert first_call == second_call # Aynı tokenizer üzerinde encode() defalarca çağrıldığında aynı sonucu döndürmelidir
 
 
def test_byte_level_bpe_tokenizer_tie_breaking_is_deterministic() -> None:
    """
    Aynı frekansta birden fazla pair olduğunda seçimin deterministik olduğunu test eder.
 
    Senaryo:
        text = "abcd"
 
        Pair frekansları:
            (97, 98) -> 1   # ('a', 'b')
            (98, 99) -> 1   # ('b', 'c')
            (99, 100) -> 1  # ('c', 'd')
 
        Tüm pair'ler aynı frekansta (1) -> tie
 
    Tie-breaking kuralı:
        max() ile (frequency, pair) karşılaştırılır.
        En büyük pair seçilir: (99, 100) -> ('c', 'd')
 
    Çünkü:
        Tie durumlarında rastgele seçim yapılırsa her train çağrısı farklı
        sonuç verebilir. Bu, tokenizer'ı non-deterministic hale getirir.
 
    Bu test determinism'in tie durumunda da korunduğunu doğrular.
    İki ayrı instance aynı tie scenario'sunda aynı seçimi yapmalıdır.
    """
    tokenizer_a = TokenizerFactory.create("byte_level_bpe", num_merges=1)
    tokenizer_b = TokenizerFactory.create("byte_level_bpe", num_merges=1)
 
    tokenizer_a.train("abcd")
    tokenizer_b.train("abcd")
 
    # Tie durumunda bile iki instance aynı sonucu üretmeli.
    assert tokenizer_a.merge_steps == tokenizer_b.merge_steps
 
 
# ---------------------------------------------------------
# MERGE STEP DATACLASS TESTS
# ---------------------------------------------------------
 
def test_byte_level_bpe_merge_dataclass_is_frozen() -> None:
    """
    ByteLevelBPEMerge dataclass'ının frozen olduğunu test eder.
 
    Çünkü:
        ByteLevelBPEMerge `@dataclass(frozen=True)` decorator ile tanımlanmıştır.
        Bu, instance oluşturulduktan sonra alanlarının değiştirilemeyeceği
        anlamına gelir.
 
    Frozen olması neden önemli?
        - Merge kuralları öğrenildikten sonra değişmemelidir
        - Determinism garantisinin bir parçasıdır
        - Yanlışlıkla mutation yapılması engellenir
 
    Test stratejisi:
        Bir merge instance'ı oluşturulur, sonra alanı değiştirilmeye çalışılır.
        FrozenInstanceError veya benzer bir Exception fırlatılmalıdır.
 
    Beklenen:
        Attribute değiştirme denemesi Exception fırlatır.
    """
    merge = ByteLevelBPEMerge(pair=(97, 98), merged_token_id=256, frequency=3)
 
    # Frozen dataclass'ta attribute değiştirme FrozenInstanceError fırlatır.
    with pytest.raises(Exception):
        merge.frequency = 999  # type: ignore[misc]
 
 
def test_byte_level_bpe_merge_dataclass_equality() -> None:
    """
    Aynı field değerlerine sahip iki ByteLevelBPEMerge instance'ının eşit
    kabul edildiğini test eder.
 
    Çünkü:
        @dataclass otomatik olarak __eq__ metodu üretir. Bu, iki dataclass
        instance'ının field-by-field karşılaştırılmasını sağlar.
 
    Bu test neden önemli?
        Determinism testlerinde iki tokenizer'ın merge_steps listeleri
        karşılaştırılır:
            tokenizer_a.merge_steps == tokenizer_b.merge_steps
 
        Bu karşılaştırma ancak ByteLevelBPEMerge eşitliği doğru çalışırsa
        anlamlıdır. Eğer eşitlik identity bazlıysa (default object eşitliği)
        iki merge instance'ı asla eşit olmaz ve determinism testleri
        yanlışlıkla başarılı olur.
 
    Beklenen:
        Aynı field'lara sahip iki instance birbirine == olarak eşit.
    """
    merge_a = ByteLevelBPEMerge(pair=(97, 98), merged_token_id=256, frequency=3)
    merge_b = ByteLevelBPEMerge(pair=(97, 98), merged_token_id=256, frequency=3)
 
    assert merge_a == merge_b # İki instance aynı field değerlerine sahip olduğu için eşit kabul edilmelidir
 
 
# ---------------------------------------------------------
# INTERNAL HELPERS TESTS
# ---------------------------------------------------------
 
def test_get_pair_frequencies_counts_adjacent_pairs() -> None:
    """
    _get_pair_frequencies() metodunun yan yana gelen pair'leri doğru saydığını test eder.
 
    Input:
        token_ids = [97, 98, 97, 98]
 
    Yan yana gelen pair'ler:
        (97, 98) -> 1. ve 2. token
        (98, 97) -> 2. ve 3. token
        (97, 98) -> 3. ve 4. token
 
    Frekanslar:
        (97, 98) -> 2
        (98, 97) -> 1
 
    Çünkü:
        Bu metod BPE algoritmasının kalbidir. Doğru çalışmazsa:
            - Yanlış pair'ler merge edilir
            - Sıkıştırma optimal olmaz
            - Sonraki tüm encode/decode işlemleri etkilenir
 
    Test ayrıca dönüş tipinin Counter olduğunu da doğrular.
    Counter, dict'ten farklı olarak missing key'ler için 0 döner ve
    BPE algoritmasında bu davranış kullanışlıdır.
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
 
    frequencies = tokenizer._get_pair_frequencies([97, 98, 97, 98])
 
    assert isinstance(frequencies, Counter) # Dönüş tipi Counter olmalıdır
    assert frequencies[(97, 98)] == 2 # (97, 98) pair'inin frekansı 2 olmalıdır
    assert frequencies[(98, 97)] == 1 # (98, 97) pair'inin frekansı 1 olmalıdır
 
 
def test_get_pair_frequencies_returns_empty_for_short_input() -> None:
    """
    Tek elemanlı veya boş listede _get_pair_frequencies()'in boş Counter
    döndürdüğünü test eder.
 
    Input örnekleri:
        []      -> hiç token yok, hiç pair yok
        [97]    -> tek token, hiç pair yok
 
    Çünkü:
        Pair en az 2 token gerektirir. Daha az token varsa pair oluşmaz.
 
    Bu davranış neden önemli?
        train() metodu bu Counter'ı kontrol ederek "merge edilecek pair var mı?"
        kararını verir. Boş Counter -> erken break.
 
        Eğer bu metod boş input'ta hata fırlatsaydı train() de hata fırlatırdı
        ve patolojik input'lar yönetilemez hale gelirdi.
 
    Beklenen:
        len(_get_pair_frequencies([])) == 0
        len(_get_pair_frequencies([97])) == 0
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
 
    assert len(tokenizer._get_pair_frequencies([])) == 0 # Boş input için Counter boş olmalıdır
    assert len(tokenizer._get_pair_frequencies([97])) == 0 # Tek elemanlı input için Counter boş olmalıdır
 
 
def test_merge_pair_replaces_matching_pairs_left_to_right() -> None:
    """
    _merge_pair() metodunun eşleşen pair'leri yeni id ile değiştirdiğini test eder.
 
    Input:
        token_ids = [1, 2, 1, 2, 3]
        pair = (1, 2)
        new_token_id = 256
 
    Beklenen output:
        [256, 256, 3]
 
    Soldan sağa tarama:
        i=0: (1, 2) match -> 256 ekle, i=2
        i=2: (1, 2) match -> 256 ekle, i=4
        i=4: tek token (3), match yok -> 3 ekle, i=5
        Sonuç: [256, 256, 3]
 
    Çünkü:
        Bu metod BPE'nin "merge uygulama" adımıdır. Aşağıdakiler garanti edilmeli:
            - Tüm match'ler bulunmalı
            - Match olmayan tokenlar değişmemeli
            - Sıralama korunmalı
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
 
    result = tokenizer._merge_pair([1, 2, 1, 2, 3], pair=(1, 2), new_token_id=256)
 
    assert result == [256, 256, 3] # Eşleşen pair'ler yeni id ile değiştirilmiş ve diğer tokenlar korunmuş olmalıdır
 
 
def test_merge_pair_does_not_overlap_same_token() -> None:
    """
    _merge_pair()'in non-overlapping davrandığını test eder.
 
    Input:
        token_ids = [1, 1, 1]
        pair = (1, 1)
        new_token_id = 256
 
    Beklenen output:
        [256, 1]
 
    Çünkü:
        BPE'de aynı token aynı merge adımında birden fazla pair'e dahil edilemez.
 
    Tarama detayı:
        i=0: (1, 1) match -> 256 ekle, i=2  (ortadaki token "tüketildi")
        i=2: tek token (1), match yok -> 1 ekle, i=3
        Sonuç: [256, 1]
 
    Eğer overlapping davranış olsaydı:
        - i=0: (1,1) match -> 256, i=1 (sadece bir adım atla)
        - i=1: (1,1) match -> 256, i=2
        - Sonuç: [256, 256] -> TEK BIR token üç kere kullanılmış olurdu
 
    Bu, BPE'nin standart davranışıdır ve deterministik kalmasını sağlar.
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
 
    result = tokenizer._merge_pair([1, 1, 1], pair=(1, 1), new_token_id=256)
 
    assert result == [256, 1] # Non-overlapping davranışla sadece ilk iki token merge edilmeli, üçüncü token korunmalıdır
 
 
def test_merge_pair_returns_unchanged_list_when_pair_not_found() -> None:
    """
    Pair listede hiç yoksa _merge_pair()'in orijinal listeyi değiştirmeden
    döndürdüğünü test eder.
 
    Input:
        token_ids = [1, 2, 3]
        pair = (9, 9)        -> listede hiç yok
        new_token_id = 256
 
    Beklenen output:
        [1, 2, 3]            -> hiçbir değişiklik
 
    Çünkü:
        Match olmayan tokenlar olduğu gibi output'a kopyalanır.
        Eğer hiç match yoksa output input'un kopyasıdır.
 
    Bu davranış neden önemli?
        train() döngüsünde bazı iterasyonlarda hedef pair listede olmayabilir.
        Bu durumda _merge_pair() sessizce listeyi geri döndürmelidir;
        hata fırlatmamalıdır.
 
    Beklenen:
        result == [1, 2, 3]
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
 
    result = tokenizer._merge_pair([1, 2, 3], pair=(9, 9), new_token_id=256)
 
    assert result == [1, 2, 3] # Pair listede hiç yoksa orijinal liste değiştirilmeden döndürülmelidir
 
 
# ---------------------------------------------------------
# REGISTRY / FACTORY TESTS
# ---------------------------------------------------------
 
def test_byte_level_bpe_tokenizer_is_registered_under_correct_name() -> None:
    """
    TokenizerFactory.create("byte_level_bpe", ...) çağrısının doğru sınıfı
    döndürdüğünü test eder.
 
    Çünkü:
        ByteLevelBPETokenizer @register_tokenizer("byte_level_bpe") decorator
        ile registry'ye kaydedilir. Bu kayıt yanlış olursa:
            - Factory yanlış sınıfı döndürür
            - "byte_level_bpe" ismi farklı bir tokenizer'a işaret eder
            - API/CLI seviyesinde tokenizer seçimi bozulur
 
    Doğrulanan iki property:
        - Üretilen instance ByteLevelBPETokenizer tipinde olmalı
        - tokenizer.name == "byte_level_bpe" olmalı (BaseTokenizer'dan)
 
    Bu test, registry mekanizmasının ucudan uca çalıştığını garanti eder.
    """
    tokenizer = TokenizerFactory.create("byte_level_bpe", num_merges=3)
 
    assert isinstance(tokenizer, ByteLevelBPETokenizer) # Üretilen instance ByteLevelBPETokenizer tipinde olmalıdır
    assert tokenizer.name == "byte_level_bpe" # Tokenizer'ın name attribute'u "byte_level_bpe" olmalıdır