from __future__ import annotations

import pytest

from tokenizer_workshop.api.services.tokenizer_factory import TokenizerFactory

SAMPLE_TEXT = (
    "salam dünya salam dünya "
    "bu bir test metnidir "
    "salam salam salam"
)

VOCAB_SIZE = 270

# Her test için önceden eğitilmiş tokenizer sağlar
# Bu sayede her testte tekrar train() yazmaya gerek kalmaz
@pytest.fixture
def trained_tokenizer():
    # Yeni boş tokenizer oluştur
    t = TokenizerFactory.create("regex_bpe")
    
    # SAMPLE_TEXT ile eğit, VOCAB_SIZE=270 token öğren
    t.train(SAMPLE_TEXT, VOCAB_SIZE)
    
    # Eğitilmiş tokenizeri teste gönder
    return t

# --- Train Testleri ---

# vocab_size 256'dan küçük olunca ValueError fırlatıldığını test eder
def test_train_vocab_size_minimum_256() -> None:
    # Yeni boş tokenizer oluştur
    t = TokenizerFactory.create("regex_bpe")
    
    # Bu blok içinde ValueError bekliyoruz
    # Gelmezse → test başarısız 
    # Gelirse  → test başarılı 
    with pytest.raises(ValueError):
        # vocab_size=100 → 256'dan küçük
        # train() ValueError fırlatmalıdır
        t.train(SAMPLE_TEXT, vocab_size=100)

# Train sonrası vocab'ın dolu olduğunu test eder
def test_train_builds_vocab(trained_tokenizer) -> None:
    # trained_tokenizer → fixture'den gelir, zaten eğitilmiş
    # len(vocab) → kaç token var?
    # > 0 → boş olmamalıdır
    # Train sonrası en az 256 token olmalıdır
    assert len(trained_tokenizer.vocab) > 0


# Train sonrası en az 1 birleştirme kuralı öğrenildiğini test eder
def test_train_learns_merges(trained_tokenizer) -> None:
    # get_merges_count() → kaç kural öğrenildi?
    # merges = {(115,97):256, (256,108):257} → 2 kural
    # >= 1 → en az 1 kural olmalı
    assert trained_tokenizer.get_merges_count() >= 1


# Train sonrası vocab boyutunun VOCAB_SIZE'dan büyük olmadığını test eder
def test_train_vocab_size_correct(trained_tokenizer) -> None:
    # get_vocab_size() → kaç token var?
    # vocab = {0:b'a', ..., 255:b'xff', 256:b'sa', ...}
    # VOCAB_SIZE = 270
    # <= VOCAB_SIZE → 270'den büyük olmamalıdır
    # 258 <= 270 → True  test geçti
    # 300 <= 270 → False test başarısız
    assert trained_tokenizer.get_vocab_size() <= VOCAB_SIZE

# --- Encode Testleri ---

# encode() metodunun list döndürdüğünü test eder
def test_encode_returns_list(trained_tokenizer) -> None:
    # encode("salam") çağır
    # result = [258, 109] gibi bir liste olmalıdır
    result = trained_tokenizer.encode("salam")
    
    # isinstance → "result bir list midir?" kontrol eder
    # isinstance([258,109], list) → True  
    # isinstance("258 109", list) → False 
    assert isinstance(result, list)

# encode() sonucundaki tüm elementlerin tam sayı olduğunu test eder
def test_encode_returns_integers(trained_tokenizer) -> None:
    # encode("salam") → [258, 109] gibi bir liste döndürür
    result = trained_tokenizer.encode("salam")
    
    # for i in result → her elementi tek tek kontrol et
    # isinstance(i, int) → "bu tam sayı mı?"
    # i=258 → isinstance(258, int) → True  
    # i=109 → isinstance(109, int) → True  
    # all() → hepsi True mı?
    # all([True, True])  → True  test geçti
    # all([True, False]) → False test başarısız
    assert all(isinstance(i, int) for i in result)

def test_encode_nonempty_text(trained_tokenizer) -> None:
    result = trained_tokenizer.encode("salam")
    assert len(result) > 0

# Boş metin gelince encode() boş liste döndürdüğünü test eder
def test_encode_empty_string(trained_tokenizer) -> None:
    # Boş metin veriyoruz → ""
    # pre_tokenize("") → [] ← regex hiçbir şey bulamadı
    # all_ids = []     ← hiçbir şey eklenmedi
    result = trained_tokenizer.encode("")
    
    # result boş liste olmalıdır
    # [] == [] → True test geçti
    # [32] == [] → False test başarısız
    assert result == []


# Train edilmeden encode() çağrılırsa RuntimeError fırlatıldığını test eder
def test_encode_without_train_raises() -> None:
    # Yeni boş tokenizer oluştur — train() çağrılmadı!
    # vocab  = {} ← boş
    # merges = {} ← boş
    t = TokenizerFactory.create("regex_bpe")
    
    # Bu blok içinde RuntimeError bekliyoruz
    # Gelmezse → test başarısız 
    # Gelirse  → test başarılı 
    with pytest.raises(RuntimeError):
        # vocab ve merges boş → tokenizer eğitilmemiş
        # encode() → RuntimeError fırlatmalıdır
        t.encode("salam")

# encode() sonucundaki tüm id'lerin vocab'da olduğunu test eder
def test_encode_ids_in_vocab(trained_tokenizer) -> None:
    # encode("salam") → [258, 109]
    ids = trained_tokenizer.encode("salam")
    
    # Her id'yi tek tek kontrol et
    for i in ids:
        # "bu id vocab'da var mı?"
        # 258 in vocab → True  
        # 999 in vocab → False 
        # vocab'da olmayan id → decode() çalışmaz!
        assert i in trained_tokenizer.vocab

# --- Decode Testleri ---

# decode() metodunun string döndürdüğünü test eder
def test_decode_returns_string(trained_tokenizer) -> None:
    # Önce encode et → id listesi al
    # encode("salam") → [258, 109]
    ids = trained_tokenizer.encode("salam")
    
    # Sonra decode et → string al
    # decode([258, 109]) → "salam"
    result = trained_tokenizer.decode(ids)
    
    # isinstance(result, str) → "string midir?"
    # isinstance("salam", str) → True  
    # isinstance([258,109], str) → False 
    assert isinstance(result, str)

# Train edilmeden decode() çağrılırsa RuntimeError fırlatıldığını test eder
def test_decode_without_train_raises() -> None:
    # Yeni boş tokenizer oluştur — train() çağrılmadı!
    # vocab = {} ← boş
    t = TokenizerFactory.create("regex_bpe")
    
    # Bu blok içinde RuntimeError bekliyoruz
    # Gelmezse → test başarısız 
    # Gelirse  → test başarılı 
    with pytest.raises(RuntimeError):
        # vocab boş → decode() RuntimeError fırlatmalıdır
        t.decode([65, 66])

# Boş liste gelince decode() boş string döndürdüğünü test eder
def test_decode_empty_list(trained_tokenizer) -> None:
    # Boş liste veriyoruz → []
    # byte_parts = [] ← hiçbir şey yok
    # b"".join([]) → b""
    # b"".decode() → ""
    result = trained_tokenizer.decode([])
    
    # Sonuç boş string olmalıdır
    # "" == "" → True  
    assert result == ""

    # --- Round-trip Testleri ---

# encode → decode → orijinal metin geri gelmeli (round-trip)
def test_roundtrip_simple(trained_tokenizer) -> None:
    text = "salam"
    # encode("salam") → [258, 109]
    # decode([258, 109]) → "salam"
    # "salam" == "salam" → True 
    assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text


# Cümle için encode → decode → orijinal metin
def test_roundtrip_sentence(trained_tokenizer) -> None:
    text = "salam dünya"
    # encode("salam dünya") → [258, 109, 32, ...]
    # decode([258, 109, 32, ...]) → "salam dünya"
    # "salam dünya" == "salam dünya" → True 
    assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text


# Noktalama işaretli metin için encode → decode → orijinal metin
def test_roundtrip_with_punctuation(trained_tokenizer) -> None:
    text = "salam, dünya!"
    # encode("salam, dünya!") → [258, 109, 44, ...]
    # decode([258, 109, 44, ...]) → "salam, dünya!"
    # "salam, dünya!" == "salam, dünya!" → True 
    assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text

# --- Vocab Testleri ---

# vocab'da 0-255 arası tüm byte tokenların olduğunu test eder
def test_base_vocab_has_all_bytes(trained_tokenizer) -> None:
    # range(256) → 0, 1, 2, ... 255
    # Her byte token vocab'da olmalıdır
    # 0 in vocab → True  
    # 255 in vocab → True 
    # Biri eksikse → encode/decode çalışmaz 
    for i in range(256):
        assert i in trained_tokenizer.vocab


# vocab'ın en az 1 token içerdiğini test eder
def test_get_vocab_size(trained_tokenizer) -> None:
    # get_vocab_size() → len(vocab)
    # En az 256 token olmalıdır (byte tokenları)
    # 256 > 0 → True 
    # 0 > 0   → False  vocab boş!
    assert trained_tokenizer.get_vocab_size() > 0


# merges sayısının negatif olmadığını test eder
def test_get_merges_count(trained_tokenizer) -> None:
    # get_merges_count() → len(merges)
    # 0 veya daha fazla olmalıdır
    # 0 >= 0 → True  (birleştirme olmadı ama sorun değil)
    # 5 >= 0 → True   (5 kural öğrenildi)
    # -1 >= 0 → False  (imkansız ama yine de kontrol ederiz)
    assert trained_tokenizer.get_merges_count() >= 0