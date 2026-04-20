

import pytest
from tokenizer_workshop.tokenizers.word_tokenizer import WordTokenizer


def test_vocab_size_reflects_unique_tokens():
    t = WordTokenizer()
    t.train("merhaba dünya merhaba")
    assert t.vocab_size == 2  # "merhaba" ve "dünya"


def test_encode_returns_correct_ids():
    t = WordTokenizer()
    t.train("merhaba dünya")
    ids = t.encode("merhaba dünya")
    assert isinstance(ids, list)
    assert len(ids) == 2


def test_decode_returns_correct_text():
    t = WordTokenizer()
    t.train("merhaba dünya")
    ids = t.encode("merhaba dünya")
    result = t.decode(ids)
    assert "merhaba" in result
    assert "dünya" in result


def test_encode_decode_roundtrip():
    t = WordTokenizer()
    text = "the cat sat on the mat"
    t.train(text)
    ids = t.encode(text)
    decoded = t.decode(ids)
    assert decoded.split() == text.split()


def test_punctuation_is_separate_token(): # noktalama işaretleri ayrı token olarak alınması kontrolü
    t = WordTokenizer()
    t.train("merhaba dünya!")
    ids = t.encode("merhaba dünya!")
    assert len(ids) == 3  # "merhaba", "dünya", "!"


def test_encode_raises_on_unknown_token():
    """
    OOV hatası kontrolü
    """
    t = WordTokenizer()
    t.train("merhaba dünya")
    with pytest.raises(ValueError, match="Unknown token"):
        t.encode("görülmeyen")


def test_decode_raises_on_unknown_id():
    t = WordTokenizer()
    t.train("merhaba dünya")
    with pytest.raises(ValueError, match="Unknown token id"):
        t.decode([9999])


def test_encode_raises_before_training():
    t = WordTokenizer()
    with pytest.raises(ValueError, match="not been trained"):
        t.encode("merhaba")


def test_decode_raises_before_training():
    t = WordTokenizer()
    with pytest.raises(ValueError, match="not been trained"):
        t.decode([0])


def test_train_raises_on_empty_text():
    t = WordTokenizer()
    with pytest.raises(ValueError, match="empty"):
        t.train("")


def test_vocabulary_is_deterministic():
    """
    Aynı text 2 kez train edilince id eşleşmesi olmalı..!
    """
    t1 = WordTokenizer()
    t2 = WordTokenizer()
    t1.train("elma armut kiraz")
    t2.train("elma armut kiraz")
    assert t1._token_to_id == t2._token_to_id