from __future__ import annotations

import pytest

from tokenizer_workshop.trainers import BPETrainer, MergeStep


def test_get_pair_stats_counts_adjacent_pairs_correctly() -> None:
    tokens = ["a", "b", "a", "b"]

    stats = BPETrainer.get_pair_stats(tokens)

    assert stats == {
        ("a", "b"): 2,
        ("b", "a"): 1,
    }


def test_get_pair_stats_returns_empty_dict_for_short_input() -> None:
    assert BPETrainer.get_pair_stats([]) == {}
    assert BPETrainer.get_pair_stats(["a"]) == {}


def test_merge_pair_replaces_matching_pairs_left_to_right() -> None:
    tokens = ["a", "b", "a", "b"]

    merged = BPETrainer.merge_pair(tokens, ("a", "b"), "ab")

    assert merged == ["ab", "ab"]


def test_merge_pair_does_not_merge_non_matching_tokens() -> None:
    tokens = ["a", "b", "c"]

    merged = BPETrainer.merge_pair(tokens, ("b", "b"), "bb")

    assert merged == ["a", "b", "c"]


def test_merge_pair_handles_overlapping_pattern_greedily() -> None:
    tokens = ["a", "a", "a"]

    merged = BPETrainer.merge_pair(tokens, ("a", "a"), "aa")

    assert merged == ["aa", "a"]


def test_merge_pair_raises_error_for_invalid_pair_length() -> None:
    with pytest.raises(ValueError, match="exactly two tokens"):
        BPETrainer.merge_pair(["a", "b"], ("a",), "a")


def test_train_raises_error_for_empty_text() -> None:
    trainer = BPETrainer()

    with pytest.raises(ValueError, match="Training text cannot be empty"):
        trainer.train("", num_merges=3)


def test_train_raises_error_for_invalid_num_merges() -> None:
    trainer = BPETrainer()

    with pytest.raises(ValueError, match="num_merges must be at least 1"):
        trainer.train("abababa", num_merges=0)


def test_train_learns_merge_steps_in_order() -> None:
    trainer = BPETrainer()

    merges = trainer.train("abababa", num_merges=3)

    assert len(merges) >= 1
    assert isinstance(merges[0], MergeStep)
    assert merges[0].pair == ("a", "b")
    assert merges[0].merged_token == "ab"
    assert merges[0].frequency == 3


def test_train_stores_merge_steps_on_trainer_instance() -> None:
    trainer = BPETrainer()

    merges = trainer.train("abababa", num_merges=2)

    assert trainer.merge_steps == merges


def test_train_is_deterministic_for_same_input() -> None:
    trainer_a = BPETrainer()
    trainer_b = BPETrainer()

    merges_a = trainer_a.train("abababa", num_merges=3)
    merges_b = trainer_b.train("abababa", num_merges=3)

    assert merges_a == merges_b


def test_train_stops_early_when_no_more_pairs_exist() -> None:
    trainer = BPETrainer()

    merges = trainer.train("a", num_merges=5)

    assert merges == []