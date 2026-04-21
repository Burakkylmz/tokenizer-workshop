# SimpleBPETokenizer

## 1. Purpose

`SimpleBPETokenizer` is a tokenizer type included in this project to teach the idea of **subword tokenization**.

The main purpose of this class is to enable the learner to clearly answer the following question:

> If frequently repeating character pairs are merged, can the text be represented more efficiently?

This tokenizer is not just a conceptual bridge between `CharTokenizer` and `ByteTokenizer`; it is a more powerful step beyond them. Because here, it is no longer only about splitting text into smaller parts, but about **learning repeating structures**.

---

## 2. Why This Tokenizer Exists

This tokenizer fills a very critical gap in the project.

### a) Goes beyond the limitations of the char-level approach

`CharTokenizer` tokenizes text in an understandable way, but since it treats each character separately, it may produce unnecessarily long sequences.

### b) Makes the idea of efficiency visible

`SimpleBPETokenizer` attempts to produce shorter token sequences by merging frequently occurring neighboring parts.

### c) Provides a transition to modern tokenizer logic

A significant portion of real-world tokenizers operate with subword logic. This class presents the simplest and most educational version of that idea.

In other words, the role of this tokenizer in the project is:

> Tokenization is not only about splitting; sometimes it is about merging parts for a better representation.

---

## 3. What “BPE” Means in This Project

BPE is handled here as a merge approach inspired by the idea of **Byte Pair Encoding**.

However, the version in this project is intentionally simplified:

* not byte-level
* uses character-level initialization
* does not include regex pre-tokenization
* does not include special token management
* does not aim for production parity

That is why it is named `SimpleBPETokenizer`.

The goal is not to exactly replicate industrial tokenizer systems, but to make **BPE logic teachable**.

---

## 4. Core Idea

This tokenizer works with the following logic:

1. split the text into character tokens
2. count consecutive token pairs
3. find the most frequent pair
4. merge this pair into a new token
5. repeat this process a specified number of times
6. apply the learned merge rules in order during encoding

Example:

```text
"abababa"
```

Initial tokens:

```text id="kz9vfb"
["a", "b", "a", "b", "a", "b", "a"]
```

Consecutive pairs:

```text id="2m0x4p"
("a", "b") -> 3
("b", "a") -> 3
```

According to the tie-break rule, the first selected pair:

```text id="xq5k9m"
("a", "b") -> "ab"
```

After merging:

```text id="k4n9r1"
["ab", "ab", "ab", "a"]
```

This example shows the learner:

> BPE transforms frequently repeating local patterns into larger tokens.

---

## 5. Separation of Responsibilities

An important architectural decision is made in this project:

* `BPETrainer` performs the learning
* `SimpleBPETokenizer` handles encode/decode behavior

This separation is valuable because it clearly distinguishes two responsibilities.

### `BPETrainer`

* calculates pair frequencies
* selects the best merge
* produces merge order

### `SimpleBPETokenizer`

* stores merge rules
* builds vocabulary
* applies merges during encoding
* reconstructs strings during decoding

This design teaches the learner:

> “What the model learns” and “how learned rules are applied” are not the same thing.

This is an important lesson not only for tokenizers but for software architecture in general.

---

## 6. Training Logic

The `train()` method is the core of this tokenizer.

During training:

### a) Merge rules are learned

`BPETrainer.train(text, num_merges=...)` is called.

From this process, we obtain:

* which pair was selected
* what it was merged into
* its frequency at that step

These are stored as `MergeStep` objects.

### b) Base vocabulary is created

Unique characters from the training data are collected.

### c) Learned merged tokens are added to the vocabulary

New tokens learned during training are added to the vocabulary.

This is an important design decision because:

* some tokens remain as characters
* some become merged tokens

The tokenizer must recognize both.

---

## 7. Why Merge Order Matters

One of the most important concepts in this tokenizer is **merge order**.

BPE merges are not just a set of rules.
Applying the same pairs in different orders can produce different outputs.

Therefore:

* merges are stored in order
* they are applied in the same order during encoding

This is critical because learners often assume:

> “Once we learn all frequent pairs, order does not matter.”

In reality, order matters significantly.
This tokenizer makes that fact visible.

---

## 8. Determinism and Tie-Breaking

During BPE training, two pairs may have the same frequency.

Example:

```text id="k7y5pz"
"abababa"
```

Here:

* `("a", "b")`
* `("b", "a")`

may appear equally often.

If selection is ambiguous, results may differ between runs.

To avoid this, a deterministic rule is used:

* select the highest frequency
* if equal, choose the lexicographically smaller pair

This ensures:

* same input → same merge order
* same merge order → same output
* reproducible experiments

---

## 9. Encode Logic

The `encode()` method works as follows:

1. split text into character tokens
2. apply learned merge steps in order
3. convert resulting tokens into integer IDs

Conceptually:

```text id="x8n6ab"
text -> char tokens -> merged tokens -> token ids
```

The most important concept:

> Encoding does not relearn tokenization; it only applies previously learned merge rules.

Training and inference are separate.

---

## 10. Decode Logic

The `decode()` method converts token IDs back to string pieces and joins them.

Because tokens are strings, decoding is straightforward:

```text id="a7t6dm"
[id_ab, id_ab, id_a] -> ["ab", "ab", "a"] -> "ababa"
```

Important point:

Decoding works regardless of token granularity.
Tokens may represent:

* single characters
* merged pairs
* larger structures

---

## 11. Vocabulary Behavior

The vocabulary consists of two parts:

### Base tokens

Unique characters from the training text

### Merged tokens

Tokens learned during training

Therefore, `vocab_size` is usually larger than in `CharTokenizer`.

The key idea:

> A larger vocabulary can be a deliberate trade-off to achieve shorter token sequences.

---

## 12. Compression Behavior

One of the biggest advantages of this tokenizer is reducing token count in repetitive structures.

Example:

```text id="e4q9lm"
"abababa"
```

With `CharTokenizer`:

* 7 characters → 7 tokens

With `SimpleBPETokenizer`:

* some parts are merged
* total token count may decrease

This makes efficiency measurable.

---

## 13. Strengths

The strengths of `SimpleBPETokenizer`:

### a) Teaches subword logic

### b) Uses repeating structures

### c) Makes merge order visible

### d) Inspectable via `MergeStep`

### e) Ideal for learning

---

## 14. Limitations

### a) Starts at character level

Modern BPE systems often start at byte level.

### b) No regex pre-tokenization

No handling for whitespace, punctuation, etc.

### c) No unknown token strategy

No fallback behavior for unseen tokens.

### d) No save/load mechanism

State is not persisted.

### e) Not optimized for scale

Designed for clarity, not performance.

---

## 15. Comparison with Other Tokenizers

### SimpleBPETokenizer vs CharTokenizer

* CharTokenizer does not merge
* BPE merges frequent parts

Result:

* CharTokenizer → simpler
* BPE → potentially more efficient

### SimpleBPETokenizer vs ByteTokenizer

* ByteTokenizer → fixed byte space
* BPE → learns new tokens

Result:

* ByteTokenizer → more inclusive
* BPE → better at exploiting repetition

### SimpleBPETokenizer vs real BPE systems

This is a simplified version for educational clarity.
It is a stepping stone toward more advanced tokenizers.

---

## 16. Design Decisions in This Project

Key decisions:

* character-level initialization
* separate `BPETrainer`
* preserve merge order
* deterministic tie-breaking
* combined vocabulary (base + merged)
* transparent encode/decode behavior

---

## 17. Testing Perspective

Tests validate:

* invalid `num_merges`
* empty text handling
* pre-training usage errors
* vocabulary creation
* encode output format
* decode correctness
* merge learning behavior
* token reduction in repetition
* deterministic merge order

---

## 18. When to Use

Useful for:

* teaching subword tokenization
* demonstrating merge logic
* explaining limitations of char-level tokenization
* introducing modern tokenizer concepts

Not sufficient for:

* production-grade systems
* multilingual complex behavior
* regex-based tokenization
* byte-level robustness

---

## 19. Final Takeaway

`SimpleBPETokenizer` is the class that deepens tokenization understanding in this project.

Its key lesson:

> Good tokenization is not just splitting text, but transforming repeating structures into more meaningful and efficient units.

