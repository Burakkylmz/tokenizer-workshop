# SimpleBPETokenizer

## 1. Purpose

`SimpleBPETokenizer` is the tokenizer class included in this project to introduce the concept of **subword tokenization**.

Its principal objective is to enable the learner to answer the following question clearly:

> Can text be represented more efficiently by merging frequently occurring adjacent character pairs?

Rather than serving merely as a conceptual bridge between `CharTokenizer` and `ByteTokenizer`; this tokenizer represents a substantive extension of them. Its scope is not limited to decomposing text into smaller units; it also involves **learning recurring patterns**.

---

## 2. Why This Tokenizer Exists

This tokenizer addresses a critical gap within the project.

### a) It transcends the limitations of the character-level approach
Although `CharTokenizer` tokenizes text in an intuitive manner, its treatment of each character as an independent unit inevitably yields unnecessarily long sequences.

### b) It renders the notion of efficiency explicit
`SimpleBPETokenizer` seeks to produce more compact token sequences by merging frequently co-occurring adjacent units.

### c) It provides a transition toward modern tokenizer design
A substantial proportion of contemporary tokenizers operate on subword principles. This class presents the simplest and most pedagogically instructive version of that approach.

In summary, the role of this tokenizer within the project can be described as follows:

> Tokenization is not concerned solely with segmentation; at times, it entails the merging of units to achieve a more effective representation.

---

## 3. What "BPE" Means in This Project

In this project, BPE is treated as a merging approach inspired by **Byte Pair Encoding**. 

The version implemented here, however, is deliberately simplified:

- it does not operate at the byte level
- it is initialized at the character level
- it omits regex-based pre-tokenization
- it does not incorporate special-token management
- it does not aim for production parity
  
For this reason, the class is named `SimpleBPETokenizer`.

The objective is not to replicate industrial tokenizer systems, but to **make the logic of BPE pedagogically accessible**.

---

## 4. Core Idea

The tokenizer operates according to the following logic:

1. The text is first decomposed into a sequence of characters.
2. Consecutive token pairs are counted.
3. The most frequent pair is identified.
4. This pair is merged into a new token.
5. The procedure is repeated for a fixed number of iterations.
6. During encoding, the learned merge rules are applied in order.

Example:

```text
"abababa"
```

Initial tokens:

```text
["a", "b", "a", "b", "a", "b", "a"]
```

Consecutive pairs:

```text
("a", "b") -> 3
("b", "a") -> 3
```

According to the tie-breaking rule, the pair selected first is:

```text
("a", "b") -> "ab"
```

The resulting token sequence after merging:

```text
["ab", "ab", "ab", "a"]
```

This example illustrates the following principle:

> BPE transforms frequently recurring local patterns into larger tokens.

---

## 5. Separation of Responsibilities

A deliberate architectural decision has been made in this project:

* `BPETrainer` is responsible for the learning process.
* `SimpleBPETokenizer` is responsible for encoding and decoding.

This separation is valuable because it clearly delineates two distinct responsibilities.

### `BPETrainer`

* computes pair frequencies
* selects the optimal merge
* determines the merge order

### `SimpleBPETokenizer`

* stores the merge rules
* constructs the vocabulary
* applies merges during encoding
* reconstructs the original string during decoding

This design enables the learner to recognize the following distinction:

> "What is the model learning?" and "How are the learned rules applied?" are fundamentally different questions.

This lesson extends well beyond tokenizers; it constitutes a core principle of software architecture.

---

## 6. Training Logic

The `train()` method constitutes the core of this tokenizer.

Training proceeds through the following stages:

### a) Merge rules are learned

`BPETrainer.train(text, num_merges=...)` is invoked

The following information is returned:

* the pair that was selected
* the token it was merged into
* its frequency at the moment of selection

This information is stored as `MergeStep` objects.

### b) The base vocabulary is constructed

The unique characters observed in the training data are extracted and used to form the base vocabulary.

### c) Learned merged tokens are added to the vocabulary

The new tokens learned during training are appended to the vocabulary.

This is a deliberate design decision. During encoding, some characters may be merged while others remain unchanged, and the tokenizer must therefore recognize both categories simultaneously:

* base character tokens
* merged tokens

---

## 7. Why Merge Order Matters

One of the most fundamental concepts in this tokenizer is the **merge order**.

BPE merges are not merely a set of rules. Applying the same rules in a different order can yield different tokenization outputs.

For this reason, merges are:

* stored in the order in which they were learned
* applied in that same order during encoding

This is a critical point, as learners often begin with the following misconception:

> "Once all frequently occurring pairs have been learned, their order is of little importance."

In reality, the order is highly consequential, and this tokenizer makes that fact explicit.

---

## 8. Determinism and Tie-Breaking

During BPE training, two distinct pairs may occasionally exhibit identical frequencies.

Example:

```text
"abababa"
```

Here, the pairs

* `("a", "b")`
* `("b", "a")`

share the same frequency.

If the selection were left ambiguous in such cases, different runs could produce different results, which would be detrimental to both training and testing.

For this reason, a deterministic tie-breaking rule is adopted in this project:

* the pair with the highest frequency is selected first
* in the case of a tie, the lexicographically smaller pair is chosen

This decision yields the following benefits:

* same input → same merge order
* same merge order → same encoding output
* reproducible experiments

Reproducibility is especially important in an educational project.Share

---

## 9. Encode Logic

The `encode()` method operates according to the following procedure:

1. The text is split into character tokens.
2. The learned merge steps are applied in order.
3. The resulting tokens are converted into integer ids.

The overall flow can be summarized as:

```text
text -> char tokens -> merged tokens -> token ids
```

The most critical point for the learner to grasp is the following:

> Tokenization is not re-learned during encoding; only the previously learned merge rules are applied.

In other words, training and inference are distinct phases. This distinction is essential to understanding how tokenizers operate.

---

## 10. Decode Logic

The `decode()` method converts integer token ids back into  their corresponding string pieces and concatenates them.

Since tokens are stored as strings in this tokenizer, the decoding process is relatively straightforward:

```text
[id_ab, id_ab, id_a] -> ["ab", "ab", "a"] -> "ababa"
```

An important observation follows from this:

Decoding operates independently of the granularity at which tokens are stored. A token may be a single character, a two-character merge, or a larger unit, without affecting the decoding procedure.

This property makes the tokenizer particularly well suited for illustrating subword logic.

---

## 11. Vocabulary Behavior

The vocabulary of `SimpleBPETokenizer` consists of two components:

### Base tokens

The unique characters observed in the training text.

### Merged tokens

The new tokens learned during training.

As a consequence, `vocab_size` is typically larger than that of `CharTokenizer`. The critical factor, however, is not the size of the vocabulary but its representational power.

This tokenizer illustrates the following idea:

> A larger vocabulary can represent a deliberate trade-off made in pursuit of shorter token sequences.

This perspective is foundational to tokenizer design from an engineering standpoint.

---

## 12. Compression Behavior

The principal advantage of this tokenizer lies in its ability to reduce the number of tokens in recurring structures.

Example:

```text
"abababa"
```

With `CharTokenizer`:

* 7 characters
* 7 tokens

With `SimpleBPETokenizer`:

* several units are merged
* the total number of tokens is reduced

This effect is also reflected in the metrics produced by the evaluation layer.

This enables the learner to ask not only "Is the algorithm correct?" but also:

> Does this tokenizer actually yield a more efficient representation?

---

## 13. Strengths

The strengths of `SimpleBPETokenizer` can be summarized as follows:

### a) It introduces subword logic

It extends beyond the character-level approach.

### b) It exploits recurring structures

It improves efficiency in tokenization.

### c) It makes the concept of merge order explicit

This is essential for approaching the logic of real-world tokenizers.

### d) It is both testable and observable

Training steps can be inspected through `MergeStep` objects.

### e) It is pedagogically effective

It presents the core idea without unnecessary complexity.
---

## 14. Limitations

This tokenizer operates under a set of deliberately accepted constraints.

### a) It is initialized at the character level

Most modern BPE systems operate at the byte level; this tokenizer does not adopt that approach.

### b) It lacks regex-based pre-tokenization

Whitespace, punctuation, numerical characters, and word boundaries are not handled separately.

### c) It has no unknown-token strategy

No advanced fallback mechanism is provided for tokens unseen after training.

### d) It lacks a save/load mechanism

The tokenizer state is currently intended for training-oriented use.

### e) It is not optimized for large-scale use

The objective is clarity rather than performance.

These limitations should not be interpreted as deficiencies; rather, they reflect **deliberate decisions of scope**.

---

## 15. Comparison with Other Tokenizers

### SimpleBPETokenizer vs CharTokenizer

* `CharTokenizer` performs no merging.
* `SimpleBPETokenizer` merges frequently occurring units.

In consequence:

* `CharTokenizer` is simpler
* `SimpleBPETokenizer` can achieve greater efficiency.

### SimpleBPETokenizer vs ByteTokenizer

* `ByteTokenizer` relies on a fixed and comprehensive byte space.
* `SimpleBPETokenizer` learns new tokens through training.

Result:

* `ByteTokenizer` offers broader coverage.
* `SimpleBPETokenizer` exploits recurring structures more effectively.

### SimpleBPETokenizer vs real-world Regex/Byte BPE systems

This class is a simplified counterpart of the tokenizers used in practice. Its primary purpose is pedagogical clarity.

Accordingly, this tokenizer is not a final destination, but a stepping stone toward more advanced tokenizer designs.

---

## 16. Design Decisions in This Project

The fundamental design decisions adopted for `SimpleBPETokenizer` in this project are as follows:

* initialization at the character level is preferred
* the merge-learning logic is isolated in a separate `BPETrainer` class
* the merge order is preserved
* deterministic tie-breaking is applied
* base tokens and merged tokens are integrated into a single vocabulary
* encode and decode behavior is kept clear and inspectable

Each of these decisions reflects a balance between educational value and architectural clarity.

---

## 17. Testing Perspective

The core behaviors verified by the tests for this tokenizer are as follows:

* an error is raised when an invalid `num_merges` is supplied
* training on empty text is rejected
* invoking encode or decode prior to training raises an error
* the vocabulary is constructed successfully after training
* the encode output is a list of integer ids
* the original text is faithfully recovered after decoding
* merge steps are indeed learned during training
* the token count decreases in the presence of recurring structures
* the same input produces an identical sequence of merge steps

These tests are valuable because this class is no longer merely a mapping-based tokenizer, but **a tokenizer with learning behavior**.

---

## 18. When to Use

`SimpleBPETokenizer` is particularly well suited to the following contexts:

* introducing the concept of subword tokenization
* demonstrating the logic of merge operations
* explaining why the character-level approach is insufficient
* reflecting on tokenizer efficiency
* serving as an introduction to modern LLM tokenizer design

It is not suitable in the following contexts:

* when production-grade tokenizer parity is required
* when robust multilingual behavior is required
* when regex-based boundary control is needed
* when byte-level robustness is required

These cases call for more advanced architectures.Share

---

## 19. Final Takeaway

`SimpleBPETokenizer` is the class that most substantially enriches the project's treatment of tokenization.

The most important lesson it offers is the following:

> Effective tokenization is not merely the segmentation of text, but the transformation of recurring structures into more meaningful and more efficient units.

Once this principle is internalized, the learner's perspective on modern tokenizer design is fundamentally transformed.