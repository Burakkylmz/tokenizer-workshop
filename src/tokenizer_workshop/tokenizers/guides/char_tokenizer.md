# CharTokenizer

## 1. Purpose

`CharTokenizer` is the simplest tokenizer class, operating at the **character level**.

Its purpose within this project is to illustrate the concept of tokenization in its most transparent and accessible form. Each unique character is treated as a distinct token.

Example:

```text
"merhaba" -> ["m", "e", "r", "h", "a", "b", "a"]
```

This approach is particularly valuable from a pedagogical standpoint, as it allows the learner to address the following foundational questions directly:

* What is a token?
* How is a vocabulary constructed?
* What does `encode` do?
* What does `decode` do?
* Why is a mapping table required?

---

## 2. Why This Tokenizer Exists

This tokenizer does not exist to deliver real-world performance; it exists for **pedagogical purposes and conceptual clarity**.

Within this project, `CharTokenizer` fulfills the following roles:

* it serves as the entry point for understanding tokenization
* it provides a reference behavior against which more advanced tokenizers can be understood
* it establishes a basis for comparison with `ByteTokenizer` and `SimpleBPETokenizer`

In other words, without this class, evaluating more advanced tokenizers would be considerably more difficult for the learner. The reason is straightforward: unless the simplest form is examined first, the motivation for more complex methods cannot be fully grasped.

---

## 3. Core Idea

The logic of this tokenizer is straightforward:

1. Collect every unique character in the training data.
2. Assign an integer identifier to each character.
3. Convert the text into these identifiers.
4. Reconstruct the text from identifiers when needed.

Example:

```text
text = "aba"

unique characters = ["a", "b"]
stoi = {"a": 0, "b": 1}
itos = {0: "a", 1: "b"}

encode("aba") -> [0, 1, 0]
decode([0, 1, 0]) -> "aba"
```

Two elements are central here:

* `stoi`: string to integer
* `itos`: integer to string

A tokenizer does not merely decompose text; it must also be able to reconstruct it.

---

## 4. Training Logic

For `CharTokenizer`, the notion of "training" does not correspond to machine-learning training in the conventional sense. Training here refers to the extraction of a vocabulary from a given text.

In code, this is accomplished as follows:

```python
unique_chars = sorted(set(text))
```

This single line encodes two important decisions:

### a) `set(text)`

This collects the unique characters present in the text.

### b) `sorted(...)`

This arranges the characters in a deterministic order.

This matters because the output of the tokenizer must be reproducible. When the same text is used for training twice, each character must be assigned the same identifier.

Without `sorted`, the ordering of the mapping can become unpredictable under certain conditions, rendering the training output unstable.

---

## 5. Encode Logic

The `encode()` method converts each character in the input text into an integer token id.

Example:

```text
"merhaba" -> [id_m, id_e, id_r, id_h, id_a, id_b, id_a]
```

A deliberate design decision has been made at this stage: when the tokenizer encounters a character not seen during training, it **does not pass over it silently** and **does not fabricate a fallback**. Instead, it raises an error directly.

This decision is pedagogically justified, as it makes the following problem explicit:

> How should a tokenizer behave when confronted with characters outside its coverage?

In practice, this problem is addressed through mechanisms such as `unknown token`, `fallback`, and `byte fallback`. Here, however, the aim is first to expose the problem in its most basic form.

---

## 6. Decode Logic

The `decode()` method converts a list of integer tokens back into text.

At this stage, the `_itos` mapping is used:

```text
[0, 1, 0] -> "aba"
```

An important observation is the following: decoding demonstrates that the tokenizer truly operates in both directions. Many learners understand the encode side but fail to recognize why the decode side is necessary.

In reality, decoding is essential for inspecting, debugging, and verifying the behavior of a tokenizer.

---

## 7. Vocabulary Behavior

For `CharTokenizer`, the vocabulary size is defined as follows:

> The number of unique characters in the training data.

This has several consequences:

* the vocabulary is data-dependent
* different corpora produce different vocabularies
* a small text yields a small vocabulary
* previously unseen characters require retraining

This behavior is pedagogically instructive, as it prompts the learner to consider a fundamental question in tokenizer design: is the vocabulary fixed, or is it learned?

---

## 8. Strengths

The strengths of `CharTokenizer` can be summarized as follows:

* it is conceptually clear
* it is straightforward to implement
* it illustrates the logic of encoding and decoding
* it makes vocabulary construction explicit
* it is easy to debug
* it is pedagogically effective

For these reasons, its selection as the first tokenizer in the project is well justified.

---

## 9. Limitations

This tokenizer has several significant limitations.

### a) Sequence length can grow considerably

Because each character constitutes a separate token, the text may be transformed into very long token sequences.

### b) It does not exploit structural regularities

For instance, it does not specifically learn frequent patterns such as the word `"token"` or the suffix `"ing"`.

### c) It fails on unseen characters

If a previously unseen character is encountered, the tokenizer cannot encode it.

### d) Its real-world efficiency is limited

In modern LLM systems, more advanced tokenizers are generally preferred.

In summary, this tokenizer is pedagogically strong but practically inefficient.

---

## 10. Comparison with Other Tokenizers

### CharTokenizer vs. ByteTokenizer

* `CharTokenizer` operates on characters.
* `ByteTokenizer` operates on UTF-8 bytes.

`ByteTokenizer` offers broader coverage, as it can represent any UTF-8 text. However, `CharTokenizer` is conceptually easier to understand.

### CharTokenizer vs. SimpleBPETokenizer

* `CharTokenizer` performs no merging.
* `SimpleBPETokenizer` merges frequently occurring units.

Consequently, `SimpleBPETokenizer` can produce shorter token sequences for certain texts.

---

## 11. Design Decisions in This Project

The key design decisions adopted for `CharTokenizer` in this project are as follows:

* the vocabulary is learned from the text
* character ordering is established deterministically
* encode and decode cannot be invoked prior to training
* an error is raised when unseen characters are encountered
* pedagogical value is prioritized over performance

These decisions have been made deliberately for educational purposes rather than for production use.

---

## 12. Testing Perspective

The core behaviors verified by the tests for this tokenizer are as follows:

* the vocabulary is constructed after training
* the encode output is a list of integer ids
* the original text is faithfully recovered after decoding
* invoking the tokenizer before training raises an error
* an error is raised when unseen characters are encountered
* the same input produces an identical vocabulary

These tests verify correctness while also preserving the underlying design contract.

---

## 13. When to Use

`CharTokenizer` is meaningful in the following contexts:

* introducing the concept of tokenization
* illustrating the fundamental logic of tokenization
* conducting small, transparent experiments
* explaining the structure of encode and decode mappings

It is generally insufficient in the following contexts:

* large-scale NLP systems
* scenarios requiring efficient sequence representation
* multilingual or otherwise complex data
* modern LLM pipelines

---

## 14. Final Takeaway

Although `CharTokenizer` is the simplest tokenizer in this project, it is by no means the least significant. On the contrary, it provides the conceptual framework necessary for understanding all other tokenizers.

The essential value of this class can be stated as follows:

> The essence of tokenization first emerges here, in its most elementary form.