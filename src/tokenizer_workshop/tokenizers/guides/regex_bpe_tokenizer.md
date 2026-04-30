# RegexBPETokenizer

## 1. Purpose

`RegexBPETokenizer` is the tokenizer class included in this project to introduce the concept of **regex-bounded Byte Pair Encoding** — a hybrid approach in which regex-based pre-tokenization and byte-level BPE merge learning operate as a single, integrated pipeline.

Its principal objective is to enable the learner to answer the following question clearly:

> Once linguistically meaningful boundaries have been established, how can the byte sequences within each region be compressed in a way that respects those boundaries?

This question lies at the heart of every modern production tokenizer. `RegexBPETokenizer` represents the closest approximation in this project to the tokenizers used in real-world systems such as GPT-2 and Llama — albeit in a simplified and pedagogically transparent form.

Example:

```text
"Salam, dünya!" -> regex chunks: ["Salam", ",", " ", "dünya", "!"]
                -> per-chunk byte BPE
                -> token ids: [258, 44, 32, 261, 33]   (illustrative)
```

The defining characteristic of this tokenizer is the following:

> Merges are confined within regex-defined chunks; they never span across them.

This is precisely the design that prevents pathological behaviors such as merging the final letter of one word with the leading character of the next.

---

## 2. Why This Tokenizer Exists

This tokenizer occupies a privileged position within the project, because it is the first instance in which two previously isolated mechanisms are deliberately combined.

### a) It unifies two prior tokenizers into a coherent pipeline

`RegexTokenizer` introduced the concept of regex-based pre-tokenization but produced no learned subword structure. `ByteBPETokenizer` (or `ByteLevelBPETokenizer`) introduced byte-level merge learning but operated on the entire text without respect for word boundaries.

`RegexBPETokenizer` is the synthesis of these two ideas:

* **regex pre-tokenization** → defines chunk boundaries
* **byte-level BPE within each chunk** → learns subword compression

Each of these mechanisms is necessary, but neither is sufficient on its own. Their combination is the principal contribution of this tokenizer.

### b) It corrects a subtle pathology of pure BPE

When BPE is applied directly to a long text without any pre-tokenization, the algorithm can — and frequently does — learn merges that span across word boundaries.

Example:

```text
"hello world hello world"
```

A pure BPE trainer might select the byte pair `(o, w)` (the final letter of `hello` and the first letter of `world`) as a frequent pair, producing a token that is linguistically nonsensical.

`RegexBPETokenizer` prevents this by construction: regex pre-tokenization splits the input into chunks before merge learning begins, and merge counts are aggregated **within** each chunk only. Consequently, no merge can ever cross a chunk boundary.

This is precisely the design choice adopted by GPT-2's tokenizer.

### c) It demonstrates the architecture of modern production tokenizers

The internal pipeline of most contemporary tokenizers can be summarized as follows:

```text
text -> normalization -> pre-tokenization -> subword learning -> token ids
```

`RegexBPETokenizer` implements steps two through four in a clean and readable form, omitting only the normalization layer (which the project's other tokenizers also do not implement).

For this reason, this class can be regarded as:

> A simplified counterpart of the tokenizers used in real-world systems, in which the same architectural principles are made transparent.

---

## 3. What "Regex-Bounded BPE" Means in This Project

In this project, regex-bounded BPE is treated as a hybrid mechanism in which two distinct ideas operate in concert:

* **regex pre-tokenization defines the boundaries within which BPE can operate**
* **BPE compresses the byte sequence within each boundary independently**

The implementation, however, remains deliberately simplified:

* it does not perform Unicode normalization (NFC, NFKC, etc.)
* it does not handle special tokens (BOS, EOS, PAD, MASK)
* it does not implement save/load functionality
* it does not employ optimized data structures for large-scale training
* its default regex pattern recognizes only Latin, Turkish, and Azerbaijani letters

The class is therefore not a production tokenizer; rather, it is a **conceptually faithful** simplification.

The objective is not to replicate industrial systems, but to render the principles underlying those systems pedagogically accessible.

---

## 4. Core Idea

The tokenizer operates according to the following logic:

1. The text is partitioned into chunks using a regex pattern.
2. Each chunk is encoded into UTF-8 bytes.
3. Adjacent byte pairs are counted **per chunk** and aggregated globally.
4. The most frequent pair is selected and merged into a new token id.
5. The merge is applied within every chunk simultaneously.
6. Steps three through five are repeated for `vocab_size − 256` iterations.
7. During encoding, the same regex pre-tokenization is applied, and the learned merges are applied in order within each chunk.
8. During decoding, token ids are mapped to byte sequences and concatenated.

Example:

```text
text = "salam dünya salam"

regex chunks = ["salam", " ", "dünya", " ", "salam"]

each chunk -> UTF-8 bytes:
    "salam" -> [115, 97, 108, 97, 109]
    " "     -> [32]
    "dünya" -> [100, 195, 188, 110, 121, 97]

pair counts (aggregated):
    (115, 97) -> appears in "salam" twice -> count 2
    (97, 108) -> appears in "salam" twice -> count 2
    ...

selected pair: (115, 97)  ->  new token id 256

merged chunks:
    "salam" -> [256, 108, 97, 109]
    "salam" -> [256, 108, 97, 109]
    "dünya" -> unchanged
```

Two observations are essential here.

First, byte pairs from different chunks are counted separately but aggregated globally. The frequency of `(o, w)` in `"hello world"` is zero, even though those bytes are adjacent in the raw text — because they belong to different chunks.

Second, when a merge is applied, it is applied **within** each chunk in isolation. A merge can never produce a token that spans across chunks.

This is the structural property that distinguishes regex-bounded BPE from pure BPE.

---

## 5. The Default Regex Pattern

The default pattern adopted by this tokenizer is as follows:

```text
[a-zA-ZğüşıöçĞÜŞİÖÇə]+ | \d+ | [^\s\w]+ | \s+
```

This pattern partitions the text into four mutually exclusive token classes:

* `[a-zA-ZğüşıöçĞÜŞİÖÇə]+` — sequences of Latin, Turkish, or Azerbaijani letters
* `\d+` — sequences of digits
* `[^\s\w]+` — sequences of punctuation and other non-word characters
* `\s+` — sequences of whitespace

This decomposition has two important consequences.

### a) Whitespace is preserved as its own token class

Unlike `RegexTokenizer`, which discards whitespace during tokenization, `RegexBPETokenizer` retains whitespace as a separate token class. This is what makes lossless round-trip possible: the original spacing of the text is recoverable from the chunk sequence.

This is also the convention followed by GPT-2's tokenizer, in which whitespace is encoded into token sequences alongside word tokens.

### b) The pattern is not language-neutral

The letter class includes only the alphabets of English, Turkish, and Azerbaijani. As a result:

| Input | Behavior |
|---|---|
| `"hello"` | Captured as a single chunk |
| `"dünya"` | Captured as a single chunk |
| `"привет"` (Cyrillic) | Not matched; passes through as raw bytes via the punctuation class or is dropped entirely |
| `"你好"` (CJK) | Not matched |
| `"😊"` | Captured by the punctuation class, since emoji are non-word, non-whitespace |

This is a deliberate scoping decision. Languages outside the project's pedagogical focus are not handled, and learners should be aware of this limitation.

A more general pattern would replace the Latin–Turkish letter class with `\w+` (which is Unicode-aware in Python by default). This is a one-line change for users requiring broader language coverage.

---

## 6. Vocabulary Behavior

The vocabulary of `RegexBPETokenizer` consists of two components.

### Base byte vocabulary

```text
0, 1, 2, ..., 255
```

The first 256 ids are reserved for raw byte values and are always present, regardless of the training data. This guarantees that any UTF-8 byte can be represented, even if no merge has been learned that covers it.

### Learned merge vocabulary

Beginning at id 256, each learned merge contributes one additional token. The total vocabulary size after training is therefore:

```text
256 + number_of_learned_merges
```

The user controls this through the `vocab_size` parameter:

```python
tokenizer.train(text, vocab_size=270)   # learns 14 merges
tokenizer.train(text, vocab_size=1000)  # learns up to 744 merges
```

A `vocab_size` smaller than 256 is rejected as an error, since the base byte vocabulary is non-negotiable.

This dual structure — fixed base plus learned extensions — is identical in spirit to that of `ByteBPETokenizer`. The distinguishing feature of `RegexBPETokenizer` lies in **how** merges are learned, not in the structure of the vocabulary itself.

---

## 7. Separation of Responsibilities

A clear architectural separation is observed within the class itself, even though the implementation is not split into separate trainer and tokenizer objects.

The following responsibilities are partitioned across distinct internal methods:

| Method | Responsibility |
|---|---|
| `_pre_tokenize` | Applies the regex to partition the text into chunks |
| `_text_to_ids` | Converts a chunk into a list of UTF-8 byte ids |
| `_get_stats` | Computes adjacent pair frequencies within a chunk |
| `_merge` | Applies a single merge rule to a chunk |
| `_build_vocab` | Constructs the id-to-bytes mapping after training |
| `train` | Orchestrates the merge-learning loop |
| `encode` | Applies pre-tokenization and learned merges in inference mode |
| `decode` | Reconstructs text from token ids |

Each method has a single, well-defined purpose. The merge-learning logic is contained within `train`; the pure transformations on byte sequences are isolated as helpers.

This design teaches the following principle:

> The boundary between training-time logic and inference-time logic should be visible in the code.

In `RegexBPETokenizer`, this boundary is made explicit: `train` mutates `self.merges`, while `encode` and `decode` only read it.

---

## 8. Training Logic

The `train()` method is the most substantive component of this tokenizer.

Training proceeds through the following stages.

### a) Validation

```python
if vocab_size < 256:
    raise ValueError(...)
```

A `vocab_size` below 256 cannot accommodate the base byte vocabulary and is therefore rejected.

### b) Pre-tokenization

The text is partitioned into chunks using the regex pattern. Each chunk is converted into a list of byte ids:

```text
"salam dünya" -> [
    [115, 97, 108, 97, 109],
    [32],
    [100, 195, 188, 110, 121, 97],
]
```

### c) Merge-learning loop

The loop runs `vocab_size − 256` iterations. At each iteration:

1. Adjacent pair frequencies are counted within each chunk.
2. The counts are aggregated across all chunks.
3. The most frequent pair is selected.
4. A new token id is assigned (`256 + iteration_index`).
5. The merge is applied within every chunk.
6. The new rule is recorded in `self.merges`.

If at any point no further pairs can be found, the loop terminates early.

### d) Vocabulary construction

After all merges have been learned, `_build_vocab()` constructs the final id-to-bytes mapping. Base byte ids (0–255) map to single-byte sequences; learned merge ids map to the concatenation of their constituents.

This last step is essential: without it, the tokenizer would have learned merges but would have no efficient way to decode token ids back into bytes.

---

## 9. Why Per-Chunk Aggregation Matters

The detail that distinguishes `RegexBPETokenizer` from a naive byte-level BPE implementation lies in a single line of the training loop:

```python
for ids in ids_list:
    for pair, count in self._get_stats(ids).items():
        total_stats[pair] += count
```

Pair frequencies are computed **per chunk** and then summed. The key fact is that `_get_stats` operates on a single chunk in isolation — it never observes the boundary between two chunks.

The consequence is that pairs spanning chunk boundaries are never counted, and therefore never selected for merging.

To appreciate the importance of this detail, consider what would happen without it. A pure BPE algorithm applied to:

```text
"hello world hello world hello world"
```

would observe the pair `(o, ` ` `)` (the letter `o` followed by a space) as one of the most frequent in the corpus. It might then learn a merge that produces a token containing both word-internal and word-external bytes, which is precisely the kind of token that real-world tokenizers are carefully designed to avoid.

Per-chunk aggregation prevents this category of merges entirely.

This is an example of a phenomenon that recurs throughout tokenizer design:

> A small change in where statistics are computed can produce a large change in the structure of the learned vocabulary.

---

## 10. Encode Logic

The `encode()` method follows a procedure structurally similar to that of training:

1. The input text is pre-tokenized into chunks using the same regex.
2. Each chunk is converted into a list of byte ids.
3. The learned merges are applied within each chunk, in the order in which they were learned.
4. The resulting byte ids are concatenated across chunks.

The most critical point for the learner to grasp is the following:

> No new merges are learned during encoding; only the rules learned during training are applied.

Equally important is that the regex pre-tokenization is applied **with the same pattern that was used during training**. If the pattern were different at inference time, the chunks would differ, and the learned merges would no longer apply correctly.

This is why `self.pattern` is stored on the tokenizer instance and reused in `encode`.

---

## 11. Decode Logic

The `decode()` method reconstructs the original byte sequence from a list of token ids and decodes it as UTF-8.

The procedure is:

1. Each id is mapped to its byte representation via `self.vocab`.
2. The byte representations are concatenated.
3. The resulting byte sequence is decoded using UTF-8.

A subtle but important property follows from this design:

> Decoding is lossless because the regex pattern preserves whitespace as its own token class.

Unlike `RegexTokenizer`, which discards whitespace and reconstructs it heuristically, `RegexBPETokenizer` retains the original spacing through the chunk sequence. As a result, encode followed by decode reproduces the input exactly — not because the tokenizer happens to handle a particular case correctly, but because the design guarantees it.

The single concession to robustness is the use of `errors="replace"` during UTF-8 decoding. This substitutes a replacement character for any byte sequence that is not valid UTF-8, rather than raising an exception. In a tokenizer that operates exclusively on its own merge rules, such cases should not arise; the safeguard is present primarily as a defense against external misuse (e.g., manually constructed id sequences).

---

## 12. Strengths

The strengths of `RegexBPETokenizer` can be summarized as follows.

### a) It combines the strengths of two prior tokenizers

It inherits the linguistic awareness of `RegexTokenizer` and the compression behavior of byte-level BPE.

### b) It prevents the pathological merges of pure BPE

By confining merges within regex-defined chunks, it eliminates a category of nonsensical tokens that pure BPE can produce.

### c) It supports lossless round-trip

Because whitespace is preserved as a token class, encode followed by decode reproduces the input exactly.

### d) It approximates production tokenizer behavior

Its architecture is structurally similar to that of GPT-2's tokenizer, which makes it a meaningful reference point for comparison studies.

### e) It is configurable

The regex pattern is exposed as a constructor parameter, allowing the learner to experiment with alternative tokenization rules.

### f) Its vocabulary size is parameterized

The `vocab_size` argument provides direct control over the trade-off between compression efficiency and vocabulary size.

---

## 13. Limitations

This tokenizer also operates under several deliberately accepted constraints.

### a) The default pattern is not language-neutral

Languages outside English, Turkish, and Azerbaijani are not recognized by the default pattern. Users requiring broader coverage must supply a custom pattern.

### b) No special-token mechanism is provided

There is no support for `[CLS]`, `[SEP]`, `[BOS]`, `[EOS]`, or similar control tokens. This makes the tokenizer unsuitable for direct integration with transformer model inputs without further wrapping.

### c) No persistence layer

The tokenizer cannot be saved to or loaded from disk. The trained merges exist only for the lifetime of the instance.

### d) Greedy merge selection without tie-breaking specification

The merge-learning loop selects the most frequent pair using `max(...)` without an explicit tie-breaking rule. In the presence of ties, the behavior depends on Python's dictionary iteration order. While deterministic in modern Python (3.7+), this is more fragile than an explicit tie-breaking rule of the kind adopted by `ByteLevelBPETokenizer`.

### e) Errors are signaled with `RuntimeError`

Most other tokenizers in this project signal "tokenizer not yet trained" via `ValueError`. `RegexBPETokenizer` uses `RuntimeError` instead. This is a minor inconsistency in the public error surface, and callers must accommodate both forms.

### f) Optimization is not a goal

The implementation prioritizes clarity over performance. Each merge iteration scans every chunk in full; this is acceptable for the small corpora used in the workshop but would not scale to large training datasets.

These limitations should not be interpreted as deficiencies; rather, they reflect **deliberate decisions of scope**.

---

## 14. Comparison with Other Tokenizers

### RegexBPETokenizer vs RegexTokenizer

* `RegexTokenizer` performs only pre-tokenization; it does not learn subword structure.
* `RegexBPETokenizer` performs pre-tokenization **and** subword learning.

In consequence:

* `RegexTokenizer` produces shorter token sequences than character-level approaches but suffers from out-of-vocabulary errors on unseen words.
* `RegexBPETokenizer` decomposes unseen words into learned subwords (or, in the limit, into raw bytes), and therefore has no out-of-vocabulary failure mode.

### RegexBPETokenizer vs ByteBPETokenizer (or ByteLevelBPETokenizer)

* `ByteBPETokenizer` applies BPE to the entire text without word boundary awareness.
* `RegexBPETokenizer` applies BPE within regex-defined chunks only.

In consequence:

* `ByteBPETokenizer` may learn merges spanning word boundaries.
* `RegexBPETokenizer` is structurally prevented from doing so.

This is the single most important architectural difference in the project's BPE family.

### RegexBPETokenizer vs SimpleBPETokenizer

* `SimpleBPETokenizer` is character-level and ignores word boundaries.
* `RegexBPETokenizer` is byte-level and respects regex-defined boundaries.

The two tokenizers occupy opposite corners of the design space.

### RegexBPETokenizer vs GPT-2's tokenizer

GPT-2's tokenizer can be viewed as an industrial counterpart of this class, with three key additions:

* a more sophisticated default regex pattern (the well-known GPT-2 pattern)
* a byte-to-unicode remapping for printable token representations
* persistence and special-token support

`RegexBPETokenizer` deliberately omits these additions in pursuit of pedagogical clarity, but its core algorithm is the same.

---

## 15. Design Decisions in This Project

The fundamental design decisions adopted for `RegexBPETokenizer` in this project are as follows:

* a regex pattern is used to enforce chunk boundaries before BPE
* pair statistics are aggregated globally but **computed per chunk**
* whitespace is retained as a distinct token class to ensure lossless round-trip
* the base byte vocabulary is fixed at 256 entries
* `vocab_size` is exposed as a training parameter to control the merge budget
* merges are stored and applied in their learned order
* configurability is preserved through an exposed regex parameter
* educational clarity is prioritized over production-grade robustness

Each of these decisions reflects a balance between architectural realism and pedagogical accessibility.

---

## 16. Testing Perspective

The core behaviors verified by the tests for this tokenizer are as follows:

* `vocab_size` below 256 is rejected
* the vocabulary is constructed correctly after training
* at least one merge rule is learned for any non-trivial corpus
* the final vocabulary size does not exceed the requested `vocab_size`
* the encode output is a list of integers
* empty inputs to encode return an empty list
* invoking encode or decode prior to training raises a `RuntimeError`
* every id produced by encode is present in the vocabulary
* decode returns a string and handles empty input gracefully
* encode followed by decode reproduces the input exactly, including punctuation and whitespace
* the base vocabulary always contains all 256 byte values

These tests are pedagogically valuable because they verify both the **structural invariants** of the tokenizer (vocabulary completeness, id ranges) and its **behavioral contracts** (lossless round-trip, error handling). The combination of the two ensures that the tokenizer is both internally consistent and externally well-behaved.

---

## 17. When to Use

`RegexBPETokenizer` is particularly well suited to the following contexts:

* explaining the architecture of modern production tokenizers
* demonstrating the importance of pre-tokenization in BPE
* illustrating why merges should respect linguistic boundaries
* providing a realistic point of comparison for the project's simpler tokenizers
* serving as an entry point for studying GPT-2-family tokenization

It is not suitable in the following contexts:

* applications requiring multilingual support beyond English, Turkish, and Azerbaijani (without a custom pattern)
* systems requiring special tokens or persistence
* large-scale training pipelines requiring optimized data structures
* production deployments where edge cases and adversarial inputs must be handled robustly

These cases call for industrial tokenizers; `RegexBPETokenizer` provides the conceptual foundation for understanding them, not a substitute.

---

## 18. Final Takeaway

`RegexBPETokenizer` is the most architecturally complete tokenizer in this project.

Because it teaches the following essential principle:

> Effective tokenization is neither pure segmentation nor pure compression; it is the disciplined combination of the two, in which each operates within boundaries the other has defined.

Once this principle is internalized, the design of every modern production tokenizer becomes legible from a single perspective.
