# NgramTokenizer

## 1. Purpose

`NgramTokenizer` is the tokenizer class included in this project to introduce the concept of **context-aware tokenization** through fixed-window word groupings.

Its principal objective is to enable the learner to answer the following question clearly:

> If individual words are themselves a coarse unit of meaning, can a tokenizer that groups consecutive words into fixed-size windows capture local context as a first-class property of the token itself?

This question motivates a tokenization approach in which a token is no longer a single linguistic unit, but a **sequence** of units — typically two, three, or four consecutive words. Such tokens carry richer information than isolated words because they preserve immediate surrounding context.

Example:

```text
"the cat sat" with n=2 (bigram)
    -> ["the cat", "cat sat"]
    -> [0, 1]
```

The defining characteristic of this tokenizer can be stated as follows:

> A token is no longer a unit of language; it is a window over a sequence of units.

This shift in perspective opens the door to context-sensitive analyses that no single-unit tokenizer in this project supports.

---

## 2. Why This Tokenizer Exists

This tokenizer fulfills several distinct pedagogical roles within the project.

### a) It introduces context as a property of the token itself

Every other tokenizer in this project produces tokens that are independent of their neighbors. A `WordTokenizer` token captures a single word in isolation. A `ByteTokenizer` token captures a single byte. A `SimpleBPETokenizer` token captures a learned subword.

`NgramTokenizer`, by contrast, produces tokens that **embed local context by construction**. The token `"the cat"` is not merely the word `"cat"`; it is the fact that `"cat"` was preceded by `"the"`.

This shift is small in implementation but substantial in pedagogical implication.

### b) It exposes the limitations of single-unit tokenization

By constructing tokens that span multiple words, `NgramTokenizer` makes visible what single-unit tokenizers cannot represent: collocation, word order, and immediate adjacency. The learner can directly contrast the two paradigms by training a `WordTokenizer` and an `NgramTokenizer` on the same text.

### c) It connects to a foundational tradition in NLP

N-gram models predate modern neural language modeling by decades. Long before transformers and word embeddings, n-gram statistics formed the backbone of speech recognition, machine translation, and information retrieval systems.

`NgramTokenizer` provides a hands-on entry point into this tradition. Although the tokenizer itself does not compute probabilities or build a language model, it produces the exact token units on which classical n-gram models operate.

### d) It illustrates the trade-off between context and combinatorial growth

A bigram vocabulary is roughly the square of a unigram vocabulary; a trigram vocabulary roughly the cube. `NgramTokenizer` makes this combinatorial explosion observable in concrete terms, motivating the design constraints of every tokenizer that succeeded the n-gram era.

---

## 3. What "N-gram" Means in This Project

In this project, an n-gram is treated as a **fixed-width window** over a sequence of whitespace-separated words.

The implementation, however, is deliberately simplified:

* it does not perform punctuation handling — punctuation attached to words remains attached
* it does not apply lowercasing or any other normalization
* it does not handle character-level n-grams (only word-level)
* it does not compute n-gram frequencies or probabilities
* it does not implement smoothing, backoff, or any classical language-model machinery

The objective is therefore not to construct an n-gram language model, but to expose the **tokenization aspect** of n-grams in a clear and inspectable form.

The class produces n-gram tokens; what one does with them — counting, modeling, comparing — is left to downstream code.

---

## 4. Core Idea

The tokenizer operates according to the following logic:

1. The text is split into words using whitespace.
2. A sliding window of size `n` traverses the word sequence.
3. Each window position produces one n-gram token (words joined by single spaces).
4. The unique n-gram tokens form the vocabulary.
5. Each unique token is assigned an integer identifier.
6. During encoding, the same n-gram extraction is applied, and tokens are mapped to ids.
7. During decoding, ids are mapped back to n-gram strings and concatenated with single spaces.

Example with `n = 2`:

```text
text = "the cat sat on the mat"

word sequence:
    ["the", "cat", "sat", "on", "the", "mat"]

sliding window positions:
    i=0 -> "the cat"
    i=1 -> "cat sat"
    i=2 -> "sat on"
    i=3 -> "on the"
    i=4 -> "the mat"

n-gram tokens:
    ["the cat", "cat sat", "sat on", "on the", "the mat"]
```

A general invariant follows directly from the sliding-window construction:

> For an input of `W` words, the number of n-gram tokens is `max(W − n + 1, W)`.

The fallback to `W` applies when `W < n`; this case is examined below.

---

## 5. Behavior on Short Inputs

A subtle but important behavior governs what happens when the input contains fewer words than `n`.

The naive sliding-window implementation would produce an empty list:

```text
text = "hello"
n = 3

words = ["hello"]
windows = []   # no valid window position
```

`NgramTokenizer` does **not** produce an empty list in this case. Instead, it falls back to returning the available words as-is:

```text
text = "hello"
n = 3

result: ["hello"]
```

This fallback is a deliberate design choice. The rationale is as follows.

### a) Empty output is rarely useful

A tokenizer that returns no tokens for a non-empty input violates a basic expectation: that meaningful input produces meaningful output. The fallback ensures that any non-trivial input produces at least some tokens.

### b) Downstream code can rely on a non-empty invariant

API layers, comparison managers, and reporting components frequently iterate over tokenizer output. A fallback that guarantees at least one token (for any non-empty input) simplifies their handling logic.

### c) The fallback is honest, not silent

The result is a unigram-flavored degradation, not a fabricated value. The token `"hello"` is genuinely the only meaningful unit available, and the tokenizer simply returns it.

This behavior trades formal n-gram purity for practical robustness — a trade-off explicitly aligned with the project's pedagogical orientation.

---

## 6. Vocabulary Behavior

For `NgramTokenizer`, the vocabulary is defined as follows:

> The number of unique n-gram tokens observed in the training data, in the order in which they were first encountered.

Two implications are worth noting.

### a) The vocabulary is data-dependent

Different corpora produce different vocabularies. A small text yields a small vocabulary. A previously unseen n-gram raises an error during encoding.

This is the same dependency observed in `WordTokenizer` and `RegexTokenizer`, but the rate at which the vocabulary grows is markedly steeper. For a corpus of `W` distinct words, a unigram vocabulary contains at most `W` entries. A bigram vocabulary may contain up to `W²` entries; a trigram vocabulary up to `W³`. In practice the figures are smaller, but the trend is unmistakable.

### b) Insertion order is preserved, not lexicographic order

Unlike `WordTokenizer`, which sorts unique tokens before assigning identifiers, `NgramTokenizer` preserves the order in which n-grams first appear in the training text:

```python
unique_tokens = list(dict.fromkeys(ngrams))
```

This is also deterministic — `dict.fromkeys` guarantees insertion order in Python 3.7 and later — but it produces a different mapping from the lexicographic ordering used by other tokenizers. Two tokenizers trained on the same text will produce identical mappings; two tokenizers trained on the same set of words in different sentence orders will not.

This is a consequential decision and is examined further in the design-decisions section.

---

## 7. Training Logic

For this tokenizer, "training" refers to the construction of a vocabulary from a given text.

Training proceeds through the following stages.

### a) Validation

```python
if not text or not text.strip():
    raise ValueError(...)
```

Empty and whitespace-only inputs are rejected. This is a more defensive check than the one used by `WordTokenizer`, which accepts whitespace-only strings.

### b) Word segmentation

The text is split using Python's default `str.split()` semantics — that is, on any run of whitespace:

```python
words = text.split()
```

Note that this segmentation is purely whitespace-based. Punctuation is not treated as a separate token; trailing punctuation remains attached to the preceding word. The bigram of `"the cat."` is `"the cat."`, not `"the cat"` followed by a separate `"."`.

### c) N-gram extraction

The internal `_build_ngrams` helper applies the sliding window:

```python
ngrams = self._build_ngrams(words)
```

### d) Order-preserving deduplication

Duplicate n-grams are removed while preserving first-appearance order:

```python
unique_tokens = list(dict.fromkeys(ngrams))
```

### e) Bidirectional mapping construction

Forward and reverse mappings are constructed from the deduplicated list:

```python
self._token_to_id = {token: i for i, token in enumerate(unique_tokens)}
self._id_to_token = {i: token for token, i in self._token_to_id.items()}
```

### f) Trained-state transition

The `_trained` flag is set to `True`, enabling subsequent encode and decode calls.

---

## 8. Encode Logic

The `encode()` method converts each n-gram in the input text into an integer token id.

A deliberate strict design has been adopted: when the tokenizer encounters an n-gram not seen during training, it raises a `ValueError` rather than substituting a fallback.

Example:

```text
training text:  "the cat sat"
encoding text:  "unknown text here"

result: ValueError("Unknown token: unknown text")
```

This strictness is consistent with the policies of `WordTokenizer` and `RegexTokenizer` in the project. The pedagogical motivation is identical:

> What should a tokenizer do when confronted with an n-gram it has never seen?

For n-gram tokenizers, this question is even more pressing than for word-level tokenizers. The combinatorial space of possible n-grams is vast, and even moderately long encoding inputs will frequently contain n-grams unseen during training. The strict failure mode makes this fragility explicit.

This is one of the most important lessons that `NgramTokenizer` conveys:

> Context-rich tokens carry more information per unit, but they fail more often on unseen input.

---

## 9. Decode Logic

The `decode()` method reconstructs a string from a list of n-gram token ids:

1. The trained-state precondition is verified.
2. Each id is mapped back to its n-gram string via `_id_to_token`.
3. The n-gram strings are joined with single spaces.

A subtle but consequential property follows from this design:

> Decoding does **not** invert encoding. The original text cannot be recovered from the token ids.

This is a direct consequence of n-gram overlap. Each adjacent pair of bigrams shares a word; decoding without overlap-resolution duplicates that shared word in the output.

Example:

```text
input:       "the cat sat"

bigrams:     ["the cat", "cat sat"]

naive join:  "the cat cat sat"   # the word "cat" appears twice
```

The decoder makes no attempt to resolve this overlap. This is a deliberate simplification, motivated by the following considerations.

### a) Overlap resolution is not unique

Multiple input texts can produce the same bigram sequence under specific corner cases. A perfectly invertible decoder would need to enforce assumptions about the input that the tokenizer does not make.

### b) The decoded form remains useful for inspection

Although `"the cat cat sat"` is not the original input, it is a readable representation that makes the tokenization visible. For debugging and reporting purposes, this is more valuable than a faithfully reconstructed but opaque output.

### c) The lossy behavior is honest

The test suite explicitly verifies that `decode(encode("the cat sat"))` produces `"the cat cat sat"` — that is, the lossiness is documented and tested rather than hidden. Learners are not misled into thinking n-gram tokenization is reversible.

This makes explicit a critical design principle:

> Some tokenizers are intentionally lossy. The honest course of action is to document, test, and contain the loss — not to disguise it.

---

## 10. The `tokenize()` Method

`NgramTokenizer.tokenize()` differs from its counterparts in `WordTokenizer` and `ByteLevelBPETokenizer`. It performs raw n-gram segmentation **without requiring training**:

```python
def tokenize(self, text: str) -> list[str]:
    if not text or not text.strip():
        return []
    words = text.split()
    return self._build_ngrams(words)
```

The rationale is that n-gram extraction is a pure function of the input text and the parameter `n`. No vocabulary is needed to produce the n-gram strings themselves; the vocabulary is only needed to assign identifiers via `encode`.

The consequences of this design are:

* `tokenize()` can be invoked at any time, on any tokenizer instance
* `tokenize()` never raises an error on legitimate input
* `tokenize()` produces strings; `encode()` produces integer ids

This contrasts with `WordTokenizer.tokenize()`, which is implemented as a wrapper around `encode()` and therefore inherits its training requirement and OOV behavior. The two designs reflect different priorities, and learners should examine both to appreciate the trade-offs.

---

## 11. Strengths

The strengths of `NgramTokenizer` can be summarized as follows.

### a) Local context is captured by construction

A bigram token records both a word and its immediate predecessor. This is a property that no single-unit tokenizer in the project provides.

### b) The implementation is conceptually transparent

The sliding window is a single comprehension. The vocabulary mapping is a dictionary. There are no learned parameters and no statistical heuristics. Every step can be inspected in isolation.

### c) The parameter `n` is exposed as a tunable knob

The constructor accepts `n` as an argument. Learners can train tokenizers with `n = 1, 2, 3, …` on the same corpus and directly observe how vocabulary size and token count scale.

### d) Determinism is guaranteed

Both the sliding window and `dict.fromkeys` deduplication are deterministic. Identical inputs yield identical mappings across runs.

### e) The class connects to a foundational NLP tradition

For learners with broader NLP exposure, `NgramTokenizer` provides a hands-on entry point into the tokenization layer of classical n-gram language models.

---

## 12. Limitations

This tokenizer also operates under several deliberately accepted constraints.

### a) Out-of-vocabulary failure is exceptionally severe

Because the combinatorial space of possible n-grams is vast, the OOV failure mode is far more frequent than in single-unit tokenizers. Even encoding inputs drawn from the same domain as the training data will routinely contain unseen n-grams.

### b) Decoding is intentionally lossy

The original text cannot be recovered from the token ids. For applications requiring round-trip fidelity, `NgramTokenizer` is unsuitable.

### c) Vocabulary growth is combinatorial

For a corpus of `W` distinct words, the bigram vocabulary may approach `W²` entries and the trigram vocabulary `W³`. Even moderately sized corpora can produce tokenizers with impractically large vocabularies.

### d) Whitespace is the only segmentation rule

The tokenizer relies on `str.split()` for word boundaries. Punctuation attached to words is not separated, contractions are not handled, and Unicode-aware word boundaries are not applied. For text containing rich punctuation, the resulting n-grams may be linguistically odd (`"the cat."` rather than `"the cat"`).

### e) Insertion-order ids are non-canonical

Because the vocabulary is assigned in first-appearance order, the same set of words encountered in different sentence orderings produces different mappings. This makes cross-corpus comparison of token ids meaningless without rebuilding the vocabulary.

### f) No lowercasing or normalization

`The` and `the` are treated as distinct units. `The cat` and `the cat` are therefore different bigrams. For NLP applications this is generally a liability, but for the project's pedagogical purposes the absence of normalization keeps the algorithm's behavior visible.

These limitations should not be interpreted as deficiencies; rather, they reflect **deliberate decisions of scope**.

---

## 13. Comparison with Other Tokenizers

### NgramTokenizer vs WordTokenizer

* `WordTokenizer` produces single-word tokens.
* `NgramTokenizer` produces multi-word tokens.

In consequence:

* `WordTokenizer` produces shorter sequences with smaller vocabularies.
* `NgramTokenizer` produces tokens with embedded local context but with a far larger vocabulary.

A particularly instructive observation: `NgramTokenizer` with `n = 1` is structurally equivalent to `WordTokenizer` operating on whitespace-split input. The unigram case provides a natural baseline against which higher-order n-grams can be compared.

### NgramTokenizer vs RegexTokenizer

* `RegexTokenizer` separates words and punctuation as distinct tokens.
* `NgramTokenizer` treats punctuation as part of the surrounding word, since it relies on whitespace segmentation only.

In consequence:

* `RegexTokenizer` produces linguistically cleaner units.
* `NgramTokenizer` produces tokens whose composition depends on superficial whitespace.

### NgramTokenizer vs SimpleBPETokenizer

* `SimpleBPETokenizer` learns subword units that capture intra-word patterns.
* `NgramTokenizer` constructs supra-word units that capture inter-word patterns.

The two tokenizers operate at opposite scales. BPE asks: "what fragment within a word is reusable?" N-grams ask: "what window across words is recurrent?"

### NgramTokenizer vs classical n-gram language models

Classical n-gram models extend the tokenization done here with frequency counting, conditional probability estimation, and smoothing techniques (Laplace, Kneser-Ney, etc.). `NgramTokenizer` provides the segmentation layer of such models — the layer that determines what counts as a "unit" for subsequent statistical analysis.

For learners pursuing classical NLP, `NgramTokenizer` is the natural entry point. The tokens it produces are the inputs that frequency-counting code would consume.

---

## 14. Design Decisions in This Project

The fundamental design decisions adopted for `NgramTokenizer` in this project are as follows:

* whitespace-based word segmentation is preferred for its simplicity and transparency
* the parameter `n` is exposed as a constructor argument
* the vocabulary preserves first-appearance order rather than sorted order
* short inputs (with fewer words than `n`) fall back to the available words
* `tokenize()` is independent of training, while `encode()` and `decode()` require it
* unseen n-grams raise an error rather than being silently substituted
* decoding is intentionally lossy and the loss is documented and tested
* educational clarity is prioritized over linguistic refinement

Each of these decisions reflects a balance between architectural simplicity and pedagogical accessibility.

---

## 15. Testing Perspective

The core behaviors verified by the tests for this tokenizer are as follows:

* invalid `n` values (zero or negative) are rejected
* empty training input is rejected
* training builds a vocabulary whose size matches the unique n-gram count
* encode and decode raise errors before training
* encode produces a list of integer ids
* unseen n-grams raise a `ValueError` during encoding
* unknown ids raise a `ValueError` during decoding
* decode produces a string, with the documented lossy reconstruction
* the lossy round-trip is explicitly verified: `decode(encode("the cat sat"))` produces `"the cat cat sat"`
* `tokenize()` operates without requiring training
* short inputs trigger the fallback behavior
* identical inputs yield identical encoded outputs

It is worth noting that the round-trip test verifies a specific lossy outcome rather than equality with the input. This is an honest reflection of the tokenizer's behavior: full round-trip fidelity is not guaranteed, and the test is designed accordingly.

---

## 16. When to Use

`NgramTokenizer` is particularly well suited to the following contexts:

* introducing the concept of context-aware tokenization
* explaining the distinction between unit-level and window-level tokenization
* providing a baseline for comparison with subword and word-level tokenizers
* small experiments on closed corpora where the n-gram space can be enumerated
* educational settings exploring classical NLP foundations

It is generally insufficient in the following contexts:

* applications requiring robustness to unseen text
* large corpora where combinatorial vocabulary growth becomes prohibitive
* systems requiring round-trip text reconstruction
* modern neural NLP pipelines, in which subword approaches have largely supplanted n-gram tokenization

These cases call for more advanced architectures, several of which are introduced elsewhere in the project.

---

## 17. Final Takeaway

`NgramTokenizer` is the tokenizer that most directly challenges the assumption that a token must be a single linguistic unit.

Because it teaches the following essential principle:

> A token is whatever the tokenizer decides to count as one — and treating multi-word windows as tokens reframes the entire question of what tokenization is for.

Once this principle is internalized, the design space of tokenization expands considerably, and the rationale behind every more constrained tokenizer in the project becomes legible from a richer perspective.
