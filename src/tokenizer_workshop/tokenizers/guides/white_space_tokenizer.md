# WhitespaceTokenizer

## 1. Purpose

`WhitespaceTokenizer` is the tokenizer class included in this project to introduce **the simplest possible tokenization rule**: split on whitespace, take the rest as tokens.

Its principal objective is to enable the learner to answer the following question clearly:

> Before any regex, before any morpheme analysis, before any learned model — what does tokenization look like when its only rule is to break on whitespace, and what does the resulting baseline tell us about what every other tokenizer adds?

This question is foundational. Every tokenizer in this project does something more than `WhitespaceTokenizer`. Some discriminate punctuation; some normalize case; some learn subword boundaries; some operate on bytes rather than characters. `WhitespaceTokenizer` does none of these things. It applies a single rule — `str.split()` — and stops.

Example:

```text
"hello world tokenizer"
    -> ["hello", "world", "tokenizer"]
    -> [0, 1, 2]
```

The defining characteristic of this tokenizer can be stated as follows:

> Tokenization reduced to its absolute minimum: a single call to Python's built-in whitespace splitter, with no preprocessing, no postprocessing, and no algorithmic content beyond what the standard library already provides.

This is what makes it valuable. As a baseline, it isolates every other tokenizer's contribution.

---

## 2. Why This Tokenizer Exists

This tokenizer occupies a specific architectural position within the project: it is the **null hypothesis** of tokenization.

### a) It serves as a baseline against which every other tokenizer can be measured

When the learner trains a `WordTokenizer`, a `RegexTokenizer`, a `SimpleBPETokenizer`, or any other tokenizer in the catalog, the natural question is: how much does the more sophisticated approach actually contribute? The answer is meaningful only against a baseline, and `WhitespaceTokenizer` is the most honest baseline possible.

If a more elaborate tokenizer cannot beat whitespace splitting on a given task, it is not earning its complexity.

### b) It exposes what whitespace alone cannot solve

By applying its single rule consistently, `WhitespaceTokenizer` reveals where that rule falls short:

* punctuation is not separated from words (`"hello,"` is a single token)
* casing is preserved as-is (`"Hello"` and `"hello"` are distinct)
* whitespace variants are normalized (tabs and newlines collapse to single spaces during decoding)
* OOV failures are catastrophic (any unseen string fails encoding)

Each of these failures motivates a corresponding refinement in another tokenizer in the project. `WhitespaceTokenizer` is the catalog entry that makes those motivations concrete.

### c) It demonstrates that tokenization need not be algorithmically elaborate

For applications operating on clean, well-formatted text — log lines, structured input, command-line arguments, controlled corpora — whitespace splitting can be entirely sufficient. The algorithm is so simple that there is nothing to debug and almost nothing to misunderstand.

This is a useful lesson in itself: the right tokenizer for a given task is not always the most sophisticated one available.

### d) It is the conceptual ancestor of `WordTokenizer`

`WordTokenizer` adds a regex (`\w+|[^\w\s]`) on top of what `WhitespaceTokenizer` does. The two tokenizers share their data-dependent vocabulary, their strict OOV behavior, and their position in the project's progression. Reading them in sequence — `WhitespaceTokenizer`, then `WordTokenizer` — makes the regex-driven refinement legible as exactly that: a refinement of the simpler whitespace baseline.

---

## 3. What "Whitespace" Means in This Project

In this project, whitespace tokenization is treated as **the application of `str.split()` without arguments**.

This single fact carries several consequences derived from Python's standard-library specification:

* the splitter recognizes spaces, tabs, newlines, carriage returns, and form feeds
* runs of consecutive whitespace are treated as a single delimiter
* leading and trailing whitespace are ignored
* the resulting tokens contain no whitespace by construction

The implementation, however, is deliberately constrained:

* it does not separate punctuation from words
* it does not normalize case
* it does not handle Unicode normalization (NFC, NFKC)
* it does not preserve the original whitespace structure
* it does not implement any `[UNK]` mechanism
* it does not learn subword units

The objective is therefore not to provide a useful tokenizer in any rich sense, but to provide the simplest possible reference point — the one against which every more sophisticated tokenizer in the project can be compared.

---

## 4. Core Idea

The tokenizer operates according to the following logic:

1. The input text is passed to `str.split()` with no arguments.
2. The resulting list of substrings is the token sequence.
3. During training, the unique tokens form the vocabulary.
4. Each unique token is assigned an integer identifier in first-appearance order.
5. During encoding, each token is mapped to its identifier.
6. During decoding, identifiers are mapped back to tokens and joined with single spaces.

Example:

```text
text = "hello world hello"

step 1 (str.split):
    ["hello", "world", "hello"]

step 2 (deduplication, order-preserving):
    ["hello", "world"]

step 3 (id assignment):
    {"hello": 0, "world": 1}

encode("hello world"):
    [0, 1]

decode([0, 1]):
    "hello world"
```

A general invariant follows directly from the splitting procedure:

> For any input, the number of tokens equals the number of whitespace-delimited substrings.

This is so straightforward that it produces a cleaner formal property than any other tokenizer in the project. There are no edge cases to enumerate beyond empty input.

---

## 5. Behavior on Whitespace Variants

A subtle but important property concerns how `str.split()` handles different whitespace forms.

The Python specification guarantees the following behavior:

```text
"hello   world"     -> ["hello", "world"]   # multiple spaces collapse
"hello\tworld"      -> ["hello", "world"]   # tab is whitespace
"hello\nworld"      -> ["hello", "world"]   # newline is whitespace
"  hello world  "   -> ["hello", "world"]   # leading/trailing stripped
"\u00A0hello"       -> ["\u00A0hello"]      # NO-BREAK SPACE not treated as whitespace by str.split
```

The first four behaviors are intuitive. The last is a subtle gotcha: not every Unicode space character qualifies as whitespace under `str.split()`'s definition. The U+00A0 NO-BREAK SPACE, U+202F NARROW NO-BREAK SPACE, and several other "space-like" characters are excluded from the delimiter set.

For typical input — text typed at a keyboard, parsed from a UTF-8 source — this distinction is invisible. For pathological input, it can produce tokens that contain embedded space-like characters. Learners should be aware of this property even if they do not encounter it directly.

The behavior is inherited from Python's standard library, not chosen by this implementation. Replicating the exact whitespace classification rules manually would defeat the purpose of using `str.split()`.

---

## 6. Vocabulary Behavior

For `WhitespaceTokenizer`, the vocabulary is defined as follows:

> The set of unique whitespace-delimited tokens observed in the training data, in the order in which they were first encountered.

Three implications are worth noting.

### a) The vocabulary is data-dependent

Different training texts produce different vocabularies. This parallels `WordTokenizer`, `RegexTokenizer`, `NgramTokenizer`, and `SubwordTokenizer`.

### b) The vocabulary grows linearly with corpus diversity

Every distinct whitespace-delimited substring becomes its own vocabulary entry. There is no learning, no merging, no compression. A corpus with `N` distinct whitespace-delimited strings produces a vocabulary of `N` entries.

This property makes `WhitespaceTokenizer` impractical for production use on large or diverse corpora. For a corpus containing `"hello"`, `"hello,"`, `"hello!"`, `"hello?"` as distinct strings, the vocabulary contains four separate entries — even though three of them differ from the first only by punctuation.

### c) Insertion order is preserved, not lexicographic

```python
unique_tokens = list(dict.fromkeys(tokens))
```

This is the same `dict.fromkeys` pattern used by `NgramTokenizer` and `SubwordTokenizer`, and for the same reason: it preserves first-appearance order while still producing a deterministic mapping.

The contrast with `WordTokenizer`, which uses `sorted(set(...))`, is consequential. Two `WhitespaceTokenizer` instances trained on the same text will produce identical mappings. Two trained on the same set of words in different sentence orders will not.

---

## 7. Training Logic

For this tokenizer, "training" does not correspond to machine-learning training in any meaningful sense. It refers to the construction of a vocabulary mapping from `str.split()` output.

Training proceeds through the following stages.

### a) Validation

```python
if not text or not text.strip():
    raise ValueError(...)
```

Empty and whitespace-only inputs are rejected. This is the same defensive check applied by `RegexTokenizer`, `NgramTokenizer`, `SubwordTokenizer`, `UnigramTokenizer`, and others. It is more conservative than `WordTokenizer`'s validation, which rejects only empty strings.

### b) Tokenization

The input is processed by `tokenize()`, which delegates to `str.split()`.

Notably, `tokenize()` does **not** require training. The vocabulary is needed only for `encode()` and `decode()`; whitespace splitting itself has no state.

### c) Order-preserving deduplication

Duplicate tokens are removed while preserving first-appearance order via `dict.fromkeys(...)`.

### d) Bidirectional mapping construction

Forward (`token → id`) and reverse (`id → token`) mappings are built from the deduplicated list.

### e) Trained-state transition

The `_trained` flag is set to `True`, enabling subsequent encode and decode calls.

The training procedure performs no iterative optimization, no frequency analysis, and no probability assignment. The vocabulary is whatever `str.split()` produces on the training input, with duplicates removed.

---

## 8. The `tokenize()` Method

`WhitespaceTokenizer.tokenize()` is independent of training, in the same pattern adopted by `RegexTokenizer`, `NgramTokenizer`, and `SubwordTokenizer`:

```python
def tokenize(self, text: str) -> list[str]:
    if not text or not text.strip():
        return []
    return text.split()
```

The rationale is that whitespace splitting is a pure function of the input text. No vocabulary is needed to produce the tokens themselves; the vocabulary is only needed to assign integer ids via `encode`.

The consequences of this design are:

* `tokenize()` is always available, even on a fresh tokenizer instance
* `tokenize()` never raises on legitimate input
* `tokenize()` produces strings; `encode()` produces integer ids
* `tokenize()` and `encode()` can disagree on the same input — `tokenize` always succeeds, while `encode` may raise on unseen tokens

This design contrasts with `WordTokenizer.tokenize()`, which is a wrapper around `encode()` and therefore inherits its training requirement and OOV behavior. The two designs reflect different priorities.

---

## 9. Encode Logic

The `encode()` method converts text into integer token ids:

1. The trained-state precondition is verified.
2. The input is tokenized via `str.split()`.
3. Each token is mapped to its integer id.

A deliberate strict design has been adopted: when the tokenizer encounters a token not seen during training, it raises a `ValueError` rather than substituting a fallback.

Example:

```text
training text:    "hello world"
training vocab:   {"hello": 0, "world": 1}

encoding "hello kitap":
    "hello" -> id 0
    "kitap" -> ValueError: Unknown token: kitap
```

This strict failure mode is consistent with `WordTokenizer`, `RegexTokenizer`, `NgramTokenizer`, and `SubwordTokenizer`. It contrasts with the `[UNK]`-based graceful degradation adopted by `UnigramTokenizer` and `WordPieceTokenizer`.

The pedagogical value is straightforward:

> A tokenizer that splits on whitespace alone has no machinery for handling unseen content. The strict failure mode makes this fragility visible, motivating the more sophisticated approaches introduced elsewhere in the project.

---

## 10. Decode Logic

The `decode()` method reconstructs a string from a list of integer token ids:

1. The trained-state precondition is verified.
2. Each id is mapped back to its token via `_id_to_token`.
3. Unknown ids raise a `ValueError`.
4. The tokens are joined with single spaces.

A subtle but consequential property follows from this design:

> Decoding is lossy with respect to whitespace structure but lossless with respect to token content.

The loss manifests as follows:

```text
input:    "hello   world\thello"
tokens:   ["hello", "world", "hello"]
decoded:  "hello world hello"
```

Multiple consecutive spaces, tabs, newlines, and other whitespace variations are all reduced to single spaces. The original whitespace formatting is irrecoverable.

In contrast, the **content** of each token is preserved exactly. Punctuation that was attached to a word during training is preserved through round-trip; capitalization is preserved exactly; embedded characters are passed through verbatim.

This is a different kind of lossy behavior from that of `UnigramTokenizer` (which loses whitespace **and** can replace content with `[UNK]`) or `SubwordTokenizer` (which loses whitespace **and** lowercases). `WhitespaceTokenizer` loses only whitespace structure — nothing else.

For applications where the input is known to be normalized (single-spaced, no tabs, no leading/trailing whitespace), this is effectively lossless.

---

## 11. Strengths

The strengths of `WhitespaceTokenizer` can be summarized as follows.

### a) The algorithm is the simplest possible

A single call to `str.split()`. There is no regex to misunderstand, no merge schedule to track, no probability table to maintain.

### b) It is fast and predictable

Tokenization is linear in the input length, with the constant factor of Python's optimized C-level `str.split()` implementation. Training is dominated by the deduplication step.

### c) Token content is preserved exactly

Apart from whitespace structure, every character in the input is preserved through round-trip on vocabulary-covered text. Capitalization, punctuation, and embedded characters all survive intact.

### d) The implementation is essentially impossible to misimplement

`str.split()` is a well-tested standard-library function. The wrapper around it adds a thin layer of vocabulary management, which is also straightforward.

### e) `tokenize()` is independent of training

Reporting and comparison layers can use the tokenizer immediately, without orchestrating a training step.

### f) It serves as a clean baseline

For benchmarking other tokenizers in the project, `WhitespaceTokenizer` provides the cleanest possible reference point.

---

## 12. Limitations

This tokenizer also operates under several deliberately accepted constraints.

### a) Punctuation is not separated from words

`"hello,"` is a single token. `"hello"` and `"hello,"` are distinct vocabulary entries. For text containing typical punctuation, this multiplies the vocabulary considerably.

### b) Case sensitivity is preserved

`"Hello"` and `"hello"` are distinct tokens. While technically faithful to the input, this is a liability for most NLP applications.

### c) The vocabulary grows linearly with corpus diversity

Every distinct whitespace-delimited substring becomes its own entry. For real-world corpora, this produces vocabularies that are too large to be useful.

### d) OOV failure is severe

Any input containing a token not seen during training causes `encode()` to fail. Because the vocabulary lacks structure, even minor variations (`"hello,"` vs `"hello"`) trigger this failure.

### e) Whitespace structure is lost in decoding

Multiple spaces, tabs, and newlines all collapse to single spaces. The original formatting is irrecoverable from the encoded representation.

### f) Unicode whitespace classification is inherited from Python

Some Unicode "space-like" characters (such as NO-BREAK SPACE, U+00A0) are not treated as delimiters by `str.split()`. This can produce tokens with embedded space-like characters in pathological input.

### g) No subword fallback, no `[UNK]`, no learned structure

Every aspect of the tokenizer is fixed at construction. There is nothing to tune, nothing to learn, and nothing to fall back to.

These limitations should not be interpreted as deficiencies; rather, they reflect **deliberate decisions of scope**.

---

## 13. Comparison with Other Tokenizers

### WhitespaceTokenizer vs WordTokenizer

These two are the project's pair of word-level tokenizers, and their differences are pedagogically central.

* `WhitespaceTokenizer` splits on whitespace only, attaching punctuation to adjacent words.
* `WordTokenizer` applies a regex (`\w+|[^\w\s]`) that separates punctuation from words.

In consequence:

* `"hello, world!"` → `["hello,", "world!"]` under `WhitespaceTokenizer`
* `"hello, world!"` → `["hello", ",", "world", "!"]` under `WordTokenizer`

The two tokenizers also differ in their vocabulary ordering: `WhitespaceTokenizer` preserves first-appearance order via `dict.fromkeys`; `WordTokenizer` uses `sorted(set(...))`. This produces different mappings on the same input.

`WordTokenizer` should be regarded as `WhitespaceTokenizer` augmented with a punctuation-separating regex. The two classes form a natural progression.

### WhitespaceTokenizer vs PunctuationTokenizer

* `WhitespaceTokenizer` does not separate punctuation.
* `PunctuationTokenizer` separates punctuation explicitly as its central feature.

In consequence, the two tokenizers represent opposite extremes of punctuation handling:

* `WhitespaceTokenizer` is punctuation-insensitive (treats it as part of words)
* `PunctuationTokenizer` is punctuation-aware (treats it as distinct tokens)

### WhitespaceTokenizer vs CharTokenizer

* `CharTokenizer` operates on individual characters.
* `WhitespaceTokenizer` operates on whitespace-delimited substrings.

In consequence:

* `CharTokenizer` produces sequences typically ten to fifty times longer.
* `CharTokenizer` has a much smaller vocabulary, bounded by the alphabet size of the training corpus.

The two tokenizers represent opposite ends of the granularity spectrum among the project's non-learned tokenizers.

### WhitespaceTokenizer vs the BPE family

* The BPE family (`SimpleBPETokenizer`, `ByteBPETokenizer`, `ByteLevelBPETokenizer`, `RegexBPETokenizer`) learns subword merges to produce compact, OOV-resistant vocabularies.
* `WhitespaceTokenizer` learns nothing.

The contrast quantifies what learning contributes. If a task can be solved with `WhitespaceTokenizer`, the BPE family adds nothing. If `WhitespaceTokenizer` fails (typically through OOV or vocabulary growth), the BPE family's added complexity becomes justified.

### WhitespaceTokenizer vs UnigramTokenizer / WordPieceTokenizer

The probabilistic-style tokenizers handle unseen content gracefully via `[UNK]`. `WhitespaceTokenizer` raises on the slightest mismatch.

This is the largest behavioral gap in the project. It is also the gap that `[UNK]` mechanisms exist to bridge.

---

## 14. Design Decisions in This Project

The fundamental design decisions adopted for `WhitespaceTokenizer` in this project are as follows:

* `str.split()` is used directly for tokenization, with no preprocessing
* the vocabulary preserves first-appearance order via `dict.fromkeys`
* `tokenize()` is independent of training, while `encode()` and `decode()` require it
* unseen tokens raise a `ValueError` rather than degrading via `[UNK]`
* decoding joins tokens with single spaces, normalizing the original whitespace structure
* punctuation is left attached to adjacent words
* case is preserved exactly
* educational simplicity is prioritized over linguistic refinement

Each of these decisions reflects the choice to provide an absolute baseline rather than a useful tokenizer.

---

## 15. Testing Perspective

The core behaviors verified by the tests for this tokenizer are as follows:

* empty and whitespace-only training inputs are rejected
* training builds a vocabulary whose size matches the unique whitespace-delimited token count
* identical inputs yield identical mappings (determinism)
* `tokenize()` splits on whitespace correctly
* multiple consecutive spaces collapse to single delimiters
* tabs and newlines are treated as whitespace
* empty and whitespace-only inputs to `tokenize()` return empty lists
* punctuation is NOT separated from adjacent words
* `encode()` and `decode()` raise before training
* `encode()` produces integer ids
* unseen tokens raise `ValueError` during encoding
* `encode()` returns an empty list on empty input
* `decode()` returns a string and handles empty list input gracefully
* unknown ids raise `ValueError` during decoding
* round-trip preserves token content exactly
* multiple spaces in the input do not survive round-trip (decoded as single spaces)
* Turkish and other Unicode characters are handled correctly (no regex involvement, so this is automatic)
* `vocab_size` is zero before training

These tests are pedagogically valuable because they verify both the **trivial structural invariants** of the tokenizer (it is, after all, just `str.split`) and the **subtle behavioral edges** (whitespace normalization, attached punctuation, OOV handling) that distinguish it from more sophisticated alternatives.

---

## 16. When to Use

`WhitespaceTokenizer` is particularly well suited to the following contexts:

* providing an absolute baseline for benchmarking other tokenizers
* tokenizing clean, normalized input (logs, command-line arguments, controlled corpora)
* educational settings exploring what tokenization is **before** any refinement is applied
* introducing the `tokenize` / `encode` / `decode` contract in its simplest form
* situations where speed and predictability matter more than linguistic accuracy

It is generally insufficient in the following contexts:

* applications operating on arbitrary natural-language text
* corpora containing punctuation, contractions, or compound words
* multilingual pipelines where Unicode normalization matters
* systems requiring graceful handling of unseen input
* any production NLP task where a more refined tokenizer is available

These cases call for more sophisticated tokenizers; `WhitespaceTokenizer` provides the conceptual baseline against which their contributions can be measured, not a substitute.

---

## 17. Final Takeaway

`WhitespaceTokenizer` is the tokenizer that establishes what tokenization looks like before any refinement is applied.

Because it teaches the following essential principle:

> The simplest possible tokenization rule — split on whitespace — is also the clearest baseline against which every more sophisticated tokenizer can be measured; understanding what whitespace splitting fails to do is the most efficient way to understand what every other tokenizer in the catalog is for.

Once this principle is internalized, the entire project reveals itself as a structured response to the limitations of this single, irreducible starting point.
