# WordTokenizer

## 1. Purpose

`WordTokenizer` is the tokenizer class included in this project to introduce the concept of **word-level tokenization** in its purest and most direct form.

Its principal objective is to enable the learner to answer the following question clearly:

> If words are treated as the natural units of language, can a tokenizer that operates exclusively at the word level be both effective and complete?

This question is foundational to the history of natural language processing. For decades, the prevailing assumption was that words constitute the fundamental units of meaning. `WordTokenizer` makes this assumption operational, exposing both its appeal and its limitations.

Example:

```text
"Merhaba dünya!" -> ["Merhaba", "dünya", "!"] -> [0, 1, 2]
```

This approach is particularly valuable from a pedagogical standpoint, because it allows the learner to confront the central trade-off of word-level tokenization directly:

> A coarser unit produces shorter sequences, but at the cost of an unbounded vocabulary and brittle behavior on unseen input.

---

## 2. Why This Tokenizer Exists

This tokenizer fulfills several distinct pedagogical roles within the project.

### a) It establishes the natural human notion of a token

Before encountering subword or byte-level approaches, the learner naturally assumes that a token is a word. `WordTokenizer` formalizes this intuition and demonstrates what a tokenizer based on it actually looks like in code.

### b) It exposes the out-of-vocabulary problem in its starkest form

Because `WordTokenizer` treats each distinct word as an independent token, any word not present in the training data cannot be encoded. The tokenizer raises an error rather than silently substituting a fallback.

This behavior, while inconvenient in practice, is pedagogically valuable. It makes the OOV problem visible in a way that more sophisticated tokenizers obscure.

### c) It motivates the entire subword tokenization tradition

The historical emergence of BPE, WordPiece, and Unigram tokenization is largely a response to the limitations of word-level approaches. Without first appreciating those limitations, the motivation for subword methods remains abstract.

`WordTokenizer` provides this motivation in concrete form.

### d) It mirrors `nltk.word_tokenize()` in its underlying logic

The default behavior of `WordTokenizer` is conceptually aligned with that of NLTK's `word_tokenize()`: words and punctuation marks are treated as distinct tokens, and the regex-based segmentation rule is essentially the same.

For learners with prior exposure to NLP libraries, this alignment offers a familiar reference point.

---

## 3. What "Word-Level" Means in This Project

In this project, word-level tokenization is treated as a **rule-based segmentation** approach applied at a coarser granularity than character-level or byte-level tokenizers.

The implementation, however, is deliberately simplified:

* it does not perform lemmatization or stemming
* it does not apply lowercasing or other normalization
* it does not handle contractions (`don't`, `we'll`) as multi-token units
* it does not address compound words or hyphenation
* it does not maintain whitespace information

In particular, the absence of lowercasing is a consequential decision. Under the default configuration, `Merhaba` and `merhaba` are treated as two distinct tokens. This is a faithful reflection of how naive word-level tokenizers behave, and it is one of the first surprises encountered by learners experimenting with the class.

The objective is not to replicate the behavior of production-grade preprocessing pipelines, but to render the principle of word-level tokenization pedagogically accessible.

---

## 4. Core Idea

The tokenizer operates according to the following logic:

1. The text is segmented into words and punctuation marks using a regex pattern.
2. The unique tokens are extracted from the training data.
3. Each unique token is assigned an integer identifier.
4. During encoding, each token is mapped to its identifier.
5. During decoding, identifiers are mapped back to tokens and joined with single spaces.

The regex pattern used is:

```text
\w+ | [^\w\s]
```

This pattern partitions the input into two mutually exclusive token classes:

* `\w+` — sequences of letters, digits, or underscores (i.e., words and numbers)
* `[^\w\s]` — single non-word, non-whitespace characters (i.e., punctuation)

Whitespace is consumed but not preserved, which has consequences examined later in this document.

Example:

```text
text = "Merhaba dünya!"

regex matches:
    "Merhaba" -> word
    "dünya"   -> word
    "!"       -> punctuation

unique tokens (sorted) = ["!", "Merhaba", "dünya"]

token_to_id = {
    "!":       0,
    "Merhaba": 1,
    "dünya":   2,
}

encode("Merhaba dünya!") -> [1, 2, 0]
```

A second example, illustrating the OOV problem:

```text
training text:  "Merhaba dünya"
encoding text:  "Merhaba kitap"

"kitap" was never seen during training -> ValueError
```

---

## 5. Why Unicode-Aware Regex Matters

The regex implementation in this tokenizer is Unicode-aware. This is not an arbitrary detail; it is a deliberate design decision.

In Python, `\w` does not refer only to ASCII letters by default. Within a Unicode-aware regex, characters such as:

* Turkish characters (`ç`, `ğ`, `ü`, `ş`, `ö`, `ı`, `İ`)
* characters from Cyrillic, Greek, and other scripts
* characters from CJK writing systems

are also recognized as word characters.

Example:

```text
"dünya" -> ["dünya"]
```

If the regex were ASCII-restricted, the character `ü` would be excluded from the word class and treated as punctuation, fragmenting the word into multiple tokens.

This property is essential for any tokenizer intended for use with non-English text. Pedagogically, it conveys an important lesson:

> A tokenizer's apparent simplicity often conceals deep dependencies on language and encoding assumptions.

---

## 6. Vocabulary Behavior

For `WordTokenizer`, the vocabulary is defined as follows:

> The number of unique word and punctuation tokens observed in the training data.

The implications of this are as follows:

* the vocabulary is **data-dependent**
* different corpora produce different vocabularies
* a small text yields a small vocabulary
* a previously unseen word raises an error during encoding
* the vocabulary can grow rapidly with corpus size

This behavior parallels that of `CharTokenizer` and `RegexTokenizer`, but operates at a much coarser granularity:

| Tokenizer | Token unit | Typical vocabulary size |
|---|---|---|
| `CharTokenizer` | a single character | small (tens to hundreds) |
| `ByteTokenizer` | a single byte | fixed at 256 |
| `RegexTokenizer` | a word, number, or punctuation mark | medium |
| `WordTokenizer` | a word, number, or punctuation mark | medium to large |

The distinction between `RegexTokenizer` and `WordTokenizer` deserves attention. The two tokenizers use the same regex pattern; their difference lies elsewhere — in particular, in their handling of the decoded output and in their public API. This contrast is examined further in the comparison section.

---

## 7. Training Logic

For this tokenizer, "training" does not correspond to machine-learning training in the conventional sense. It refers to the construction of a vocabulary from a given text.

Training proceeds through the following stages.

### a) Validation

```python
if not text:
    raise ValueError(...)
```

An empty training text is rejected. Note, however, that whitespace-only input is **not** rejected by the same check; this is a minor difference from some of the project's other tokenizers and is documented further in the limitations section.

### b) Segmentation

The text is segmented using the regex pattern:

```python
tokens = re.findall(r"\w+|[^\w\s]", text)
```

### c) Vocabulary construction

The unique tokens are extracted, sorted, and assigned integer identifiers:

```python
unique_tokens = sorted(set(tokens))
token_to_id = {token: idx for idx, token in enumerate(unique_tokens)}
```

The `sorted(...)` step is essential. Without it, the ordering of the mapping could become non-deterministic in certain configurations, rendering the training output unstable across runs.

### d) Bidirectional mapping

A reverse mapping is also constructed to support decoding:

```python
id_to_token = {idx: token for token, idx in token_to_id.items()}
```

This bidirectional structure is shared with `CharTokenizer` and `RegexTokenizer`; only the granularity of the keys differs.

---

## 8. Encode Logic

The `encode()` method converts each regex token into an integer token id.

A deliberate design decision has been made at this stage: when the tokenizer encounters a token not seen during training, it **does not pass over it silently**, **does not fabricate a fallback**, and **does not substitute an unknown-token marker**. Instead, it raises a `ValueError` directly.

Example:

```text
training text:  "Merhaba dünya"
encoding text:  "Merhaba kitap"  # "kitap" never seen

result: ValueError("Unknown token encountered ... kitap")
```

This decision is pedagogically justified, as it makes the following problem explicit:

> What should a tokenizer do when confronted with a word it has never seen?

In practice, this problem is addressed through mechanisms such as `<UNK>` tokens, subword fallback (BPE, WordPiece), or byte-level fallback (byte BPE). The strict failure mode of `WordTokenizer` exposes the problem in its starkest form, motivating the more sophisticated approaches introduced later in the project.

---

## 9. Decode Logic

The `decode()` method converts a list of integer token ids back into text.

The procedure is straightforward:

1. Each id is mapped back to its token via `id_to_token`.
2. The tokens are joined with single spaces.

A subtle but important property follows from this design:

> Decoding is lossy. The original spacing and the precise placement of punctuation are not preserved.

This manifests in two specific ways.

### a) Punctuation acquires leading whitespace

```text
input:    "merhaba dünya!"
encoded:  [..., id("dünya"), id("!")]
decoded:  "merhaba dünya !"   # note the space before "!"
```

Unlike `RegexTokenizer`, which removes the unwanted whitespace before punctuation through a post-processing step, `WordTokenizer` performs no such cleanup. The exclamation mark is reproduced as a separate space-delimited token.

### b) Repeated whitespace collapses to a single space

```text
input:    "hello       world"
decoded:  "hello world"
```

Multiple consecutive spaces, tabs, and newlines are all reduced to a single space character.

These properties make explicit a critical principle of tokenizer design:

> Round-trip fidelity is not an automatic property of a tokenizer; it is a design goal that must be deliberately pursued.

`WordTokenizer` does not pursue round-trip fidelity; it accepts a degree of loss in exchange for a simple and transparent decoding procedure. The project's tests acknowledge this directly: the round-trip test verifies word ordering via `decoded.split() == text.split()`, not byte-for-byte equality.

---

## 10. The `tokenize()` Method

A subtle but important architectural detail concerns the `tokenize()` method.

Unlike some of the project's other tokenizers, in which `tokenize()` performs pure regex segmentation and operates without prior training, `WordTokenizer.tokenize()` is implemented as a wrapper around `encode()`:

```python
def tokenize(self, text: str) -> list[str]:
    token_ids = self.encode(text)
    return [self._id_to_token[token_id] for token_id in token_ids]
```

The rationale is alignment with the project's `CompareManager`, which expects tokenizers to return the **vocabulary form** of each token rather than the raw segmentation.

The consequence of this design is that:

* `tokenize()` requires training (because `encode()` does)
* `tokenize()` raises a `ValueError` on unseen tokens (because `encode()` does)
* `tokenize()` only returns tokens that appear in the trained vocabulary

This contrasts with `RegexTokenizer.tokenize()`, which performs raw regex segmentation and never raises errors. The distinction is small but consequential, and learners should be aware of it.

---

## 11. Strengths

The strengths of `WordTokenizer` can be summarized as follows.

### a) Sequence length is significantly reduced

Compared to `CharTokenizer` or `ByteTokenizer`, the same text is represented in far fewer tokens. A sentence of fifty characters might collapse into eight to ten word tokens.

### b) Tokens correspond to meaningful linguistic units

Each token is, by construction, either a word or a punctuation mark. This granularity is closely aligned with human intuition about language.

### c) The implementation is conceptually transparent

Every step — regex segmentation, sorting, identifier assignment, lookup — can be inspected and understood independently. There are no learned parameters, no statistical heuristics, and no opaque state.

### d) The output is interpretable

A learner inspecting `_token_to_id` can immediately understand the tokenizer's behavior. This is rarely possible with subword tokenizers.

### e) Determinism is straightforward to guarantee

The sorted unique tokens produce a deterministic mapping, and the regex segmentation is itself deterministic. Identical inputs yield identical outputs across runs.

---

## 12. Limitations

This tokenizer also operates under several deliberately accepted constraints.

### a) Out-of-vocabulary words are not handled

Any word not present in the training data causes encoding to fail. There is no `<UNK>` token, no subword fallback, and no byte-level fallback. This is the single most significant limitation, and it is precisely what motivates the subword tokenizers introduced later in the project.

### b) The vocabulary grows linearly with the corpus

In a sufficiently large training set, virtually every distinct word becomes a separate vocabulary entry. This makes the vocabulary impractically large for real-world corpora — a problem that is structurally absent in fixed-vocabulary or subword approaches.

### c) Case sensitivity is preserved

`Merhaba` and `merhaba` are treated as distinct tokens. While this is technically correct, in most NLP applications it is a liability rather than a feature, since it doubles or triples the vocabulary without adding semantic value.

### d) Round-trip fidelity is not preserved

The original spacing of the text is lost during tokenization, and punctuation acquires unwanted leading spaces during decoding. For applications requiring byte-for-byte reconstruction, `WordTokenizer` is not suitable.

### e) Whitespace-only training input is not detected

The empty-input check (`if not text:`) does not trigger on whitespace-only strings. A training call such as `train("   ")` succeeds silently and produces a tokenizer with an empty vocabulary, which then fails at the first `encode()` call. A more defensive check (`if not text.strip():`) would make this failure mode explicit at training time.

### f) Compound and inflected forms are unrelated

The words `run`, `running`, and `runner` share no token-level relationship under this tokenizer. Each is treated as a fully independent vocabulary entry. This is precisely the structural redundancy that subword tokenizers are designed to eliminate.

These limitations should not be interpreted as deficiencies; rather, they reflect **deliberate decisions of scope**.

---

## 13. Comparison with Other Tokenizers

### WordTokenizer vs CharTokenizer

* `CharTokenizer` operates on individual characters.
* `WordTokenizer` operates on words and punctuation.

In consequence:

* `CharTokenizer` produces longer sequences but tolerates unseen characters within its alphabet.
* `WordTokenizer` produces much shorter sequences but fails on any unseen word.

The two tokenizers represent opposite ends of the granularity spectrum.

### WordTokenizer vs ByteTokenizer

* `ByteTokenizer` provides universal coverage with a fixed vocabulary.
* `WordTokenizer` provides interpretable units with a data-dependent vocabulary.

In consequence:

* `ByteTokenizer` never fails on unseen input.
* `WordTokenizer` fails immediately on any unseen word.

This is a clear trade-off between robustness and interpretability.

### WordTokenizer vs RegexTokenizer

These two tokenizers are structurally similar — they share the same regex pattern and the same data-dependent vocabulary. The differences are subtler.

* `RegexTokenizer.tokenize()` performs raw regex segmentation without requiring training.
* `WordTokenizer.tokenize()` is a wrapper around `encode()` and therefore requires training.

* `RegexTokenizer.decode()` removes whitespace before punctuation, producing more readable output.
* `WordTokenizer.decode()` performs no such cleanup.

* `RegexTokenizer` exposes its mappings as public attributes (`token_to_id`, `id_to_token`).
* `WordTokenizer` exposes them as private attributes (`_token_to_id`, `_id_to_token`).

The two tokenizers occupy similar pedagogical territory, but with subtly different API contracts. Learners should examine both to appreciate how small implementation choices shape user-facing behavior.

### WordTokenizer vs SimpleBPETokenizer

* `SimpleBPETokenizer` learns recurring subword patterns through merge operations.
* `WordTokenizer` learns nothing; it merely segments.

In consequence:

* `SimpleBPETokenizer` adapts its tokenization to the data.
* `WordTokenizer` applies a fixed segmentation rule.

`SimpleBPETokenizer` represents the natural successor to `WordTokenizer` in the project's progression.

### WordTokenizer vs nltk.word_tokenize

NLTK's `word_tokenize` is the canonical real-world counterpart of this class. The two share the same conceptual foundation, but NLTK's implementation incorporates a large body of language-specific heuristics (handling of contractions, hyphenated compounds, abbreviations, etc.) that this class deliberately omits.

`WordTokenizer` should therefore be regarded as a pedagogical simplification of word-level tokenization, not a substitute for production NLP toolkits.

---

## 14. Design Decisions in This Project

The fundamental design decisions adopted for `WordTokenizer` in this project are as follows:

* a Unicode-aware regex pattern is used to capture words and punctuation
* the vocabulary is constructed deterministically through sorted unique tokens
* unseen tokens raise an error rather than being silently substituted
* `tokenize()` is implemented as a wrapper around `encode()` for consistency with `CompareManager`
* whitespace is not preserved in the decoded output
* case sensitivity is preserved (no normalization is applied)
* internal mappings are exposed as private attributes
* educational clarity is prioritized over production-grade robustness

Each of these decisions reflects a balance between architectural simplicity and pedagogical accessibility.

---

## 15. Testing Perspective

The core behaviors verified by the tests for this tokenizer are as follows:

* `vocab_size` reflects the number of unique tokens after training
* the encode output is a list of integer ids
* decoded output contains the original word forms
* the round-trip preserves word order (verified via `decoded.split() == text.split()`)
* punctuation marks are produced as separate tokens
* unseen tokens raise a `ValueError` during encoding
* unknown ids raise a `ValueError` during decoding
* invoking encode or decode prior to training raises a `ValueError`
* empty training input raises a `ValueError`
* identical training inputs yield identical encoding outputs

It is worth noting that the round-trip test verifies word ordering rather than byte-for-byte equality. This is an honest reflection of the tokenizer's actual behavior: full round-trip fidelity is not guaranteed, and the test is designed accordingly.

---

## 16. When to Use

`WordTokenizer` is particularly well suited to the following contexts:

* introducing the concept of word-level tokenization
* demonstrating the out-of-vocabulary problem in concrete form
* providing a reference behavior for comparison with subword tokenizers
* small experiments on closed corpora where every word is known in advance
* educational settings where transparency is more important than coverage

It is generally insufficient in the following contexts:

* applications requiring robustness to unseen words
* large corpora where vocabulary growth becomes prohibitive
* systems requiring byte-for-byte round-trip fidelity
* multilingual pipelines where case and morphology matter
* modern NLP pipelines, in which subword approaches have largely supplanted word-level tokenization

These cases call for more advanced architectures, several of which are introduced later in the project.

---

## 17. Final Takeaway

`WordTokenizer` is the tokenizer that most closely matches naive intuitions about language, and for that reason it is also the one whose limitations are most instructive.

Because it teaches the following essential principle:

> Words may feel like the natural units of language, but treating them as fixed tokens forces a tokenizer into an unbounded vocabulary and a brittle relationship with unseen input.

Once this principle is internalized, the design of every subword tokenizer — BPE, WordPiece, Unigram — becomes legible as a response to the precise limitations exposed here.
