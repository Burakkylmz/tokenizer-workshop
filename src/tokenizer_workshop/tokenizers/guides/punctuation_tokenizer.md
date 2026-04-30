# PunctuationTokenizer

## 1. Purpose

`PunctuationTokenizer` is the tokenizer class included in this project to introduce **punctuation-aware tokenization** as an explicit refinement of whitespace-only segmentation.

Its principal objective is to enable the learner to answer the following question clearly:

> Once whitespace alone is no longer trusted to determine word boundaries, what is the smallest possible refinement to the rule, and what does that refinement add?

This question matters because the boundary between `WhitespaceTokenizer` and more sophisticated tokenizers is not crossed all at once. The first refinement — separating punctuation from the words it touches — is small enough to specify in a single regex pattern, yet substantial enough to change the meaning of the resulting token sequence.

Example:

```text
"Hello, world!"

WhitespaceTokenizer:
    -> ["Hello,", "world!"]   # punctuation glued to words

PunctuationTokenizer:
    -> ["hello", ",", "world", "!"]   # punctuation as separate tokens
```

The defining characteristic of this tokenizer can be stated as follows:

> A minimum-effort refinement of whitespace tokenization that preserves the simplicity of the rule while restoring the semantic distinction between content and punctuation.

---

## 2. Why This Tokenizer Exists

This tokenizer fulfills several distinct pedagogical roles within the project.

### a) It bridges whitespace tokenization and regex tokenization

`WhitespaceTokenizer` operates on a single principle: split on whitespace. `RegexTokenizer` operates on a more refined principle: separate words from non-word characters via regex. `PunctuationTokenizer` occupies the conceptual midpoint — it uses the same regex as `RegexTokenizer` but applies a deliberately simpler post-processing pipeline.

This intermediate position is pedagogically valuable. It lets the learner compare three tokenizers that differ only in small, identifiable ways, isolating the contribution of each design decision.

### b) It exposes the cost of "punctuation glued to words"

When `WhitespaceTokenizer` produces tokens like `"Hello,"` and `"world!"`, the trailing punctuation is treated as part of the word. This has a measurable consequence: `"Hello,"` and `"Hello"` become distinct tokens, even though they refer to the same underlying word.

`PunctuationTokenizer` makes this cost explicit by removing it. When punctuation is split off, `"hello"` is a single token regardless of what follows it. The vocabulary shrinks, repetition increases, and the tokenization becomes more linguistically faithful — but at the cost of decode fidelity, as examined later.

### c) It demonstrates that "case-insensitive" is a separate design axis

A subtle but consequential property: `PunctuationTokenizer` applies lowercasing as part of tokenization. This is **not** required by the punctuation-separation goal — `RegexTokenizer` performs the same regex-based splitting without lowercasing. Including lowercasing here is an independent design choice, and one that the learner can compare against tokenizers that omit it.

### d) It serves as a baseline for comparison studies

Because the tokenization rule is simple and inspectable, `PunctuationTokenizer` provides a stable reference point against which more sophisticated tokenizers can be evaluated. When a subword or learned tokenizer outperforms it on some metric, the gap is a measurement of what the additional sophistication contributes.

---

## 3. What "Punctuation-Aware" Means in This Project

In this project, punctuation-awareness is treated as a **regex-driven separation** rather than a learned or curated mechanism.

The default tokenization rule is:

```text
\w+ | [^\w\s]
```

This pattern partitions the text into two mutually exclusive token classes:

* `\w+` — sequences of letters, digits, or underscores (i.e., words and numbers)
* `[^\w\s]` — single non-word, non-whitespace characters (i.e., punctuation)

The implementation, however, remains deliberately simplified:

* it does not handle contractions (`don't`, `we'll`) as multi-token units
* it does not parse hyphenated compounds intelligently
* it does not distinguish abbreviations (`U.S.A.`) from sentence-ending periods
* it does not implement Unicode-aware punctuation classification beyond what `[^\w\s]` provides
* it does not expose a configurable pattern (unlike `RegexTokenizer`)

The objective is therefore not to provide industrial-grade punctuation handling, but to demonstrate the **minimum increment** over whitespace tokenization — and the consequences that follow from even this minimum.

---

## 4. Core Idea

The tokenizer operates according to the following logic.

### Tokenization phase (no training required)

1. The input text is checked for emptiness.
2. The text is lowercased.
3. The regex pattern partitions the text into word and punctuation tokens.
4. Whitespace is consumed but not preserved.

### Training phase (for encode/decode only)

5. The training text is tokenized via the procedure above.
6. Unique tokens are extracted while preserving first-appearance order.
7. Each unique token is assigned an integer id.
8. Forward and reverse mappings are constructed.

Example with input `"Hello, world!"`:

```text
step 2 (lowercase):
    "hello, world!"

step 3 (regex segmentation):
    "hello"  -> word
    ","      -> punctuation
    "world"  -> word
    "!"      -> punctuation

result:
    ["hello", ",", "world", "!"]
```

A general invariant follows from this construction:

> The token count is exactly the number of word-blocks plus the number of standalone punctuation characters, with whitespace contributing nothing.

This makes the token count predictable from the input's structure alone, which is one of the tokenizer's pedagogical virtues.

---

## 5. The Three-Way Comparison: Word, Regex, and Punctuation Tokenizers

`PunctuationTokenizer` shares the same underlying regex pattern (`\w+|[^\w\s]`) with `WordTokenizer` and `RegexTokenizer`. The three tokenizers diverge in subtle but consequential ways. This comparison is central to understanding what `PunctuationTokenizer` is.

### Tokenization differences

| Tokenizer | Lowercasing | Regex pattern |
|---|---|---|
| `WordTokenizer` | No | `\w+\|[^\w\s]` |
| `RegexTokenizer` | No | `\w+\|[^\w\s]` (configurable) |
| `PunctuationTokenizer` | Yes | `\w+\|[^\w\s]` (fixed) |

### Decoding differences

| Tokenizer | Decode behavior on `["hello", ",", "world", "!"]` |
|---|---|
| `WordTokenizer` | `"hello , world !"` (simple join) |
| `RegexTokenizer` | `"hello, world!"` (cleans whitespace before punctuation) |
| `PunctuationTokenizer` | `"hello , world !"` (simple join) |

### `tokenize()` method differences

| Tokenizer | Requires training? |
|---|---|
| `WordTokenizer` | Yes (wraps `encode()`) |
| `RegexTokenizer` | No |
| `PunctuationTokenizer` | No |

This three-way comparison reveals an important pedagogical truth:

> Identical regex patterns can produce meaningfully different tokenizers when accompanied by different normalization, decoding, and lifecycle decisions.

The differences between these three tokenizers are not algorithmic — they are decisions about what to normalize, what to preserve, and when training is required. Each decision is small. Each decision changes the tokenizer's identity.

---

## 6. Why Lowercasing Is Applied

The decision to lowercase during tokenization is a deliberate one, and it deserves explicit treatment.

The implications are as follows.

### a) Vocabulary size shrinks

Without lowercasing, `"Hello"`, `"hello"`, and `"HELLO"` would each become a distinct vocabulary entry. With lowercasing, they collapse to a single entry. For a typical corpus with mixed casing, this can reduce vocabulary size by 20–40%.

### b) Token frequency consolidates

The token `"the"` may appear 1000 times in a corpus with mixed case, but it would be split across `"the"`, `"The"`, and occasionally `"THE"`. Lowercasing consolidates these counts into a single statistic, producing more stable vocabulary rankings.

### c) Semantic equivalence is enforced at the tokenizer level

A learner comparing `"hello"` and `"Hello"` recognizes them as the same word. A case-sensitive tokenizer does not — it treats them as unrelated tokens that happen to share a suffix. Lowercasing pushes this semantic equivalence into the tokenization layer.

### d) The cost is irreversibility

The trade-off is clear: case information is lost permanently. Decoding produces lowercase output regardless of the original input. For applications where case matters (named entity recognition, sentence-initial capitalization, acronym handling), this is a significant limitation.

This trade-off is examined again in the limitations section, but the decision itself is principled. Lowercasing simplifies the tokenizer's vocabulary at the cost of one specific kind of information.

---

## 7. Vocabulary Behavior

For `PunctuationTokenizer`, the vocabulary is defined as follows:

> The set of unique lowercase word and punctuation tokens observed in the training data, in the order in which they were first encountered.

Three implications are worth noting.

### a) The vocabulary is data-dependent

Different training texts produce different vocabularies. This parallels the behavior of `WordTokenizer`, `RegexTokenizer`, and `NgramTokenizer`.

### b) Lowercasing reduces vocabulary growth

Compared to a case-sensitive tokenizer with the same regex, `PunctuationTokenizer`'s vocabulary grows more slowly. This is because case-only variants collapse into single entries.

### c) Insertion-order ids are deterministic but not canonical

The implementation uses `dict.fromkeys` to deduplicate while preserving first-appearance order. Two tokenizers trained on the same text produce identical mappings. Two trained on the same set of tokens in different sentence orders do not.

This is consistent with `NgramTokenizer` and `SubwordTokenizer`, and contrasts with `WordTokenizer`, which uses `sorted(set(...))` and produces mappings independent of input order.

---

## 8. Training Logic

For this tokenizer, "training" does not correspond to machine-learning training in any meaningful sense. It refers to the construction of a vocabulary mapping from the tokens produced by `tokenize()`.

Training proceeds through the following stages.

### a) Validation

Empty and whitespace-only inputs are rejected with a `ValueError`. This is consistent with the project's other tokenizers that include defensive validation.

### b) Tokenization

The input is processed by `tokenize()`, which applies lowercasing and regex segmentation. The result is a list of word and punctuation tokens.

### c) Order-preserving deduplication

```python
unique_tokens = list(dict.fromkeys(tokens))
```

Duplicates are removed; first-appearance order is preserved.

### d) Bidirectional mapping construction

Forward (`token → id`) and reverse (`id → token`) mappings are built.

### e) Trained-state transition

The `_trained` flag is set to `True`, enabling subsequent encode and decode operations.

The training procedure performs no iteration, no frequency analysis, no probability assignment. The vocabulary is whatever tokens `tokenize()` produces, in whatever order.

---

## 9. The `tokenize()` Method

A noteworthy architectural detail concerns `tokenize()`.

Unlike `encode()` and `decode()`, which require training, `tokenize()` operates on any input without prior training:

```python
tokenizer = PunctuationTokenizer()
tokenizer.tokenize("hello, world!")  # works, no training required
tokenizer.encode("hello, world!")    # raises: not trained
```

The rationale is that punctuation-aware segmentation is a pure function of the input text and the regex pattern. No vocabulary is needed to produce the tokens themselves; the vocabulary is only needed to assign integer ids.

This design has three consequences.

### a) `tokenize()` is always available

Reporting and comparison layers can invoke `tokenize()` immediately, without orchestrating a training step.

### b) `tokenize()` never raises on legitimate input

Empty input returns an empty list. All other inputs produce a token list.

### c) `tokenize()` and `encode()` can produce different results on the same input

Specifically: `tokenize("foo")` may produce a token that is not in the trained vocabulary, in which case `encode("foo")` raises while `tokenize("foo")` succeeds. This is a feature, not an inconsistency: the two methods answer different questions.

This pattern is shared with `RegexTokenizer`, `NgramTokenizer`, and `SubwordTokenizer`. It contrasts with `WordTokenizer`, where `tokenize()` is implemented as a wrapper around `encode()` and therefore inherits its training requirement.

---

## 10. Encode Logic

The `encode()` method converts text into integer token ids:

1. The trained-state precondition is verified.
2. The input is tokenized via the lowercase + regex pipeline.
3. Each token is mapped to its integer id.

A deliberate strict design has been adopted: when the tokenizer encounters a token not seen during training, it raises a `ValueError` rather than substituting a fallback.

Example:

```text
training text:  "Hello, world!"
training vocab: {"hello": 0, ",": 1, "world": 2, "!": 3}

encoding "Hello, planet!":
    tokenized:   ["hello", ",", "planet", "!"]
    "hello" -> 0   (in vocab)
    "," -> 1       (in vocab)
    "planet" -> ValueError: Unknown token: planet
```

This strict failure mode is consistent with the project's other vocabulary-based tokenizers (`WordTokenizer`, `RegexTokenizer`, `NgramTokenizer`, `SubwordTokenizer`). The pedagogical motivation is identical:

> What should a tokenizer do when confronted with a token it has never seen?

`PunctuationTokenizer` adopts the strict-failure policy, leaving the OOV problem visible. This is one of the most important lessons that the simpler tokenizers in this project convey, and `PunctuationTokenizer` is no exception.

---

## 11. Decode Logic

The `decode()` method reconstructs a string from a list of token ids:

1. The trained-state precondition is verified.
2. Each id is mapped back to its token via `_id_to_token`.
3. Unknown ids raise a `ValueError`.
4. The tokens are joined with single spaces.

A consequential property follows from this design:

> Decoding is intentionally lossy. Whitespace, capitalization, and punctuation positioning are not preserved.

The loss manifests in three specific ways.

### a) Capitalization is lost permanently

The lowercase normalization applied during tokenization is irreversible. `"Hello"` and `"hello"` produce identical token sequences and identical decoded outputs.

### b) Whitespace is normalized

Multiple spaces, tabs, and newlines all collapse to single spaces in the decoded output:

```text
"Hello,    world!" -> tokenize -> decode -> "hello , world !"
```

### c) Punctuation acquires unwanted leading whitespace

The simple `" ".join(...)` decoding strategy treats every token as space-separated, including punctuation. The result is:

```text
["hello", ",", "world", "!"]  ->  "hello , world !"
```

Note the spaces before the comma and the exclamation mark. This is a deliberate simplification — `RegexTokenizer` performs additional cleanup to remove these spaces, but `PunctuationTokenizer` does not.

This contrast with `RegexTokenizer` is examined explicitly in the limitations section. The decision to keep the join simple is a pedagogical one: it makes the decoder's behavior easy to predict and trivial to reason about. The cost is decode fidelity, which is sacrificed in exchange for clarity.

---

## 12. Strengths

The strengths of `PunctuationTokenizer` can be summarized as follows.

### a) The algorithm is conceptually transparent

Lowercasing, regex segmentation, and id mapping. Three steps. No iteration, no learning, no probability.

### b) Vocabulary is smaller and more stable than case-sensitive alternatives

Lowercasing consolidates case variants, producing a more compact vocabulary and more stable token frequencies.

### c) Punctuation is preserved as semantic content

Unlike `WhitespaceTokenizer`, where punctuation is glued to words, `PunctuationTokenizer` treats punctuation as first-class tokens that can appear in their own right.

### d) `tokenize()` is independent of training

Reporting and comparison layers can use the tokenizer immediately, without orchestrating a training step.

### e) Determinism is straightforward

Identical inputs produce identical mappings across runs.

### f) It serves as a clean midpoint in the project's progression

Between `WhitespaceTokenizer` (no punctuation handling) and `RegexTokenizer` (configurable patterns and punctuation cleanup), `PunctuationTokenizer` occupies a precise intermediate position.

---

## 13. Limitations

This tokenizer also operates under several deliberately accepted constraints.

### a) Capitalization is lost permanently

The implicit lowercasing is irreversible. For applications that depend on case information, this is a fatal limitation.

### b) Whitespace and punctuation positioning are not preserved

The `" ".join(...)` decoder treats all tokens as space-separated, producing decoded text with visible spaces around punctuation marks.

### c) The regex is fixed, not configurable

Unlike `RegexTokenizer`, which exposes its pattern as a constructor parameter, `PunctuationTokenizer` hardcodes its regex. Users requiring a different segmentation rule must use `RegexTokenizer` instead.

### d) Contractions and hyphenated compounds are not handled specially

`"don't"` is tokenized as `["don", "'", "t"]` rather than `["do", "n't"]` or `["don't"]`. `"state-of-the-art"` becomes a sequence of words and hyphens rather than a single compound. These are linguistic refinements that fall outside the tokenizer's scope.

### e) There is no `[UNK]` mechanism

Out-of-vocabulary tokens raise hard errors rather than degrading gracefully. Pipelines encountering unfamiliar tokens must either retrain or fail.

### f) Unicode punctuation handling depends on the regex's behavior

The `[^\w\s]` class in Python's regex is Unicode-aware in a specific way: it matches anything that is neither a word character nor whitespace under the engine's Unicode tables. This is generally correct for common punctuation but may behave unexpectedly for edge cases (combining marks, exotic Unicode classes).

These limitations should not be interpreted as deficiencies; rather, they reflect **deliberate decisions of scope**.

---

## 14. Comparison with Other Tokenizers

### PunctuationTokenizer vs WhitespaceTokenizer

* `WhitespaceTokenizer` splits only on whitespace; punctuation stays attached to words.
* `PunctuationTokenizer` separates punctuation into its own tokens.

In consequence:

* `WhitespaceTokenizer` produces inflated vocabulary (`"hello,"` and `"hello"` are distinct).
* `PunctuationTokenizer` produces compact vocabulary by separating punctuation.

This is the single refinement that defines `PunctuationTokenizer`'s identity.

### PunctuationTokenizer vs WordTokenizer

* Both use the same regex (`\w+|[^\w\s]`).
* `WordTokenizer` is case-sensitive; `PunctuationTokenizer` lowercases.
* `WordTokenizer.tokenize()` requires training; `PunctuationTokenizer.tokenize()` does not.

In consequence:

* The two tokenizers produce token streams that differ only in case and lifecycle behavior.
* `PunctuationTokenizer` is essentially `WordTokenizer` with two specific design choices changed.

### PunctuationTokenizer vs RegexTokenizer

* Both use the same default regex.
* `RegexTokenizer` is case-sensitive; `PunctuationTokenizer` lowercases.
* `RegexTokenizer` exposes its pattern as a parameter; `PunctuationTokenizer` does not.
* `RegexTokenizer.decode()` cleans whitespace around punctuation; `PunctuationTokenizer.decode()` does not.

In consequence:

* `RegexTokenizer` is more configurable and produces cleaner decoded output.
* `PunctuationTokenizer` is simpler and more predictable, at the cost of decode fidelity.

The two tokenizers occupy adjacent positions in the project's progression. Choosing between them is a choice about which simplifications to accept.

### PunctuationTokenizer vs SubwordTokenizer

* `PunctuationTokenizer` produces one token per word.
* `SubwordTokenizer` further splits each word into fixed-length chunks.

In consequence:

* `PunctuationTokenizer` produces shorter sequences with larger per-token semantic content.
* `SubwordTokenizer` produces longer sequences with smaller, less meaningful chunks.

The two tokenizers operate at different granularities. They are not direct competitors but complementary points in the granularity spectrum.

---

## 15. Design Decisions in This Project

The fundamental design decisions adopted for `PunctuationTokenizer` in this project are as follows:

* the same regex used by `WordTokenizer` and `RegexTokenizer` is adopted, but fixed rather than configurable
* lowercasing is applied implicitly during tokenization
* `tokenize()` is independent of training, while `encode()` and `decode()` require it
* the vocabulary preserves first-appearance order via `dict.fromkeys`
* unseen tokens raise a `ValueError` rather than degrading via `[UNK]`
* decoding uses simple `" ".join(...)` without punctuation-aware cleanup
* punctuation is preserved as a distinct token class, not glued to surrounding words
* educational clarity and minimal sophistication are prioritized over linguistic refinement

Each of these decisions reflects a balance between the simplicity of `WhitespaceTokenizer` and the configurability of `RegexTokenizer`.

---

## 16. Testing Perspective

The core behaviors verified by the tests for this tokenizer are as follows:

* empty and whitespace-only training inputs are rejected
* training builds a vocabulary whose size matches the unique token count
* encode and decode raise before training
* `tokenize()` operates without requiring training
* word tokens are returned in lowercase form
* punctuation tokens are preserved as distinct tokens
* whitespace is consumed but not represented in the token stream
* unseen tokens raise `ValueError` during encoding
* unknown ids raise `ValueError` during decoding
* round-trip is lossy with respect to whitespace, capitalization, and punctuation positioning
* identical inputs yield identical encoded outputs
* Unicode characters (Turkish, etc.) are handled via the regex's Unicode-awareness

These tests are pedagogically valuable because they verify both the **structural invariants** of the tokenizer (vocabulary growth, deterministic mapping, lifecycle states) and its **behavioral contracts** (lossy reconstruction, strict OOV failure, lowercase normalization).

---

## 17. When to Use

`PunctuationTokenizer` is particularly well suited to the following contexts:

* introducing the concept of punctuation-aware tokenization
* explaining the cost of "punctuation glued to words" in whitespace-only tokenization
* providing a baseline that separates content from punctuation
* small experiments where lowercase normalization is acceptable
* educational settings exploring the relationship between regex patterns and tokenization behavior

It is generally insufficient in the following contexts:

* applications requiring case-sensitive output (named entity recognition, etc.)
* systems requiring exact whitespace and punctuation reconstruction
* multilingual pipelines requiring linguistically refined punctuation handling
* production deployments where contraction handling, hyphen handling, or abbreviation handling are needed

These cases call for more advanced tokenizers; `PunctuationTokenizer` provides the conceptual foundation against which they can be understood, not a substitute.

---

## 18. Final Takeaway

`PunctuationTokenizer` is the tokenizer that most clearly illustrates how small changes in normalization and decoding decisions transform a tokenizer's identity.

Because it teaches the following essential principle:

> Two tokenizers can share the same regex, the same algorithmic core, and the same vocabulary structure — yet behave differently in ways that matter, simply because they make different choices about what to preserve and what to discard.

Once this principle is internalized, the apparent multiplicity of simple regex-based tokenizers in modern NLP toolkits becomes legible as a small number of orthogonal design axes — case sensitivity, punctuation cleanup, configurability, lifecycle requirements — that can be mixed and matched independently.
