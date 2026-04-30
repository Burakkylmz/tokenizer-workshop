# SubwordTokenizer

## 1. Purpose

`SubwordTokenizer` is the tokenizer class included in this project to introduce **fixed-length subword chunking** as the simplest possible form of sub-word tokenization.

Its principal objective is to enable the learner to answer the following question clearly:

> Before any learned merge rules, before any probabilistic segmentation, before any morphological analysis — what does sub-word tokenization look like in its most stripped-down form, and what does that simplest form already gain over word-level tokenization?

This question matters because the term "subword" in modern NLP is heavily loaded. It evokes BPE, WordPiece, Unigram, SentencePiece — sophisticated algorithms with learned vocabularies. `SubwordTokenizer` deliberately strips all of this away: words are sliced into fixed-length chunks, and that is the entire algorithm.

Example:

```text
"tokenization" with subword_size=3
    -> ["tok", "eni", "zat", "ion"]

"Hello, world!" with subword_size=3
    -> ["hel", "lo", ",", "wor", "ld", "!"]
```

The defining characteristic of this tokenizer can be stated as follows:

> Sub-word tokenization without learning, without probability, without linguistic awareness — fixed-length chunking is the irreducible minimum that "sub-word" can mean.

This honesty about what the tokenizer is **not** is essential. The name "subword" is suggestive; the implementation is intentionally far more modest.

---

## 2. Why This Tokenizer Exists

This tokenizer fulfills several distinct pedagogical roles within the project.

### a) It exposes the difference between "subword" as concept and "subword" as algorithm

A learner encountering BPE, WordPiece, or Unigram for the first time may conflate two ideas: that subwords are **smaller than words**, and that subwords are **linguistically motivated**. `SubwordTokenizer` separates these. Its tokens are smaller than words. Its tokens are not linguistically motivated. Both of these can be true at the same time.

This separation matters because it isolates the contribution of learning. When `SimpleBPETokenizer` produces `["token", "izer"]` and `SubwordTokenizer` produces `["tok", "eni", "zer"]`, the difference is **not** that one is sub-word and the other is not — they both are. The difference is that one chose its boundaries via learned merge frequencies, and the other chose them via fixed-width slicing.

### b) It provides a baseline for measuring what learning is worth

If a learned tokenizer (BPE, Unigram) can be compared against `SubwordTokenizer` on the same corpus, the difference in token count, vocabulary size, and reconstruction quality is a direct measurement of what the learning procedure adds.

In other words: this tokenizer is a control group. It is what subword tokenization looks like with the learning component removed.

### c) It demonstrates the algorithmic core of any subword tokenizer

Even sophisticated tokenizers ultimately produce a list of subword pieces from a word. `SubwordTokenizer` shows that the **act of producing a list of pieces** is trivial; the difficulty lies in choosing **which** pieces to produce.

This reframing is pedagogically valuable. It clarifies that BPE, Unigram, and their relatives are not difficult because of how they slice — they all slice. They are difficult because of how they decide where to slice.

### d) It provides a fast and predictable reference point

Because the algorithm has no training in any meaningful sense (training only constructs a vocabulary mapping), `SubwordTokenizer` is essentially constant-time. For workshop comparisons against learned tokenizers, this provides a fast and reproducible reference.

---

## 3. What "Subword" Means in This Project

In this project, `SubwordTokenizer` interprets "subword" in a deliberately literal sense:

> A subword is a contiguous, fixed-length chunk of a word, with no requirement that the chunk correspond to any linguistic, statistical, or morphological unit.

The implementation, however, is deliberately constrained:

* it does not learn merge rules
* it does not assign token probabilities
* it does not perform morphological analysis
* it does not optimize any objective function
* it does not maintain a fixed-size vocabulary budget
* it does not preserve whitespace or capitalization
* it does not implement an `[UNK]` mechanism

The objective is therefore not to provide a meaningful subword tokenizer in the sense of BPE or Unigram, but to provide the **algorithmic skeleton** of subword tokenization — the shared substrate from which more sophisticated algorithms diverge.

---

## 4. Core Idea

The tokenizer operates according to the following logic.

1. The input text is lowercased.
2. A regex (`\w+|[^\w\s]`) partitions the text into word tokens and punctuation tokens.
3. Each word token is sliced into chunks of fixed length (`subword_size`).
4. Each punctuation token is preserved as-is.
5. Whitespace is consumed but not preserved.

For training, the tokenizer additionally:

6. Collects all unique tokens produced by step 5.
7. Assigns each unique token an integer id, in first-appearance order.
8. Constructs forward and reverse mappings.

Example with `subword_size = 3`:

```text
text = "Hello, world!"

step 1 (lowercase):
    "hello, world!"

step 2 (regex segmentation):
    ["hello", ",", "world", "!"]

step 3 (chunk word tokens):
    "hello"  -> ["hel", "lo"]
    "world"  -> ["wor", "ld"]

step 4 (preserve punctuation):
    ","      -> ","
    "!"      -> "!"

final token sequence:
    ["hel", "lo", ",", "wor", "ld", "!"]
```

A general invariant follows directly from the chunking procedure:

> For a word of length `W`, the number of subword tokens is `ceil(W / subword_size)`.

The final chunk may be shorter than `subword_size` when the word length is not an exact multiple — for example, `"hello"` with `subword_size = 3` yields `["hel", "lo"]`, where `"lo"` is a partial chunk.

---

## 5. Behavior on Short Words

A subtle but important detail concerns words shorter than `subword_size`.

When `len(word) < subword_size`, the chunking procedure produces a single chunk — the entire word.

Example with `subword_size = 5`:

```text
"hi" -> ["hi"]
```

This is not a special case in the implementation; it falls out naturally from the `range(0, len(word), subword_size)` loop. But its behavioral consequence is worth noting: short words are passed through intact, regardless of `subword_size`.

This means that `SubwordTokenizer` is **not strictly constant-granularity**. A long word produces many small chunks, a short word produces one chunk that may be shorter than the configured size, and the boundary between these two regimes is not sharp.

For pedagogical purposes, this is a feature rather than a bug. It mirrors the behavior of every real subword tokenizer: short words are typically left as single tokens, while long words are decomposed.

---

## 6. Vocabulary Behavior

For `SubwordTokenizer`, the vocabulary is defined as follows:

> The set of unique subword chunks observed in the training data, in the order in which they were first encountered.

Three implications are worth noting.

### a) The vocabulary is data-dependent

Different training texts produce different vocabularies. This parallels `WordTokenizer`, `RegexTokenizer`, and `NgramTokenizer`, but with markedly different scaling properties.

### b) The vocabulary is bounded by the corpus's character variety, not its word count

Because every subword chunk is at most `subword_size` characters long, the maximum possible vocabulary size is bounded by:

```text
max_vocab_size = (alphabet_size) ^ (subword_size)
```

For a Latin alphabet of 26 lowercase letters and `subword_size = 3`, this yields a hard ceiling of approximately 17,576 distinct chunks — regardless of how large the training corpus grows. In practice, the actual vocabulary is much smaller, since most three-letter combinations never appear.

This is fundamentally different from `WordTokenizer`, where vocabulary grows linearly with the corpus, and from `NgramTokenizer`, where it grows combinatorially.

### c) Insertion order is preserved, not lexicographic order

```python
unique_tokens = list(dict.fromkeys(tokens))
```

This is the same `dict.fromkeys` pattern used by `NgramTokenizer`, and for the same reason: it preserves first-appearance order while still producing a deterministic mapping.

The contrast with `WordTokenizer`, which uses `sorted(set(...))`, is consequential. Two `SubwordTokenizer` instances trained on the same text will produce identical mappings. Two trained on the same set of words in different orders will not.

---

## 7. Training Logic

For this tokenizer, "training" does not correspond to machine-learning training in any meaningful sense. It refers to the construction of a vocabulary mapping from the tokens produced by `tokenize()`.

Training proceeds through the following stages.

### a) Validation

```python
if not text or not text.strip():
    raise ValueError(...)
```

Empty and whitespace-only inputs are rejected. This is consistent with the project's other tokenizers that include defensive validation.

### b) Tokenization

The input is processed by `tokenize()`, which performs the lowercase, regex, and chunking pipeline described in Section 4.

Notably, `tokenize()` itself does **not** require training. This separation of responsibilities — `tokenize()` for segmentation, `train()` for vocabulary mapping — is a small but consequential design choice, examined further in the next section.

### c) Order-preserving deduplication

Duplicate chunks are removed while preserving first-appearance order via `dict.fromkeys(...)`.

### d) Bidirectional mapping construction

Forward (`token → id`) and reverse (`id → token`) mappings are built from the deduplicated list.

### e) Trained-state transition

The `_trained` flag is set to `True`, enabling subsequent encode and decode calls.

The training procedure performs no iterative optimization, no frequency analysis, and no probability assignment. The vocabulary is whatever chunks `tokenize()` happens to produce, in whatever order.

---

## 8. The `tokenize()` Method

A noteworthy architectural detail concerns `tokenize()`.

Unlike `encode()` and `decode()`, which require training, `tokenize()` operates on any input without prior training:

```python
tokenizer = SubwordTokenizer(subword_size=3)
tokenizer.tokenize("hello world")  # works, no training required
tokenizer.encode("hello world")    # raises: not trained
```

The rationale is that subword chunking is a pure function of the input text and the `subword_size` parameter. No vocabulary is needed to produce the chunks themselves; the vocabulary is only needed to assign integer ids to them.

This design has three consequences.

### a) `tokenize()` is always available

Reporting and comparison layers can invoke `tokenize()` immediately, without orchestrating a training step.

### b) `tokenize()` never raises on legitimate input

Empty input returns an empty list. All other inputs produce a token list, since chunking cannot fail.

### c) `tokenize()` and `encode()` can produce different results on the same input

Specifically: `tokenize("foo")` may produce a chunk that is not in the trained vocabulary, in which case `encode("foo")` raises while `tokenize("foo")` succeeds. This is a feature, not an inconsistency: the two methods answer different questions.

This pattern is shared with `NgramTokenizer` and contrasts with `WordTokenizer`, where `tokenize()` is implemented as a wrapper around `encode()` and therefore inherits its training requirement.

---

## 9. Encode Logic

The `encode()` method converts text into integer token ids:

1. The trained-state precondition is verified.
2. The input is tokenized via the `tokenize()` pipeline.
3. Each token is mapped to its integer id.

A deliberate strict design has been adopted: when the tokenizer encounters a chunk not seen during training, it raises a `ValueError` rather than substituting a fallback.

Example:

```text
training text:  "tokenization"
training vocab: {"tok": 0, "eni": 1, "zat": 2, "ion": 3}

encoding "tokens":
    tokenized:   ["tok", "ens"]
    "tok" -> id 0  (in vocab)
    "ens" -> ValueError: Unknown token: ens
```

This strict failure mode is consistent with `WordTokenizer`, `RegexTokenizer`, and `NgramTokenizer`, but it carries different practical implications. Because subword chunks are short and combinatorial, even small vocabulary gaps can cause encoding to fail on familiar-looking input.

For pedagogical purposes, this strictness is valuable. It exposes the limitation of fixed-length chunking — that the chunks themselves are not transferable across vocabularies — in a way that learned subword tokenizers (which mitigate this problem through different strategies) deliberately hide.

---

## 10. Decode Logic

The `decode()` method reconstructs a string from a list of token ids:

1. The trained-state precondition is verified.
2. Each id is mapped back to its chunk via `_id_to_token`.
3. Unknown ids raise a `ValueError`.
4. The chunks are concatenated with empty string (`""`).

A consequential property follows from this design:

> Decoding is intentionally lossy. Whitespace, capitalization, and punctuation context are not preserved.

The loss manifests in three specific ways.

### a) Capitalization is lost

The lowercase normalization applied at the start of `tokenize()` is irreversible. `"Hello"` and `"hello"` produce identical token sequences, so decoding cannot distinguish them.

### b) Whitespace is not preserved

The regex segmentation discards whitespace entirely. There are no whitespace tokens to decode, and no boundaries are reconstructed.

```text
"Hello, world!" -> tokenize -> decode -> "hello,world!"
```

The original space between the comma and `"world"` is gone.

### c) Punctuation merges with adjacent words

Because chunks are concatenated with empty string, punctuation tokens attach to whatever surrounds them in the decoded output. The result is readable but visually unconventional.

These properties make explicit a critical design principle:

> A tokenizer's lossiness is not a bug to be mitigated; it is a contract to be documented and tested.

`SubwordTokenizer` accepts substantial information loss in exchange for algorithmic simplicity. The trade-off is honest, and learners should appreciate it as a design choice rather than a defect.

---

## 11. Strengths

The strengths of `SubwordTokenizer` can be summarized as follows.

### a) The algorithm is conceptually transparent

Fixed-length chunking is the simplest possible procedure for producing sub-word tokens. There are no learned parameters, no probability tables, no optimization loops. Every step can be inspected in isolation.

### b) The implementation is fast and predictable

Tokenization is linear in the input length. Training is dominated by the deduplication step. There is no iteration, no convergence criterion, and no failure mode beyond input validation.

### c) The vocabulary has a hard ceiling

Unlike `WordTokenizer` and `NgramTokenizer`, where vocabulary grows without bound, `SubwordTokenizer`'s vocabulary is bounded by `(alphabet_size) ^ (subword_size)`. For typical Latin alphabets and small chunk sizes, this is a manageable upper bound.

### d) Determinism is straightforward

The chunking procedure is deterministic, and `dict.fromkeys` preserves first-appearance order. Identical inputs yield identical mappings across runs.

### e) `tokenize()` is independent of training

Reporting and comparison layers can use the tokenizer immediately, without a training step. This makes integration straightforward.

### f) It serves as a baseline for learned subword tokenizers

When BPE, WordPiece, or Unigram is compared against `SubwordTokenizer` on the same corpus, the difference quantifies what learning contributes.

---

## 12. Limitations

This tokenizer also operates under several deliberately accepted constraints.

### a) Chunk boundaries are linguistically arbitrary

The chunk `"eni"` from `"tokenization"` corresponds to no morpheme, no syllable, and no statistically frequent unit. It exists only because of where the fixed-length window happens to fall. This is precisely the limitation that learned subword tokenizers address.

### b) Encoding is fragile

A small change in the input (a different prefix, a missing letter) can shift the chunk boundaries entirely, producing a token sequence with little overlap with the training vocabulary. This makes encoding brittle on text that differs even slightly from the training distribution.

### c) Capitalization is lost permanently

The implicit lowercasing in `tokenize()` is irreversible. For applications that depend on case information (named entity recognition, for example), this is a fatal limitation.

### d) Whitespace is not preserved

Decoded text omits all spacing, producing visually compressed output. Round-trip fidelity in any meaningful sense is impossible.

### e) There is no `[UNK]` mechanism

Out-of-vocabulary chunks raise a hard error rather than degrading gracefully. Pipelines that encounter unfamiliar text must either retrain or fail.

### f) The name "subword" is suggestive rather than descriptive

In the broader NLP literature, "subword tokenization" usually implies BPE, WordPiece, Unigram, or similar learned approaches. `SubwordTokenizer` adopts the term in its narrowest, most literal sense — sub-units of words — without the learning that the term commonly implies. Learners encountering the class for the first time should be aware of this distinction.

These limitations should not be interpreted as deficiencies; rather, they reflect **deliberate decisions of scope**.

---

## 13. Comparison with Other Tokenizers

### SubwordTokenizer vs WordTokenizer

* `WordTokenizer` produces one token per word.
* `SubwordTokenizer` produces multiple chunks per word.

In consequence:

* `WordTokenizer` produces shorter sequences with smaller vocabularies for short corpora.
* `SubwordTokenizer` produces longer sequences with bounded vocabulary growth on large corpora.

A particularly instructive observation: for words shorter than `subword_size`, `SubwordTokenizer` produces output identical to a `WordTokenizer` with lowercasing. The two tokenizers converge in the small-word regime and diverge as words lengthen.

### SubwordTokenizer vs SimpleBPETokenizer

* `SimpleBPETokenizer` learns chunk boundaries by merging frequent character pairs.
* `SubwordTokenizer` chooses chunk boundaries by fixed offsets.

In consequence:

* BPE chunks tend to correspond to morphologically meaningful units (`token` + `izer`, `run` + `ning`).
* `SubwordTokenizer` chunks correspond to nothing in particular (`tok` + `eni` + `zer`).

The algorithmic skeletons are similar — both produce a list of chunks per word. The difference lies entirely in **how the chunk boundaries are chosen**. This contrast is the central pedagogical value of including both tokenizers in the project.

### SubwordTokenizer vs UnigramTokenizer

* `UnigramTokenizer` selects the highest-probability segmentation via Viterbi DP.
* `SubwordTokenizer` selects the segmentation determined by fixed-length offsets.

The `UnigramTokenizer` design space includes `SubwordTokenizer`'s output as one of many possible segmentations, but it would never be selected by a probability-aware algorithm — fixed-length chunks are not the most probable parsing of typical text.

This makes `SubwordTokenizer` a useful "worst-case-but-still-functional" baseline against which Unigram's probability optimization can be quantified.

### SubwordTokenizer vs SentencePieceTokenizer

* `SentencePieceTokenizer` wraps an industrial subword tokenizer (Unigram or BPE).
* `SubwordTokenizer` is the simplest possible from-scratch subword tokenizer.

The two represent extreme ends of the subword-tokenizer spectrum: maximal sophistication and minimal sophistication. They share an interface and a conceptual commitment (subwords are smaller than words) but nothing else.

### SubwordTokenizer vs CharTokenizer

* `CharTokenizer` operates on individual characters.
* `SubwordTokenizer` operates on fixed-length character sequences.

In consequence:

* `CharTokenizer` is `SubwordTokenizer` with `subword_size = 1`.

This is not metaphorical — it is literally true. `SubwordTokenizer` with `subword_size = 1` produces tokens identical to `CharTokenizer`'s tokens (modulo lowercasing). The two tokenizers form a continuous parameterization of granularity, with `SubwordTokenizer` as the generalization.

---

## 14. Design Decisions in This Project

The fundamental design decisions adopted for `SubwordTokenizer` in this project are as follows:

* fixed-length chunking is preferred over learned segmentation
* the chunk size is exposed as a constructor parameter (`subword_size`)
* lowercasing is applied implicitly during tokenization
* punctuation is preserved as a separate token class
* whitespace is consumed silently and not represented in the token stream
* `tokenize()` is independent of training, while `encode()` and `decode()` require it
* the vocabulary preserves first-appearance order via `dict.fromkeys`
* unseen chunks raise a `ValueError` rather than degrading via `[UNK]`
* decoding produces a lossy reconstruction that is honest about its limitations
* educational clarity and minimal sophistication are prioritized over algorithmic refinement

Each of these decisions reflects a balance between architectural simplicity and pedagogical accessibility.

---

## 15. Testing Perspective

The core behaviors verified by the tests for this tokenizer are as follows:

* invalid `subword_size` (less than 1) is rejected
* empty and whitespace-only training inputs are rejected
* training builds a vocabulary whose size matches the unique chunk count
* encode and decode raise before training
* `tokenize()` operates without requiring training
* word tokens are split into fixed-length chunks
* punctuation tokens are preserved as distinct tokens
* short words shorter than `subword_size` produce a single chunk
* lowercasing is applied during tokenization
* unseen chunks raise `ValueError` during encoding
* unknown ids raise `ValueError` during decoding
* round-trip is lossy with respect to whitespace and capitalization
* identical inputs yield identical encoded outputs

These tests are pedagogically valuable because they verify both the **structural invariants** of the tokenizer (chunk size, vocabulary growth, punctuation handling) and its **behavioral contracts** (lossy reconstruction, strict OOV failure, deterministic mapping).

---

## 16. When to Use

`SubwordTokenizer` is particularly well suited to the following contexts:

* introducing the algorithmic skeleton of subword tokenization
* providing a baseline for measuring what learned subword tokenizers contribute
* small experiments where vocabulary boundedness matters
* educational settings exploring the relationship between chunk granularity and sequence length
* scenarios where reproducible, training-free chunk-level segmentation is sufficient

It is generally insufficient in the following contexts:

* applications requiring linguistically meaningful subword boundaries
* systems requiring case-sensitive or whitespace-sensitive output
* multilingual pipelines where chunk-level encoding fails to capture morphological structure
* production deployments where BPE, WordPiece, Unigram, or SentencePiece are appropriate

These cases call for learned subword tokenizers; `SubwordTokenizer` provides the conceptual foundation against which they can be understood, not a substitute.

---

## 17. Final Takeaway

`SubwordTokenizer` is the tokenizer that most clearly illustrates what subword tokenization is **before** the learning component is added.

Because it teaches the following essential principle:

> The act of slicing a word into smaller pieces is trivial; the substantial difficulty in subword tokenization lies entirely in choosing where the slices fall — and every learned subword tokenizer in modern NLP is, at its core, an answer to that question.

Once this principle is internalized, the design of every learned subword tokenizer — BPE, WordPiece, Unigram, SentencePiece — becomes legible as a different answer to the same question that `SubwordTokenizer` declines to ask.
