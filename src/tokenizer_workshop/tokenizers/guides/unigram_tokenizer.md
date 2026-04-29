# UnigramTokenizer

## 1. Purpose

`UnigramTokenizer` is the tokenizer class included in this project to introduce **probabilistic subword tokenization** through a simplified Unigram Language Model.

Its principal objective is to enable the learner to answer the following question clearly:

> If multiple ways of segmenting a word into subwords are possible, can the tokenizer be made to choose the segmentation that is most probable under a learned model — rather than the one that happens to be discovered first by a greedy procedure?

This question marks a fundamental departure from every other tokenizer in this project. Every preceding tokenizer is deterministic in the sense that it applies fixed rules: split here, merge that, accept this regex, reject the rest. `UnigramTokenizer`, by contrast, frames tokenization as a **search over candidate segmentations**, scored by learned probabilities, with the highest-scoring segmentation selected via dynamic programming.

Example:

```text
"tokenizer" can be segmented in multiple ways:
    ["token", "izer"]
    ["tok", "en", "izer"]
    ["tokenizer"]

The model selects the segmentation with the highest total log-probability.
```

The defining characteristic of this tokenizer can be stated as follows:

> Tokenization is no longer the application of a rule; it is the optimization of a score over a structured search space.

---

## 2. Why This Tokenizer Exists

This tokenizer occupies a unique position within the project, because it represents the only fully probabilistic approach in the catalog.

### a) It introduces probability as a tokenization criterion

`WordTokenizer`, `RegexTokenizer`, `CharTokenizer`, and `ByteTokenizer` all answer the question "what is a token?" with a fixed rule. BPE-family tokenizers (`SimpleBPETokenizer`, `ByteBPETokenizer`, `ByteLevelBPETokenizer`) answer it with a learned ordered sequence of merges, applied deterministically.

`UnigramTokenizer` answers the question differently: a token is whichever segmentation yields the highest total probability under a learned model. This shift — from rule-based to score-based tokenization — is one of the most important conceptual transitions in modern NLP.

### b) It approximates the SentencePiece Unigram algorithm

The Unigram tokenizer used in SentencePiece is the basis for the tokenizers shipped with models such as XLNet, ALBERT, T5, and many multilingual systems. `UnigramTokenizer` is a deliberately simplified version of that algorithm — preserving its essential structure (candidate generation, frequency-based scoring, Viterbi segmentation) while omitting its production complexity (EM training, vocabulary pruning, marginal likelihood optimization).

This makes the tokenizer pedagogically valuable in its own right, while also providing a stepping stone toward understanding production Unigram tokenizers.

### c) It introduces the `[UNK]` token as a first-class concept

Throughout the project, most tokenizers respond to unseen input with a `ValueError`. `UnigramTokenizer` is the first to formalize **graceful degradation** through an explicit unknown-token mechanism: ids that have no segmentation map to `[UNK]`, and downstream code can continue operating.

This is the convention adopted by every transformer tokenizer in production. Encountering it for the first time within this project marks a meaningful shift in tokenizer maturity.

### d) It exposes the Viterbi algorithm in a tokenization context

The dynamic-programming machinery used to find the optimal segmentation is the same Viterbi algorithm used in speech recognition, part-of-speech tagging, and many sequence labeling tasks. `UnigramTokenizer` provides a self-contained and inspectable implementation, making the algorithm legible at the level of code rather than equation.

---

## 3. What "Unigram" Means in This Project

In this project, "Unigram" refers to a tokenization approach in which:

* a vocabulary of subword candidates is learned from training data
* each candidate is assigned a probability based on its frequency
* segmentation of a new word is performed by selecting the highest-scoring decomposition under those probabilities

The implementation, however, is deliberately simplified:

* it does not perform Expectation-Maximization training
* it does not apply iterative vocabulary pruning
* it does not optimize the marginal likelihood
* it uses Laplace (add-one) smoothing rather than more sophisticated estimators
* it does not handle byte fallback for characters absent from the training data

The objective is therefore not to replicate SentencePiece's industrial Unigram implementation, but to render the **conceptual core** of probabilistic subword tokenization pedagogically accessible.

What remains, after these simplifications, is enough: candidate generation, frequency counting, log-probability assignment, Viterbi segmentation, and an explicit `[UNK]` mechanism. These are the invariants of every Unigram tokenizer; everything else is tuning.

---

## 4. Core Idea

The tokenizer operates according to the following logic.

### Training phase

1. The text is segmented into words and punctuation using a basic regex (`\w+|[^\w\s]`), with lowercasing applied.
2. For each word, every contiguous substring up to `max_subword_length` characters is generated as a candidate token.
3. The frequency of each candidate across the training corpus is counted.
4. The most frequent candidates are retained, up to the target vocabulary size.
5. An `[UNK]` token is prepended to the vocabulary at id 0.
6. Each token is assigned a log-probability based on its frequency, with Laplace smoothing applied.

### Inference phase

1. The input text is segmented into words using the same basic regex.
2. For each word, the Viterbi algorithm finds the segmentation maximizing the sum of log-probabilities.
3. The selected tokens are mapped to their integer ids.
4. Any token absent from the vocabulary maps to the `[UNK]` id.

Example:

```text
training text: "tokenizer tokens token"

candidate substrings (excerpt):
    "t", "to", "tok", "toke", "token", "tokens", "tokenize", "tokenizer"
    "o", "on", "ons", ...
    "k", "ke", "ken", ...
    "iz", "ize", "izer", ...

after frequency selection (with target_vocab_size = 10):
    vocabulary = ["[UNK]", "token", "iz", "izer", "tok", "en", "to", "s", "er", ...]

inference on "tokenizer":
    candidate segmentations include:
        ["tokenizer"]                     (single token, may not be in vocab)
        ["token", "izer"]                 (two tokens)
        ["tok", "en", "izer"]             (three tokens)

    Viterbi selects the one with highest total log-probability.
```

Three observations are essential here.

First, candidate generation is **exhaustive within a length limit**. Every possible substring of every word becomes a candidate. This is what gives the tokenizer the flexibility to discover non-obvious subword boundaries.

Second, the vocabulary is **fixed-size by construction**. Unlike `WordTokenizer` and `NgramTokenizer`, where the vocabulary grows with the corpus, `UnigramTokenizer` truncates to the most frequent candidates, ensuring bounded memory regardless of training data size.

Third, the segmentation chosen at inference time is **not necessarily the one with the longest tokens, nor the fewest tokens**. It is the one with the highest joint probability — and the difference is consequential.

---

## 5. Why Log-Probabilities Matter

The decision to operate on log-probabilities rather than raw probabilities is not incidental.

The implications are as follows.

### a) Numerical stability

Probabilities of subword tokens, especially across many tokens in a long word, can become arbitrarily small. Multiplying them in floating-point arithmetic produces underflow rapidly. Log-probabilities convert multiplication into addition, eliminating this failure mode entirely.

### b) Algorithmic convenience

The Viterbi algorithm seeks to maximize a product of probabilities along a segmentation path. In log-space, this becomes a sum, which is both easier to compute and easier to compare across candidate paths.

### c) Laplace smoothing prevents zero probabilities

Each token's probability is computed as:

```text
P(token) = (frequency(token) + 1) / (total_frequency + 1)
```

The "+1" in the numerator ensures that even tokens with zero training frequency receive a non-zero probability. Without this smoothing, the logarithm of zero would propagate as negative infinity through the Viterbi DP, breaking any segmentation that involves an unseen token.

### d) Comparison with raw frequencies

A naive alternative would rank candidates by raw frequency and select the most frequent segmentation. This approach is simpler, but it ignores the **joint** probability of an entire segmentation. A long word segmented as `[A, B]` may have a higher joint score than `[C]` alone, even if `C` is more frequent than either `A` or `B` individually.

Log-probabilities make this joint reasoning natural and efficient.

---

## 6. The Viterbi Segmentation Algorithm

The choice of segmentation at inference time is performed by a Viterbi-style dynamic programming algorithm. This is the structural heart of `UnigramTokenizer`.

The algorithm operates on two parallel tables.

```text
dp[i]            : the highest score achievable for the prefix word[:i]
backpointer[i]   : the start position j of the last token that achieved dp[i]
```

The recurrence is:

```text
dp[i] = max over valid j of:
    dp[j] + log_probability(word[j:i])

provided word[j:i] is in the vocabulary
and (i - j) <= max_subword_length
```

After the table is filled, the optimal segmentation is recovered by tracing the backpointers from `dp[n]` down to `dp[0]`.

Example for `word = "tokenizer"` and the vocabulary `{"token", "izer", "tokenizer", ...}`:

```text
dp[0]  = 0           (empty prefix)
dp[5]  = log P("token")                 via j=0, piece="token"
dp[9]  = max(
            dp[5] + log P("izer"),      via j=5, piece="izer"
            dp[0] + log P("tokenizer"), via j=0, piece="tokenizer"
         )
         = whichever is higher

backtrace: starting from dp[9], follow backpointers
```

A property worth noting:

> The fallback to `[UNK]` is structural, not heuristic.

If `dp[n]` remains `-infinity` after the DP completes, no valid segmentation of the word exists under the current vocabulary. The implementation returns `[UNK]` — not because of a special case, but because no path through the DP reached the end. This is a clean separation between the algorithm's success case and its failure case.

---

## 7. Vocabulary Behavior

The vocabulary of `UnigramTokenizer` consists of two components.

### Reserved `[UNK]` token

The `[UNK]` token is always present at id 0, regardless of the training data. This is the first token in the vocabulary by construction, and it serves as the fallback for any segmentation that cannot be completed.

### Learned subword vocabulary

The remaining `target_vocab_size - 1` slots are filled with the most frequent substring candidates from the training corpus. Each receives a unique id from 1 upward, in order of frequency.

The total vocabulary size after training is therefore exactly `target_vocab_size`, regardless of how rich or sparse the training corpus is — provided the corpus contains enough distinct substrings.

This is a fundamental departure from data-dependent tokenizers like `WordTokenizer`, where vocabulary grows with corpus size, and from byte-level tokenizers, where the base vocabulary is fixed at 256 plus learned merges. `UnigramTokenizer` enforces a strict vocabulary budget upfront, and the training procedure adapts to that budget rather than the other way around.

---

## 8. Training Logic

The `train()` method is the most substantive component of this tokenizer.

Training proceeds through the following stages.

### a) Validation

Empty and whitespace-only inputs are rejected with a `ValueError`.

### b) Basic tokenization

The text is segmented using a regex (`\w+|[^\w\s]`), with lowercasing applied. Each resulting word becomes the source of substring candidates.

### c) Candidate generation

For each word, every contiguous substring up to `max_subword_length` characters is generated:

```python
for i in range(len(word)):
    for j in range(i + 1, min(len(word), i + self.max_subword_length) + 1):
        candidate_counter[word[i:j]] += 1
```

This produces a `Counter` of substring frequencies across the entire corpus.

### d) Vocabulary selection

The `target_vocab_size - 1` most frequent candidates are selected. Combined with the reserved `[UNK]` token, this produces exactly `target_vocab_size` vocabulary entries.

### e) Log-probability assignment

For every selected token, a Laplace-smoothed log-probability is computed:

```python
total_freq = sum(freq for _, freq in most_common) + 1
self._token_logprob[token] = log((counter[token] + 1) / total_freq)
```

The `[UNK]` token receives a log-probability based on its (zero) training frequency, smoothed by the add-one rule. This ensures that even `[UNK]` participates meaningfully in Viterbi scoring rather than being treated as a hard error.

### f) Bidirectional mapping construction

Forward (`token → id`) and reverse (`id → token`) mappings are built from the final vocabulary.

### g) Trained-state transition

The `_trained` flag is set to `True`, enabling subsequent encode and decode operations.

---

## 9. Encode Logic

The `encode()` method converts text into a list of integer token ids:

1. The trained-state precondition is verified.
2. The input is tokenized via Viterbi segmentation (or basic tokenization, if the tokenizer has not been trained — but encode itself rejects this case).
3. Each token is mapped to its id; any token absent from the vocabulary maps to the `[UNK]` id (0).

A subtle but important property follows from this design:

> Encoding never raises an error for unseen content.

Unlike `WordTokenizer`, `RegexTokenizer`, or `NgramTokenizer`, all of which raise `ValueError` on unseen tokens, `UnigramTokenizer` substitutes the `[UNK]` id and continues. This is the first tokenizer in the project that **gracefully degrades** rather than failing fast.

The trade-off is straightforward:

* Strict tokenizers expose unseen content as a hard signal during development
* `UnigramTokenizer` allows pipelines to continue operating, with `[UNK]` as a measurable signal of degradation

For production NLP, the latter behavior is generally preferred — and `UnigramTokenizer`'s adoption of it reflects the maturity of the design.

---

## 10. Decode Logic

The `decode()` method reconstructs a string from a list of token ids:

1. The trained-state precondition is verified.
2. Each id is mapped to its token via `_id_to_token`.
3. Unknown ids (those not present in the vocabulary) raise a `ValueError`.
4. The tokens are concatenated with empty string (`""`).

A subtle but consequential property follows from this design:

> Decoding is lossy. The original whitespace and punctuation context are not preserved, and `[UNK]` tokens cannot be restored to their original content.

This loss manifests in three specific ways.

### a) Whitespace is not preserved

The basic tokenizer used during training discards whitespace, treating it only as a delimiter. The decoder, lacking information about the original spacing, joins tokens directly without separators.

### b) Subword tokens concatenate without separators

A word segmented as `["token", "izer"]` decodes to `"tokenizer"`, which is what one wants. But a sentence segmented as `["the", "cat", "sat"]` decodes to `"thecatsat"` — losing word boundaries entirely.

### c) `[UNK]` tokens reveal information loss

Once a piece of text has been mapped to `[UNK]`, it cannot be recovered. Decoding produces the literal string `[UNK]` in place of the original content.

These properties make explicit a critical design principle:

> Probabilistic tokenizers prioritize segmentation quality over reconstruction fidelity.

The test suite explicitly verifies this lossy behavior, ensuring that learners understand the trade-off rather than mistaking it for a bug.

---

## 11. The `tokenize()` Method and Its Pre-Training Fallback

A noteworthy behavior governs `tokenize()` when called on an untrained tokenizer.

```python
def tokenize(self, text: str) -> list[str]:
    if not text or not text.strip():
        return []
    if not self._trained:
        return self._basic_tokenize(text)
    # ... Viterbi segmentation
```

Unlike `encode()` and `decode()`, which require training and raise `ValueError` if invoked beforehand, `tokenize()` falls back to basic regex tokenization in the untrained state. This produces useful output even before the model has been trained.

The rationale is alignment with reporting and comparison layers: a tool that displays tokenization for many tokenizers may legitimately want to compare an untrained Unigram baseline against trained alternatives. The fallback ensures that `UnigramTokenizer` participates in such comparisons gracefully.

The consequences of this design are:

* `tokenize()` always produces output for valid input
* Pre-training output corresponds to basic word/punctuation segmentation
* Post-training output corresponds to Viterbi-optimal subword segmentation
* The transition between the two is invisible to callers, except for the change in token granularity

This is a deliberate accommodation of pedagogical and reporting use cases, distinct from the strict training requirement enforced by `encode` and `decode`.

---

## 12. Strengths

The strengths of `UnigramTokenizer` can be summarized as follows.

### a) Probabilistic segmentation is a novel paradigm in the project

It is the only tokenizer in the catalog that selects tokens by optimization rather than by rule application.

### b) The vocabulary size is bounded upfront

Memory consumption is predictable. Unlike `WordTokenizer`, which can grow without limit, `UnigramTokenizer` always produces a vocabulary of exactly `target_vocab_size` entries.

### c) Unseen content does not cause failures

The `[UNK]` mechanism allows graceful degradation. Pipelines continue operating on unfamiliar input, with the `[UNK]` token serving as a measurable degradation signal.

### d) The segmentation is principled

Viterbi finds the global optimum over all possible segmentations under the learned probabilities. There is no greedy bias, no order-dependence, and no arbitrary tie-breaking — only the highest-scoring path.

### e) The implementation reveals classical NLP machinery

Dynamic programming, log-probabilities, smoothing, and `[UNK]` handling are all visible in the source. For learners pursuing classical or modern NLP, this tokenizer is a pedagogical bridge into a wider technical landscape.

### f) Pre-training fallback enables uniform reporting

The `tokenize()` method's regex fallback ensures that the tokenizer participates meaningfully in comparison and reporting layers even before training.

---

## 13. Limitations

This tokenizer also operates under several deliberately accepted constraints.

### a) Training is single-pass and frequency-based, not EM

Production Unigram tokenizers (such as SentencePiece) train iteratively via Expectation-Maximization, refining vocabulary and probabilities until convergence. This implementation performs a single frequency-counting pass, which is conceptually simpler but can produce suboptimal vocabularies on diverse corpora.

### b) Decoding is intentionally lossy

Whitespace, capitalization, and punctuation context are not preserved. The original text cannot be recovered from token ids alone.

### c) Smoothing is rudimentary

Laplace smoothing is the simplest of all smoothing techniques and tends to over-allocate probability mass to rare events. More refined approaches (Kneser-Ney, Good-Turing) are not implemented.

### d) Candidate generation is exhaustive within a length limit

Memory consumption during training scales with the number of substring candidates, which can grow rapidly for long words and high `max_subword_length`. For corpora with very long tokens, this becomes a practical bottleneck.

### e) No byte fallback for unseen characters

If a character appears at inference time that did not appear during training (e.g., an emoji in a model trained on plain English), the Viterbi DP cannot find a path covering it, and the entire word collapses to `[UNK]`. Production tokenizers typically combine Unigram with byte-level fallback to eliminate this failure mode.

### f) Lowercasing is applied implicitly during basic tokenization

Case information is lost at the very first stage of training. This makes the tokenizer unsuitable for case-sensitive tasks without modification.

These limitations should not be interpreted as deficiencies; rather, they reflect **deliberate decisions of scope**.

---

## 14. Comparison with Other Tokenizers

### UnigramTokenizer vs SimpleBPETokenizer

* `SimpleBPETokenizer` learns merges greedily, applying them in fixed order.
* `UnigramTokenizer` learns probabilities globally, applying Viterbi to find the optimal segmentation.

The contrast is fundamental:

* BPE is order-dependent and deterministic.
* Unigram is order-independent and optimization-based.

A particularly instructive observation: BPE and Unigram often produce different tokenizations of the same word, even when trained on the same corpus. The differences reflect the algorithms' different optimization criteria, not any inherent flaw in either approach.

### UnigramTokenizer vs WordTokenizer

* `WordTokenizer` produces one token per word, with strict OOV behavior.
* `UnigramTokenizer` produces multiple subword tokens per word, with `[UNK]` fallback.

In consequence, `UnigramTokenizer` handles unseen words by decomposing them into known subwords; `WordTokenizer` simply rejects them.

### UnigramTokenizer vs ByteLevelBPETokenizer

* `ByteLevelBPETokenizer` cannot fail, because every byte is in the base vocabulary.
* `UnigramTokenizer` can fail (returning `[UNK]`), but at the word level rather than the byte level.

The two tokenizers represent two distinct strategies for handling out-of-vocabulary content:

* Byte-level: structural elimination via universal coverage.
* Unigram: graceful degradation via an explicit fallback token.

Production systems often combine the two, applying Unigram to typical text and byte-level fallback for exotic input.

### UnigramTokenizer vs SentencePiece's Unigram

SentencePiece's Unigram is the production counterpart of this class, with three principal additions:

* iterative EM-based training with vocabulary pruning
* native handling of spaces via the `▁` prefix marker
* persistence and special-token machinery

`UnigramTokenizer` deliberately omits these additions in pursuit of pedagogical clarity, but its core algorithm — candidate generation, frequency-based vocabulary, log-probability assignment, Viterbi segmentation — is structurally identical.

---

## 15. Design Decisions in This Project

The fundamental design decisions adopted for `UnigramTokenizer` in this project are as follows:

* probabilistic segmentation is preferred over deterministic rule application
* candidates are generated exhaustively within a length limit, then ranked by frequency
* Laplace smoothing is applied to prevent zero-probability tokens
* log-probabilities replace raw probabilities for numerical stability and algorithmic convenience
* Viterbi dynamic programming finds the optimal segmentation
* `[UNK]` is reserved at id 0 and serves as both the fallback for unknown content and an explicit marker of segmentation failure
* `encode()` gracefully maps unseen tokens to `[UNK]` rather than raising
* `tokenize()` falls back to basic regex segmentation when called before training
* basic tokenization applies lowercasing, sacrificing case information for vocabulary compactness
* educational clarity is prioritized over production-grade probabilistic refinement

Each of these decisions reflects a balance between architectural fidelity to real-world Unigram tokenizers and the pedagogical accessibility that this project requires.

---

## 16. Testing Perspective

The core behaviors verified by the tests for this tokenizer are as follows:

* invalid `vocab_size` (less than 2) and `max_subword_length` (less than 1) values are rejected
* empty and whitespace-only training inputs are rejected
* the trained vocabulary always contains the `[UNK]` token
* the vocabulary size never exceeds the configured target
* frequent substrings from the training corpus appear in the vocabulary
* training is deterministic across instances given the same input
* `tokenize()` falls back to basic tokenization before training
* `tokenize()` produces subword segmentations after training
* unknown words during tokenization map to `[UNK]`
* `encode()` raises before training but never raises on unseen content after training
* `encode()` substitutes the `[UNK]` id for unseen tokens
* encoding is deterministic for the same input
* `decode()` raises before training and on unknown ids
* the round-trip is lossless **only** for content already in the vocabulary; whitespace is not preserved
* Turkish and other Unicode characters are handled (via the regex's Unicode-aware `\w`)
* Viterbi prefers the highest-scoring segmentation over alternatives
* Viterbi returns `[UNK]` when no valid path exists

These tests are pedagogically valuable because they verify both the **algorithmic invariants** of the Unigram approach (vocabulary bounds, `[UNK]` semantics, Viterbi optimality) and its **behavioral contracts** (graceful degradation, deterministic training, lossy reconstruction).

---

## 17. When to Use

`UnigramTokenizer` is particularly well suited to the following contexts:

* introducing the concept of probabilistic tokenization
* explaining the Viterbi algorithm in a tokenization context
* demonstrating graceful degradation through `[UNK]` mechanisms
* comparing rule-based and score-based tokenization paradigms
* providing a stepping stone toward understanding SentencePiece's Unigram tokenizer
* educational settings exploring classical NLP probability machinery

It is generally insufficient in the following contexts:

* applications requiring lossless round-trip
* case-sensitive tasks (without modification to disable lowercasing)
* corpora requiring byte-level fallback for unseen characters
* large-scale pipelines requiring EM-trained vocabulary refinement
* production deployments where SentencePiece or equivalent libraries should be used

These cases call for industrial Unigram implementations; `UnigramTokenizer` provides the conceptual foundation for understanding them, not a substitute.

---

## 18. Final Takeaway

`UnigramTokenizer` is the tokenizer that most fully shifts the question of tokenization from "what rule applies here?" to "what segmentation is most probable?".

Because it teaches the following essential principle:

> Tokenization is not a mechanical procedure to be applied once; it is an optimization problem with a structured search space, learned scores, and a principled algorithm for finding the best answer.

Once this principle is internalized, the design of every modern subword tokenizer — including SentencePiece, WordPiece, and the Unigram variants used in multilingual transformer models — becomes legible from the same probabilistic perspective.
