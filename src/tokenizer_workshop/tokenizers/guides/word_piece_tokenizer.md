# WordPieceTokenizer

## 1. Purpose

`WordPieceTokenizer` is the tokenizer class included in this project to introduce the **WordPiece subword tokenization** approach used in BERT and its descendants.

Its principal objective is to enable the learner to answer the following question clearly:

> If a word can be decomposed into many possible sequences of subword pieces, can a tokenizer that selects pieces by greedy longest-match — choosing the longest available vocabulary entry at each step — produce useful segmentations efficiently, and how does this strategy compare to the probabilistic approach of Unigram tokenization?

This question marks an important pedagogical pivot. `UnigramTokenizer` introduced **score-based** segmentation via Viterbi dynamic programming. `WordPieceTokenizer` introduces an alternative: **greedy** segmentation, where each step makes the locally optimal choice without backtracking or considering alternatives.

Example:

```text
"tokenization" with a vocabulary containing "token" and "##ization"
    -> ["token", "##ization"]
    -> [42, 91]
```

The defining characteristic of this tokenizer can be stated as follows:

> Subword segmentation by repeatedly matching the longest available piece, with continuation pieces marked by the `##` prefix to preserve word-boundary information.

This is the strategy adopted by BERT, DistilBERT, ELECTRA, and many other transformer models. Its enduring popularity rests on a balance: greedy is computationally efficient, the `##` convention enables clean decoding, and the algorithm is straightforward to implement correctly.

---

## 2. Why This Tokenizer Exists

This tokenizer occupies a privileged position within the project, because it is the only implementation of the **greedy longest-match** strategy.

### a) It introduces greedy segmentation as an alternative to probabilistic optimization

`UnigramTokenizer` selects segmentations by maximizing total log-probability across all candidate decompositions of a word. `WordPieceTokenizer`, by contrast, makes one local decision at a time: at the current position, find the longest piece in the vocabulary, advance past it, repeat.

The two algorithms can produce different segmentations of the same word. Neither is universally better; they reflect different design priorities:

* Unigram optimizes globally, at the cost of additional algorithmic machinery (Viterbi DP)
* WordPiece optimizes locally, at the cost of occasional suboptimal segmentations

The pedagogical value lies in seeing both approaches operate on the same problem, in the same project, with the same interface.

### b) It introduces the `##` continuation marker

This is a notational convention with substantial architectural consequences. By distinguishing "word-initial" pieces from "word-continuation" pieces at the vocabulary level, the tokenizer encodes word-boundary information into the tokens themselves.

The `##` convention is what enables WordPiece to perform **lossless decoding** — a property that `UnigramTokenizer` (which discards whitespace) does not provide.

### c) It is the production tokenizer of BERT and its descendants

BERT, DistilBERT, ELECTRA, MobileBERT, and many domain-specific BERT variants all use WordPiece. Familiarity with WordPiece is therefore not optional for any learner pursuing modern NLP — it is part of the working vocabulary of the field.

### d) It demonstrates a different vocabulary-construction philosophy

In WordPiece, candidates are generated from **every contiguous substring** of every word (subject to a length cap), with both word-initial and word-continuation variants enumerated. The most frequent candidates win vocabulary slots. This is a different procedure from the BPE merge loop or the Unigram EM algorithm, and it is worth seeing in code.

---

## 3. What "WordPiece" Means in This Project

In this project, WordPiece is treated as a **subword tokenization approach** in which:

* a vocabulary contains both word-initial pieces (`token`) and word-continuation pieces (`##ization`)
* candidates are generated exhaustively from substrings during training
* segmentation is performed by greedy longest-match at inference time
* unknown content is mapped to the `[UNK]` token

The implementation, however, is deliberately simplified:

* it does not implement BERT's exact training procedure (which uses likelihood-based merge selection)
* it does not perform character-level fallback within unknown words
* it does not handle special tokens (`[CLS]`, `[SEP]`, `[MASK]`, `[PAD]`)
* it does not implement save/load functionality
* it does not include normalization steps such as accent stripping or punctuation splitting variants

The objective is therefore not to replicate Hugging Face's WordPiece tokenizer or BERT's `tokenization.py` byte-for-byte. It is to expose the **conceptual core** of WordPiece: greedy longest-match plus the `##` continuation convention.

---

## 4. Core Idea

The tokenizer operates according to the following logic.

### Training phase

1. The text is segmented into words and punctuation using a basic regex (`\w+|[^\w\s]`), with lowercasing applied.
2. For each word, all contiguous substrings (up to `max_subword_length` characters) are enumerated as candidates.
3. Word-initial candidates are recorded as-is; word-continuation candidates are recorded with the `##` prefix.
4. Candidate frequencies are counted across the corpus.
5. The most frequent `target_vocab_size - 1` candidates are retained.
6. The `[UNK]` token is prepended at id 0.

### Inference phase

1. The input text is segmented into words via the same basic regex.
2. For each word:
   * if the entire word is in the vocabulary, it is emitted as a single token
   * otherwise, greedy longest-match is applied from left to right
3. If at any position no valid piece can be found, the entire word collapses to `[UNK]`.

Example with vocabulary `{"token", "##ization", "##s", "tokenize", "[UNK]"}`:

```text
input: "tokenization tokens"

word-by-word processing:
    "tokenization":
        try longest piece starting at 0: "tokenization" (not in vocab)
        try "tokenizatio" (not in vocab)
        ...
        try "token" (in vocab!) -> emit "token", advance 5 positions
        remaining: "ization"
        try longest piece with "##" prefix: "##ization" (in vocab!) -> emit
        result: ["token", "##ization"]

    "tokens":
        full word check: "tokens" not in vocab
        greedy: "tokenize" doesn't fit, "token" matches -> emit
        remaining: "s"
        try "##s" (in vocab!) -> emit
        result: ["token", "##s"]

final output: ["token", "##ization", "token", "##s"]
```

Three observations are essential here.

First, the greedy strategy is **myopic**: it commits to the longest match at each step, even when a shorter match might have led to a better overall segmentation. This is the central trade-off of WordPiece.

Second, the `##` prefix is **mandatory** for any piece that does not begin a word. This is what distinguishes a word-initial `"er"` from a word-continuation `"##er"` — they receive separate vocabulary entries even though they consist of the same characters.

Third, the algorithm has a **first-position fast path**: if the entire word appears in the vocabulary as-is, it is emitted as a single token without attempting any decomposition.

---

## 5. The `##` Continuation Marker Convention

The most distinctive feature of WordPiece, and a deliberate departure from most other tokenizers in this project, is its use of the `##` prefix to mark continuation pieces.

The convention is as follows:

> A piece without `##` prefix begins a word; a piece with `##` prefix continues a word.

The implications of this design are far-reaching.

### a) Decoding becomes lossless without auxiliary information

Because the marker is present at every word-continuation boundary, the decoder can reconstruct word boundaries deterministically:

```text
tokens:  ["the", "token", "##ization", "is", "good"]
decoded: "the tokenization is good"
```

The `##` on `"##ization"` signals that it attaches to the preceding `"token"`. The absence of `##` on `"is"` signals that a space precedes it. No heuristics are needed.

### b) Word-initial and word-internal occurrences are vocabulary-distinct

The string `"er"` may appear at the start of words (`"errand"`) or in the middle (`"runner"`). Under the `##` convention, these are different vocabulary entries: `"er"` and `"##er"`. They may be selected with different frequencies during training, and they participate in different segmentations during inference.

This duplication has a cost — it doubles the vocabulary's potential for any given character sequence — but it produces cleaner segmentations that respect word morphology.

### c) The marker is a structural element, not a hint

Unlike some conventions where a marker can be ignored or stripped, the `##` prefix is essential to WordPiece's operation. Removing it would collapse the distinction between word-initial and word-continuation pieces, producing ambiguous segmentations and broken decoding.

The contrast with the project's other tokenizers is instructive:

| Tokenizer | Word-boundary handling |
|---|---|
| `WordTokenizer` | One token per word; boundaries are token boundaries |
| `RegexBPETokenizer` | Whitespace as separate token |
| `UnigramTokenizer` | Boundaries are not preserved at all |
| `SentencePieceTokenizer` | `▁` marker on first piece of each word |
| `WordPieceTokenizer` | `##` marker on continuation pieces |

The SentencePiece and WordPiece conventions are mirror images of each other: SentencePiece marks the **start** of words; WordPiece marks the **continuation**. Both achieve lossless decoding through different signaling strategies.

---

## 6. Greedy Longest-Match: Algorithm and Trade-offs

The algorithmic heart of `WordPieceTokenizer` is the greedy longest-match procedure.

```text
function greedy_wordpiece_tokenize(word):
    if word in vocabulary:
        return [word]                          # fast path

    tokens = []
    start = 0

    while start < len(word):
        end = min(len(word), start + max_subword_length)
        matched = None

        while end > start:
            piece = word[start:end]
            if start > 0:
                piece = "##" + piece
            if piece in vocabulary:
                matched = piece
                break
            end -= 1

        if matched is None:
            return ["[UNK]"]                   # collapse entire word

        tokens.append(matched)
        start = end

    return tokens
```

Three properties of this procedure deserve attention.

### a) The outer loop advances; the inner loop shrinks

The outer loop walks through positions in the word from left to right. The inner loop, at each position, tries the longest possible piece first and shortens it on failure. This produces longest-match-first behavior at every step.

### b) Failure cascades to `[UNK]` for the entire word

If at any point no piece matches — including no single character — the procedure returns `["[UNK]"]`, discarding any tokens it had already accumulated. This is a significant departure from BPE-family tokenizers, which fall back to byte-level or character-level pieces, and from Unigram, where a similar fallback occurs only if the Viterbi DP cannot find any path.

### c) The algorithm is **not optimal**

Greedy longest-match does not necessarily produce the segmentation with the fewest tokens, or the highest joint probability, or any other globally optimal property. It produces the segmentation that local maximization happens to construct.

Consider a hypothetical vocabulary `{"abc", "ab", "##c", "##bc", "##abc"}` and the word `"abc"`. The fast path returns `["abc"]` (one token). But if `"abc"` were not in the vocabulary, greedy would emit `["ab", "##c"]` (two tokens), even though `["a", "##bc"]` might also be valid.

The trade-off is straightforward: greedy is **simple and fast**. It runs in linear time per word and requires only vocabulary lookups. The cost is occasional non-optimal segmentations, which in practice are rare enough that production WordPiece tokenizers (including BERT's) accept the trade-off.

---

## 7. Vocabulary Behavior

The vocabulary of `WordPieceTokenizer` consists of two components.

### Reserved `[UNK]` token

The `[UNK]` token is always present at id 0, regardless of the training data. This is the same convention adopted by `UnigramTokenizer`, and it serves the same purpose: graceful degradation on unseen content.

### Learned subword vocabulary

The remaining `target_vocab_size - 1` slots are filled with the most frequent candidates from the training corpus. Each receives a unique id from 1 upward, in order of frequency.

The total vocabulary size after training is bounded by `target_vocab_size`, but may be smaller if the training corpus does not produce enough distinct candidates.

The implementation produces both word-initial and word-continuation forms during candidate generation. A character sequence that appears both at word starts and word interiors will therefore have both forms eligible for the vocabulary, and both may be selected if frequent enough.

---

## 8. Training Logic

The `train()` method is the most substantive component of this tokenizer.

Training proceeds through the following stages.

### a) Validation

Empty and whitespace-only inputs are rejected with a `ValueError`. A secondary check rejects inputs that produce no words after basic tokenization (an edge case that the primary check might miss).

### b) Basic tokenization

The text is segmented using a regex (`\w+|[^\w\s]`), with lowercasing applied. Each resulting word becomes the source of subword candidates.

### c) Candidate generation

For each word, every contiguous substring up to `max_subword_length` characters is enumerated:

```python
for start in range(len(word)):
    for end in range(start + 1, max_end + 1):
        piece = word[start:end]
        if start == 0:
            candidates.append(piece)              # word-initial
        else:
            candidates.append(f"##{piece}")       # word-continuation
```

This produces a `Counter` of candidate frequencies across the corpus. Note that a single occurrence of `"token"` in training generates many candidates: `"t"`, `"to"`, `"tok"`, `"toke"`, `"token"`, `"##o"`, `"##ok"`, `"##oke"`, `"##oken"`, `"##k"`, `"##ke"`, `"##ken"`, `"##e"`, `"##en"`, `"##n"`. Frequency counts are tracked across all such candidates.

### d) Vocabulary selection

The `target_vocab_size - 1` most frequent candidates are selected. Combined with the reserved `[UNK]` token, this produces a vocabulary of at most `target_vocab_size` entries.

### e) Bidirectional mapping construction

Forward (`token → id`) and reverse (`id → token`) mappings are built from the final vocabulary.

### f) Trained-state transition

The `_is_trained` flag is set to `True`, enabling subsequent encode and decode operations.

This procedure is simpler than BERT's actual WordPiece training — which uses an iterative likelihood-based merge selection — but it preserves the essential structure: candidates are generated, frequencies are counted, the top-N are retained, and `[UNK]` is reserved.

---

## 9. Encode Logic

The `encode()` method converts text into a list of integer token ids:

1. The trained-state precondition is verified.
2. The input is tokenized via the greedy WordPiece pipeline.
3. Each token is mapped to its id; any token absent from the vocabulary maps to the `[UNK]` id.

A subtle but important property follows from this design:

> Encoding never raises an error for unseen content.

This is the same graceful-degradation convention adopted by `UnigramTokenizer`. Like Unigram, WordPiece allows pipelines to continue operating on unfamiliar input, with `[UNK]` serving as a measurable degradation signal.

The contrast with strict tokenizers (`WordTokenizer`, `NgramTokenizer`, `SubwordTokenizer`) is consequential. Production WordPiece tokenizers, including BERT's, must operate on text drawn from distributions that may differ from training data. The graceful fallback is essential for that operating regime.

---

## 10. Decode Logic

The `decode()` method is the most algorithmically distinctive component of this tokenizer.

The procedure is:

1. The trained-state precondition is verified.
2. Each id is mapped back to its token via `_id_to_token`.
3. Unknown ids raise a `ValueError`.
4. Tokens are reassembled with **continuation-aware merging**:
   * `[UNK]` is preserved verbatim
   * tokens with `##` prefix have their `##` stripped and are appended to the previous piece
   * other tokens become new pieces in the output, joined with single spaces

```python
for token in decoded_tokens:
    if token == "[UNK]":
        pieces.append("[UNK]")
    elif token.startswith("##"):
        if pieces:
            pieces[-1] = pieces[-1] + token[2:]
        else:
            pieces.append(token[2:])
    else:
        pieces.append(token)

return " ".join(pieces)
```

This logic produces nearly-lossless output for typical inputs:

```text
encoded:  [42, 91, 8, 17]
decoded:  ["token", "##ization", "is", "good"]
joined:   "tokenization is good"
```

Three properties are worth noting.

### a) Decoding is lossless for vocabulary-covered text

If every piece of the input was in the training vocabulary, the decoded output reconstructs the original (after lowercasing).

### b) Lossy with respect to lowercasing only

Capitalization is lost during the basic-tokenization step. Otherwise, decoding preserves the input.

### c) `[UNK]` tokens are retained as visible markers

Unlike BPE-family tokenizers (where unknown content is decomposed into recognized pieces) or `UnigramTokenizer` (where `[UNK]` is similarly retained), WordPiece produces `[UNK]` strings in decoded output as explicit indicators of information loss. This is honest and traceable.

The contrast with `UnigramTokenizer` is particularly instructive:

| Aspect | UnigramTokenizer | WordPieceTokenizer |
|---|---|---|
| Decode join | `""` (empty string) | `" "` (space), with `##` merge |
| Whitespace | Lost entirely | Preserved via `##` convention |
| Round-trip on known content | Lossy | Lossless (modulo case) |

This is one of the largest behavioral distinctions between the two probabilistic-style tokenizers in the project.

---

## 11. The `tokenize()` Method and Its Pre-Training Fallback

A noteworthy behavior governs `tokenize()` when called on an untrained tokenizer.

```python
def tokenize(self, text: str) -> list[str]:
    if not text or not text.strip():
        return []
    words = self._basic_tokenize(text)
    if not self._is_trained:
        return words
    # ... greedy WordPiece tokenization
```

Unlike `encode()` and `decode()`, which require training, `tokenize()` falls back to basic regex tokenization in the untrained state. This is the same pattern adopted by `UnigramTokenizer`.

The rationale is alignment with reporting and comparison layers: a tool that displays tokenization for many tokenizers may legitimately want to compare an untrained WordPiece baseline against trained alternatives. The fallback ensures that `WordPieceTokenizer` participates in such comparisons gracefully.

The consequences of this design are:

* `tokenize()` always produces output for valid input
* Pre-training output corresponds to basic word/punctuation segmentation (no `##` prefixes)
* Post-training output corresponds to greedy WordPiece segmentation (with `##` prefixes on continuation pieces)
* The shift in token granularity between the two is observable to callers

---

## 12. Strengths

The strengths of `WordPieceTokenizer` can be summarized as follows.

### a) The algorithm is simple and efficient

Greedy longest-match runs in linear time per word, with only vocabulary lookups. There is no dynamic programming, no probability table, no iterative optimization at inference time.

### b) The `##` convention enables lossless decoding

For any input that decomposes entirely into vocabulary pieces, the round-trip preserves the original (modulo case). This is a property that `UnigramTokenizer` cannot offer.

### c) Graceful degradation via `[UNK]`

Unseen content does not cause encoding to fail; it is mapped to the `[UNK]` token. Pipelines remain operational on unfamiliar input.

### d) The vocabulary is bounded upfront

`target_vocab_size` provides a hard upper bound on the vocabulary, regardless of training-corpus size.

### e) It corresponds to a real production tokenizer

BERT, DistilBERT, ELECTRA, and many downstream models all use WordPiece. Exposure to this tokenizer is exposure to a working component of the modern NLP stack.

### f) Pre-training fallback enables uniform reporting

Like `UnigramTokenizer`, the `tokenize()` method's regex fallback ensures that the tokenizer participates meaningfully in comparison layers even before training.

---

## 13. Limitations

This tokenizer also operates under several deliberately accepted constraints.

### a) Greedy segmentation is not always optimal

Local longest-match at each step can produce segmentations that are suboptimal globally. Unigram's Viterbi DP, by contrast, is guaranteed to find the globally optimal segmentation under its scoring function.

### b) The training procedure is frequency-based, not likelihood-based

Production WordPiece training (in BERT) uses an iterative algorithm that selects pieces based on their effect on the corpus's likelihood. This implementation uses simple frequency counting.

### c) No character-level fallback within unknown words

When greedy fails, the entire word collapses to `[UNK]`. Production WordPiece tokenizers typically fall back to character-level or byte-level pieces, ensuring that no information is fully discarded.

### d) Lowercasing is applied implicitly

Case information is lost at the very first stage of training. This makes the tokenizer unsuitable for case-sensitive tasks without modification.

### e) No special-token machinery

`[CLS]`, `[SEP]`, `[MASK]`, `[PAD]` are not supported. Direct integration with BERT-family models would require a wrapper handling these tokens.

### f) Candidate generation is exhaustive

For each word, all substrings (up to `max_subword_length`) are enumerated. This scales as `O(W × L²)` per word, where `W` is corpus size and `L` is word length. For long words, this becomes prohibitive.

These limitations should not be interpreted as deficiencies; rather, they reflect **deliberate decisions of scope**.

---

## 14. Comparison with Other Tokenizers

### WordPieceTokenizer vs UnigramTokenizer

These two are the project's pair of probabilistic-style subword tokenizers, and their differences are pedagogically central:

* `UnigramTokenizer` selects segmentations via Viterbi DP optimization.
* `WordPieceTokenizer` selects segmentations via greedy longest-match.

In consequence:

* `UnigramTokenizer` finds the globally optimal segmentation under its scoring function.
* `WordPieceTokenizer` finds a locally optimal segmentation, often with fewer total tokens but no global optimality guarantee.

* `UnigramTokenizer` decodes lossily (via empty-string join).
* `WordPieceTokenizer` decodes losslessly (via `##` continuation merging).

The contrast is not a matter of one being "better" — both algorithms are used in production transformer models. They reflect different design priorities.

### WordPieceTokenizer vs BPE family

* `SimpleBPETokenizer`, `ByteBPETokenizer`, `ByteLevelBPETokenizer`, `RegexBPETokenizer` learn merge rules and apply them in fixed order.
* `WordPieceTokenizer` learns a vocabulary and applies greedy longest-match against it.

In consequence:

* BPE produces deterministic segmentations driven by merge order.
* WordPiece produces deterministic segmentations driven by vocabulary content and longest-match.

A subtle observation: both BPE and WordPiece can be implemented to produce identical vocabularies in some cases. The difference is more in how segmentation is performed at inference time than in what gets segmented.

### WordPieceTokenizer vs SentencePieceTokenizer

* `SentencePieceTokenizer` (as a wrapper) supports four model types including BPE and Unigram, but not WordPiece directly.
* `WordPieceTokenizer` is the project's from-scratch WordPiece implementation.

The two tokenizers complement each other: SentencePiece provides industrial alternatives to BPE and Unigram, while WordPiece fills the gap for the BERT-style algorithm.

### WordPieceTokenizer vs SubwordTokenizer

* `SubwordTokenizer` chunks words at fixed lengths.
* `WordPieceTokenizer` chunks words at frequent vocabulary boundaries.

A particularly instructive comparison: if the WordPiece vocabulary happened to contain only fixed-length pieces, the two would produce similar output. The difference is that WordPiece **learns which lengths to use** through frequency analysis, whereas `SubwordTokenizer` uses a single fixed length without learning.

This makes `SubwordTokenizer` a useful baseline against which WordPiece's frequency-driven boundaries can be measured.

### WordPieceTokenizer vs BERT's tokenizer

BERT's `BertTokenizer` is the production counterpart of this class, with several principal additions:

* iterative likelihood-based vocabulary training (rather than simple frequency)
* character-level fallback within unknown words
* full special-token machinery (`[CLS]`, `[SEP]`, `[MASK]`, `[PAD]`, etc.)
* basic tokenization with optional accent stripping and Chinese character splitting
* persistence and vocabulary file loading

`WordPieceTokenizer` deliberately omits these additions in pursuit of pedagogical clarity. Its core algorithm — greedy longest-match with `##` continuation pieces — is structurally identical.

---

## 15. Design Decisions in This Project

The fundamental design decisions adopted for `WordPieceTokenizer` in this project are as follows:

* greedy longest-match is preferred over probabilistic optimization for segmentation
* `##` is used as the continuation prefix, mirroring BERT's convention
* candidates are generated exhaustively from substrings during training
* the most frequent candidates are retained, up to `target_vocab_size - 1` slots
* `[UNK]` is reserved at id 0 and used as fallback for unseen content
* `encode()` gracefully maps unseen tokens to `[UNK]` rather than raising
* `tokenize()` falls back to basic regex segmentation when called before training
* basic tokenization applies lowercasing
* greedy failure collapses the entire word to `[UNK]`, with no character-level fallback
* decoding strips `##` prefixes and joins continuation pieces to the preceding token
* educational clarity is prioritized over BERT-byte-for-byte fidelity

Each of these decisions reflects a balance between architectural fidelity to BERT's WordPiece and the pedagogical accessibility that this project requires.

---

## 16. Testing Perspective

The core behaviors verified by the tests for this tokenizer are as follows:

* invalid `vocab_size` (less than 2) and `max_subword_length` (less than 1) values are rejected
* empty and whitespace-only training inputs are rejected
* the trained vocabulary always contains the `[UNK]` token
* the vocabulary size never exceeds the configured target
* `tokenize()` falls back to basic tokenization before training
* full-word matches use the entire word when present in vocabulary
* greedy decomposition produces subword tokens otherwise
* unknown words map to `[UNK]`
* `encode()` raises before training but never raises on unseen content after training
* `encode()` substitutes the `[UNK]` id for unseen tokens
* `decode()` raises before training and on unknown ids
* decoding merges continuation pieces correctly (`["token", "##ization"]` → `"tokenization"`)
* `[UNK]` tokens are preserved through round-trip
* basic tokenization applies lowercasing and splits punctuation
* candidate generation respects the `max_subword_length` constraint
* greedy longest-match prefers longer pieces over shorter alternatives
* greedy returns `[UNK]` when no piece matches
* training is deterministic across instances given the same input

These tests are pedagogically valuable because they verify both the **algorithmic invariants** of WordPiece (greedy longest-match, `##` continuation handling, `[UNK]` semantics) and its **behavioral contracts** (graceful degradation, lossless decoding for vocabulary-covered text, deterministic training).

---

## 17. When to Use

`WordPieceTokenizer` is particularly well suited to the following contexts:

* explaining BERT-family tokenization in a working implementation
* introducing greedy longest-match as an alternative to probabilistic optimization
* demonstrating the `##` continuation marker convention
* providing a concrete reference point for comparing against `UnigramTokenizer`
* educational settings exploring the trade-offs between greedy and optimal subword segmentation

It is generally insufficient in the following contexts:

* applications requiring BERT byte-for-byte tokenization compatibility (use Hugging Face's tokenizer)
* systems requiring character-level fallback within unknown words
* pipelines requiring special tokens (`[CLS]`, `[SEP]`, etc.)
* case-sensitive tasks (without modification to disable lowercasing)
* production deployments where training-time likelihood optimization matters

These cases call for industrial WordPiece implementations; `WordPieceTokenizer` provides the conceptual foundation for understanding them, not a substitute.

---

## 18. Final Takeaway

`WordPieceTokenizer` is the tokenizer that most directly answers a question that `UnigramTokenizer` posed: if probabilistic optimization is one valid strategy for subword segmentation, what does the greedy alternative look like, and what does each give up?

Because it teaches the following essential principle:

> Subword tokenization can be approached as an optimization problem (Unigram) or as a greedy procedure (WordPiece); the right choice depends on the operating constraints, but both choices have informed real-world systems used by millions, and understanding the difference is part of understanding modern NLP.

Once this principle is internalized, the design space of subword tokenizers reveals itself as a coherent landscape — with greedy and optimal procedures, with `##` and `▁` boundary conventions, with frequency-based and likelihood-based training — rather than a confusing collection of similar-looking algorithms.
