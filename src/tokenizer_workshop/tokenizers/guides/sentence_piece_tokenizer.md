# SentencePieceTokenizer

## 1. Purpose

`SentencePieceTokenizer` is the tokenizer class included in this project to introduce **production-grade subword tokenization** through Google's SentencePiece library.

Its principal objective is to enable the learner to answer the following question clearly:

> Once the conceptual foundations of subword tokenization have been established through hand-rolled implementations, how does an industrial-strength tokenizer differ — and what does the learner gain by stepping from a pedagogical implementation to a production one?

This question marks a deliberate transition within the project. Every preceding tokenizer has been written from scratch, with simplifications chosen for educational clarity. `SentencePieceTokenizer` instead **wraps a mature external library**, exposing the same API surface but delegating the algorithmic work to code engineered for real-world deployment.

Example:

```text
"Hello world"
    -> ["▁Hello", "▁world"]   # note the whitespace marker
    -> [27, 41]
```

The defining characteristic of this tokenizer can be stated as follows:

> A wrapper that brings industrial-strength subword tokenization into the workshop, without abandoning the project's uniform tokenizer contract.

---

## 2. Why This Tokenizer Exists

This tokenizer occupies a unique position within the project, because it is one of only two adapter-style tokenizers in the catalog (the other being `PreTrainedTokenizerWrapper`).

### a) It connects pedagogical implementations to industrial reality

`SimpleBPETokenizer`, `ByteLevelBPETokenizer`, `UnigramTokenizer`, and the other from-scratch tokenizers in the project are conceptually faithful but deliberately simplified. They omit Expectation-Maximization training, vocabulary pruning, byte fallback, and many other refinements found in production systems.

`SentencePieceTokenizer` provides the missing reference point. By wrapping the same library that powers tokenization in T5, ALBERT, XLNet, and many multilingual models, it allows learners to compare hand-rolled implementations against the industrial tokenizer they were designed to teach.

### b) It introduces the whitespace-as-marker convention

All preceding tokenizers in the project either preserve whitespace as a separate token (`RegexBPETokenizer`, `ByteLevelBPETokenizer`) or discard it entirely (`WordTokenizer`, `RegexTokenizer`, `UnigramTokenizer`). SentencePiece adopts a third strategy: whitespace is encoded **into** the tokens themselves via a special marker character (`▁`, U+2581).

This convention is not cosmetic. It is what enables truly lossless round-trip in a tokenizer that operates without explicit whitespace tokens — and it is the convention adopted by every modern subword tokenizer descended from SentencePiece.

### c) It exposes four tokenization paradigms through a single interface

Through the `model_type` parameter, `SentencePieceTokenizer` provides access to four distinct algorithms:

* `unigram` — probabilistic subword model (the production counterpart of `UnigramTokenizer`)
* `bpe` — Byte Pair Encoding (the production counterpart of `SimpleBPETokenizer` and the BPE family)
* `char` — character-level tokenization (the production counterpart of `CharTokenizer`)
* `word` — whitespace-based word tokenization (the production counterpart of `WordTokenizer`)

A single class therefore gives the learner production-grade access to four of the project's pedagogical tokenizers. This makes side-by-side comparison straightforward and meaningful.

### d) It demonstrates the adapter pattern at a higher level of complexity

Unlike `PreTrainedTokenizerWrapper`, which delegates to a tokenizer that has already been trained elsewhere, `SentencePieceTokenizer` performs **training and inference** through the wrapped library. This is a fuller adapter pattern, and it raises issues — temporary file management, lazy imports, configuration translation — that simpler wrappers do not encounter.

---

## 3. What "SentencePiece" Means in This Project

In this project, SentencePiece is treated as **an externally-developed, production-grade subword tokenization toolkit** whose capabilities are exposed to the workshop through a thin adapter.

The implementation, however, remains deliberately constrained:

* it does not expose every SentencePiece option (only the essentials: `vocab_size`, `model_type`, `character_coverage`)
* it does not support custom special tokens (BOS, EOS, PAD are deliberately disabled)
* it does not implement save/load — every training call rebuilds the model from scratch in a temporary directory
* it does not provide access to lattice-based sampling for noisy training (`encode_with_alpha`, n-best segmentation, etc.)
* it does not expose vocabulary pruning, byte-fallback configuration, or denormalization rules

The objective is therefore not to expose every SentencePiece feature, but to provide a clean adapter that fits the project's tokenizer contract while preserving SentencePiece's most important architectural property — **lossless round-trip through whitespace markers**.

For users requiring the full feature set, the underlying library can be invoked directly. For learners, the adapter is sufficient.

---

## 4. Core Idea

The tokenizer operates as a thin layer above the SentencePiece library, with the following pipeline.

### Training phase

1. The input text is validated (rejecting empty or whitespace-only input).
2. A temporary directory is created.
3. The training text is written to a temporary file (SentencePiece does not accept in-memory strings).
4. `SentencePieceTrainer.train(...)` is invoked with the configured parameters.
5. The resulting `.model` file is loaded into a `SentencePieceProcessor` instance.
6. The temporary directory is automatically cleaned up.

### Inference phase

1. The trained-state precondition is verified.
2. `tokenize()`, `encode()`, and `decode()` delegate to `_processor.encode(...)` and `_processor.decode(...)`.

Example for `model_type="unigram"`, `vocab_size=200`:

```text
training corpus: a Turkish news article

after training:
    vocabulary contains pieces such as:
        "▁ve", "▁bir", "▁için", "lar", "ler", "ı", ...

inference on "Bu bir test cümlesidir":
    tokens: ["▁Bu", "▁bir", "▁test", "▁cümle", "sidir"]
    ids:    [27, 41, 88, 91, 152]   (illustrative)

decode back:
    "Bu bir test cümlesidir"   # exactly the input, including spaces
```

Three observations are essential here.

First, the algorithm is not implemented in this class. The class is an **adapter**; the algorithmic work is delegated to SentencePiece. This is a fundamental architectural difference from every from-scratch tokenizer in the project.

Second, the temporary file workflow is mandatory. SentencePiece's training API accepts only file paths, not in-memory strings. The wrapper hides this by managing the temporary directory itself, but the constraint shapes the design.

Third, decoding produces output that is byte-for-byte identical to the input — including whitespace and capitalization. This is achieved through the `▁` whitespace marker, examined below.

---

## 5. The Whitespace Marker Convention

The most distinctive feature of SentencePiece, and a deliberate departure from other tokenizers in this project, is its handling of whitespace.

The convention is as follows:

> Each piece that begins a word (i.e., immediately follows a whitespace boundary) is prefixed with the marker character `▁` (U+2581, "Lower One Eighth Block").

The implications of this design are far-reaching.

### a) Whitespace is encoded into the tokens themselves

Rather than producing a separate token for each space, SentencePiece embeds whitespace information into the leading character of every word. The token `"▁hello"` carries two pieces of information at once: the word `"hello"`, and the fact that it was preceded by whitespace.

### b) Decoding becomes lossless without special handling

Because the marker is always present at word boundaries, the decoder can reconstruct whitespace exactly:

```text
tokens:  ["▁Hello", ",", "▁world", "!"]
decoded: "Hello, world!"
```

The leading `▁` of `"▁world"` is decoded back to a space; the comma's absence of `▁` correctly produces no space before it. No heuristics, no post-processing — the marker carries all the information needed.

### c) The convention extends to tokens that begin sentences

The very first token in a text also receives a leading `▁`, since SentencePiece treats the start-of-text position as a whitespace boundary. This is why `"Hello"` becomes `"▁Hello"` rather than `"Hello"`.

This is the property that makes SentencePiece the de facto standard for subword tokenization in modern multilingual language models. It eliminates an entire category of bugs related to whitespace reconstruction — the kind of bugs that `UnigramTokenizer` exhibits explicitly through its lossy decode.

The contrast with the project's other tokenizers is instructive:

| Tokenizer | Whitespace handling |
|---|---|
| `WordTokenizer` | Discarded; reconstructed heuristically |
| `RegexTokenizer` | Discarded; reconstructed with punctuation cleanup |
| `RegexBPETokenizer` | Preserved as a separate token class |
| `ByteLevelBPETokenizer` | Preserved as raw bytes (`0x20`) |
| `UnigramTokenizer` | Discarded; not reconstructed |
| `SentencePieceTokenizer` | Embedded into tokens via `▁` marker |

Each strategy is principled. SentencePiece's choice is the most compact and the most lossless — at the cost of producing tokens whose visual form is slightly unusual.

---

## 6. The Four Model Types

The `model_type` parameter governs the underlying algorithm:

### unigram

The probabilistic Unigram Language Model approach, trained via Expectation-Maximization. Selects the segmentation maximizing the likelihood under the learned vocabulary. This is the default and the most modern of SentencePiece's four algorithms.

The from-scratch counterpart in this project is `UnigramTokenizer`, which implements the conceptual core (Viterbi segmentation, log-probabilities) without the EM training that gives SentencePiece its production strength.

### bpe

Byte Pair Encoding, the algorithm popularized by GPT-2. Iteratively merges the most frequent adjacent pair until the vocabulary reaches the target size.

The from-scratch counterparts are `SimpleBPETokenizer` (character-level), `ByteBPETokenizer` (byte-level), `ByteLevelBPETokenizer` (byte-level with frozen-dataclass merges), and `RegexBPETokenizer` (regex-bounded BPE).

### char

Character-level tokenization. Each character becomes a token directly.

The from-scratch counterpart is `CharTokenizer`.

### word

Whitespace-based word tokenization. Each whitespace-delimited unit becomes a token.

The from-scratch counterpart is `WordTokenizer`.

This four-way coverage is what makes `SentencePieceTokenizer` particularly valuable for cross-tokenizer benchmarking. The same wrapper, with only the `model_type` parameter changed, can produce production-grade comparisons against any of the project's hand-rolled tokenizers.

---

## 7. Vocabulary Behavior

For `SentencePieceTokenizer`, the vocabulary is **bounded by the `vocab_size` parameter** and constructed entirely by the underlying library.

Two implications follow.

### a) The vocabulary is fixed-size, not data-dependent in the usual sense

Unlike `WordTokenizer` and `RegexTokenizer`, where the vocabulary grows with corpus size, SentencePiece always converges to exactly `vocab_size` entries — provided the training corpus is rich enough to support that many distinct subwords.

This convergence is an algorithmic property of the training procedure, not an artifact of the wrapper. SentencePiece's training is iterative and target-aware: it grows or prunes candidates until the target is reached.

### b) The vocabulary contents are not directly inspectable through the wrapper

The wrapper exposes `vocab_size` (the count) but does not expose the vocabulary contents themselves. Users requiring this information can access `_processor.id_to_piece(...)` or use the underlying library's serialization features.

The wrapper deliberately keeps this surface narrow. The objective is to fit the project's `BaseTokenizer` contract, not to expose every SentencePiece capability.

---

## 8. Training Logic

The `train()` method orchestrates the SentencePiece training pipeline.

Training proceeds through the following stages.

### a) Validation

Empty and whitespace-only inputs are rejected with a `ValueError`. This check is consistent with the project's other tokenizers.

### b) Temporary directory creation

A `TemporaryDirectory` context is opened. All training artifacts will live inside it and be deleted automatically when the context exits.

### c) Input file preparation

The training text is written to a UTF-8-encoded file inside the temporary directory:

```python
input_path.write_text(text, encoding="utf-8")
```

This indirection is necessary because SentencePiece's training API does not accept in-memory strings. The wrapper handles this transparently.

### d) Training invocation

`SentencePieceTrainer.train(...)` is called with the configured parameters and several hard-coded defaults:

| Parameter | Value | Rationale |
|---|---|---|
| `bos_id` | -1 | BOS token disabled — out of scope for this wrapper |
| `eos_id` | -1 | EOS token disabled — out of scope for this wrapper |
| `pad_id` | -1 | Padding disabled — handled at the model level, not tokenizer |
| `unk_id` | 0 | Unknown token reserved at id 0, mirroring the `UnigramTokenizer` convention |
| `hard_vocab_limit` | False | Prevents crashes on small training corpora |

The choice to disable BOS, EOS, and PAD is deliberate. These are model-input formatting concerns, not tokenization concerns. Including them by default would produce token sequences that are not faithful comparisons to the project's other tokenizers.

### e) Model loading

After training, the resulting `.model` file is loaded into the `SentencePieceProcessor` instance:

```python
self._processor.load(str(model_path))
```

Once loaded, the model lives in memory and is independent of the temporary directory, which is then cleaned up automatically.

### f) Trained-state transition

The `_trained` flag is set to `True`, enabling subsequent encode and decode operations.

The combination of temporary file creation, in-memory model loading, and automatic cleanup ensures that no disk artifacts persist beyond the training call. This is a deliberate design property: training is reproducible and idempotent, and the file system is never polluted.

---

## 9. Encode and Decode

The encode and decode methods are nearly trivial wrappers around the library's API:

```python
def encode(self, text: str) -> list[int]:
    return list(self._processor.encode(text, out_type=int))

def decode(self, token_ids: list[int]) -> str:
    return self._processor.decode(token_ids)
```

Three properties of the underlying library are inherited automatically.

### a) Lossless round-trip

```text
text -> encode -> decode -> text
```

For typical input — text, punctuation, whitespace, capitalization — the round-trip is byte-for-byte exact. This is the property that the `▁` whitespace marker is designed to provide.

### b) Graceful handling of unseen content

SentencePiece does not raise on unseen input. Unfamiliar characters are mapped to the `unk` token, and decoding produces an `[UNK]` placeholder where the original content stood. This mirrors the convention of `UnigramTokenizer`, but with library-grade implementation rather than a hand-written approximation.

### c) Determinism

For a given training corpus and configuration, encode produces identical outputs across runs and across machines. SentencePiece's training and inference are both deterministic, and the wrapper introduces no randomness.

The wrapper applies one defensive measure beyond the library's behavior: empty and whitespace-only inputs to `encode` and `tokenize` return empty lists rather than invoking the library. This protects against edge cases where the library might behave unexpectedly on degenerate input, and it keeps behavior aligned with the rest of the project.

---

## 10. The Lazy Import Pattern

The wrapper imports the `sentencepiece` library lazily, inside `__init__`:

```python
try:
    import sentencepiece as spm
except ImportError as exc:
    raise ImportError(
        "SentencePieceTokenizer requires the 'sentencepiece' package. ..."
    ) from exc
```

This pattern has three benefits.

### a) The library becomes a soft dependency

The project does not require `sentencepiece` to be installed unless this specific tokenizer is constructed. Users working only with the from-scratch tokenizers can avoid the dependency entirely.

### b) The error message is actionable

A user attempting to use the wrapper without the library installed receives a clear, actionable error message — not the deeper `ModuleNotFoundError` that Python would otherwise raise. The message includes the installation command (`uv add sentencepiece`).

### c) Unit tests of the wrapper's interface remain runnable

Tests that exercise the wrapper's validation logic (e.g., rejecting invalid `vocab_size`) can be written without requiring the library, provided they avoid construction. In practice, the tests in this project do require the library, but the architecture leaves the option open.

This pattern is shared with `PreTrainedTokenizerWrapper`, which makes the same trade-off for the `transformers` library. The two wrappers form a small but coherent family of "soft-dependency adapters" within the project.

---

## 11. Strengths

The strengths of `SentencePieceTokenizer` can be summarized as follows.

### a) Production-grade tokenization with minimal code

The wrapper is a thin layer; the algorithmic work is performed by a mature, well-tested library. This brings real-world tokenization quality to the workshop without reimplementing it from scratch.

### b) Lossless round-trip via whitespace markers

The `▁` convention eliminates an entire class of whitespace-handling bugs that the project's other tokenizers exhibit explicitly.

### c) Four algorithms behind one interface

`unigram`, `bpe`, `char`, and `word` are accessible through a single class, enabling principled comparison against the project's hand-rolled tokenizers.

### d) Determinism and reproducibility

SentencePiece's training and inference are deterministic. Combined with the wrapper's stateless design (training rebuilds the model from scratch), every operation is reproducible.

### e) No persistent disk state

Temporary files are managed via `TemporaryDirectory`, ensuring automatic cleanup. No artifacts accumulate across training calls.

### f) Soft-dependency architecture

The lazy import keeps `sentencepiece` an optional dependency, with clear error messages when it is missing.

---

## 12. Limitations

This tokenizer also operates under several deliberately accepted constraints.

### a) Special tokens are disabled by default

BOS, EOS, and PAD are configured with `id=-1` (disabled). For users requiring these tokens for transformer model inputs, the wrapper must be subclassed or the underlying library used directly.

### b) The vocabulary contents are not exposed

Only the size is accessible. Inspection of individual pieces requires reaching past the wrapper to the underlying processor.

### c) No persistence

Every training call rebuilds the model from scratch. Saving a trained tokenizer to disk for later reuse is not supported by this wrapper.

### d) The library is an external dependency

`sentencepiece` is a C++ library with Python bindings. It must be installed separately, and its installation can fail on uncommon platforms.

### e) Training requires writing to disk

Even though the wrapper hides this, the underlying library writes a temporary file for every training call. For very large corpora, this introduces I/O overhead that an in-memory training path would not.

### f) The four model types are exposed without their advanced options

Each algorithm in SentencePiece supports many parameters beyond what the wrapper accepts. Users requiring advanced configuration (e.g., custom split rules, byte fallback, normalization variants) cannot access them through this interface.

These limitations should not be interpreted as deficiencies; rather, they reflect **deliberate decisions of scope**.

---

## 13. Comparison with Other Tokenizers

### SentencePieceTokenizer vs UnigramTokenizer

* `UnigramTokenizer` is a from-scratch implementation of the Unigram algorithm.
* `SentencePieceTokenizer` (with `model_type="unigram"`) wraps Google's production Unigram.

In consequence:

* `UnigramTokenizer` exposes the Viterbi DP and log-probability machinery for inspection.
* `SentencePieceTokenizer` is a black-box but offers EM-trained vocabularies and lossless round-trip.

The two tokenizers form a natural pedagogical pairing: study `UnigramTokenizer` to understand the algorithm, use `SentencePieceTokenizer` for real comparisons.

### SentencePieceTokenizer vs the BPE family

* `SimpleBPETokenizer`, `ByteBPETokenizer`, `ByteLevelBPETokenizer`, `RegexBPETokenizer` are from-scratch BPE implementations.
* `SentencePieceTokenizer` (with `model_type="bpe"`) wraps SentencePiece's production BPE.

In consequence:

* The from-scratch BPE family teaches the algorithm in stages of increasing realism.
* `SentencePieceTokenizer` provides a single industrial reference point against which the entire family can be compared.

### SentencePieceTokenizer vs PreTrainedTokenizerWrapper

* `PreTrainedTokenizerWrapper` adapts already-trained Hugging Face tokenizers.
* `SentencePieceTokenizer` performs training and inference through the wrapped library.

The two wrappers are siblings in the project's adapter family, but they differ in scope:

* `PreTrainedTokenizerWrapper` is read-only with respect to training.
* `SentencePieceTokenizer` is full-lifecycle, handling training and inference symmetrically.

### SentencePieceTokenizer vs SentencePiece (direct use)

The direct library exposes hundreds of training options, multiple sampling strategies, lattice-based n-best segmentation, save/load, and many other capabilities. The wrapper exposes a narrow subset chosen for compatibility with the project's `BaseTokenizer` contract.

For learners, the wrapper is sufficient. For applications requiring production-level control, the direct library is the right tool.

---

## 14. Design Decisions in This Project

The fundamental design decisions adopted for `SentencePieceTokenizer` in this project are as follows:

* SentencePiece is wrapped as a soft dependency via lazy import
* training writes to a temporary directory, automatically cleaned up after use
* every training call rebuilds the model from scratch — no persistence
* BOS, EOS, and PAD tokens are deliberately disabled
* `unk_id` is fixed at 0 to align with `UnigramTokenizer`
* `hard_vocab_limit=False` prevents training failures on small corpora
* the four model types (`unigram`, `bpe`, `char`, `word`) are exposed but no model-type-specific parameters
* empty and whitespace-only inputs are normalized to empty output rather than passed to the library
* educational comparison with the project's hand-rolled tokenizers takes priority over feature completeness

Each of these decisions reflects a balance between architectural fidelity to SentencePiece and the constrained, comparable interface that the rest of the project requires.

---

## 15. Testing Perspective

The core behaviors verified by the tests for this tokenizer are as follows:

* invalid `vocab_size` (less than 2), `model_type` (outside the four valid values), and `character_coverage` (outside `(0, 1]`) are rejected
* empty and whitespace-only training inputs are rejected
* the vocabulary is built after training and reflects the configured target
* `vocab_size` returns 0 before training
* `tokenize`, `encode`, and `decode` raise before training
* empty inputs to `tokenize` and `encode` return empty lists rather than invoking the library
* `tokenize` produces string tokens, including the `▁` whitespace marker
* `encode` produces integer ids deterministically
* `decode` produces a string and preserves the input through round-trip
* Turkish and other Unicode characters are handled correctly
* all four model types (`unigram`, `bpe`, `char`, `word`) are supported via parametrized tests

These tests are pedagogically valuable because they verify both the **wrapper invariants** (input validation, lifecycle states, empty-input handling) and the **delegated behaviors** (whitespace marker presence, lossless round-trip, multi-model-type support).

---

## 16. When to Use

`SentencePieceTokenizer` is particularly well suited to the following contexts:

* benchmarking the project's hand-rolled tokenizers against an industrial reference
* demonstrating the whitespace-marker convention in a working implementation
* exploring how the choice of `model_type` affects tokenization quality on a fixed corpus
* multilingual experiments requiring robust character coverage
* providing a concrete entry point into modern subword tokenization for learners

It is generally insufficient in the following contexts:

* applications requiring fine-grained control over SentencePiece training parameters
* systems requiring tokenizer persistence (saved models, checkpoints)
* pipelines requiring BOS, EOS, or PAD tokens for direct model integration
* environments where adding a C++ dependency is undesirable

These cases call for direct use of the SentencePiece library; the wrapper provides a comparable interface for the project, not a substitute for the full toolkit.

---

## 17. Final Takeaway

`SentencePieceTokenizer` is the tokenizer that most clearly illustrates what the workshop's hand-rolled implementations were teaching toward.

Because it teaches the following essential principle:

> Pedagogical clarity and production strength are not opposed, but they live in different places: the from-scratch implementations expose the algorithm; the production wrapper exposes the consequences of having engineered around every edge case the algorithm encounters in the real world.

Once this principle is internalized, the project's two-track design — hand-rolled tokenizers for understanding, library-wrapped tokenizers for benchmarking — becomes legible as a single coherent pedagogical strategy rather than two disconnected efforts.
