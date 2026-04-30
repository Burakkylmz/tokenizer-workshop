# ByteLevelBPETokenizer

## 1. Purpose

`ByteLevelBPETokenizer` is the tokenizer class included in this project to introduce **byte-level Byte Pair Encoding** in its purest and most architecturally faithful form.

Its principal objective is to enable the learner to answer the following question clearly:

> If every text is reducible to a sequence of UTF-8 bytes, can a tokenizer that operates exclusively on byte identifiers — without any string-level intermediate representation — provide both universal coverage and learned compression?

This question lies at the heart of the most influential modern tokenizers, including those used by GPT-2, GPT-3, Llama, and Mistral. `ByteLevelBPETokenizer` represents the cleanest pedagogical embodiment of this idea within the project: integer byte identifiers in, integer token identifiers out, with no symbolic detour through strings during the training loop.

Example:

```text
"abababa"
    -> UTF-8 bytes: [97, 98, 97, 98, 97, 98, 97]
    -> learned merges (illustrative)
    -> token ids: [97, 258]
```

The defining characteristic of this tokenizer can be stated as follows:

> The training loop, the encoding logic, and the merge representations all operate on integer byte identifiers, never on character strings.

This design choice is what distinguishes `ByteLevelBPETokenizer` from `ByteBPETokenizer` (which performs BPE on a string-mapped representation of bytes) and from `SimpleBPETokenizer` (which operates on character strings entirely).

---

## 2. Why This Tokenizer Exists

This tokenizer occupies a privileged position within the project, because it represents the architectural reference point against which every other BPE-family tokenizer can be measured.

### a) It eliminates a layer of indirection present in earlier tokenizers

`SimpleBPETokenizer` operates on character strings. `ByteBPETokenizer` operates on byte values mapped through a single-character symbol table. `ByteLevelBPETokenizer` removes the symbol table entirely: pairs are `tuple[int, int]`, merges produce new integer ids, and decoding reconstructs byte sequences directly.

This makes the tokenizer's behavior easier to reason about and faster in principle, but more importantly:

> It demonstrates that the symbolic intermediate representation in earlier BPE implementations was an implementation convenience, not a conceptual necessity.

### b) It provides a faithful approximation of production tokenizer cores

The training loop of `ByteLevelBPETokenizer` is structurally aligned with that of the byte-level BPE used in GPT-2 and its successors. The frequencies are counted on integer pairs, the merge ids start at 256, and the merge schedule is applied in order during inference.

What is omitted, deliberately, is everything around this core:

* the GPT-2-style regex pre-tokenization (provided separately by `RegexBPETokenizer`)
* the printable byte-to-unicode remapping
* persistence and special-token machinery

The omissions clarify rather than dilute the algorithm.

### c) It exposes the OOV-free property of byte-level approaches at maximum strength

Because the base vocabulary covers all 256 possible byte values from construction, `ByteLevelBPETokenizer` cannot fail on any UTF-8 input — including emoji, scripts unseen during training, control characters, and arbitrary binary data.

This is the property that makes byte-level tokenizers preferred in modern LLM pipelines, and `ByteLevelBPETokenizer` makes it visible without distractions.

---

## 3. What "Byte-Level BPE" Means in This Project

In this project, byte-level BPE is treated as a tokenization approach in which two principles operate jointly:

* **the base vocabulary is fixed at 256 byte values**, ensuring universal coverage of UTF-8
* **merge learning operates on integer byte identifiers**, producing new ids that begin at 256

The implementation, however, is deliberately simplified:

* it does not perform regex pre-tokenization
* it does not handle special tokens
* it does not implement save/load functionality
* it does not employ priority-queue optimization for merge selection
* it does not include the GPT-2 byte-to-unicode remapping for printable token display

The class is therefore not a production tokenizer. It is a **conceptually faithful** simplification.

The objective is not to replicate industrial systems verbatim, but to render the principles underlying those systems pedagogically transparent.

---

## 4. Core Idea

The tokenizer operates according to the following logic:

1. The text is encoded as a UTF-8 byte sequence.
2. Each byte is treated as an independent token id in the range `0..255`.
3. Adjacent pair frequencies are counted across the entire token sequence.
4. The most frequent pair is selected, with deterministic tie-breaking.
5. A new token id is assigned, beginning at 256.
6. The pair is merged throughout the sequence, non-overlapping and left-to-right.
7. Steps three through six are repeated up to `num_merges` times.
8. During encoding, the same merge schedule is applied in order.
9. During decoding, token ids are mapped to byte sequences and concatenated.

Example:

```text
text = "abababa"

initial bytes:    [97, 98, 97, 98, 97, 98, 97]

iteration 1:
    pair frequencies:
        (97, 98) -> 3
        (98, 97) -> 3            # tie

    tie-breaking selects (98, 97)
    new token id = 256
    sequence after merge: [97, 256, 256, 256]

iteration 2:
    pair frequencies:
        (97, 256) -> 1
        (256, 256) -> 2

    selected pair: (256, 256)
    new token id = 257
    sequence after merge: [97, 257, 256]

iteration 3:
    pair frequencies:
        (97, 257) -> 1
        (257, 256) -> 1          # tie

    tie-breaking selects (257, 256)
    new token id = 258
    sequence after merge: [97, 258]
```

Three observations are essential here.

First, the algorithm operates entirely on integers from start to finish. There is no point at which bytes are reinterpreted as characters or strings.

Second, the tie-breaking rule produces a specific, deterministic outcome — and that outcome can be counterintuitive. In iteration 1, both `(97, 98)` (`'a','b'`) and `(98, 97)` (`'b','a'`) appear with the same frequency. The tie-breaking rule selects the **larger** pair under tuple comparison, which is `(98, 97)`. This detail is examined further in the determinism section.

Third, merges interact: a token produced in one iteration becomes available as a participant in pairs in subsequent iterations. This is precisely how higher-order patterns are progressively captured.

---

## 5. Why Byte-Level Initialization Matters

The decision to begin from a 256-entry byte vocabulary is the single most consequential design choice in this tokenizer.

The implications are as follows.

### a) The OOV problem is structurally eliminated

For `WordTokenizer` and `RegexTokenizer`, an unseen word at inference time is a fatal error. For `CharTokenizer`, an unseen character is a fatal error. For `SimpleBPETokenizer`, the same problem persists for unseen characters.

For `ByteLevelBPETokenizer`, no such failure mode exists. Every UTF-8 byte is, by construction, already in the vocabulary.

```text
training text:  "abababa"        # only bytes 97 and 98 observed
encoding text:  "merhaba 😊 dünya" # bytes never seen during training

result: encoded successfully, no error
```

This property is not a feature added on top of the algorithm; it is a consequence of the choice of base vocabulary.

### b) Multi-byte characters are decomposed naturally

A Turkish character such as `ç` is represented as two UTF-8 bytes. A 4-byte emoji such as `😊` is represented as four. The tokenizer treats these multi-byte sequences as ordinary token sequences during training, and merge learning may or may not eventually combine them into single tokens.

This decomposition is automatic. The learner does not need to specify language-specific rules; the byte-level base handles them all.

### c) Vocabulary growth is bounded by `num_merges`

Because the base vocabulary is fixed and learned tokens accumulate at `256, 257, 258, …`, the final vocabulary size is exactly `256 + len(merge_steps)`. This invariant — verified by the test suite — gives the user precise control over the model's vocabulary footprint.

---

## 6. Vocabulary Behavior

The vocabulary of `ByteLevelBPETokenizer` consists of two components.

### Base byte vocabulary

```text
0, 1, 2, ..., 255
```

The first 256 ids are reserved for raw byte values and are present from construction, even before training. This means:

```text
tokenizer = ByteLevelBPETokenizer(num_merges=3)
tokenizer.vocab_size  # 256, before train() is ever called
```

This is a structural difference from `WordTokenizer`, `CharTokenizer`, and `RegexTokenizer`, all of which have a vocabulary size of zero before training.

### Learned merge vocabulary

Beginning at id 256, each successful merge contributes one additional entry. After training:

```text
vocab_size = 256 + len(merge_steps)
```

The number of entries in `merge_steps` is bounded by `num_merges`, but may be smaller if the training corpus runs out of mergeable pairs (for example, in trivial inputs such as a single character).

This dual structure — fixed base plus learned extensions — is a hallmark of byte-level BPE. It is what `ByteBPETokenizer` and `RegexBPETokenizer` also employ, but `ByteLevelBPETokenizer` exhibits it most cleanly because it is unobscured by symbolic mappings or regex pre-tokenization.

---

## 7. The `ByteLevelBPEMerge` Dataclass

A small but architecturally significant choice: each learned merge is represented as a frozen dataclass.

```python
@dataclass(frozen=True)
class ByteLevelBPEMerge:
    pair: tuple[int, int]
    merged_token_id: int
    frequency: int
```

Three properties are worth noting.

### a) The dataclass is frozen

Once a merge has been recorded, its fields cannot be modified. This guarantees that the merge schedule learned during training cannot be silently mutated afterward — a property that downstream determinism guarantees rely upon.

### b) Structural equality is automatic

`@dataclass` generates an `__eq__` method that compares fields rather than identities. This is what makes the determinism tests in the suite meaningful: comparing `merge_steps` lists across two independently trained tokenizers is a valid test only because `ByteLevelBPEMerge` instances are compared by value.

### c) Frequency is recorded but unused at inference time

The `frequency` field is preserved purely for inspection and reporting. It is not consulted during encoding. Including it nevertheless reflects a deliberate transparency principle: a tokenizer should be inspectable, not just functional.

---

## 8. Separation of Responsibilities

A clear architectural separation is maintained within the class, even though the implementation is not split into separate trainer and tokenizer objects.

The following responsibilities are partitioned across distinct internal methods:

| Method | Responsibility |
|---|---|
| `_reset_training_state` | Clears prior training artifacts before a new run |
| `_get_pair_frequencies` | Computes adjacent pair frequencies as a `Counter` |
| `_merge_pair` | Applies a single merge rule to a token id list |
| `train` | Orchestrates the merge-learning loop |
| `encode` | Applies the learned merge schedule at inference time |
| `decode` | Reconstructs byte sequences and decodes to UTF-8 |
| `tokenize` | Returns the human-readable form of each token |

The two helpers — `_get_pair_frequencies` and `_merge_pair` — are pure functions over `list[int]`. They take token ids in, return token ids or counters out, and have no dependency on the tokenizer's training state.

This purity is not incidental; it is the property that makes them independently testable. The test suite exercises both helpers in isolation, on synthetic input lists that have nothing to do with `train()`. A regression in either helper fails its own dedicated test rather than only manifesting as a downstream training-level symptom.

This teaches the following principle:

> The most testable code is code that does not need a class instance's state to function.

---

## 9. Training Logic

The `train()` method is the most substantive component of this tokenizer.

Training proceeds through the following stages.

### a) Validation

Empty and whitespace-only inputs are rejected with a `ValueError`. The check uses `not text or not text.strip()`, which catches both literal empty strings and inputs containing only whitespace characters.

This is more defensive than `WordTokenizer`'s training validation, which accepts whitespace-only strings silently.

### b) State reset

`_reset_training_state()` is invoked at the start of every training call. This rebuilds the base byte vocabulary, clears the merge schedule, resets the merge id counter to 256, and marks the tokenizer as untrained.

The consequence is that retraining a tokenizer instance produces the same result as constructing a fresh one — a property verified explicitly by the test suite.

### c) Initial byte sequence

The text is encoded as UTF-8 and converted to a list of byte ids:

```python
token_ids = list(text.encode("utf-8"))
```

### d) Merge-learning loop

The loop runs at most `num_merges` iterations. At each iteration:

1. Adjacent pair frequencies are computed via `_get_pair_frequencies`.
2. If no pairs remain, the loop terminates early.
3. The most frequent pair is selected with deterministic tie-breaking.
4. A new id is assigned (`self._next_token_id`, then incremented).
5. The new id-to-bytes mapping is recorded by concatenating the components.
6. A `ByteLevelBPEMerge` is appended to `merge_steps`.
7. The merge is applied across the sequence via `_merge_pair`.

### e) Trained-state transition

After the loop, `self._trained = True` enables `encode`, `decode`, and `tokenize` to operate. Prior to this transition, all three methods raise `ValueError`.

---

## 10. Determinism and Tie-Breaking

A specific tie-breaking rule governs what happens when two or more pairs share the maximum frequency.

The rule, implemented via:

```python
key=lambda item: (item[1], item[0])  # (frequency, pair)
```

with `max(...)`, can be summarized as follows:

* the pair with the highest frequency is selected
* in case of a tie in frequency, the lexicographically **larger** pair is selected

The choice of "larger" rather than "smaller" is deliberate, but it produces outcomes that can surprise readers familiar with other BPE references.

For the input `"abababa"`, the pairs `(97, 98)` and `(98, 97)` both occur three times. Under this rule, `(98, 97)` is selected first — that is, the pair `('b', 'a')` rather than `('a', 'b')`.

This has cascading consequences for the entire merge schedule. The test suite documents and verifies this behavior explicitly, ensuring that future modifications to the tie-breaking rule do not silently shift downstream outputs.

The benefits of this design are:

* same input → same merge schedule, deterministically
* same merge schedule → same encoding output
* reproducible experiments and comparisons

In an educational project, where reproducibility of test fixtures and example outputs is essential, deterministic tie-breaking is non-negotiable.

---

## 11. Encode Logic

The `encode()` method follows a procedure structurally similar to that of training, but without learning new rules.

1. The trained-state precondition is verified.
2. Empty and whitespace-only inputs return an empty list rather than raising.
3. The input text is encoded to UTF-8 bytes.
4. The learned merges are applied in their training order.

A subtle but important asymmetry in input handling deserves attention:

* `train("")` raises `ValueError`
* `encode("")` returns `[]`

The asymmetry is deliberate. An empty corpus produces no learnable model, so training on it is meaningless and signaled as an error. An empty input to encode, by contrast, has a natural and unambiguous result: an empty token list. Pipelines that legitimately encounter empty strings — at sentence boundaries, in JSON fields, in web form input — should not need to special-case this.

The most critical point for the learner to grasp is the following:

> No new merges are learned during encoding; only the rules learned during training are applied, in their original order.

This is a property shared with all BPE-family tokenizers, but it is worth restating because it underlies a common confusion among learners encountering BPE for the first time.

---

## 12. Decode Logic

The `decode()` method reconstructs text from a list of token ids:

1. The trained-state precondition is verified.
2. Each id is mapped to its byte sequence via `_id_to_bytes`.
3. The byte sequences are concatenated.
4. The resulting byte sequence is UTF-8 decoded.

Two failure modes are handled explicitly.

### a) Unknown token id

If an id appears in the input that is not present in `_id_to_bytes`, a `ValueError` is raised. This catches cases such as:

* manually constructed id sequences containing typos
* id sequences from a tokenizer trained with different merges
* corrupted or out-of-range data

### b) Invalid UTF-8 sequence

If the concatenated bytes do not form a valid UTF-8 sequence, the underlying `UnicodeDecodeError` is wrapped in a more informative `ValueError`:

```text
"Token ids do not form a valid UTF-8 byte sequence"
```

This wrapping serves a deliberate purpose: it keeps the public error surface uniform. Callers need to handle only `ValueError`, not the deeper codec-level exception.

The lossless round-trip property — `decode(encode(text)) == text` for any UTF-8 input — follows directly from the byte-level design. Because every byte is preserved through encoding and recovered through decoding, no information about whitespace, punctuation, or character composition is lost.

This is in sharp contrast to `WordTokenizer` and `RegexTokenizer`, both of which discard whitespace information and reconstruct the decoded text heuristically.

---

## 13. The `tokenize()` Method and the `repr()` Fallback

A small but instructive detail concerns the `tokenize()` method.

Unlike `encode()`, which returns integer ids, `tokenize()` returns the human-readable string form of each token. The implementation invokes `encode()` and then converts each id to its byte representation, attempting UTF-8 decoding:

```python
try:
    tokens.append(token_bytes.decode("utf-8"))
except UnicodeDecodeError:
    tokens.append(repr(token_bytes))
```

The `repr()` fallback is necessary because individual byte-level tokens are not always valid UTF-8 strings on their own. Consider the Turkish character `ç`, whose two-byte UTF-8 representation is `[0xC3, 0xA7]`. If the BPE algorithm has not yet merged these two bytes into a single token, they exist as separate base byte tokens. Decoding either byte alone raises `UnicodeDecodeError`, because each is a fragment of a multi-byte sequence rather than a complete character.

The fallback to `repr()` produces a display form such as `b'\xc3'`, which is pedagogically transparent: it reveals to the learner that the token is a partial UTF-8 fragment, not a representable character.

This is a place where simplification chooses transparency over polish. Production tokenizers typically apply the GPT-2 byte-to-unicode remapping, which produces aesthetically clean printable representations for every byte. `ByteLevelBPETokenizer` deliberately omits this layer.

---

## 14. Strengths

The strengths of `ByteLevelBPETokenizer` can be summarized as follows.

### a) The OOV failure mode is structurally absent

Every possible UTF-8 input can be encoded. There is no scenario in which the tokenizer fails because of unseen characters or words.

### b) Round-trip is lossless

For any input text, encode followed by decode reproduces the original byte-for-byte. Whitespace, punctuation, and exotic characters are all preserved.

### c) The implementation is uniformly integer-based

Pairs, merges, ids, and counts are all integers. There is no point at which the algorithm switches between string and integer representations, which makes the code easier to audit and extend.

### d) Internal helpers are independently testable

`_get_pair_frequencies` and `_merge_pair` are pure functions over token id lists. They can be tested without ever invoking `train`, which produces precise, narrowly-scoped failure signals when something regresses.

### e) Tie-breaking is deterministic and documented

The selection rule is explicit, the tests verify its behavior, and the merge schedule is reproducible across runs and across machines.

### f) Re-training is safe

`_reset_training_state` ensures that retraining a tokenizer instance produces the same result as constructing a new one. State leakage between training runs is structurally prevented.

---

## 15. Limitations

This tokenizer also operates under several deliberately accepted constraints.

### a) No regex pre-tokenization

Without pre-tokenization, the algorithm may learn merges that span what humans would consider word boundaries. For inputs such as `"hello world hello world"`, a pair such as `(o, ` ` `)` (the letter `o` followed by a space) can be learned as a frequent pair. This is precisely the pathology that `RegexBPETokenizer` was designed to address.

### b) No special-token machinery

There is no support for `[BOS]`, `[EOS]`, `[PAD]`, or similar control tokens. This makes the tokenizer unsuitable for direct integration with transformer model inputs without further wrapping.

### c) No persistence

The tokenizer cannot be saved to or loaded from disk. The trained merge schedule exists only for the lifetime of the instance.

### d) No printable byte remapping

Tokens that are not valid UTF-8 fragments are displayed via `repr()`, producing forms such as `b'\xc3'`. Production tokenizers typically apply a byte-to-unicode mapping to produce cleaner display forms.

### e) Performance is O(num_merges × len(text))

Each merge iteration scans the full token sequence. For training corpora of moderate size this is acceptable; for large-scale corpora it would require a priority-queue-based optimization.

### f) Tie-breaking selects the lexicographically larger pair

This is the documented behavior, but it differs from the convention adopted by some BPE references (which select the smaller pair). Learners porting reference implementations should be aware of this choice.

These limitations should not be interpreted as deficiencies; rather, they reflect **deliberate decisions of scope**.

---

## 16. Comparison with Other Tokenizers

### ByteLevelBPETokenizer vs SimpleBPETokenizer

* `SimpleBPETokenizer` operates on character strings, suffering OOV failures on unseen characters.
* `ByteLevelBPETokenizer` operates on byte ids, with no possible OOV failure.

The two tokenizers occupy opposite ends of the BPE robustness spectrum.

### ByteLevelBPETokenizer vs ByteBPETokenizer

These two are the closest siblings in the project. Both operate on UTF-8 bytes and learn merges from byte data. The difference lies in the internal representation:

* `ByteBPETokenizer` maps each byte to a single-character symbol and runs a string-level BPE.
* `ByteLevelBPETokenizer` operates on integer byte ids end-to-end, with no symbolic intermediate.

In consequence:

* `ByteBPETokenizer` is conceptually closer to `SimpleBPETokenizer` and serves as a bridge from string-level to byte-level thinking.
* `ByteLevelBPETokenizer` is the cleaner end state, structurally closer to GPT-2's tokenizer core.

### ByteLevelBPETokenizer vs RegexBPETokenizer

* `RegexBPETokenizer` applies regex pre-tokenization before BPE, confining merges within chunks.
* `ByteLevelBPETokenizer` applies no pre-tokenization; merges may span any byte boundary in the input.

This is a clear trade-off:

* `ByteLevelBPETokenizer` is simpler and more general but may learn cross-word merges.
* `RegexBPETokenizer` is more constrained but produces linguistically cleaner tokens.

The two tokenizers represent two distinct architectural commitments. Production systems such as GPT-2 combine both.

### ByteLevelBPETokenizer vs GPT-2's tokenizer

GPT-2's tokenizer is essentially `RegexBPETokenizer` + a byte-to-unicode remapping for display + persistence. The merge-learning core is structurally the same as that of `ByteLevelBPETokenizer`. The omitted features can be added incrementally; they are not part of the tokenization algorithm itself.

This means `ByteLevelBPETokenizer` is an honest pedagogical simplification of GPT-2's core, not a fundamentally different design.

---

## 17. Design Decisions in This Project

The fundamental design decisions adopted for `ByteLevelBPETokenizer` in this project are as follows:

* the base vocabulary is fixed at 256 byte values from construction
* training operates exclusively on integer byte identifiers
* tie-breaking selects the lexicographically larger pair, deterministically
* learned merges are stored as frozen dataclass instances
* `_reset_training_state` ensures retraining produces the same result as constructing a fresh instance
* internal helpers (`_get_pair_frequencies`, `_merge_pair`) are pure functions over token id lists
* `train` rejects empty and whitespace-only input, while `encode` accepts and returns `[]` for them
* decode wraps `UnicodeDecodeError` in `ValueError` to keep the public error surface uniform
* `tokenize` falls back to `repr()` for byte fragments that are not valid UTF-8 on their own
* educational clarity is prioritized over production-grade performance

Each of these decisions reflects a balance between architectural realism and pedagogical accessibility.

---

## 18. Testing Perspective

The core behaviors verified by the tests for this tokenizer are as follows:

* `num_merges` validation rejects zero and negative values
* the initial vocabulary is exactly 256 entries before training
* training rejects empty and whitespace-only input
* training learns at least one merge for any non-trivial corpus
* the first selected merge corresponds to the documented tie-breaking rule
* learned ids are sequential and begin at 256
* training stops early when no pairs remain
* `num_merges` acts as an upper bound, not an exact target
* retraining clears prior state and resets the id counter
* `vocab_size = 256 + len(merge_steps)` invariant holds after training
* encode and decode raise `ValueError` before training
* encode returns `[]` on empty and whitespace-only input
* encode produces at least one merge id (>= 256) on trained input that contains a learned pattern
* token count after encoding is strictly less than raw byte length on repetitive inputs
* characters never seen during training can still be encoded and decoded losslessly
* lossless round-trip holds for ASCII, multi-byte Turkish characters, 4-byte emoji, mixed content, and inputs the tokenizer was never trained on
* `tokenize()` produces a list of strings whose concatenation equals the original input
* determinism holds across instances, across repeated calls, and in tie-breaking scenarios
* `ByteLevelBPEMerge` is frozen and supports structural equality
* `_get_pair_frequencies` and `_merge_pair` produce correct results on synthetic inputs

These tests are pedagogically valuable because they verify both the **structural invariants** of the tokenizer (vocabulary completeness, id ranges, frozen merges) and its **behavioral contracts** (lossless round-trip, deterministic tie-breaking, error handling). The combination of the two ensures that the tokenizer is internally consistent and externally well-behaved across a wide range of inputs.

---

## 19. When to Use

`ByteLevelBPETokenizer` is particularly well suited to the following contexts:

* explaining the algorithmic core of byte-level BPE in its purest form
* providing a reference point for understanding GPT-2-family tokenization
* studying the trade-offs between coverage, compression, and structural constraints
* serving as a baseline against which `RegexBPETokenizer` can be evaluated
* demonstrating how a fixed base vocabulary eliminates the OOV failure mode

It is not suitable in the following contexts:

* applications requiring linguistically clean tokens (use `RegexBPETokenizer` instead)
* systems requiring special tokens, persistence, or printable byte remapping
* large-scale training pipelines requiring optimized merge selection
* production deployments where every implementation detail matters

These cases call for industrial tokenizers; `ByteLevelBPETokenizer` provides the conceptual foundation for understanding them, not a substitute.

---

## 20. Final Takeaway

`ByteLevelBPETokenizer` is the algorithmic heart of the project's BPE family.

Because it teaches the following essential principle:

> Byte-level BPE is not a mere implementation choice; it is a structural commitment that simultaneously eliminates the out-of-vocabulary problem and grounds compression in the most universal representation a tokenizer can have — the raw byte.

Once this principle is internalized, the entire landscape of modern subword tokenizers — and the careful engineering that surrounds their byte-level cores — becomes legible from a single perspective.
