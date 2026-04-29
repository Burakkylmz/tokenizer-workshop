# MorphemeTokenizer

## 1. Purpose

`MorphemeTokenizer` is the tokenizer class included in this project to introduce **linguistically-motivated tokenization** through suffix-based morphological decomposition.

Its principal objective is to enable the learner to answer the following question clearly:

> When morphologically rich languages such as Turkish encode substantial grammatical information through suffixation, can a tokenizer that deliberately separates stems from their suffixes produce more meaningful and more efficient tokens than a tokenizer that treats words as atomic units?

This question matters because it represents a different philosophical commitment from every other tokenizer in this project. Every preceding tokenizer answers "what is a token?" with either a fixed rule (character, byte, word, regex chunk, n-gram) or a learned pattern (BPE merges, Unigram probabilities). `MorphemeTokenizer` answers it with a **linguistic claim**: that words have internal structure, and that this structure should be reflected in the tokenization.

Example:

```text
"Books are running."
    -> ["book", "s", "are", "runn", "ing", "."]

"Çocuklar okulda."
    -> ["çocuk", "lar", "okul", "da", "."]

"evlerimizde"
    -> ["ev", "ler", "imiz", "de"]
```

The defining characteristic of this tokenizer can be stated as follows:

> Tokenization here is no longer a purely mechanical or statistical procedure; it is an attempt — however simplified — to align token boundaries with the morphological boundaries that linguists themselves recognize.

---

## 2. Why This Tokenizer Exists

This tokenizer fulfills several distinct pedagogical roles within the project.

### a) It introduces linguistic motivation as a tokenization criterion

Every preceding tokenizer is **language-agnostic**. `CharTokenizer` cuts at every character; `ByteTokenizer` cuts at every byte; `RegexTokenizer` cuts where its pattern matches; BPE and Unigram cut where statistics suggest. None of these procedures consult any linguistic theory.

`MorphemeTokenizer` does. It encodes a specific linguistic hypothesis — that suffixes are decomposable units worth separating from stems — directly into the algorithm. The list of recognized suffixes is, in effect, a small grammar embedded in the tokenizer.

### b) It is particularly well-suited to morphologically rich languages

Turkish is an agglutinative language: a single Turkish word can encode information that English would express through several separate words, prepositions, or auxiliary verbs. The Turkish word `evlerimizde` corresponds, in English, to "in our houses" — a stem (`ev`, "house") followed by three suffixes encoding plurality, possession, and locative case.

For such languages, treating each surface form as an atomic token is strikingly inefficient. The vocabulary explodes, OOV failures become routine, and meaningful word relationships are lost. `MorphemeTokenizer` mitigates this by separating stems from suffixes, dramatically reducing both vocabulary size and OOV frequency for typical Turkish text.

### c) It exposes a linguistically grounded baseline

When statistical tokenizers (BPE, Unigram, WordPiece) are applied to morphologically rich languages, their learned segmentations sometimes — but not always — align with morpheme boundaries. `MorphemeTokenizer` provides a hand-built reference point against which such alignment can be measured.

If a learned tokenizer produces `["ev", "ler", "imiz", "de"]` for `evlerimizde`, it has discovered the morpheme structure on its own. If it produces something else, the difference quantifies how far statistical learning lies from linguistic ground truth.

### d) It demonstrates a deliberately bounded form of linguistic awareness

`MorphemeTokenizer` is not a morphological analyzer in the sense of Zemberek, the Helsinki Finite-State Toolkit, or production lemmatization libraries. It does not handle vowel harmony, consonant assimilation, irregular forms, or any of the phenomena that make real Turkish morphology challenging.

What it does provide is the simplest non-trivial linguistic procedure: **longest-suffix-first decomposition under a fixed suffix table**. Even this minimal commitment is enough to expose the trade-offs that all linguistically-motivated tokenizers face.

---

## 3. What "Morpheme" Means in This Project

In this project, "morpheme" is interpreted in a deliberately narrow sense:

> A morpheme is, for the purposes of this tokenizer, a recognized suffix from a fixed list. Stems are whatever remains after suffixes have been peeled off.

The implementation, however, omits the substantial machinery of real morphological analysis:

* it does not handle vowel harmony (Turkish: `evlerde` vs `okullarda`)
* it does not detect consonant changes (Turkish: `kitap → kitabı`)
* it does not handle irregular forms or suppletion
* it does not perform lemmatization in the linguistic sense
* it does not recognize prefixes, infixes, or circumfixes
* it does not distinguish derivational from inflectional morphology
* it does not detect compounds
* it does not validate morpheme combinations against any morphological grammar

The objective is therefore not to perform morphological analysis in any rigorous sense, but to render the **idea** of morphologically-aware tokenization concrete enough to be inspected, tested, and compared.

What remains, after these simplifications, is enough for pedagogical purposes: a small suffix list, a longest-match decomposition rule, and a minimum-stem-length guard. These three components capture the essential trade-offs of morpheme-based tokenization without the complexity of a real morphological grammar.

---

## 4. Core Idea

The tokenizer operates according to the following logic:

1. The input text is lowercased.
2. A regex (`\w+|[^\w\s]`) partitions the text into word tokens and punctuation tokens.
3. Each word token is recursively decomposed by stripping the longest matching suffix, while leaving a stem of at least `min_stem_length` characters.
4. Each punctuation token is preserved as-is.
5. Whitespace is consumed but not preserved.

Example with the default suffix list:

```text
text = "Books are running."

step 1 (lowercase + regex):
    ["books", "are", "running", "."]

step 2 (decompose word tokens):
    "books" -> match "s" suffix -> stem "book" (length 4 >= 2)
            -> ["book", "s"]

    "are"   -> no suffix matches that leaves a valid stem
            -> ["are"]

    "running" -> match "ing" suffix -> stem "runn" (length 4 >= 2)
              -> ["runn", "ing"]

    "." preserved as-is

final token sequence:
    ["book", "s", "are", "runn", "ing", "."]
```

A second example, illustrating multiple suffix iteration:

```text
text = "evlerimizde"

iteration 1: peel "de"   -> stem "evlerimiz" (length 9 >= 2)
iteration 2: peel "imiz" -> stem "evler"     (length 5 >= 2)
iteration 3: peel "ler"  -> stem "ev"        (length 2 >= 2)
iteration 4: no suffix matches with valid stem -> stop

reverse order (suffixes were collected outside-in):
    ["ev", "ler", "imiz", "de"]
```

Three observations are essential here.

First, suffix decomposition is **iterative**. A single word may shed multiple suffixes in sequence, each peeled from the diminishing remainder. This is what allows a complex Turkish form like `evlerimizde` to be decomposed into four meaningful pieces.

Second, decomposition is **longest-first**. The suffix list is sorted by length in descending order so that longer suffixes are tried before shorter ones. This is examined in detail in the next section.

Third, the algorithm is **bounded by `min_stem_length`**. If peeling a suffix would leave a stem shorter than this threshold, the suffix is rejected. This guards against linguistically implausible decompositions of short words.

---

## 5. Why Longest-Suffix-First Matters

The decision to try longer suffixes before shorter ones is not incidental. It encodes a specific linguistic intuition.

Consider the English word `kindness` and a suffix list containing both `s` and `ness`.

If suffixes were tried in alphabetical order (or insertion order without sorting), the algorithm might encounter `s` first:

```text
"kindness" + suffix "s"  -> stem "kindnes", suffix "s"  -> ["kindnes", "s"]
```

This decomposition is linguistically wrong. The word's morphology is `kind + ness`, not `kindnes + s`. Yet the algorithm has no way to know this without additional information.

By sorting suffixes longest-first, the algorithm tries `ness` before `s`:

```text
"kindness" + suffix "ness" -> stem "kind", suffix "ness" -> ["kind", "ness"]
```

This decomposition aligns with linguistic reality — not because the algorithm has become smarter, but because longer suffixes are more **specific** and therefore more reliable.

The same principle applies in Turkish. The suffix `ımız` should be preferred over `ım` when both are candidates, because `ımız` carries more information and is less likely to be a coincidental ending.

This is one of the few cases in the project where a simple sorting decision encodes a substantive linguistic claim:

> Longer suffixes are, all else equal, more reliable. Try them first.

This rule is not perfect. There are cases where a short suffix is actually the right choice and a longer one is a misleading coincidence. But for the simplified scope of this tokenizer, the longest-first heuristic is a reasonable default.

---

## 6. The `min_stem_length` Guard

A second linguistically-motivated heuristic governs what happens when peeling a suffix would leave a stem too short to be plausible.

Without such a guard, the algorithm would happily decompose any word ending in a recognized suffix, regardless of how little remained:

```text
"as" + suffix "s" -> stem "a" (length 1)
```

Decomposing `as` into `a + s` is linguistically nonsensical. The "s" here is not a plural suffix; it is part of an indivisible word.

The `min_stem_length` parameter (default value: 2) prevents this category of decomposition. If peeling a suffix would leave a stem with fewer than `min_stem_length` characters, the suffix is rejected and the word is preserved intact.

```text
"as" with min_stem_length=2 -> ["as"]   (suffix "s" rejected)
"books" with min_stem_length=2 -> ["book", "s"]   (stem length 4, accepted)
```

This guard is consequential because it transforms the tokenizer from a pure mechanical procedure into a procedure with a built-in plausibility check. Words below a threshold of "decomposability" are protected from over-segmentation.

The trade-off is straightforward:

* A higher `min_stem_length` reduces false-positive decompositions but also blocks legitimate ones for short words.
* A lower `min_stem_length` permits more decompositions but increases the rate of linguistically implausible ones.

The default of 2 strikes a balance suitable for educational use. For production morphological analysis, more sophisticated criteria — typically involving frequency or grammatical validity — are required.

---

## 7. Vocabulary Behavior

For `MorphemeTokenizer`, the vocabulary is defined as follows:

> The set of unique stems and suffixes observed in the training data, in the order in which they were first encountered.

Three implications are worth noting.

### a) The vocabulary is data-dependent

Different training texts produce different vocabularies. A training corpus rich in Turkish text will produce a vocabulary heavy with Turkish stems and recognized suffixes; an English corpus will produce one heavy with English equivalents.

### b) The vocabulary is typically smaller than `WordTokenizer`'s

Because morphologically related words share stems, the vocabulary grows more slowly than for word-level tokenization. A corpus containing `ev`, `evler`, `evlerimiz`, and `evlerimizde` produces only four word-level vocabulary entries but only one stem (`ev`) plus recurring suffixes — a much smaller footprint as the corpus grows.

This is precisely the property that makes morpheme-aware tokenization attractive for agglutinative languages.

### c) Insertion order is preserved

The vocabulary is constructed in first-appearance order, not lexicographic order. This is consistent with `NgramTokenizer` and `SubwordTokenizer`, both of which use the same `dict.fromkeys` pattern for deterministic but order-preserving deduplication.

---

## 8. Suffix List Normalization

A small but consequential detail concerns how the suffix list is processed during construction.

The user-provided suffix list (or the default list) is normalized through three steps:

1. Empty strings and `None` values are removed.
2. All suffixes are lowercased.
3. Duplicates are removed.
4. The remaining suffixes are sorted by length in descending order.

```python
normalized = list(set(s.lower() for s in suffixes if s))
return sorted(normalized, key=len, reverse=True)
```

This normalization is necessary because the suffix list directly governs algorithm behavior. A duplicated suffix wastes loop iterations; a misordered suffix produces wrong decompositions; an upper-case suffix never matches because the input is lowercased before tokenization.

The longest-first sort is what makes the algorithm work correctly. Without it, the longest-suffix-first principle would become an accident of insertion order rather than a guarantee.

---

## 9. Training Logic

For this tokenizer, "training" does not correspond to machine-learning training in any meaningful sense. It refers to the construction of a vocabulary mapping from the tokens produced by `tokenize()`.

Training proceeds through the following stages.

### a) Validation

Empty and whitespace-only inputs are rejected with a `ValueError`. This is consistent with the project's other tokenizers that include defensive validation.

### b) Tokenization

The input is processed by `tokenize()`, which performs the lowercase, regex, and suffix-decomposition pipeline.

Notably, `tokenize()` itself does **not** require training. The decomposition rules are static (the suffix list and `min_stem_length`) and do not depend on any learned state.

### c) Order-preserving deduplication

Duplicate tokens are removed while preserving first-appearance order via `dict.fromkeys(...)`.

### d) Bidirectional mapping construction

Forward (`token → id`) and reverse (`id → token`) mappings are built from the deduplicated list.

### e) Trained-state transition

The `_trained` flag is set to `True`, enabling subsequent encode and decode operations.

The training procedure performs no iterative optimization, no frequency analysis, and no probability assignment. The vocabulary contents are entirely determined by what `tokenize()` produces from the input.

---

## 10. The `tokenize()` Method

A consistent architectural detail across the project's tokenizers concerns whether `tokenize()` requires training.

`MorphemeTokenizer` follows the **training-independent** convention:

```python
tokenizer = MorphemeTokenizer()
tokenizer.tokenize("evlerimizde")  # works, no training required
tokenizer.encode("evlerimizde")    # raises: not trained
```

The rationale is that morpheme decomposition is a pure function of the input text and the configured suffix list. No vocabulary is needed to produce the decomposition; the vocabulary is only needed to assign integer ids.

This pattern is shared with `NgramTokenizer`, `SubwordTokenizer`, and `RegexTokenizer`. The grouping is not coincidental: all four tokenizers perform their core segmentation through fixed rules rather than learned parameters, which makes pre-training output meaningful.

The contrast is with `WordTokenizer`, where `tokenize()` is implemented as a wrapper around `encode()` and therefore inherits the training requirement. The difference is small in code but consequential in usage: tokenizers in the first group can be inspected and compared without orchestrating training; tokenizers in the second group cannot.

---

## 11. Encode and Decode

The encode and decode methods are nearly trivial.

`encode()` tokenizes the input and maps each token to its integer id. Tokens not present in the trained vocabulary raise a `ValueError`. There is no `[UNK]` mechanism.

`decode()` maps each id back to its token and joins the tokens with a single space. Unknown ids raise a `ValueError`.

A consequential property follows from this design:

> Decoding is intentionally lossy. Whitespace, capitalization, and the original boundaries between morphemes are not preserved.

The loss manifests in three specific ways.

### a) Capitalization is lost

The implicit lowercasing during tokenization is irreversible.

### b) Whitespace is not preserved exactly

The decoder joins tokens with a single space, regardless of how the original text was spaced. Multiple consecutive spaces, tabs, and newlines all collapse to a single space.

### c) Morpheme boundaries are visible in the output

Because tokens are joined with spaces, a decomposed word reappears as separate space-delimited pieces:

```text
"running" -> ["runn", "ing"] -> decoded -> "runn ing"
```

This is not the same as the original input. The space introduced between `runn` and `ing` is an artifact of the join procedure, not of the original text.

This is an honest reflection of the tokenizer's design:

> A tokenizer that decomposes words cannot trivially recompose them, because the space-delimited output preserves the decomposition rather than concealing it.

For users requiring the original text back, decode is not the right tool. For users wishing to see the morpheme structure made explicit, decode is exactly the right tool.

---

## 12. Strengths

The strengths of `MorphemeTokenizer` can be summarized as follows.

### a) It captures linguistic structure that statistical tokenizers may miss

For languages with rich agglutinative morphology, the morpheme decomposition often aligns better with semantic units than the segmentation produced by frequency-based tokenizers.

### b) Vocabulary growth is sub-linear in corpus size

Stems recur across morphologically related words, so the vocabulary grows much more slowly than for word-level tokenization. This is particularly valuable for Turkish and similar languages.

### c) The algorithm is conceptually transparent

Every decision — which suffix to try, when to stop, how short the stem can be — is a human-readable rule. There are no learned parameters, no probability tables, and no optimization procedures.

### d) Both the suffix list and the stem-length threshold are configurable

Users requiring different linguistic conventions (English-only, German, custom domain-specific suffixes) can supply their own suffix list at construction. The `min_stem_length` is similarly tunable.

### e) The tokenizer is fast and deterministic

Suffix matching is a single-pass operation per word. No probabilities, no DP tables, no merge schedules. Identical inputs yield identical outputs.

### f) `tokenize()` is independent of training

Reporting and comparison layers can use the tokenizer immediately, without orchestrating a training step.

---

## 13. Limitations

This tokenizer also operates under several deliberately accepted constraints.

### a) The suffix list is fixed

The tokenizer cannot learn new suffixes from data. If the training corpus contains morphology not covered by the suffix list (a less common Turkish suffix, an English suffix not on the list, any non-default language), the tokenizer treats those words as atomic. Updating the suffix list requires manual configuration.

### b) Phonological phenomena are ignored

Turkish exhibits vowel harmony, consonant alternation, and other phonological processes that produce surface forms differing from their underlying morphological structure. The tokenizer does not attempt to model any of these.

```text
"kitap" + accusative -> "kitabı"   (phonological change: p -> b)
```

The tokenizer would fail to recognize this relationship — it would treat `kitap` and `kitabı` as unrelated stems.

### c) Greedy decomposition can produce wrong answers

The longest-suffix-first heuristic is not infallible. Words that happen to end in a recognized suffix without actually containing that suffix as a morpheme will be incorrectly decomposed. (For example, the English word `address` does not end in the suffix `s`, but a naive algorithm might claim otherwise.)

### d) No prefix or infix handling

Only suffixes are recognized. Prefixes (`un-`, `re-`), circumfixes (German `ge-...-t`), and infixes (Tagalog) are not handled.

### e) No `[UNK]` mechanism

Unseen tokens during encoding raise a hard error rather than degrading gracefully.

### f) Decoding is intentionally lossy

The original text cannot be recovered exactly from token ids. Whitespace, capitalization, and morpheme-internal boundaries are not preserved.

### g) The tokenizer is not a morphological analyzer

For applications requiring real morphological analysis — lemmatization, root identification, derivational chain inference — this class is insufficient. Production tools (Zemberek for Turkish, the Helsinki Finite-State Toolkit, spaCy's morphological analyzers) should be used instead.

These limitations should not be interpreted as deficiencies; rather, they reflect **deliberate decisions of scope**.

---

## 14. Comparison with Other Tokenizers

### MorphemeTokenizer vs WordTokenizer

* `WordTokenizer` produces one token per surface form, with no internal structure.
* `MorphemeTokenizer` produces multiple tokens per word, exposing internal morphological structure.

In consequence:

* For agglutinative languages, `WordTokenizer` produces an explosively large vocabulary.
* `MorphemeTokenizer` keeps the vocabulary small by sharing stems across morphologically related words.

A particularly instructive observation: for a language with no morphology at all (a hypothetical pure isolating language with no affixes), `MorphemeTokenizer` would degrade to `WordTokenizer` exactly. The two differ only when morphology exists to be analyzed.

### MorphemeTokenizer vs SimpleBPETokenizer

* `SimpleBPETokenizer` discovers segmentations through statistical merging.
* `MorphemeTokenizer` applies a hand-built linguistic rule.

In consequence:

* BPE adapts to whatever data it is trained on, but its segmentations may or may not align with morphology.
* `MorphemeTokenizer` has fixed linguistic commitments but does not adapt to new languages without manual configuration.

A particularly instructive comparison: for a Turkish corpus, BPE may eventually learn merges that approximate morpheme boundaries, but it will also learn merges that do not. `MorphemeTokenizer` will produce morpheme-aligned segmentations from the start, but only for the suffixes it knows about.

### MorphemeTokenizer vs UnigramTokenizer

* `UnigramTokenizer` selects segmentations by maximizing learned probabilities.
* `MorphemeTokenizer` selects segmentations by applying a linguistic rule.

In consequence:

* Unigram is data-driven: its segmentations reflect the statistical structure of the training corpus.
* `MorphemeTokenizer` is theory-driven: its segmentations reflect the linguistic theory encoded in the suffix list.

The two represent different epistemological commitments. Neither is universally correct; they answer different questions.

### MorphemeTokenizer vs Production Morphological Analyzers (Zemberek, FST toolkits)

`MorphemeTokenizer` is to production morphological analysis what a glossary is to a dictionary: it provides surface-level recognition without the depth, regularity, or completeness of the real thing.

Production tools handle vowel harmony, irregular forms, derivational chains, and grammatical validity. `MorphemeTokenizer` handles none of these. It captures the **idea** of morpheme-aware tokenization without the engineering required to make that idea work in practice.

For learners, this simplification is valuable. For applications, the production tools are required.

---

## 15. Design Decisions in This Project

The fundamental design decisions adopted for `MorphemeTokenizer` in this project are as follows:

* a fixed suffix list is preferred over learned morpheme discovery
* longest-suffix-first matching encodes the linguistic principle that longer suffixes are more reliable
* `min_stem_length` guards against linguistically implausible over-decomposition
* the suffix list is normalized (lowercased, deduplicated, length-sorted) at construction
* lowercasing is applied implicitly during tokenization
* punctuation is preserved as a separate token class
* whitespace is consumed silently and not represented in the token stream
* `tokenize()` is independent of training, while `encode()` and `decode()` require it
* the vocabulary preserves first-appearance order via `dict.fromkeys`
* unseen tokens raise an error rather than degrading via `[UNK]`
* educational clarity and minimal linguistic commitment are prioritized over morphological completeness

Each of these decisions reflects a balance between linguistic motivation and architectural simplicity.

---

## 16. Testing Perspective

The core behaviors verified by the tests for this tokenizer are as follows:

* invalid `min_stem_length` values are rejected
* the suffix list is normalized at construction (case, duplicates, ordering)
* empty and whitespace-only training inputs are rejected
* training builds a vocabulary whose contents reflect the morpheme-decomposed input
* training is deterministic across instances given the same input
* simple suffixes (`-s`, `-ed`, `-ing`) are correctly stripped from English words
* Turkish plural suffixes (`-lar`, `-ler`) are correctly recognized
* multiple suffixes are stripped iteratively (`evlerimizde -> ev/ler/imiz/de`)
* `min_stem_length` correctly blocks over-decomposition of short words
* punctuation is preserved as a separate token class
* empty and whitespace-only inputs to `tokenize()` return empty lists
* encode and decode raise before training
* unseen tokens during encoding raise `ValueError`
* unknown ids during decoding raise `ValueError`
* `decode([])` returns the empty string
* round-trip on a single word is lossless when the word contains no morphology
* round-trip loses whitespace information
* `vocab_size` returns 0 before training

These tests are pedagogically valuable because they verify both the **algorithmic invariants** of suffix-based decomposition (longest-first, stem-length guard, iterative peeling) and its **behavioral contracts** (deterministic output, lossy reconstruction, strict OOV failure).

---

## 17. When to Use

`MorphemeTokenizer` is particularly well suited to the following contexts:

* introducing the concept of linguistically-motivated tokenization
* demonstrating morpheme-aware segmentation on Turkish or other agglutinative-language corpora
* providing a baseline against which statistical tokenizers can be compared on morphologically rich text
* educational settings exploring the relationship between linguistic theory and tokenization design
* small experiments where the suffix list can be hand-curated for the target domain

It is generally insufficient in the following contexts:

* applications requiring genuine morphological analysis (lemmatization, root finding, derivational chain inference)
* multilingual pipelines covering languages whose morphology is not captured by the suffix list
* systems requiring round-trip text reconstruction
* production deployments where Zemberek, FST toolkits, or transformer-based morphological analyzers are appropriate

These cases call for production-grade morphological analysis tools; `MorphemeTokenizer` provides the conceptual foundation against which such tools can be understood, not a substitute.

---

## 18. Final Takeaway

`MorphemeTokenizer` is the only tokenizer in the project that begins from a linguistic claim rather than a mechanical or statistical one.

Because it teaches the following essential principle:

> Tokenization is not exclusively a problem of frequency, geometry, or rule application; it can also be a problem of theory — and once linguistic structure is admitted as a legitimate criterion for token boundaries, the question of what a token should be becomes inseparable from the question of what the language itself is doing.

Once this principle is internalized, the entire landscape of tokenization design becomes legible as a set of choices about which kind of structure — statistical, geometric, linguistic — the tokenizer is willing to recognize.
