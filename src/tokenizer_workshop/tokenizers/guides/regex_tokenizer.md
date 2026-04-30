# RegexTokenizer

## 1. Purpose

`RegexTokenizer` is the tokenizer class included in this project to introduce the concept of **pattern-based pre-tokenization**.

Its principal objective is to enable the learner to answer the following question clearly:

> Can a regular expression alone determine where a token begins and ends, and is this sufficient to obtain a meaningful tokenization?

Unlike `CharTokenizer`, which operates on individual characters, or `ByteTokenizer`, which operates on raw bytes, `RegexTokenizer` works at a substantially higher level: it segments text into **words and punctuation marks** based on a configurable regex pattern.

Example:

```text
"Hello world!" -> ["Hello", "world", "!"]
```

This approach is particularly valuable because it represents the first stage at which the learner observes that **the boundary of a token can be defined externally**, rather than being implicit in the data itself.

---

## 2. Why This Tokenizer Exists

This tokenizer fills a critical pedagogical gap within the project.

### a) It introduces the notion of pre-tokenization

`CharTokenizer` and `ByteTokenizer` operate on inputs in their most atomic form. They make no decisions regarding what constitutes a "meaningful unit"; they simply traverse every character or byte.

`RegexTokenizer`, by contrast, operates at a higher level: it actively decides what counts as a token before any further processing occurs.

This is precisely the role that pre-tokenization plays in modern tokenizer pipelines.

### b) It separates words from punctuation

Most modern tokenizers (BERT, GPT-2, RoBERTa, etc.) treat punctuation marks as separate tokens during their pre-tokenization stage.

`RegexTokenizer` makes this behavior explicit:

```text
"Hello, world!" -> ["Hello", ",", "world", "!"]
```

This treatment is essential, because in natural language a comma or an exclamation mark generally carries semantic value distinct from the surrounding word.

### c) It demonstrates the configurability of tokenization

`RegexTokenizer` accepts a custom regex pattern. This means the learner can directly observe how altering the tokenization rule changes the output.

This makes the relationship between **rule** and **behavior** transparent in a way that is difficult to achieve with fixed-rule tokenizers.

---

## 3. What "Regex" Means in This Project

In this project, regex is treated not merely as a string-matching tool, but as a mechanism for **defining token boundaries**.

The default pattern adopted by `RegexTokenizer` is approximately as follows:

```text
\w+ | [^\w\s]
```

This pattern operates on two fundamental principles:

* `\w+` matches sequences consisting of letters, digits, or underscores. In practice, this captures words and numbers as single tokens.
* `[^\w\s]` matches a single non-word, non-whitespace character. In practice, this captures punctuation marks and most symbols.

This combination effectively communicates to the learner the following idea:

> A token is either a sequence of word characters (a word or a number), or a single non-word, non-whitespace character.

Note that whitespace itself is **not** captured. It functions as a delimiter, but is not preserved as a token.

This design decision has significant implications and is examined in greater detail in the section on decoding.

---

## 4. Core Idea

The tokenizer operates according to the following logic:

1. The text is scanned using a regex pattern.
2. All matches are collected as tokens.
3. Whitespace between matches is silently discarded.
4. The unique tokens form the vocabulary.
5. Each token is assigned an integer identifier.
6. During decoding, tokens are joined to reconstruct a readable form of the text.

Example:

```text
text = "Hello, world!"

regex matches:
    "Hello" -> word
    ","     -> punctuation
    "world" -> word
    "!"     -> punctuation

tokens = ["Hello", ",", "world", "!"]
```

A second example, including digits:

```text
text = "abc 123 def"

regex matches:
    "abc" -> word characters
    "123" -> word characters (digits are matched by \w)
    "def" -> word characters

tokens = ["abc", "123", "def"]
```

The critical observation here is the following:

> The tokenizer no longer traverses every character; it scans the text in larger, semantically motivated chunks.

This represents a substantive shift in granularity compared to `CharTokenizer` and `ByteTokenizer`.

---

## 5. Why Unicode-Aware Regex Matters

The regex implementation in this tokenizer is Unicode-aware. This is not an arbitrary detail; it is a deliberate design decision.

In Python, `\w` does not refer only to ASCII letters by default. Within a Unicode-aware regex, characters such as:

* Turkish characters (`ç`, `ğ`, `ü`, `ş`, `ö`, `ı`, `İ`)
* characters from CJK scripts
* Cyrillic characters
* characters from many other writing systems

are also recognized as word characters.

Example:

```text
"Merhaba dünya!" -> ["Merhaba", "dünya", "!"]
```

This behavior would not be available with an ASCII-only regex. As a result:

* `dünya` is captured as a single token
* the character `ü` is not erroneously treated as a separate non-word symbol

Without this property, the tokenizer would be of limited use for languages other than English. Pedagogically, this conveys an important lesson:

> Tokenizers are not language-neutral; their design decisions directly determine which languages they can serve effectively.

---

## 6. Vocabulary Behavior

For `RegexTokenizer`, the vocabulary is defined as follows:

> The number of unique regex tokens observed in the training data.

The implications of this are as follows:

* the vocabulary is **data-dependent**
* different corpora produce different vocabularies
* a small text yields a small vocabulary
* a previously unseen word raises an error during encoding

This behavior is similar to that of `CharTokenizer`, but operates at a much coarser granularity:

| Tokenizer | Token unit |
|---|---|
| `CharTokenizer` | a single character |
| `ByteTokenizer` | a single byte |
| `RegexTokenizer` | a word, a number, or a single punctuation mark |

This contrast is pedagogically valuable, because it raises the following question for the learner:

> When the granularity of the token grows, what is gained, and what is lost?

This question is examined further in the sections on strengths and limitations.

---

## 7. Training Logic

For this tokenizer, "training" does not correspond to machine-learning training in the conventional sense. Training here refers to the construction of a vocabulary from a given text.

Training proceeds through the following stages:

### a) The text is tokenized using regex

```python
tokens = self.tokenize(text)
```

### b) Unique tokens are extracted

```python
unique_tokens = sorted(set(tokens))
```

The `sorted(...)` call here is highly consequential. Without it, the ordering of the mapping could become unpredictable, rendering the training output non-reproducible.

### c) A bidirectional mapping is constructed

* `token_to_id`: token → integer
* `id_to_token`: integer → token

This bidirectional structure mirrors that of `CharTokenizer`, but operates at a different level of granularity.

---

## 8. Encode Logic

The `encode()` method converts each regex token into an integer token id.

Example:

```text
"Hello world!" -> [id_Hello, id_world, id_!]
```

A deliberate design decision has been made at this stage: when the tokenizer encounters a token not seen during training, it **does not pass over it silently** and **does not fabricate a fallback**. Instead, it raises an error directly.

This decision is pedagogically justified, as it makes the following problem explicit:

> What should a tokenizer do when encountering a word it has never seen?

In real-world systems, this problem is addressed through mechanisms such as `unknown token`, `subword fallback`, or `byte fallback`. Here, however, the aim is first to expose the problem in its starkest form.

This is precisely the limitation that motivates the more advanced tokenizers introduced later in the project: `SimpleBPETokenizer`, `ByteBPETokenizer`, and others.

---

## 9. Decode Logic

The `decode()` method converts a list of integer token ids back into text.

The procedure is as follows:

1. Each id is mapped back to its corresponding token using `id_to_token`.
2. The tokens are joined together with appropriate whitespace handling.

A subtle but important point arises here:

> `RegexTokenizer` does not preserve the original whitespace.

During tokenization, whitespace is silently discarded. As a result, the decoded text is not necessarily byte-identical to the original input.

To produce a readable result, the decoder applies a heuristic: words are joined with a single space, while punctuation marks are attached to the preceding token without a space.

Example:

```text
input:    "Hello, world!"
tokens:   ["Hello", ",", "world", "!"]
decoded:  "Hello, world!"
```

This generally produces the desired result, but it is important to note:

* multiple consecutive spaces are collapsed
* tabs and newlines are not preserved
* unconventional spacing is normalized

This makes explicit a critical property of tokenization:

> A tokenizer is not necessarily a lossless transformation.

The learner should appreciate that round-trip equivalence is a design choice rather than a universal guarantee. Some tokenizers (such as `ByteTokenizer`) are lossless by construction; others (such as `RegexTokenizer`) accept a degree of loss in exchange for a more meaningful tokenization.

---

## 10. Configurable Patterns

One of the most distinctive features of `RegexTokenizer` is that the regex pattern itself is configurable.

Example:

```python
tokenizer = RegexTokenizer(pattern=r"[A-Za-z]+")
tokenizer.tokenize("Hello, world! 123")
# -> ["Hello", "world"]
```

In this configuration:

* the comma is not captured (it is not a letter)
* the exclamation mark is not captured
* the digits `123` are not captured

This demonstrates an important principle:

> The behavior of the tokenizer is determined entirely by the regex.

This makes evident a property of fundamental importance:

> A tokenizer is not merely an algorithm; it is a contract defined by its rules.

This perspective is essential for understanding tokenizer design at a deeper level.

---

## 11. Strengths

The strengths of `RegexTokenizer` can be summarized as follows:

### a) It produces meaningful units

Unlike character-level or byte-level tokenizers, the resulting tokens generally correspond to recognizable linguistic elements such as words and punctuation marks.

### b) Sequence length is significantly reduced

Compared to `CharTokenizer`, the same text is represented in far fewer tokens.

### c) It introduces the concept of pre-tokenization

This concept forms the foundation for more advanced tokenizers such as those used in BERT and GPT-2.

### d) Its behavior is configurable

Different regex patterns produce different tokenization behaviors. This flexibility is rare among the simpler tokenizers in this project.

### e) It supports Unicode

It can handle Turkish, Cyrillic, CJK, and other non-ASCII writing systems out of the box.

### f) It is deterministic

Identical inputs yield identical outputs, which simplifies testing and debugging.

---

## 12. Limitations

This tokenizer also operates under several deliberately accepted constraints.

### a) Whitespace is not preserved

The original spacing of the input cannot be reconstructed exactly. For applications requiring lossless reconstruction (e.g., source code processing), this is a significant limitation.

### b) Unseen words cannot be encoded

`RegexTokenizer` raises an error when it encounters a word that does not appear in its vocabulary. There is no subword fallback, no byte fallback, and no unknown-token mechanism.

### c) The vocabulary can grow rapidly

In a sufficiently large corpus, virtually every distinct word becomes a separate vocabulary entry. This causes the vocabulary to grow much faster than that of a character-level tokenizer.

### d) It does not exploit recurring structures

Words that share a common root (`run`, `running`, `runner`) are treated as entirely distinct tokens. This is precisely the limitation that BPE-family tokenizers are designed to address.

### e) It is sensitive to the choice of regex

A poorly designed regex can produce unintended tokenization behavior. The flexibility offered by configurability also introduces an additional category of potential errors.

These limitations should not be interpreted as deficiencies; rather, they reflect **deliberate decisions of scope**.

---

## 13. Comparison with Other Tokenizers

### RegexTokenizer vs CharTokenizer

* `CharTokenizer` operates on individual characters.
* `RegexTokenizer` operates on words and punctuation.

In consequence:

* `CharTokenizer` produces longer sequences.
* `RegexTokenizer` produces shorter and more meaningful sequences.

However, `CharTokenizer` is more inclusive: it can encode any character, whereas `RegexTokenizer` cannot encode words it has not previously seen.

### RegexTokenizer vs ByteTokenizer

* `ByteTokenizer` provides universal coverage.
* `RegexTokenizer` provides a coarser, more meaningful granularity.

This is a clear trade-off:

* `ByteTokenizer` is robust but verbose.
* `RegexTokenizer` is concise but fragile.

### RegexTokenizer vs SimpleBPETokenizer

* `SimpleBPETokenizer` learns recurring patterns.
* `RegexTokenizer` operates entirely from a fixed rule.

In consequence:

* `SimpleBPETokenizer` adapts to the data.
* `RegexTokenizer` does not.

However, `RegexTokenizer` produces more interpretable tokens, which is particularly valuable in pedagogical settings.

### RegexTokenizer vs real-world tokenizers (BERT, GPT-2)

Real-world tokenizers typically combine regex-based pre-tokenization with a subword learning mechanism (BPE, WordPiece, Unigram).

`RegexTokenizer` represents only the first half of this pipeline. For this reason:

* it should be regarded as a **pre-tokenization** layer
* it constitutes a stepping stone toward more advanced architectures, not a final destination

---

## 14. Design Decisions in This Project

The fundamental design decisions adopted for `RegexTokenizer` in this project are as follows:

* a Unicode-aware default regex pattern is preferred
* the regex pattern is exposed as a configurable parameter
* the vocabulary is constructed deterministically from the training data
* whitespace is not preserved
* punctuation is treated as a distinct token category
* unseen tokens raise an error rather than being silently substituted
* educational clarity is prioritized over production-grade robustness

Each of these decisions reflects a balance between pedagogical value and architectural simplicity.

---

## 15. Testing Perspective

The core behaviors verified by the tests for this tokenizer are as follows:

* the vocabulary is constructed correctly after training
* unique regex tokens are reflected in `vocab_size`
* the same input produces an identical mapping
* punctuation is treated as a separate token
* Turkish and other Unicode characters are handled correctly
* empty strings and whitespace-only inputs return an empty token list
* custom regex patterns alter the tokenization behavior
* `encode()` produces a list of integer ids
* invoking `encode()` or `decode()` prior to training raises an error
* unseen tokens raise an error during encoding
* unknown ids raise an error during decoding
* simple round-trip cases yield faithful reconstruction

These tests are pedagogically valuable, because they explicitly verify the **boundary between rule and behavior** that lies at the heart of `RegexTokenizer`.

---

## 16. When to Use

`RegexTokenizer` is particularly well suited to the following contexts:

* introducing the concept of pre-tokenization
* explaining the distinction between word-level and character-level tokenization
* discussing the role of punctuation in tokenization
* demonstrating the importance of Unicode-aware patterns
* providing a reference behavior for comparison with subword tokenizers

It is generally insufficient in the following contexts:

* tokenizers that must produce identical output to the input (lossless reconstruction)
* large-scale NLP pipelines requiring out-of-vocabulary robustness
* multilingual systems with extensive vocabulary requirements
* modern LLM tokenizer pipelines

These cases call for more advanced architectures, several of which are introduced later in the project.

---

## 17. Final Takeaway

`RegexTokenizer` occupies a critical threshold within the project's tokenization pipeline.

Because it teaches the following essential idea:

> A tokenizer's behavior is not derived solely from the data; it can also be defined by an explicit rule.

Once this principle is internalized, the rationale for the more sophisticated tokenizers introduced later — those that **combine** regex-based pre-tokenization with learned subword merges — becomes much easier to grasp.
