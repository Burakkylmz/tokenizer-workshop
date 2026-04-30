```

========================================================================================================================
                                              TOKENIZER EVALUATION REPORT                                               
========================================================================================================================

SOURCE TEXT
------------------------------------------------------------------------------------------------------------------------
Hello world! Tokenization is fun.

SUMMARY TABLE
------------------------------------------------------------------------------------------------------------------------
Tokenizer     | Token | Uniq | Uniq Ratio | Avg Len | Min Len | Max Len | Avg Chars/Token | Unknown | Latency  | Eff.
--------------+-------+------+------------+---------+---------+---------+-----------------+---------+----------+-----
word          | 7     | 7    | 1.00       | 4.14    | 1       | 12      | 4.71            | 0.00    | 0.000029 | 4.71
char          | 33    | 20   | 0.61       | 1.73    | 1       | 2       | 1.00            | 0.00    | 0.000037 | 1.00
byte          | 33    | 20   | 0.61       | 2.73    | 2       | 3       | 1.00            | 0.00    | 0.000024 | 1.00
byte_bpe      | 25    | 19   | 0.76       | 1.32    | 1       | 5       | 1.32            | 0.00    | 0.000169 | 1.32
simple_bpe    | 25    | 19   | 0.76       | 1.80    | 1       | 2       | 1.32            | 0.00    | 0.000125 | 1.32
regex         | 7     | 7    | 1.00       | 4.14    | 1       | 12      | 4.71            | 0.00    | 0.000018 | 4.71
regex_bpe     | 11    | 8    | 0.73       | 2.45    | 2       | 3       | 3.00            | 0.00    | 0.000384 | 3.00
ngram         | 4     | 4    | 1.00       | 13.25   | 7       | 19      | 8.25            | 0.00    | 0.000018 | 8.25
wordpiece     | 11    | 9    | 0.82       | 4.00    | 1       | 12      | 3.00            | 0.00    | 0.000048 | 3.00
unigram       | 12    | 11   | 0.92       | 2.75    | 1       | 12      | 2.75            | 0.00    | 0.000130 | 2.75
sentencepiece | 30    | 20   | 0.67       | 1.13    | 1       | 2       | 1.10            | 0.00    | 0.000097 | 1.10

HIGHLIGHTS
------------------------------------------------------------------------------------------------------------------------
Lowest Token Count   : ngram (4)
Highest Token Count  : char (33)
Best Efficiency      : ngram (8.25)
Highest Unique Count : char (20)
Fastest Tokenizer    : ngram (18µs)

INTERPRETATION
------------------------------------------------------------------------------------------------------------------------
The 'ngram' tokenizer yielded the lowest token count, suggesting a coarse-grained segmentation strategy.
The 'char' tokenizer produced the highest number of tokens, indicating a fine-grained segmentation.
The 'ngram' tokenizer achieved the highest efficiency score, (8.25), balancing token count and unknown-token behavior.
The 'char' tokenizer generated the most unique tokens (20), suggesting higher diversity.
The 'ngram' tokenizer completed tokenization in the shortest time (0.000018s), making it the fastest option for this sample.
The 'ngram' tokenizer produced the highest characters-per-token ratio (8.25), indicating larger token chunks.

Overall, these observations highlight how different tokenization strategies affect segmentation granularity, token diversity, and processing efficiency.

TOKENIZER DETAILS
------------------------------------------------------------------------------------------------------------------------

[word]
Tokens                 : ['Hello', 'world', '!', 'Tokenization', 'is', 'fun', '.']
Token Count            : 7
Unique Token Count     : 7
Unique Ratio           : 1.00
Average Token Length   : 4.14
Min Token Length       : 1
Max Token Length       : 12
Avg Chars / Token      : 4.71
Unknown Count          : 0
Unknown Rate           : 0.00
Latency                : 0.000029s
Efficiency Score       : 4.71
Top-5 Tokens           : [('Hello', 1), ('world', 1), ('!', 1), ('Tokenization', 1), ('is', 1)]
Token Length Dist.     : {1: 2, 2: 1, 3: 1, 5: 2, 12: 1}

[char]
Tokens                 : ['5', '11', '17', '17', '20', '1', '27', '20', '23', '17', '10', '2', '1', '6', '20', '16', '11', '19', '15', '30', '7', '25', '15', '20', '19', '1', '15', '24', '1', '12', '26', '19', '3']
Token Count            : 33
Unique Token Count     : 20
Unique Ratio           : 0.61
Average Token Length   : 1.73
Min Token Length       : 1
Max Token Length       : 2
Avg Chars / Token      : 1.00
Unknown Count          : 0
Unknown Rate           : 0.00
Latency                : 0.000037s
Efficiency Score       : 1.00
Top-5 Tokens           : [('20', 4), ('1', 4), ('17', 3), ('19', 3), ('15', 3)]
Token Length Dist.     : {1: 9, 2: 24}

[byte]
Tokens                 : ['72', '101', '108', '108', '111', '32', '119', '111', '114', '108', '100', '33', '32', '84', '111', '107', '101', '110', '105', '122', '97', '116', '105', '111', '110', '32', '105', '115', '32', '102', '117', '110', '46']
Token Count            : 33
Unique Token Count     : 20
Unique Ratio           : 0.61
Average Token Length   : 2.73
Min Token Length       : 2
Max Token Length       : 3
Avg Chars / Token      : 1.00
Unknown Count          : 0
Unknown Rate           : 0.00
Latency                : 0.000024s
Efficiency Score       : 1.00
Top-5 Tokens           : [('111', 4), ('32', 4), ('108', 3), ('110', 3), ('105', 3)]
Token Length Dist.     : {2: 9, 3: 24}

[byte_bpe]
Tokens                 : ['H', 'el', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', '!', ' ', 'Token', 'i', 'z', 'at', 'i', 'o', 'n ', 'i', 's ', 'f', 'u', 'n', '.']
Token Count            : 25
Unique Token Count     : 19
Unique Ratio           : 0.76
Average Token Length   : 1.32
Min Token Length       : 1
Max Token Length       : 5
Avg Chars / Token      : 1.32
Unknown Count          : 0
Unknown Rate           : 0.00
Latency                : 0.000169s
Efficiency Score       : 1.32
Top-5 Tokens           : [('o', 3), ('i', 3), ('l', 2), (' ', 2), ('H', 1)]
Token Length Dist.     : {1: 20, 2: 4, 5: 1}

[simple_bpe]
Tokens                 : ['5', '36', '17', '20', '1', '27', '20', '23', '17', '10', '2', '1', '40', '15', '30', '35', '15', '20', '37', '15', '33', '12', '26', '19', '3']
Token Count            : 25
Unique Token Count     : 19
Unique Ratio           : 0.76
Average Token Length   : 1.80
Min Token Length       : 1
Max Token Length       : 2
Avg Chars / Token      : 1.32
Unknown Count          : 0
Unknown Rate           : 0.00
Latency                : 0.000125s
Efficiency Score       : 1.32
Top-5 Tokens           : [('20', 3), ('15', 3), ('17', 2), ('1', 2), ('5', 1)]
Token Length Dist.     : {1: 5, 2: 20}

[regex]
Tokens                 : ['Hello', 'world', '!', 'Tokenization', 'is', 'fun', '.']
Token Count            : 7
Unique Token Count     : 7
Unique Ratio           : 1.00
Average Token Length   : 4.14
Min Token Length       : 1
Max Token Length       : 12
Avg Chars / Token      : 4.71
Unknown Count          : 0
Unknown Rate           : 0.00
Latency                : 0.000018s
Efficiency Score       : 4.71
Top-5 Tokens           : [('Hello', 1), ('world', 1), ('!', 1), ('Tokenization', 1), ('is', 1)]
Token Length Dist.     : {1: 2, 2: 1, 3: 1, 5: 2, 12: 1}

[regex_bpe]
Tokens                 : ['275', '32', '279', '33', '32', '268', '32', '280', '32', '282', '46']
Token Count            : 11
Unique Token Count     : 8
Unique Ratio           : 0.73
Average Token Length   : 2.45
Min Token Length       : 2
Max Token Length       : 3
Avg Chars / Token      : 3.00
Unknown Count          : 0
Unknown Rate           : 0.00
Latency                : 0.000384s
Efficiency Score       : 3.00
Top-5 Tokens           : [('32', 4), ('275', 1), ('279', 1), ('33', 1), ('268', 1)]
Token Length Dist.     : {2: 6, 3: 5}

[ngram]
Tokens                 : ['Hello world!', 'world! Tokenization', 'Tokenization is', 'is fun.']
Token Count            : 4
Unique Token Count     : 4
Unique Ratio           : 1.00
Average Token Length   : 13.25
Min Token Length       : 7
Max Token Length       : 19
Avg Chars / Token      : 8.25
Unknown Count          : 0
Unknown Rate           : 0.00
Latency                : 0.000018s
Efficiency Score       : 8.25
Top-5 Tokens           : [('Hello world!', 1), ('world! Tokenization', 1), ('Tokenization is', 1), ('is fun.', 1)]
Token Length Dist.     : {7: 1, 12: 1, 15: 1, 19: 1}

[wordpiece]
Tokens                 : ['hel', '##l', '##o', '[UNK]', '[UNK]', 'tokenization', '[UNK]', 'f', '##u', '##n', '.']
Token Count            : 11
Unique Token Count     : 9
Unique Ratio           : 0.82
Average Token Length   : 4.00
Min Token Length       : 1
Max Token Length       : 12
Avg Chars / Token      : 3.00
Unknown Count          : 0
Unknown Rate           : 0.00
Latency                : 0.000048s
Efficiency Score       : 3.00
Top-5 Tokens           : [('[UNK]', 3), ('hel', 1), ('##l', 1), ('##o', 1), ('tokenization', 1)]
Token Length Dist.     : {1: 2, 3: 5, 5: 3, 12: 1}

[unigram]
Tokens                 : ['hel', 'l', 'o', '[UNK]', '[UNK]', 'tokenization', 'i', 's', 'f', 'u', 'n', '.']
Token Count            : 12
Unique Token Count     : 11
Unique Ratio           : 0.92
Average Token Length   : 2.75
Min Token Length       : 1
Max Token Length       : 12
Avg Chars / Token      : 2.75
Unknown Count          : 0
Unknown Rate           : 0.00
Latency                : 0.000130s
Efficiency Score       : 2.75
Top-5 Tokens           : [('[UNK]', 2), ('hel', 1), ('l', 1), ('o', 1), ('tokenization', 1)]
Token Length Dist.     : {1: 8, 3: 1, 5: 2, 12: 1}

[sentencepiece]
Tokens                 : ['▁', 'H', 'el', 'l', 'o', '▁', 'w', 'o', 'r', 'l', 'd', '!', '▁', 'T', 'o', 'k', 'en', 'i', 'z', 'at', 'i', 'o', 'n', '▁', 'i', 's', '▁f', 'u', 'n', '.']
Token Count            : 30
Unique Token Count     : 20
Unique Ratio           : 0.67
Average Token Length   : 1.13
Min Token Length       : 1
Max Token Length       : 2
Avg Chars / Token      : 1.10
Unknown Count          : 0
Unknown Rate           : 0.00
Latency                : 0.000097s
Efficiency Score       : 1.10
Top-5 Tokens           : [('▁', 4), ('o', 4), ('i', 3), ('l', 2), ('n', 2)]
Token Length Dist.     : {1: 26, 2: 4}

OVERALL RANKING
------------------------------------------------------------------------------------------------------------------------
1. ngram (efficiency=8.25, latency=0.000018s)
2. regex (efficiency=4.71, latency=0.000018s)
3. word (efficiency=4.71, latency=0.000029s)
4. wordpiece (efficiency=3.00, latency=0.000048s)
5. regex_bpe (efficiency=3.00, latency=0.000384s)
6. unigram (efficiency=2.75, latency=0.000130s)
7. simple_bpe (efficiency=1.32, latency=0.000125s)
8. byte_bpe (efficiency=1.32, latency=0.000169s)
9. sentencepiece (efficiency=1.10, latency=0.000097s)
10. byte (efficiency=1.00, latency=0.000024s)
11. char (efficiency=1.00, latency=0.000037s)

PAIRWISE COMPARISONS
------------------------------------------------------------------------------------------------------------------------

[word <-> char]
Common Tokens (0)      : []
Only In word (7) : ['!', '.', 'Hello', 'Tokenization', 'fun', 'is', 'world']
Only In char (20) : ['1', '10', '11', '12', '15', '16', '17', '19', '2', '20', '23', '24', '25', '26', '27', '3', '30', '5', '6', '7']
Overlap Ratio : 0.00

[word <-> byte]
Common Tokens (0)      : []
Only In word (7) : ['!', '.', 'Hello', 'Tokenization', 'fun', 'is', 'world']
Only In byte (20) : ['100', '101', '102', '105', '107', '108', '110', '111', '114', '115', '116', '117', '119', '122', '32', '33', '46', '72', '84', '97']
Overlap Ratio : 0.00

[word <-> byte_bpe]
Common Tokens (2)      : ['!', '.']
Only In word (5) : ['Hello', 'Tokenization', 'fun', 'is', 'world']
Only In byte_bpe (17) : [' ', 'H', 'Token', 'at', 'd', 'el', 'f', 'i', 'l', 'n', 'n ', 'o', 'r', 's ', 'u', 'w', 'z']
Overlap Ratio : 0.08

[word <-> simple_bpe]
Common Tokens (0)      : []
Only In word (7) : ['!', '.', 'Hello', 'Tokenization', 'fun', 'is', 'world']
Only In simple_bpe (19) : ['1', '10', '12', '15', '17', '19', '2', '20', '23', '26', '27', '3', '30', '33', '35', '36', '37', '40', '5']
Overlap Ratio : 0.00

[word <-> regex]
Common Tokens (7)      : ['!', '.', 'Hello', 'Tokenization', 'fun', 'is', 'world']
Only In word (0) : []
Only In regex (0) : []
Overlap Ratio : 1.00

[word <-> regex_bpe]
Common Tokens (0)      : []
Only In word (7) : ['!', '.', 'Hello', 'Tokenization', 'fun', 'is', 'world']
Only In regex_bpe (8) : ['268', '275', '279', '280', '282', '32', '33', '46']
Overlap Ratio : 0.00

[word <-> ngram]
Common Tokens (0)      : []
Only In word (7) : ['!', '.', 'Hello', 'Tokenization', 'fun', 'is', 'world']
Only In ngram (4) : ['Hello world!', 'Tokenization is', 'is fun.', 'world! Tokenization']
Overlap Ratio : 0.00

[word <-> wordpiece]
Common Tokens (1)      : ['.']
Only In word (6) : ['!', 'Hello', 'Tokenization', 'fun', 'is', 'world']
Only In wordpiece (8) : ['##l', '##n', '##o', '##u', '[UNK]', 'f', 'hel', 'tokenization']
Overlap Ratio : 0.07

[word <-> unigram]
Common Tokens (1)      : ['.']
Only In word (6) : ['!', 'Hello', 'Tokenization', 'fun', 'is', 'world']
Only In unigram (10) : ['[UNK]', 'f', 'hel', 'i', 'l', 'n', 'o', 's', 'tokenization', 'u']
Overlap Ratio : 0.06

[word <-> sentencepiece]
Common Tokens (2)      : ['!', '.']
Only In word (5) : ['Hello', 'Tokenization', 'fun', 'is', 'world']
Only In sentencepiece (18) : ['H', 'T', 'at', 'd', 'el', 'en', 'i', 'k', 'l', 'n', 'o', 'r', 's', 'u', 'w', 'z', '▁', '▁f']
Overlap Ratio : 0.08

[char <-> byte]
Common Tokens (0)      : []
Only In char (20) : ['1', '10', '11', '12', '15', '16', '17', '19', '2', '20', '23', '24', '25', '26', '27', '3', '30', '5', '6', '7']
Only In byte (20) : ['100', '101', '102', '105', '107', '108', '110', '111', '114', '115', '116', '117', '119', '122', '32', '33', '46', '72', '84', '97']
Overlap Ratio : 0.00

[char <-> byte_bpe]
Common Tokens (0)      : []
Only In char (20) : ['1', '10', '11', '12', '15', '16', '17', '19', '2', '20', '23', '24', '25', '26', '27', '3', '30', '5', '6', '7']
Only In byte_bpe (19) : [' ', '!', '.', 'H', 'Token', 'at', 'd', 'el', 'f', 'i', 'l', 'n', 'n ', 'o', 'r', 's ', 'u', 'w', 'z']
Overlap Ratio : 0.00

[char <-> simple_bpe]
Common Tokens (14)      : ['1', '10', '12', '15', '17', '19', '2', '20', '23', '26', '27', '3', '30', '5']
Only In char (6) : ['11', '16', '24', '25', '6', '7']
Only In simple_bpe (5) : ['33', '35', '36', '37', '40']
Overlap Ratio : 0.56

[char <-> regex]
Common Tokens (0)      : []
Only In char (20) : ['1', '10', '11', '12', '15', '16', '17', '19', '2', '20', '23', '24', '25', '26', '27', '3', '30', '5', '6', '7']
Only In regex (7) : ['!', '.', 'Hello', 'Tokenization', 'fun', 'is', 'world']
Overlap Ratio : 0.00

[char <-> regex_bpe]
Common Tokens (0)      : []
Only In char (20) : ['1', '10', '11', '12', '15', '16', '17', '19', '2', '20', '23', '24', '25', '26', '27', '3', '30', '5', '6', '7']
Only In regex_bpe (8) : ['268', '275', '279', '280', '282', '32', '33', '46']
Overlap Ratio : 0.00

[char <-> ngram]
Common Tokens (0)      : []
Only In char (20) : ['1', '10', '11', '12', '15', '16', '17', '19', '2', '20', '23', '24', '25', '26', '27', '3', '30', '5', '6', '7']
Only In ngram (4) : ['Hello world!', 'Tokenization is', 'is fun.', 'world! Tokenization']
Overlap Ratio : 0.00

[char <-> wordpiece]
Common Tokens (0)      : []
Only In char (20) : ['1', '10', '11', '12', '15', '16', '17', '19', '2', '20', '23', '24', '25', '26', '27', '3', '30', '5', '6', '7']
Only In wordpiece (9) : ['##l', '##n', '##o', '##u', '.', '[UNK]', 'f', 'hel', 'tokenization']
Overlap Ratio : 0.00

[char <-> unigram]
Common Tokens (0)      : []
Only In char (20) : ['1', '10', '11', '12', '15', '16', '17', '19', '2', '20', '23', '24', '25', '26', '27', '3', '30', '5', '6', '7']
Only In unigram (11) : ['.', '[UNK]', 'f', 'hel', 'i', 'l', 'n', 'o', 's', 'tokenization', 'u']
Overlap Ratio : 0.00

[char <-> sentencepiece]
Common Tokens (0)      : []
Only In char (20) : ['1', '10', '11', '12', '15', '16', '17', '19', '2', '20', '23', '24', '25', '26', '27', '3', '30', '5', '6', '7']
Only In sentencepiece (20) : ['!', '.', 'H', 'T', 'at', 'd', 'el', 'en', 'i', 'k', 'l', 'n', 'o', 'r', 's', 'u', 'w', 'z', '▁', '▁f']
Overlap Ratio : 0.00

[byte <-> byte_bpe]
Common Tokens (0)      : []
Only In byte (20) : ['100', '101', '102', '105', '107', '108', '110', '111', '114', '115', '116', '117', '119', '122', '32', '33', '46', '72', '84', '97']
Only In byte_bpe (19) : [' ', '!', '.', 'H', 'Token', 'at', 'd', 'el', 'f', 'i', 'l', 'n', 'n ', 'o', 'r', 's ', 'u', 'w', 'z']
Overlap Ratio : 0.00

[byte <-> simple_bpe]
Common Tokens (1)      : ['33']
Only In byte (19) : ['100', '101', '102', '105', '107', '108', '110', '111', '114', '115', '116', '117', '119', '122', '32', '46', '72', '84', '97']
Only In simple_bpe (18) : ['1', '10', '12', '15', '17', '19', '2', '20', '23', '26', '27', '3', '30', '35', '36', '37', '40', '5']
Overlap Ratio : 0.03

[byte <-> regex]
Common Tokens (0)      : []
Only In byte (20) : ['100', '101', '102', '105', '107', '108', '110', '111', '114', '115', '116', '117', '119', '122', '32', '33', '46', '72', '84', '97']
Only In regex (7) : ['!', '.', 'Hello', 'Tokenization', 'fun', 'is', 'world']
Overlap Ratio : 0.00

[byte <-> regex_bpe]
Common Tokens (3)      : ['32', '33', '46']
Only In byte (17) : ['100', '101', '102', '105', '107', '108', '110', '111', '114', '115', '116', '117', '119', '122', '72', '84', '97']
Only In regex_bpe (5) : ['268', '275', '279', '280', '282']
Overlap Ratio : 0.12

[byte <-> ngram]
Common Tokens (0)      : []
Only In byte (20) : ['100', '101', '102', '105', '107', '108', '110', '111', '114', '115', '116', '117', '119', '122', '32', '33', '46', '72', '84', '97']
Only In ngram (4) : ['Hello world!', 'Tokenization is', 'is fun.', 'world! Tokenization']
Overlap Ratio : 0.00

[byte <-> wordpiece]
Common Tokens (0)      : []
Only In byte (20) : ['100', '101', '102', '105', '107', '108', '110', '111', '114', '115', '116', '117', '119', '122', '32', '33', '46', '72', '84', '97']
Only In wordpiece (9) : ['##l', '##n', '##o', '##u', '.', '[UNK]', 'f', 'hel', 'tokenization']
Overlap Ratio : 0.00

[byte <-> unigram]
Common Tokens (0)      : []
Only In byte (20) : ['100', '101', '102', '105', '107', '108', '110', '111', '114', '115', '116', '117', '119', '122', '32', '33', '46', '72', '84', '97']
Only In unigram (11) : ['.', '[UNK]', 'f', 'hel', 'i', 'l', 'n', 'o', 's', 'tokenization', 'u']
Overlap Ratio : 0.00

[byte <-> sentencepiece]
Common Tokens (0)      : []
Only In byte (20) : ['100', '101', '102', '105', '107', '108', '110', '111', '114', '115', '116', '117', '119', '122', '32', '33', '46', '72', '84', '97']
Only In sentencepiece (20) : ['!', '.', 'H', 'T', 'at', 'd', 'el', 'en', 'i', 'k', 'l', 'n', 'o', 'r', 's', 'u', 'w', 'z', '▁', '▁f']
Overlap Ratio : 0.00

[byte_bpe <-> simple_bpe]
Common Tokens (0)      : []
Only In byte_bpe (19) : [' ', '!', '.', 'H', 'Token', 'at', 'd', 'el', 'f', 'i', 'l', 'n', 'n ', 'o', 'r', 's ', 'u', 'w', 'z']
Only In simple_bpe (19) : ['1', '10', '12', '15', '17', '19', '2', '20', '23', '26', '27', '3', '30', '33', '35', '36', '37', '40', '5']
Overlap Ratio : 0.00

[byte_bpe <-> regex]
Common Tokens (2)      : ['!', '.']
Only In byte_bpe (17) : [' ', 'H', 'Token', 'at', 'd', 'el', 'f', 'i', 'l', 'n', 'n ', 'o', 'r', 's ', 'u', 'w', 'z']
Only In regex (5) : ['Hello', 'Tokenization', 'fun', 'is', 'world']
Overlap Ratio : 0.08

[byte_bpe <-> regex_bpe]
Common Tokens (0)      : []
Only In byte_bpe (19) : [' ', '!', '.', 'H', 'Token', 'at', 'd', 'el', 'f', 'i', 'l', 'n', 'n ', 'o', 'r', 's ', 'u', 'w', 'z']
Only In regex_bpe (8) : ['268', '275', '279', '280', '282', '32', '33', '46']
Overlap Ratio : 0.00

[byte_bpe <-> ngram]
Common Tokens (0)      : []
Only In byte_bpe (19) : [' ', '!', '.', 'H', 'Token', 'at', 'd', 'el', 'f', 'i', 'l', 'n', 'n ', 'o', 'r', 's ', 'u', 'w', 'z']
Only In ngram (4) : ['Hello world!', 'Tokenization is', 'is fun.', 'world! Tokenization']
Overlap Ratio : 0.00

[byte_bpe <-> wordpiece]
Common Tokens (2)      : ['.', 'f']
Only In byte_bpe (17) : [' ', '!', 'H', 'Token', 'at', 'd', 'el', 'i', 'l', 'n', 'n ', 'o', 'r', 's ', 'u', 'w', 'z']
Only In wordpiece (7) : ['##l', '##n', '##o', '##u', '[UNK]', 'hel', 'tokenization']
Overlap Ratio : 0.08

[byte_bpe <-> unigram]
Common Tokens (7)      : ['.', 'f', 'i', 'l', 'n', 'o', 'u']
Only In byte_bpe (12) : [' ', '!', 'H', 'Token', 'at', 'd', 'el', 'n ', 'r', 's ', 'w', 'z']
Only In unigram (4) : ['[UNK]', 'hel', 's', 'tokenization']
Overlap Ratio : 0.30

[byte_bpe <-> sentencepiece]
Common Tokens (14)      : ['!', '.', 'H', 'at', 'd', 'el', 'i', 'l', 'n', 'o', 'r', 'u', 'w', 'z']
Only In byte_bpe (5) : [' ', 'Token', 'f', 'n ', 's ']
Only In sentencepiece (6) : ['T', 'en', 'k', 's', '▁', '▁f']
Overlap Ratio : 0.56

[simple_bpe <-> regex]
Common Tokens (0)      : []
Only In simple_bpe (19) : ['1', '10', '12', '15', '17', '19', '2', '20', '23', '26', '27', '3', '30', '33', '35', '36', '37', '40', '5']
Only In regex (7) : ['!', '.', 'Hello', 'Tokenization', 'fun', 'is', 'world']
Overlap Ratio : 0.00

[simple_bpe <-> regex_bpe]
Common Tokens (1)      : ['33']
Only In simple_bpe (18) : ['1', '10', '12', '15', '17', '19', '2', '20', '23', '26', '27', '3', '30', '35', '36', '37', '40', '5']
Only In regex_bpe (7) : ['268', '275', '279', '280', '282', '32', '46']
Overlap Ratio : 0.04

[simple_bpe <-> ngram]
Common Tokens (0)      : []
Only In simple_bpe (19) : ['1', '10', '12', '15', '17', '19', '2', '20', '23', '26', '27', '3', '30', '33', '35', '36', '37', '40', '5']
Only In ngram (4) : ['Hello world!', 'Tokenization is', 'is fun.', 'world! Tokenization']
Overlap Ratio : 0.00

[simple_bpe <-> wordpiece]
Common Tokens (0)      : []
Only In simple_bpe (19) : ['1', '10', '12', '15', '17', '19', '2', '20', '23', '26', '27', '3', '30', '33', '35', '36', '37', '40', '5']
Only In wordpiece (9) : ['##l', '##n', '##o', '##u', '.', '[UNK]', 'f', 'hel', 'tokenization']
Overlap Ratio : 0.00

[simple_bpe <-> unigram]
Common Tokens (0)      : []
Only In simple_bpe (19) : ['1', '10', '12', '15', '17', '19', '2', '20', '23', '26', '27', '3', '30', '33', '35', '36', '37', '40', '5']
Only In unigram (11) : ['.', '[UNK]', 'f', 'hel', 'i', 'l', 'n', 'o', 's', 'tokenization', 'u']
Overlap Ratio : 0.00

[simple_bpe <-> sentencepiece]
Common Tokens (0)      : []
Only In simple_bpe (19) : ['1', '10', '12', '15', '17', '19', '2', '20', '23', '26', '27', '3', '30', '33', '35', '36', '37', '40', '5']
Only In sentencepiece (20) : ['!', '.', 'H', 'T', 'at', 'd', 'el', 'en', 'i', 'k', 'l', 'n', 'o', 'r', 's', 'u', 'w', 'z', '▁', '▁f']
Overlap Ratio : 0.00

[regex <-> regex_bpe]
Common Tokens (0)      : []
Only In regex (7) : ['!', '.', 'Hello', 'Tokenization', 'fun', 'is', 'world']
Only In regex_bpe (8) : ['268', '275', '279', '280', '282', '32', '33', '46']
Overlap Ratio : 0.00

[regex <-> ngram]
Common Tokens (0)      : []
Only In regex (7) : ['!', '.', 'Hello', 'Tokenization', 'fun', 'is', 'world']
Only In ngram (4) : ['Hello world!', 'Tokenization is', 'is fun.', 'world! Tokenization']
Overlap Ratio : 0.00

[regex <-> wordpiece]
Common Tokens (1)      : ['.']
Only In regex (6) : ['!', 'Hello', 'Tokenization', 'fun', 'is', 'world']
Only In wordpiece (8) : ['##l', '##n', '##o', '##u', '[UNK]', 'f', 'hel', 'tokenization']
Overlap Ratio : 0.07

[regex <-> unigram]
Common Tokens (1)      : ['.']
Only In regex (6) : ['!', 'Hello', 'Tokenization', 'fun', 'is', 'world']
Only In unigram (10) : ['[UNK]', 'f', 'hel', 'i', 'l', 'n', 'o', 's', 'tokenization', 'u']
Overlap Ratio : 0.06

[regex <-> sentencepiece]
Common Tokens (2)      : ['!', '.']
Only In regex (5) : ['Hello', 'Tokenization', 'fun', 'is', 'world']
Only In sentencepiece (18) : ['H', 'T', 'at', 'd', 'el', 'en', 'i', 'k', 'l', 'n', 'o', 'r', 's', 'u', 'w', 'z', '▁', '▁f']
Overlap Ratio : 0.08

[regex_bpe <-> ngram]
Common Tokens (0)      : []
Only In regex_bpe (8) : ['268', '275', '279', '280', '282', '32', '33', '46']
Only In ngram (4) : ['Hello world!', 'Tokenization is', 'is fun.', 'world! Tokenization']
Overlap Ratio : 0.00

[regex_bpe <-> wordpiece]
Common Tokens (0)      : []
Only In regex_bpe (8) : ['268', '275', '279', '280', '282', '32', '33', '46']
Only In wordpiece (9) : ['##l', '##n', '##o', '##u', '.', '[UNK]', 'f', 'hel', 'tokenization']
Overlap Ratio : 0.00

[regex_bpe <-> unigram]
Common Tokens (0)      : []
Only In regex_bpe (8) : ['268', '275', '279', '280', '282', '32', '33', '46']
Only In unigram (11) : ['.', '[UNK]', 'f', 'hel', 'i', 'l', 'n', 'o', 's', 'tokenization', 'u']
Overlap Ratio : 0.00

[regex_bpe <-> sentencepiece]
Common Tokens (0)      : []
Only In regex_bpe (8) : ['268', '275', '279', '280', '282', '32', '33', '46']
Only In sentencepiece (20) : ['!', '.', 'H', 'T', 'at', 'd', 'el', 'en', 'i', 'k', 'l', 'n', 'o', 'r', 's', 'u', 'w', 'z', '▁', '▁f']
Overlap Ratio : 0.00

[ngram <-> wordpiece]
Common Tokens (0)      : []
Only In ngram (4) : ['Hello world!', 'Tokenization is', 'is fun.', 'world! Tokenization']
Only In wordpiece (9) : ['##l', '##n', '##o', '##u', '.', '[UNK]', 'f', 'hel', 'tokenization']
Overlap Ratio : 0.00

[ngram <-> unigram]
Common Tokens (0)      : []
Only In ngram (4) : ['Hello world!', 'Tokenization is', 'is fun.', 'world! Tokenization']
Only In unigram (11) : ['.', '[UNK]', 'f', 'hel', 'i', 'l', 'n', 'o', 's', 'tokenization', 'u']
Overlap Ratio : 0.00

[ngram <-> sentencepiece]
Common Tokens (0)      : []
Only In ngram (4) : ['Hello world!', 'Tokenization is', 'is fun.', 'world! Tokenization']
Only In sentencepiece (20) : ['!', '.', 'H', 'T', 'at', 'd', 'el', 'en', 'i', 'k', 'l', 'n', 'o', 'r', 's', 'u', 'w', 'z', '▁', '▁f']
Overlap Ratio : 0.00

[wordpiece <-> unigram]
Common Tokens (5)      : ['.', '[UNK]', 'f', 'hel', 'tokenization']
Only In wordpiece (4) : ['##l', '##n', '##o', '##u']
Only In unigram (6) : ['i', 'l', 'n', 'o', 's', 'u']
Overlap Ratio : 0.33

[wordpiece <-> sentencepiece]
Common Tokens (1)      : ['.']
Only In wordpiece (8) : ['##l', '##n', '##o', '##u', '[UNK]', 'f', 'hel', 'tokenization']
Only In sentencepiece (19) : ['!', 'H', 'T', 'at', 'd', 'el', 'en', 'i', 'k', 'l', 'n', 'o', 'r', 's', 'u', 'w', 'z', '▁', '▁f']
Overlap Ratio : 0.04

[unigram <-> sentencepiece]
Common Tokens (7)      : ['.', 'i', 'l', 'n', 'o', 's', 'u']
Only In unigram (4) : ['[UNK]', 'f', 'hel', 'tokenization']
Only In sentencepiece (13) : ['!', 'H', 'T', 'at', 'd', 'el', 'en', 'k', 'r', 'w', 'z', '▁', '▁f']
Overlap Ratio : 0.29

========================================================================================================================
                                                     END OF REPORT                                                      
========================================================================================================================
```