```

========================================================================================================================
                                              TOKENIZER EVALUATION REPORT                                               
========================================================================================================================

SOURCE TEXT
------------------------------------------------------------------------------------------------------------------------
Hello world! Tokenization is fun.

SUMMARY TABLE
------------------------------------------------------------------------------------------------------------------------
Tokenizer | Token | Uniq | Uniq Ratio | Avg Len | Min Len | Max Len | Avg Chars/Token | Unknown | Latency  | Eff.
----------+-------+------+------------+---------+---------+---------+-----------------+---------+----------+-----
word      | 7     | 7    | 1.00       | 4.14    | 1       | 12      | 4.71            | 0.00    | 0.000016 | 4.71
char      | 33    | 20   | 0.61       | 1.73    | 1       | 2       | 1.00            | 0.00    | 0.000026 | 1.00
byte      | 33    | 20   | 0.61       | 2.73    | 2       | 3       | 1.00            | 0.00    | 0.000025 | 1.00
byte_bpe  | 25    | 19   | 0.76       | 1.32    | 1       | 5       | 1.32            | 0.00    | 0.000149 | 1.32
regex     | 7     | 7    | 1.00       | 4.14    | 1       | 12      | 4.71            | 0.00    | 0.000012 | 4.71
regex_bpe | 11    | 8    | 0.73       | 2.45    | 2       | 3       | 3.00            | 0.00    | 0.000421 | 3.00

HIGHLIGHTS
------------------------------------------------------------------------------------------------------------------------
Lowest Token Count   : word (7)
Highest Token Count  : char (33)
Best Efficiency      : word (4.71)
Highest Unique Count : char (20)
Fastest Tokenizer    : regex (12µs)

INTERPRETATION
------------------------------------------------------------------------------------------------------------------------
The 'word' tokenizer yielded the lowest token count, suggesting a coarse-grained segmentation strategy.
The 'char' tokenizer produced the highest number of tokens, indicating a fine-grained segmentation.
The 'word' tokenizer achieved the highest efficiency score, (4.71), balancing token count and unknown-token behavior.
The 'char' tokenizer generated the most unique tokens (20), suggesting higher diversity.
The 'regex' tokenizer completed tokenization in the shortest time (0.000012s), making it the fastest option for this sample.
The 'word' tokenizer produced the highest characters-per-token ratio (4.71), indicating larger token chunks.

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
Latency                : 0.000016s
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
Latency                : 0.000026s
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
Latency                : 0.000025s
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
Latency                : 0.000149s
Efficiency Score       : 1.32
Top-5 Tokens           : [('o', 3), ('i', 3), ('l', 2), (' ', 2), ('H', 1)]
Token Length Dist.     : {1: 20, 2: 4, 5: 1}

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
Latency                : 0.000012s
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
Latency                : 0.000421s
Efficiency Score       : 3.00
Top-5 Tokens           : [('32', 4), ('275', 1), ('279', 1), ('33', 1), ('268', 1)]
Token Length Dist.     : {2: 6, 3: 5}

OVERALL RANKING
------------------------------------------------------------------------------------------------------------------------
1. regex (efficiency=4.71, latency=0.000012s)
2. word (efficiency=4.71, latency=0.000016s)
3. regex_bpe (efficiency=3.00, latency=0.000421s)
4. byte_bpe (efficiency=1.32, latency=0.000149s)
5. byte (efficiency=1.00, latency=0.000025s)
6. char (efficiency=1.00, latency=0.000026s)

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

[byte <-> byte_bpe]
Common Tokens (0)      : []
Only In byte (20) : ['100', '101', '102', '105', '107', '108', '110', '111', '114', '115', '116', '117', '119', '122', '32', '33', '46', '72', '84', '97']
Only In byte_bpe (19) : [' ', '!', '.', 'H', 'Token', 'at', 'd', 'el', 'f', 'i', 'l', 'n', 'n ', 'o', 'r', 's ', 'u', 'w', 'z']
Overlap Ratio : 0.00

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

[regex <-> regex_bpe]
Common Tokens (0)      : []
Only In regex (7) : ['!', '.', 'Hello', 'Tokenization', 'fun', 'is', 'world']
Only In regex_bpe (8) : ['268', '275', '279', '280', '282', '32', '33', '46']
Overlap Ratio : 0.00

========================================================================================================================
                                                     END OF REPORT                                                      
========================================================================================================================
```