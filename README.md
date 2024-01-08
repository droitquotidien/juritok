# juritok
Tokenisation des textes du JO et des textes consolidés

## Installation

Seen during the class.

## Use

Run `jtk` command after the installation of the package.

It is going to train two models: one with all the text data of JORF, and another with only text data from the law articles (between quotes in JORF).

Then, both models are tested. Here is an output:

```
==== TESTING MODEL WITH ALL JORF ====
Number of different words:  4491
Vocab size:  1000
['▁«', '▁Le', '▁présent', '▁décret', '▁entre', '▁en', '▁vigueur', '▁le', '▁1', 'er', '▁janvier', '▁2023', '.', '▁»']
[75, 177, 224, 184, 516, 106, 717, 73, 74, 12, 714, 246, 935, 172]
« Le présent décret entre en vigueur le 1er janvier 2023. »


==== TESTING MODEL WITH LAW ARTICLES ONLY ====
Number of different words:  1624
Vocab size:  1000
['▁«', '▁Le', '▁présent', '▁décret', '▁entre', '▁en', '▁vigueur', '▁le', '▁1', 'er', '▁j', 'an', 'v', 'ier', '▁20', '2', '3', '.', '▁»']
[27, 180, 208, 433, 591, 73, 687, 53, 51, 25, 233, 20, 949, 153, 416, 962, 954, 948, 149]
« Le présent décret entre en vigueur le 1er janvier 2023. »
```
