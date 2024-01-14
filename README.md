# Juritok
Tokenisation des textes du JO et des textes consolidés

## First step : combining all JO from 2018 to 2022 (keeping 2023 to test the tokenization).
Opening csv files using the separator '|' and keeping only the last line, where the text we are interested in are contained.  
When trying to read it, it appeared that there were words with upper and lower cases that might be considered as different words when they are not. The choice to only have lower cases was made.  
Also, some identification lines looking like "fr/..../date" were splitting the tewt, when they do not bring any information to the text. They were removed.

## Second step : Training the model.
Some sentences were too long to be considered, the maximum length was increased to try to get all words.
Increasing the vocab size also helped, even if after some point it does not improve the coverage anymore (99.9551% with 1000 and 10000)  
A compromise was found with 1000.  
*(Using all files did not enable the model to properly charge, the dataset was reduced to the period 2020-2022)*

## Third step : Trying the model with a different JO (2023)
Taking the first 10 sentences, checking input and output :

The text document entitled "Tokenization_test.txt" is divided in 3 parts:
- Original sentences from the JO.
- Tokens obtained when encoding with sentence piece.
- Sentences built up after decoding the model.  

The sentences are the same, even though the syntax differ (no \n after decoding).  
The tokens are not always split perfectly (mai and rie for mairie for instance) but it seems satisfying enough for our purpose.