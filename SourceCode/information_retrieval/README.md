# AutomaticFactVerification
web search project

## Information Retrieval Part

### Document Retrieval

Used rule-based information extraction


### Sentence Retrieval

Basic method: Used TF-IDF, BOW, TF-IDF with threshold, TF-IDF with SVD to compute cosine

Advanced method: Used TF-IDF to generate training data for BERT, then use BERT fine tune to retrieve sentence


## Code and files:
Claim-TFIDF.ipynb : implementation of document retrieval and sentence retrieval
GlobalProcess.ipynb: preprocess the wiki corpus
Process_New_Wiki.ipynb: generates DataFrame for Claim-TFIDF.ipynb
Data_for_BERT.ipynb: generates training data for BERT

lrb_dictionary.json: dictionary that saved page identifier and sentence index, run Claim-TFIDF.ipynb lrb dicitonary function.
new_wiki.csv: run GlobalProcess.ipynb and can get new_wiki.csv

### To run the code:
Put the lrb_dicitionary.json and new_wiki.csv in the same directory of the Claim-TFIDF.ipynb.

Run Claim-TFIDF.ipynb
