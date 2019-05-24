#!/usr/bin/env python
# coding: utf-8

# ## Library

# In[104]:


import os
import gc
import json
import pickle
import datetime
import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import groupby
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import coo_matrix
import multiprocessing
from multiprocessing import Manager,Pool
import copy


# ## Load  dataframe
# load_processed_corpus_df = pd.read_pickle("./processed_corpus.pkl")
load_processed_corpus_df = pd.read_csv("./new_wiki.csv")


nltk.download('stopwords')

lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize(word):
    lemma = lemmatizer.lemmatize(word,'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word,'n')
    return lemma

def pre_process(comment) -> str:
    # lower cased
    comment = comment.lower()
    # tokenize
    words =  nltk.tokenize.word_tokenize(comment)
    # lemmatize 
    words = [lemmatize(w) for w in words]
    
    stop_words = nltk.corpus.stopwords.words('english')
    words = [w for w in words if not w in stop_words]
    processed_comment = " ".join(words)
    return processed_comment


# ## Build a dictionary to store index in dataframe ( only run once)
# ## Deal with LRB in dictionary. (only run once)

# In[180]:


with open('lrb_dictionary.json', 'r') as f4:  # load lrb dictioanry json
        lrb_dictionary = json.load(f4) 

#Delete keys in dictionary.

delete_words_list = ['The','Part', 'Most','Water', 'How', 'Love','Speech','American','President','German','Irish',
                     'Indian','Spanish','Japan','Califorina','Americans','Chinese','British','Monday','Tuesday',
                    'Wednesday','Thursday', 'Friday','Saturday', 'Sunday','January','February','March','April',
                     'May', 'June', 'July', 'August', 'September', 'October', 'November','December','Russian']

for word in delete_words_list:
    try:
        del lrb_dictionary[word]
    except KeyError:
        pass
    

# page keys are the page identifier, keys of the page_dictionary
page_keys = list(lrb_dictionary.keys())

def retrieve_sentenceText(claim_word,page_keys,page_dictionary,df):
    retrieved_sentence = []
    if claim_word in page_keys:
        retrieved_index = page_dictionary[claim_word]      # all indexes in the dataframe
        for index in retrieved_index:
             # retrieve all the raw doc txt
            retrieved_sentence.append(df.loc[index, 'text'])
    return retrieved_sentence


# ## Rule on Claim: find the Upper cased words.
# 
# Rule1: First Continuous Upper words.

# In[113]:


def find_upper_word(claim, page_keys):
    res_list = []
    res_index = []      
    words = claim.split()
    
    start = ""
    temp = ""
    i = 0
    while i < len(words):         
        # first step: find uppercase word in the claim
        if words[i][0].isupper():
            temp = words[i]        
            start = temp  # start(as a cache)                        
            for j in range(i,len(words)-1):
                temp = temp + '_' + words[j+1]
#                 print(temp)
                if temp in page_keys:
                    start = temp  # matchs the word as long as possible
                    i = j + 1
                
                if j - i > 2:
                    break
                
            res_list.append(start)  
        i += 1                    
    return res_list

# ## Method 6: Pure TF IDF 


def retrieval_evidence_func6(query,page_keys,page_dictionary,load_processed_corpus_df):
    res = []
    # Determine if the first word in query is "There" and "A"
    if query.split()[0] == "There":
        query = query.replace("There","there") 
        
    if query.split()[0] == "A":
        query = query.replace("A","a") 
    
    # remove all 's 
    for word in query.split():
        if (word[-2:len(word)]) == "'s":
            query = query.replace(word,word[:-2])
            
     # remove all '
    for word in query.split():
        if (word[-1:len(word)]) == "'":
            query = query.replace(word,word[:-1])

    query_corpus = []
    if len(find_upper_word(query,page_keys)) >= 1:
        for query_word in find_upper_word(query,page_keys):
            if query_word[-1:len(query_word)] == ".":
                query_word = query_word[:-1]
            if query_word[-1:len(query_word)] == ",":
                query_word = query_word[:-1]
            
            query_corpus.extend(retrieve_sentenceText(query_word, page_keys,page_dictionary,load_processed_corpus_df))
        
        if len(query_corpus) == 1:
            res.append([query_corpus[0].split()[0],int(query_corpus[0].split()[1])])
        else:
            # preprocess claim
            processed_query_claim = pre_process(query)
            # preprocess evidences
            processed_query_corpus = []
            for corpus in query_corpus:
                processed_query_corpus.append(pre_process(corpus))
            
            tfidf_vectorizer = TfidfVectorizer()
            processed_query_corpus.insert(0, processed_query_claim)

            tfidf = tfidf_vectorizer.fit_transform(processed_query_corpus)

            query_rep = tfidf[0].todense()
            docs_rep = tfidf[1:].todense()
                        # calculate cosine similarity between the query and retrieved sentences
            query_doc_cos_dist = []

            # cosine distance, and hence no need to revese argsort result
            for doc_rep in docs_rep:
                query_doc_cos_dist.append(cosine(query_rep, doc_rep))

            query_doc_sort_index = np.argsort(np.array(query_doc_cos_dist))

            count = 0
            for rank, sort_index in enumerate(query_doc_sort_index):
                res.append([query_corpus[sort_index].split()[0],int(query_corpus[sort_index].split()[1])])
                if count == 6:
                    break
                else:
                    count += 1   
        return res
        
    else:
        return []

with open('train2.json', 'r') as f9:  # load dev dataset
       train_data = json.load(f9)
print("Length of the train data is: " + str(len(train_data)))

with open('train_result.json', 'r') as f8:  # store result
       train_res_data = json.load(f8)
print("Length of the train result data is: " + str(len(train_res_data)))


global_test_data=Manager().dict(train_res_data)

i = multiprocessing.Value("i",0)
def deal_dataset(key):
   detail_dict = global_test_data[key]
   detail_dict["evidence"] = []
   query = detail_dict["claim"]

   detail_dict["evidence"] = retrieval_evidence_func6(query,page_keys,lrb_dictionary,load_processed_corpus_df)
   global_test_data[key] = detail_dict
   i.value += 1
   print(i.value)

pool=Pool(processes = 10)

for key in list(train_data):
   pool.apply_async(deal_dataset, args=(key,))

pool.close()
pool.join()

with open('result_train_bert1.json', 'w') as f:
   json.dump(dict(global_test_data), f, indent = 4)





