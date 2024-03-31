# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 03:43:59 2018

@author: mona
"""

#import os
import re
import nltk
#import pickle
import math as m 
import pandas as pd

def rocchio(q_prev,sum_relevant):
    q_updated = 0
    q_updated = q_prev + 0.65 *(1/10)*sum_relevant
    
    return q_updated
    
def computeTF(wordDict,bow):
    tfDict = {}
    bowCount = len(bow)
    for word,count in wordDict.items():
       tfDict[word] =  count / float(bowCount)
       
    return tfDict 


def computeIDF(dictionary):
    idfDict = {}
    #import math
    for word in dictionary:
        out = dictionary[word]
        idfDict[word] = m.log(82/float(out))
        
    return idfDict


def tokenizer(text):
    text = re.sub("[^a-zA-Z]+", " ", text)
    tokens = nltk.tokenize.word_tokenize(text)
    return tokens        


def preprocessing_txt(text):
    
    tokens = tokenizer(text)
    stemmer = nltk.stem.porter.PorterStemmer()
    stopwords = nltk.corpus.stopwords.words('english')
    new_text = ""
    for token in tokens:
        token = token.lower()
        if token not in stopwords:
#             print token
            new_text += stemmer.stem(token)
            new_text += " "
        
    return new_text

query_dictionary={}
def query_index():
    """
    Creates a dictionary of words as key and name of the documents as items
    """
    docs_indexed = 0
    #doc_loc = "example.txt"
    #file_doc = open(doc_loc, "r")
    for a in range(0,len(query_word)):
        
        for b in range(0,len(query_word[a])):
    #file_doc = preprocessing_txt(file_doc.read())
    #tokens = tokenizer(file_doc)
    #for wordin tokens:
            if not query_dictionary.__contains__(query_word[a][b]):
                count = 1
                doclist = 1
               # doclist[doc] = 1
                query_dictionary[query_word[a][b]] = doclist
            else:
            
                count = count + 1
                doclist = query_dictionary[query_word[a][b]]
                doclist = count
                query_dictionary[query_word[a][b]] = doclist
                    
    docs_indexed += 1
   
           
    return query_dictionary

if __name__ == "__main__":
    
    query_word ={}
    query_file = 'query.txt'
    file = open(query_file,"r")
    i = 0
    for line in file:
        file_doc = preprocessing_txt(line)
        query_word[i] = tokenizer(file_doc)
        i = i+1
        
    query_index() 
    #############################
    
    idf_query = computeIDF(query_dictionary)
    table_idf = pd .DataFrame([idf_query])
    
    ###############################
    
    tftable_query = {}    
    for i in range(0,len(query_word)):
        tftable_query[i]= computeTF(query_dictionary,query_word[i])
        
    print('done')
    
    import numpy as np
    table_tf = pd .DataFrame([tftable_query])
    tf_table_query=np.transpose(table_tf)
    
    #######################################
    
    tf_idf_query=np.zeros(shape=(len(tf_table_query),len(query_dictionary)))
    b=0
    for a in range(0,len(tf_table_query)):
        
    
        for i in query_dictionary:
        
            term = tf_table_query[0][a][i] * float(table_idf[i])
            tf_idf_query[a][b]=term
            if b == len(query_dictionary)-1:
                
                b=0
            else:
                b=b+1 
        
        print('done')
        
    ##################Rocchio's algorithm ###################
    tf_idf_query_updated=np.zeros(shape=(len(tf_table_query),len(query_dictionary)))
    
    for i in range(0,len(query_word)):
        for j in range(0,len(query_dictionary)):
             s = np.sum(tf_idf_query[i])
             p = tf_idf_query[i][j]
             
             tf_idf_query_updated[i][j] = rocchio(p,s)
        
        
        

    
