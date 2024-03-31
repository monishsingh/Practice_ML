# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 12:22:28 2018

@author: mona
"""

#import csv
#import os
#import sys
import re
import nltk
#import operator
import pickle 
#import progressbar
#import time
#import numpy as np
#import pandas as pd


def low(z):
    return z.lower()

def make(x):
    s=set(x)
    return sorted([(i,x.count(i)) for i in s],key = first)


def first(z):
    return z[0]

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

inverted = {}
import pickle
def inverted_index():
    
    """
    Creates a dictionary of words as key and name of the documents as items
    """
    docs_indexed = 0
    #doc_loc = "example.txt"
    #file_doc = open(doc_loc, "r")
    for a in range(0,len(words)):
        
        for b in range(0,len(words[a])):
    #file_doc = preprocessing_txt(file_doc.read())
    #tokens = tokenizer(file_doc)
    #for wordin tokens:
            if not inverted.__contains__(words[a][b]):
                count = 1
                doclist = 0
               # doclist[doc] = 1
                inverted[words[a][b]] = doclist
            else:
            
                count = count + 1
                doclist = inverted[words[a][b]]
                doclist = count
                inverted[words[a][b]] = doclist
                    
    docs_indexed += 1
       # i = docs_indexed
       # if(i % (point) == 0):
           # sys.stdout.write("\r[" + "=" * (i / increment) + ">" +  " " * ((total - i)/ increment) + "]" +  str(100*i / float(len(list_doc))) + "%")
           # sys.stdout.flush()
           
    return inverted

if __name__ == "__main__":
    with open("indexed_docs.pkl","wb") as handle:
        
         inverted_index()
         pickle.dump(inverted, handle)
print ("done")


###########################################################
import math as m
dic_idf=[]
def inverseDocumentFrequency(term, allDocuments):
    numDocumentsWithThisTerm = 0
    
    for a in range(0,len(allDocuments)):
            
        if allDocuments[a].__contains__(term) :
        
            numDocumentsWithThisTerm = numDocumentsWithThisTerm + 1
        
    #print( numDocumentsWithThisTerm)
    if numDocumentsWithThisTerm > 0:
        return 1.0 + m.log(float(10) / numDocumentsWithThisTerm)
    else:
        return 1.0

dic_idf.append(float(inverseDocumentFrequency('pine', inverted)))

import numpy as np

C=[]

for word in inverted:
    print(word)
    C.append(word)
print(C)    
    
import pandas as pd
pd.DataFrame([C])

