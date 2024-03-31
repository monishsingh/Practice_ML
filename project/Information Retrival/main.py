# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 12:34:36 2018

@author: mona
"""

import os
import re
import nltk
import pickle
import math as m 
import pandas as pd


def computeIDF(dictionary):
    idfDict = {}
    #import math
    for word in dictionary:
        out = len(dictionary[word])
        idfDict[word] = m.log(6377/float(out))
        
    return idfDict


def computeTF(wordDict,bow,a):
    tfDict = {}
    bowCount = len(bow)
#    for word,count in wordDict.items():
#        tfDict[word] =  count / float(bowCount)
#        
    for j in range(0,len(bow)):
            
            term = bow[j]
            count= dictionary[term][list_doc[a]]
            tfDict[term] =  count / float(bowCount)
            
        
    return tfDict 

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

def inverted_index():
    """
    Creates a dictionary of words as key and name of the documents as items
    """
    #inverted = {}
    docs_indexed = 0
    list_doc = os.listdir("alldocs")
    
    
    for doc in list_doc:
#         sys.stdout.write('\r')
        doc_loc = "alldocs/" + str(doc)
        print(doc_loc)
        file_doc = open(doc_loc, "r",encoding="utf8")
        file_doc = preprocessing_txt(file_doc.read())
        tokens = tokenizer(file_doc)
        for word in tokens:
            if not inverted.__contains__(word):
                count = 1
                doclist = {}
                doclist[doc] = 1
                inverted[word] = doclist
            else:
                if doc in inverted[word]:
                    doclist = inverted[word]
                    doclist[doc] += 1
                    inverted[word] = doclist
                else:
                    count = 1
                    doclist = inverted[word]
                    doclist[doc] = count
                    inverted[word] = doclist
                    
        docs_indexed += 1
           
    return inverted

def tf_index():
    """
    Creates a dictionary of words as key and name of the documents as items
    """
    docs_indexed = 0
    #doc_loc = "example.txt"
    #file_doc = open(doc_loc, "r")
    for a in range(0,len(wordss)):
        
        for b in range(0,len(wordss[a])):
    #file_doc = preprocessing_txt(file_doc.read())
    #tokens = tokenizer(file_doc)
    #for wordin tokens:
            if not inverted.__contains__(wordss[a][b]):
                count = 1
                doclist = 0
               # doclist[doc] = 1
                inverted[wordss[a][b]] = doclist
            else:
            
                count = count + 1
                doclist = inverted[wordss[a][b]]
                doclist = count
                inverted[wordss[a][b]] = doclist
                    
    docs_indexed += 1
   
           
    return inverted


if __name__ == "__main__":
    with open("indexed_docs.pkl","wb") as handle:
        
         inverted_index()
         pickle.dump(inverted, handle)
         
         
         
    with open("indexed_docs.pkl","rb") as dict_file:
         
         dictionary = pickle.load(dict_file)
         
         
    idf_table = computeIDF(dictionary)
    table4 = pd .DataFrame([idf_table])
    
    #############################################
    
    list_doc = os.listdir("alldocs")
    i = 0
    words = {}
    for doc in list_doc:
#         sys.stdout.write('\r')
        doc_loc = "alldocs/" + str(doc)
        print(doc_loc)
        
        file_doc = open(doc_loc, "r",encoding="utf8")
        file_doc2 = preprocessing_txt(file_doc.read())
        words[i] = tokenizer(file_doc2)
        i = i+1
#   
    #################################################
    with open("word_docs.pkl","wb") as handle:
        
         
         pickle.dump(words, handle)
         
    with open("word_docs.pkl","rb") as dict_file:
         
         wordss = pickle.load(dict_file)
         
    ######################################     
    

    tftable = {}    
    for i in range(0,len(wordss)):
        tftable[i]= computeTF(dictionary,wordss[i],i)
        
    print('done')
    
    import numpy as np
    table3 = pd .DataFrame([tftable])
    tf_table=np.transpose(table3)
    
    
    #######################################
    
    tf_idf=np.zeros(shape=(len(tf_table),len(dictionary)))
    b=0
    for a in range(0,len(tf_table)):
        
    
        for i in dictionary:
        
            term = tf_table[0][a][i] * float(table4[i])
            tf_idf[a][b]=term
            if b == len(inverted)-1:
                
                b=0
            else:
                b=b+1 
        
        print('done')

        
            
            
            
    
    
    
    
   
            
