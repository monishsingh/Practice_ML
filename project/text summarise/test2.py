# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 20:29:40 2018

@author: mona
"""

import os
import nltk
import re
import numpy as np
import math as m

from bs4 import BeautifulSoup 

text={}
a=0   

soup={}


def add_sentences(doc_loc,a):
    
    file_doc = open(doc_loc, "r",encoding="utf8")
    
    soup[a] = BeautifulSoup(file_doc.read(),"lxml")
    if soup[a].find_all('p') is not None:
        
           text = ''.join(map(lambda p: p.text,soup[a].find_all('p')))
    
    return text

#def sentences(text):
#    '''break text block in to sentences'''
#    ends = re.compile('[.?!]')
    
#    return ends.split(text)

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

def sentences(text):
    '''break text block in to sentences'''
    
    
    return sent_tokenize(text)

def termFrequency(term, document):
    
    normalizeDocument = document
    
    return str(normalizeDocument.count(term) / float(len(normalizeDocument)))
 
#def inverseDocumentFrequency(term, allDocuments):
#    numDocumentsWithThisTerm = 0
#    for doc in allDocuments:
#            numDocumentsWithThisTerm = numDocumentsWithThisTerm + 1
#           # print(numDocumentsWithThisTerm)
#            
#    if numDocumentsWithThisTerm > 0:
#        return 1.0 + m.log(float(len(allDocuments)) / numDocumentsWithThisTerm)
#    else:
#        return 1.0
    
def inverseDocumentFrequency(term, allDocuments):
    numDocumentsWithThisTerm = 0
    
    for a in range(0,len(allDocuments)):
            
        if allDocuments[a].__contains__(term) :
        
            numDocumentsWithThisTerm = numDocumentsWithThisTerm + 1
        
    #print( numDocumentsWithThisTerm)
    if numDocumentsWithThisTerm > 0:
        return 1.0 + m.log(float(len(allDocuments)) / numDocumentsWithThisTerm)
    else:
        return 1.0


        

import nltk
def preprocessing_txt(tokens):
    
   
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

def tokenizer(text):
    text = re.sub("[^a-zA-Z]+", " ", text)
    tokens = nltk.tokenize.word_tokenize(text)
    return tokens        

        
if __name__ == "__main__":
    
    list_doc = os.listdir("Topic1")

    n = len(list_doc)
    
    for a in range(0,n):
        file_text=[]
        #doc_loc = "Topic1/" + str(doc)
        doc_loc = "Topic1/" + list_doc[a]
        
        sen=add_sentences(doc_loc,a)
        
        file_text.append(sen)

        print(file_text)
        
        b=file_text        
        with open('example.txt', 'a') as myfile:
    
   #  myfile.write(' '.join(file_text).encode('utf8'))
                    
             myfile.write(' '.join(file_text))
             
             myfile.close()
    ###########################################################         
             
    file = open('example.txt', 'r') 
   # print (file.read())
    
    senc = sentences(file.read())
   # print(sen)
    from nltk.tokenize import word_tokenize

    word_sent = [word_tokenize(s.lower())for s in senc]
    
    word = [preprocessing_txt(s)for s in word_sent]
    
    words = [tokenizer(s)for s in word]
    
    tf=[]
    idf=[]
    for a in range(0,len(words)):
        
        for b in range(0,len(words[a])):
            
            tf.append( float(termFrequency(words[a][b], words[a]))) 
            
            idf.append(float(inverseDocumentFrequency(words[a][b], words)))
        
    tf_idf=[]        
    for a in range(0,len(tf)):
        
        tf_idf.append(tf[a] * idf[a])
        
        
        
       
            