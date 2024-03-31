# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 12:31:55 2018

@author: mona
"""

import os
import nltk
import re
from bs4 import BeautifulSoup

def tokenizer(text):
    text = re.sub("[^a-zA-Z]+", " ", text)
    token = nltk.tokenize.word_tokenize(text)
    return token

def pre_processing():
    
    list_doc = os.listdir("Topic1")
    
    for doc in list_doc:

        doc_loc = "Topic1/" + str(doc)
        file_doc = open(doc_loc, "r",encoding="utf8")
       # token = tokenizer(file_doc)
      #  file_doc = preprocessing_txt(file_doc.read())
      
        token = ""
        soup = BeautifulSoup(file_doc.read(),"lxml")
        
        
#        if soup is None:
#            return(None)
            
            
        text = ""
        if soup.find_all(token) is not None:
            text = ''.join(map(lambda p: p.text,soup.find_all(token)))
            
            soup2 = BeautifulSoup(text)
            if soup2.find_all('p') is not None:
                text = ''.join(map(lambda p: p.text,soup.find_all('p')))
                
            
             

if __name__ == "__main__":
        
         pre_processing()
            
#########################################################
import re
        
def sentences(text):
    '''break text block in to sentences'''
    ends = re.compile('[.?!]')
    
    return ends.split(text) 

if __name__ == "__main__":
    
    file = open('example.txt', 'r') 
   # print (file.read())
    
    senc = sentences(file.read())
   # print(sen)
###############################################################    

from nltk.tokenize import sent_tokenize
        
def sentences(text):
    '''break text block in to sentences'''
    
    
    return sent_tokenize(text)

if __name__ == "__main__":
    
    file = open('example.txt', 'r') 
   # print (file.read())
    
    senc = sentences(file.read())
       
    
################################################################ 
    
    
    for a in range(0,len(words)):
        
        for b in range(0,len(words[a])):
            
            tf.append( float(termFrequency(words[a][b], words[a]))) 
            
            idf.append(float(inverseDocumentFrequency(word[a][b], words)))
           
 ##########################################################
idf=[]
import math as m
def inverseDocumentFrequency(term, allDocuments):
    numDocumentsWithThisTerm = 0
    
    for a in range(0,len(allDocuments)):
            
        if allDocuments[a].__contains__(term) :
        
            numDocumentsWithThisTerm = numDocumentsWithThisTerm + 1
        
    print( numDocumentsWithThisTerm)
    if numDocumentsWithThisTerm > 0:
        return 1.0 + m.log(float(len(allDocuments)) / numDocumentsWithThisTerm)
    else:
        return 1.0
    
idf.append(float(inverseDocumentFrequency(words[0][0], words)))    

1.0 + m.log(float(len(words)) / 4512)
numDocumentsWithThisTerm=0
if words.__contains__("pine") :
        
            numDocumentsWithThisTerm = numDocumentsWithThisTerm + 1
else:
     print("wrong")            
print(numDocumentsWithThisTerm)   