# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 04:12:24 2018

@author: mona
"""
import codeCS
import os
import numpy as np    
import pandas as pd

import pickle
import nltk
import re
import math as m
from bs4 import BeautifulSoup 

text={}
a=0   

soup={}

inverted = {}


def cosine(vec1,vec2):
    sum1=0.0 
    sum2=0.0
    sum3=0.0
    sum1 = np.sum(tf_idf[vec1] * tf_idf[vec2])
    sum2 = np.sum(tf_idf[vec1]**2)
    sum3 = np.sum(tf_idf[vec2]**2)
    deno=m.sqrt(sum2)*m.sqrt(sum3)
    
    if deno==0:
        return 0
    else:
        return float(sum1/deno)


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

def computeTF(wordDict,bow):
    tfDict = {}
    bowCount = len(bow)
    for word,count in wordDict.items():
        tfDict[word] =  count /(float(bowCount)+1)
        
    return tfDict 

def computeIDF(inverted):
    idfDict = {}
    #import math
    for word,count in inverted.items():
        idfDict[word] = m.log(25/float(count + 1))
        
    return idfDict



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
   # import re
   # text = re.split(r'\W+',str(text))
    text = re.sub("[^a-zA-Z]+"," ", text)
    tokens = nltk.tokenize.word_tokenize(str(text))
    return tokens        


        
if __name__ == "__main__":
    
    list_doc = os.listdir("Topic2")

    n = len(list_doc)
    
    for a in range(0,n):
        file_text=[]
        #doc_loc = "Topic1/" + str(doc)
        doc_loc = "Topic2/" + list_doc[a]
        
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
    #from nltk.tokenize import word_tokenize

    word_sent = [word_tokenize(s.lower())for s in senc]
    
    word = [preprocessing_txt(s)for s in word_sent]
    
    words = [tokenizer(s)for s in word]
    
    with open("indexed_docs.pkl","wb") as handle:
        
         inverted_index()
         pickle.dump(inverted, handle)
    print ("done")
    
 ################################################################   
    wordDict={}
    for i in range(0,len(words)):
        
        wordDict[i]=dict.fromkeys(inverted,0)
        
        
    for i in range(0,len(words)):
        
        for term in words[i]:
            
            print(term)
            wordDict[i][term]+=1
            
    
    table = pd .DataFrame([wordDict])

    table1=np.transpose(table)
    
    tfTable={}
    
    
    for a in range(0,len(words)):
        
        tfTable[a] = computeTF(wordDict[a], words[a])
        
    
    table3 = pd .DataFrame([tfTable])
    tf_table=np.transpose(table3)
    
    
    idfTable = computeIDF(inverted)
    table4 = pd .DataFrame([idfTable])
    
    tf_idf=np.zeros(shape=(len(tf_table),len(inverted)))
    b=0
    for a in range(0,len(tf_table)):
        
    
        for i in inverted:
        
            term = tf_table[0][a][i] * float(table4[i])
            tf_idf[a][b]=term
            if b == len(inverted)-1:
                
                b=0
            else:
                b=b+1 
        
        print('done')

    

    import networkx as nx
    g=nx.Graph()    
    
    Matrix2 = [[0 for x in range(len(word))] for y in range(len(word))]
    Matrix = [[0 for x in range(len(word))] for y in range(len(word))] 
    ct=0
    for i in range(len(word)):
        for j in range(len(word)):
            if i!=j:
                Matrix[i][j]=cosine(i,j)
                if Matrix[i][j]>0.3:
                    
                    g.add_edge(i,j,weight=Matrix[i][j])
                    Matrix2[i][j]=Matrix[i][j]
                    print(Matrix[i][j])
                
g.print_graph()




     
    
    
    
    
    


        
            

 

    
        