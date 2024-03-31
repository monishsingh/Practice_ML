# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 21:32:28 2018

@author: mona
"""

import os
import re
import nltk
import pickle
import math as m 
import pandas as pd


def cosine(vec1,vec2):
    sum1=0.0 
    sum2=0.0
    sum3=0.0
    sum1 = np.sum(tf_idf_query[vec1] * tf_idf_glove[vec2])
    sum2 = np.sum(tf_idf_query[vec1]**2)
    sum3 = np.sum(tf_idf_glove[vec2]**2)
    deno=m.sqrt(sum2)*m.sqrt(sum3)
#    print(sum1,end=" ")
#    print(sum2,end=" ")
#    print(sum3,end=" ")
#    print(deno)
#    
    if deno==0:
        return 0
    else:
        return float(sum1/deno)


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
        
    ####################################### 
    glove_dic = []
    glove_file = 'glove.840B.300d.txt'
    file = open(glove_file,"r",encoding="utf8")
    
    a = 0
    for i in file:
        
        if a < 2000:
            glove_dic.append(i)
            a = a+1
        else:
            break
            
        
    print('done') 
    
    with open("glove_docs.pkl","wb") as handle:
        
         #inverted_index()
         pickle.dump(glove_dic, handle)
         
    with open("glove_docs.pkl","rb") as dict_file:
         
         glove_dictionary = pickle.load(dict_file)
    
    
    ################################
    glove_dictionary={}
    for i in range(0,len(glove_dic)):
        glove_dictionary[i]=glove_dic[i].split()
         
    ##############################
    tf_idf_glove=np.zeros(shape=(len(glove_dic),len(query_dictionary)))    
    
    for i in range(0,len(glove_dic)):
        for j in range(1,300):
            
            tf_idf_glove[i][j-1] = glove_dictionary[i][j]
            
    ################################
    Matrix_cosine = [[0 for x in range(len(glove_dic))] for y in range(len(query_word))]         
    
    for i in range(len(query_word)):
        for j in range(len(glove_dic)):
    
             Matrix_cosine[i][j]=cosine(i,j)
             print(Matrix_cosine[i][j])
             
    file = open("tf_idf_expansion.txt",'w')
    for i in range(0,len(Matrix_cosine)):
            file.write("%s\n" % str(Matrix_cosine[i]))
           # file.write(str(Matrix_cosine[i]))
        
         

   # Matrix = Matrix_cosine
   
    Matrix = [[0 for x in range(len(glove_dic))] for y in range(len(query_word))]         
    
    for i in range(len(query_word)):
        for j in range(len(glove_dic)):
    
             Matrix[i][j]=cosine(i,j)
             print(Matrix[i][j])

    
    ###################################
    pick = [[0 for x in range(5)] for y in range(len(Matrix))]
    for i in range(0,len(Matrix)):
        
        for j in range(0,5):
            value = max(Matrix[i])
            for k in range(0,len(Matrix[i])):
                
                if value == Matrix[i][k]:
                    line_glove = k
                    pick[i][j] = line_glove
                    Matrix[i][k]=0
                
     ######################################
    add_word = [[0 for x in range(5)] for y in range(len(Matrix))] 
    for i in range(0,len(pick)):
        for j in range(0,5):
            term = glove_dictionary[pick[i][j]][0]
            
            add_word[i][j]=term
    #########################################
    query_expand ={}
    query_file = 'query.txt'
    file = open(query_file,"r")
    i = 0
    for line in file:
        #print(line)
        query_expand[i]=line + str(add_word[0])
        i = i+1
    ###########################################
    
    f2=open('query_expansion.txt','w')
    for i in range(0,len(query_expand)):
        
        f2.write("%s\n" %query_expand[i])
        


    
        
        
            
            
        
               
    
    
        
        
    