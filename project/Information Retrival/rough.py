# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 01:48:54 2018

@author: mona
"""
#import re
#import nltk
#
#def tokenizer(text):
#    text = re.sub("[^a-zA-Z]+", " ", text)
#    tokens = nltk.tokenize.word_tokenize(text)
#    return tokens
#
#text_word ={}
#query_file = 'elastic_before_update.txt'
#file = open(query_file,"r")
#
#j = 0
#for i in file:
#    text_word[j]=i
#    j = j+1
#        
#text_words = {}    
#for k in range(0,len(text_word)):
#    text_words[k] = text_word[k].split(' ')
#    
#text_wordss = {}
#for k in range(0,len(text_words)):
#    b = len(text_words[k])
#    text_wordss[k] = text_words[k][0] +" "+ text_words[k][b-1]
#    
#file = open("elastics_before_updates.txt",'w')
##f2=open('query_expansion.txt','w')
#for i in range(0,len(text_wordss)):
#       
#     file.write(text_wordss[i])
#            
###########################################################
import pandas as pd
 
output = open("output.txt",'r')
#query = open("query.txt","r")
est_output = open("output_after_expansion.txt",'r')

query = open("query.txt","r")
query_ID = []

for line in query:
     query_ID.append(line.split()[0])
     

query_ID     
     
e_out = pd.DataFrame(columns = ["q_ID","Doc"])
o_out = pd.DataFrame(columns=["q_ID","Doc"])
#e_out ={}
#o_out ={}

query = []
docs = []
k = 0
import numpy as np
#e_out=np.zeros(shape=(len(query),2))
for line in est_output:
       
    query.append(line.split())

for k in range(len(query)):
    
    e_out.loc[k]['q_ID']=query[k][0]
    e_out.loc[k]['Doc']=query[k][1]
    
querys=[]    
for line in output:
    querys.append(line.split())
    
#for i in range(7500):
#    o_out.loc[i]=i
for k in range(len(querys)):
    
    o_out.loc[k]['q_ID']=querys[k][0]
    o_out.loc[k]['Doc']=querys[k][1]
    
prerec = open("precision_after_expand.txt","a")
prerec.write("queryID" + "     " + "precision" + "     " + "recall" +"          " +"f_score"+"\n")
       
for q_ID in query_ID:
       # estimated = list(e_out[e_out['q_ID'] == q_ID]["Doc"])
        #estimated=list[set(e_out.q_ID.isin(q_ID))]
        estimated=e_out.q_ID.isin(list(q_ID))
#         print len(estimated)
        #true = list(o_out[o_out["q_ID"] == q_ID]["Doc"])
        true=o_out.q_ID.isin(list(q_ID))
#         print len(true)
        precision = len(list(set(estimated).intersection(set(true))))/float(len(estimated)+1)
        recall = len(list(set(estimated).intersection(set(true))))/float(len(true)+1)
        f_score = (2*precision * recall)/precision + recall
        prerec.write(str(q_ID) + " " + str(precision) + " " + str(recall) +"    "+str(f_score)+"   "+ "\n")

prerec.close()
output.close()
est_output.close()
    
    
for i in range(7500):
    o_out.loc[i]=i
    
           
    
    
   