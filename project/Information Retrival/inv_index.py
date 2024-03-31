# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 12:22:19 2018

@author: mona
"""
#import csv
import os
#import sys
import re
import nltk
#import operator
import pickle 
#import progressbar

#import numpy as np



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

if __name__ == "__main__":
    with open("indexed_docs.pkl","wb") as handle:
        
         inverted_index()
         pickle.dump(inverted, handle)


with open("indexed_docs.pkl","rb") as dict_file:
    dictionary = pickle.load(dict_file)


import pandas as pd
import time
    
def process_query(query_file):
    
    f = open(query_file,"r")
    query_ID = []
    query_result = []
    Time = []
    q_res = pd.DataFrame(columns = ["queryID","Time"])
    for j,query in enumerate(f):
        result = []
        q_ID = query.split()
        query_ID.append(q_ID)
        query = preprocessing_txt(query)
        query = query.split()
        START = time.time()
        for i,word in enumerate(query):
            
            if i == 0:
                for i in dictionary[word].items():
                    result.append(i[0])
            else:
                temp = []
                for i in dictionary[word].items():
                    temp.append(i[0])
                result = list(set(result).intersection(set(temp)))
        END = time.time()
        f1 = open("output_inverted_index.txt",'a')
#         print str(q_ID) + "%d" % len(result)
        for res in result:
            f1.write(str(q_ID) + " " + str(res) + "\n")
        f1.close() 
        Time.append(float(END - START))
        query_result.append(result)
    q_res["queryID"] = query_ID
    q_res["Time"] = Time
    q_res.to_csv("inverted_index.csv",encoding='utf-8')
    result = pd.DataFrame(columns = ["query_ID","relevant_docs"])
    result["query_ID"] = query_ID
    result["relevant_docs"] = query_result
    return result

if __name__ == "__main__":
    query_file = "query.txt"
    process_query(query_file)
    
import pandas as pd
import numpy as np
              
def precision_and_recall(output_file,filename):
    """
    Args:
        output_file: file containing result for queries
    """
    
    output = open("output.txt",'r')
    query = open("query.txt","r")
    est_output = open(output_file)
    query_ID = []
    
    prerec = open(filename,"a")
    prerec.write("queryID" + " " + "precision" + " " + "recall" + "\n")
    for line in query:
        query_ID.append(line.split())
        
    e_out = pd.DataFrame(columns = ["q_ID","Doc"])
    o_out = pd.DataFrame(columns=["q_ID","Doc"])
    
    query = []
    docs = []
    for line in est_output:
        query.append(str(line.split()))
        docs.append(line.split())
    e_out['q_ID'] = query
    e_out["Doc"] = docs
    
    query = []
    docs = []
    for line in output:
        query.append(line.split()[0])
        docs.append(line.split()[1])
    o_out['q_ID'] = query
    o_out["Doc"] = docs
        
    for q_ID in query_ID:
        #estimated = list(e_out[e_out['q_ID'] == q_ID]["Doc"])
        #estimated=list[set(e_out.q_ID.isin(q_ID))]
        estimated=e_out.q_ID.isin(q_ID)
#         print len(estimated)
        #true = list(o_out[o_out["q_ID"] == q_ID]["Doc"])
        true=o_out.q_ID.isin(q_ID)
#         print len(true)
        precision = len(list(set(estimated).intersection(set(true))))/float(len(estimated))
        recall = len(list(set(estimated).intersection(set(true))))/float(len(true))
        prerec.write(str(q_ID) + " " + str(precision) + " " + str(recall) + "\n")
    prerec.close()
    output.close()
    est_output.close()

if __name__ == "__main__":
	precision_and_recall("output_inverted_index.txt","inverted_index_precision_and_recall.txt")
	precision_and_recall("grep_output.txt","grep_precision_and_recall.txt")
	precision_and_recall("output_elastic.txt","elastic_search_precision_and_recall.txt")
    
     

