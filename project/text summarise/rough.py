# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 02:58:05 2018

@author: mona
"""
#wordDict={}
#for i in range(0,len(words)):
#  wordDict[i]=dict.fromkeys(inverted,0)
#  
#for i in range(0,len(words)):
#    for term in words[i]:
#        print(term)
#        wordDict[i][term]+=1
#        
#import numpy as np    
#import pandas as pd
#table = pd .DataFrame([wordDict])
#
#table1=np.transpose(table)
#
#tfTable={}
#def computeTF(wordDict,bow):
#    tfDict = {}
#    bowCount = len(bow)
#    for word,count in wordDict.items():
#        tfDict[word] =  count / float(bowCount)
#        
#    return tfDict 
#
#
#
#for a in range(0,len(words)):
#            
#    tfTable[a] = computeTF(wordDict[0], words[a]) 
#       
#table2=np.array(table1)    



#def computeIDF(inverted):
#    idfDict = {}
#    import math
#    for word,count in inverted.items():
#        idfDict[word] = math.log(25/float(count + 1))
#        
#    return idfDict
#
#
#
#if __name__ == "__main__":
#    
#    idfTable = computeIDF(inverted)
#    
#    table4 = pd .DataFrame([idfTable])
#        
#    
#    
#        
#table3 = pd .DataFrame([tfTable])
#
#tf_table=np.transpose(table3)
#
#
#tfidfTable=np.zeros(shape=(len(words),len(idfTable)
tf_tables = pd.DataFrame(tf_table)
table4_mat=np.mat(table4)
table8=tf_table[0][0] * table4_mat 

for i in tf_table[0][0]:
    print(i)

#ifidfTable=np.zeros(shape=(len(words),len(idfTable) 
tfidfTable = {}
for a in range(0,len(words)):
    for i in tf_table[0][a]:
        tfidfTable[a]=tf_table[0][a][i] * float(table4[i])
    print('done')    


for a in range(0,len(words)):
    for b in range(0,len(idfTable)):
    
         tfidfTable[a][b]=table4_mat * 
#    
#table5 = pd .DataFrame([tfidfTable])
#tfidf_table=np.transpose(table5)
#
#
import math as m
def similarity(p,q,r):
    
    value = 0    
    value = p/(m.sqrt(q) * m.sqrt(r))
    print(value)

    return value

    
if __name__ == "__main__":
    
   edge =[] 
   for i in range(12,13):
       for j in range(0,1123):
           p = tfidf_table[0][i] * tfidf_table[0][j]
            
           sum_num = 0
           for word in p: 
               sum_num = sum_num + float(p[word])
           
           t_value=similarity(sum_num,float(denomiator[i]),float(denomiator[j]))
           if t_value > 0.1:
               
               edge.append(t_value)
           else:
               edge.append(0)

      
with open('tfidf_vector.txt', 'a') as myfile:
    
    
   #  myfile.write(' '.join(file_text).encode('utf8'))
     
     for i in edge:
         
         myfile.write(''.join(str(i)))
         myfile.write(',')
         
             
     myfile.write('\n')        
     myfile.close()
    
#    d=tfidf_table[0][0]*tfidf_table[0][i]
#
#for word in tfidf_table[0][0]:
#    sum = sum + tfidf_table[0][0][word] 
#    print(tfidf_table[0][0][word])
#    print(sum)
#sum = 0    
#for word in d:
#    sum = sum + d[word] 
#    print(d[word])
#    print(sum)
def denom(p):
    sum_deno = 0
    for word2 in p:
        sum_deno = sum_deno + 
        
    print(sum_deno)
        
    return sum_deno
    
            
if __name__ == "__main__":
    
    deno = {}
    g = tfidf_table * tfidf_table        
    for k in range(0,len(tfidfTable)):
       # for j in range(0,len(tfidf_table)):
        deno[k]= denom(tfidfTable[k])
            
    
with open('tfidfs.txt', 'a') as myfile:
    
   #  myfile.write(' '.join(file_text).encode('utf8'))
                    
             myfile.write(' '.join(str(tfidfTable)))
             
             myfile.close()
        
import numpy as np    
tfidf_TableM = np.mat(tfidf_table)    

import numpy as np
for i in range(0,len(tfidfTable)):
    
    tfidf_TableM[i]=np.array(tfidfTable[i])    


sum_deno={}
#g=tfidf_table * tfidf_table
for i in range(900,1123):
    
    count = 0
    for word in g[0][i]:
    #for word2 in g[0][0]:
       # print(word2)
        count = count + float(g[0][i][str(word)])
    print(count)    
    sum_deno[i]= count    
print(sum_deno)

table6 = pd.DataFrame([sum_deno])
table7 = np.transpose(table6)

#tfidf_table=np.transpose(table6)
   

with open('tfidf.txt', 'a') as myfile:
    
   #  myfile.write(' '.join(file_text).encode('utf8'))
       for i in range(900,1123):
             print(table7[0][i])             
             myfile.write(''.join(str(table7[0][i])))
             myfile.write('\n')
             
       myfile.close()
             
             

#count=0
#for i in g[0][0]:
#    count=count+1
#    if count>1:
#        break
#    else:
#        print(i)
count = 0               
for word in g[0][1]:
    #for word2 in g[0][0]:
       # print(word2)
        count = count + g[0][1][str(word)]
print(count)        


doc_loc = "tfidf.txt"
file_doc = open(doc_loc, "r") 
denomiator=[]
for i in file_doc:
    print(i)
    denomiator.append(i)

doc_loc = "tfidf_original.txt"
file_doc = open(doc_loc,"r")
matrix = [item.split() for item in file_doc.readline.splits('\n')[:-1]] 


rows=1123
cols=1
with open('tfidf_original.txt') as f:
   data = []
   for i in range(0, rows):
      data.append(list(map(str, f.readline().split()[:1])))
print (data) 

term = pd.DataFrame(data)

for i in term[0]:
    print(i)
    
doc_loc = "tfidf_original.txt"
file_doc = open(doc_loc, "r") 
term=[]
for i in file_doc():
    print(i)
    term.append(i)
    
################################################################    
tf_idf=np.zeros(shape=(1123,3255))    
import re
a=0
b=0
with open("tfidf_original.txt",'r') as f:
    for line in f:
        print(line)
        a=a+1
        for word in re.findall(r'\w+', line):
            b=b+1
            if b < 3256:
                
                print(word)
                tf_idf[a][b]=word
            else:
                break
f.close()
#####################################################################
import numpy as np

dataDict = idfTable
orderedNames = []

dataMatrix = np.array([dataDict[i] for i in orderedNames])  

float(table4['pine'])

count=0
with open("tfidf_original.txt",'r') as f:
    for line in f:
       # print(line)
#        a=a+1
        if count<2:
           
           for word in line:
            count+=1
            term = word
            print(word)
#######################correct########################
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
        
    print(done)
######################correct######################################
import codeCS

import networkx as nx
g=nx.Graph()    
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
    
Matrix2 = [[0 for x in range(len(word))] for y in range(len(word))]
Matrix = [[0 for x in range(len(word))] for y in range(len(word))] 
ct=0
for i in range(len(word)):
    for j in range(len(word)):
        if i!=j:
            Matrix[i][j]=cosine(i,j)
            if Matrix[i][j]>0.1:
                g.add_edge(i,j,weight=Matrix[i][j])
                Matrix2[i][j]=Matrix[i][j]
                print(Matrix[i][j])
                
g.print_graph()

#################################################################
#count = 0 
def degree(node):
    count = 0
    a = 0
    for i in Matrix2[node]:
        if i > 0:
            count = count + 1
        else:
            a = a+1
            
    return count

Degree={}
for j in range(0,len(word)):
    Degree[j]=degree(j)


#################################################
d=[]    
for i in range(0,len(word)):
    sum = 0
    for j in range(0,len(L[i])):
        sum = sum + L[i][j]
    d.append(str(sum))
    print('done')
    
    
p=[]
for i in range(0,len(word)):
    
    
    p.append(str((0.85/len(word))+(1-0.85) *float( D[i])))
    
    
    