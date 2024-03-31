# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 04:36:53 2018

@author: mona
"""

text_word = {}
query_file = 'query.txt'
file = open(query_file,"r")
j = 0
for i in file:
    text_word[j]=i
    j = j+1

from rouge import Rouge 

#scores = {}
file = open('elastics_before_updates.txt', 'r')
string = file.read()

#for i in range(0,len(text_word)):
#    
#     string1 = text_word[0]

file1 = open('output.txt','r')
string1 = file1.read()

#list_doc = os.listdir("GroundTruth")
hypothesis = string
reference = string1

rouge = Rouge()
scores = rouge.get_scores(reference,hypothesis)


print(scores)



