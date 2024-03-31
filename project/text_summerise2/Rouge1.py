# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 04:36:53 2018

@author: mona
"""

from rouge import Rouge 

file = open('summary9.txt', 'r')
string = file.read()

file1 = open('Topic3.txt', 'r')
string1 = file1.read()



#list_doc = os.listdir("GroundTruth")
hypothesis = string
reference = string1

rouge = Rouge()
scores = rouge.get_scores(reference,hypothesis)


print(scores)



