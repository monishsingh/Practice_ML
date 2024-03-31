# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 04:16:54 2018

@author: mona
"""

def summary(node):
    
    for i in node:
        text = senc[i]
        with open('Tsummary.txt', 'a') as myfile:
                    
             myfile.write(''.join(text))
             
             myfile.close()
    
    
if __name__ == "__main__":
    
    summary(pick1)
    
    
    
    
    