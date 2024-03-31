# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 16:14:27 2018

@author: mona
"""

###  text rank algorithm   #############
import numpy as np
def powerMethod(Matrix2,n):
    p={}
    p[0]=1/n
    t=0
    for i in range(0,100):
        t=t+1
        p[t]= (np.transpose(Matrix2))* p[t-1]
    return p[t]    
        
n=len(word)
def text_rank(Matrix2):
    degree=np.zeros(shape=(len(word),1))
    for i in range(0,len(word)):
        for j in range(0,len(word)):
            
            if Matrix2[i][j] > 0.1:
                Matrix2[i][j] = 1
                degree[i]=degree[i]+1
            else:
                Matrix2[i][j]=0
                
                
    for i in range(0,len(word)):
        for j in range(0,len(word)):
            
            Matrix2[i][j]=(Matrix2[i][j])/(Degree1[i]+1)
            
    L = powerMethod(Matrix2,n)
    
    return L
              
if __name__ == "__main__":
    l={}
    l = text_rank(Matrix2)
    
    def degree(node):
        
         count = 0
         a = 0
         
         for i in Matrix2[node]:
             
              if i > 0:
                  
                 count = count + 1
             else:
                 a = a+1
            
    return count

############################################
    
    page = pd.DataFrame([p])
#    for j in range(0,len(word)):
#        
#         Page[j]=P[j]
#         
#    Page=p 

    
    count =0
    temp = 0
    for i in page:
        print(i)
        
        if count > 0:
            
            if temp < float(page[i]):
                
                temp =float( page[i])
                 
        else:
            
            temp = float(page[i])
            count=count + 1
             
    print(temp)
    high = temp
    
    for i in page:
        if high == page[i]:
            
            a = i
            page[i]=0
            
    print(a)
    pick1=[]
    pick1.append(a)

###########################################
d=[]    
for i in range(0,len(word)):
    sum = 0
    for j in range(0,len(L[i])):
        sum = sum + Matrix2[i][j]
    d.append(str(sum))
    print('done')


p=[]
for i in range(0,len(word)):
    
    
    p.append(str((0.85/len(word))+(1-0.85) *float( d[i])))

     
P = pd.DataFrame([p])    
    

rank=np.transpose(page)

