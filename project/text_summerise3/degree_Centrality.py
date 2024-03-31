# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 02:50:58 2018

@author: mona
"""

def degree(node):
    count = 0
    a = 0
    for i in Matrix2[node]:
        if i > 0:
            count = count + 1
        else:
            a = a+1
            
    return count


if __name__ == "__main__":
    
    Degree={}
    for j in range(0,len(word)):
        
         Degree[j]=degree(j)
         
    Degree1=Degree 

    
    count =0
    temp = 0
    for i in Degree1:
        print(i)
        
        if count > 0:
            
            if temp < Degree1[i]:
                
                temp = Degree1[i]
                 
        else:
            
            temp = Degree1[i]
            count=count + 1
             
    print(temp)
    high = temp
    
    for i in Degree1:
        if high == Degree1[i]:
            a = i
            Degree1[i]=0
            
    print(a)
   # pick=[]
    pick.append(a)
    ###############################       
    for i in Matrix2[a]:
        print(i)
        
        if count > 0:
            
            if temp < Matrix2[a][i]:
                
                temp = Matrix2[a][i]
                 
        else:
            
            temp = Matrix2[a][i]
            count=count + 1
             
    print(temp)
    high1 = temp
    
    for i in Matrix2[a]:
        if high1 == Matrix2[a][i]:
            b = i
            
    print(b) 
    
            
             
             
             
