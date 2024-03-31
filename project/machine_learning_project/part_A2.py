# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 02:16:32 2018

@author: mona
"""

import nltk
import re
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import math as m


def tanh(x):
  return (m.exp(x)-m.exp(-x)) / (m.exp(x) + m.exp(-x))

inverted ={}
def inverted_index():
    
    """
    Creates a dictionary of words as key and name of the documents as items
    """
    docs_indexed = 0
#    
    count= 0
    for a in range(0,len(wordss)):
        
        for b in range(0,len(wordss[a])):
#            file_doc = preprocessing_txt(file_doc.read())
#            tokens = tokenizer(file_doc)
#            for wordin tokens:
    
            if not inverted.__contains__(wordss[a][b]):
                count = 1
                doclist = 1
               # doclist[doc] = 1
                inverted[wordss[a][b]] = doclist
            else:
                
                count = inverted[wordss[a][b]]
                count = count + 1
                doclist = count
                inverted[wordss[a][b]] = doclist
                    
    docs_indexed += 1
            
    return inverted

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
     lines=[]
     count=0
     file = open('Assignment_2_data.txt', 'r')
     
     
     for line in file:
         lines.append(line)
         print('done')
         
     word=[]    
     for i in range(0,len(lines)):
          word.append(tokenizer(lines[i]))
     print(word)
          
     words=[]
     for i in range(0,len(word)):
         words.append(preprocessing_txt(word[i]))
         
     wordss=[]       
     for k in range(0,len(words)):
         wordss.append(tokenizer(words[k]))
     print(wordss)
         
     inverted_index()
     
     list_term=[]
     for i in inverted:
         list_term.append(i)
     
    # import numpy as np   
     input_vector=np.zeros(shape=(len(lines),len(list_term)))
     b=0
     for a in range(0,len(lines)):
         b=0
         for c in range(0,len(list_term)):
            
        
    
             
           
                  if list_term[c] == wordss[a][b]:
                       input_vector[a][c]=1
                      # c=0
                      # b=b+1
                       if b == len(wordss[a])-1:
                           b=0
                       else:
                           b=b+1
                       
                  else:
                       input_vector[a][c]=0
                       
                       
     print('done')

     import math as m
    # no_of_input = m.floor(len(lines)*0.80)
     no_of_input = len(inverted)
     W = np.random.randn(no_of_input,100) * np.sqrt(2/100)
     W1=np.transpose(W)
     
     no_of_training = m.floor(len(lines)*0.80)
     b = np.random.randint(0,input_vector.shape[0],no_of_training)
     in_put=input_vector[b]
     
      ## assume ham = 0 and spam = 1
     Y = np.zeros(shape=(no_of_training,1))
     for i in range(0,no_of_training):
         
         if in_put[i][0] == 1:
         
            Y[i][0] = 0
         else:
            Y[i][0] = 1
    
    ################# forward propagation ##################
     Z = in_put.dot(W)
     
     act_layer1 = np.zeros(shape=(no_of_training,100))
     
     for i in range(0,no_of_training):
         for j in range(0,100):
             act_layer1[i][j] = tanh(Z[i][j])
             
             
     W2 = np.random.randn(100,50) * np.sqrt(2/100)
     
     Z1 = act_layer1.dot(W2)
     
     act_layer2 = np.zeros(shape=(no_of_training,50))
     
     for i in range(0,no_of_training):
         for j in range(0,50):
             act_layer2[i][j] = tanh(Z1[i][j])
     
        
     W3 = np.random.randn(50,1) * np.sqrt(2/100)  
     
     Z3 = act_layer2.dot(W3)
     act_layer3 = np.zeros(shape=(no_of_training,1))
     for i in range(0,no_of_training):
         for j in range(0,1):
             act_layer3[i][j] = tanh(Z3[i][j])
             
     #####################  backward propagation  ####################        
    # delta4 = act_layer3 - Y 
     delta4 = np.power((act_layer3 - Y),2)
     
     g_z3 = np.multiply(act_layer2,(1-act_layer2))
     delta3 =np.multiply((delta4.dot(np.transpose(W3))),g_z3)
     
     g_z2 = np.multiply(act_layer1,(1-act_layer1))
     delta2 =np.multiply((delta3.dot(np.transpose(W2))),g_z2)
     
     big_delta1 = (np.transpose(in_put)).dot(delta2)
     big_delta2 = (np.transpose(act_layer1)).dot(delta3)
     big_delta3 = (np.transpose(act_layer2)).dot(delta4)
     
     D1 = (1/no_of_training)*big_delta1 + 0.1*W
     D2 = (1/no_of_training)*big_delta2 + 0.1*W2 
     D3 = (1/no_of_training)*big_delta3 + 0.1*W3
             
     ################################################################       
     insample_error=[]
     for i in range(0,7):
        
        Z = in_put.dot(D1)
     
        act_layer1 = np.zeros(shape=(no_of_training,100))
     
        for i in range(0,no_of_training):
             for j in range(0,100):
                 act_layer1[i][j] = tanh(Z[i][j])
             
             
       # W2 = np.random.randn(100,50) * np.sqrt(2/100)
     
        Z1 = act_layer1.dot(D2)
     
        act_layer2 = np.zeros(shape=(no_of_training,50))
     
        for i in range(0,no_of_training):
            for j in range(0,50):
                act_layer2[i][j] = tanh(Z1[i][j])
     
        
       # W3 = np.random.randn(50,1) * np.sqrt(2/100)  
     
        Z3 = act_layer2.dot(D3)
        act_layer3 = np.zeros(shape=(no_of_training,1))
        for i in range(0,no_of_training):
            for j in range(0,1):
                act_layer3[i][j] = tanh(Z3[i][j])
                
        #####################################################
        
        output1 = np.zeros(shape=(len(act_layer3),1))
     
        for i in range(0,len(act_layer3)):
            
             for j in range(0,1):
                 
             
                 if float(act_layer3[i]) < 0.1:
                     output1[i][j] = 0
                 else:
                     output1[i][j] = 1
            
        total = 0    
        for i in range(0,len(output1)):
         
             if float(output1[i]) == float(Y[i]):
                   total = total + 1
             
        accuracy = (total/len(output1))*100
        print(accuracy)
        
        error = len(output1) - total
        
        insample_error.append(error)
        
        ############################################################
       # np.power((Zf-Zo),2)     
        delta4 = np.power((act_layer3 - Y),2) 
     
        g_z3 = np.multiply(act_layer2,(1-act_layer2))
        delta3 =np.multiply((delta4.dot(np.transpose(D3))),g_z3)
     
        g_z2 = np.multiply(act_layer1,(1-act_layer1))
        delta2 =np.multiply((delta3.dot(np.transpose(D2))),g_z2)
     
        big_delta1 = (np.transpose(in_put)).dot(delta2)
        big_delta2 = (np.transpose(act_layer1)).dot(delta3)
        big_delta3 = (np.transpose(act_layer2)).dot(delta4)
        
        D1 = (1/no_of_training)*big_delta1 + 0.1*D1
        D2 = (1/no_of_training)*big_delta2 + 0.1*D2 
        D3 = (1/no_of_training)*big_delta3 + 0.1*D3
        
     print('done') 
     
     file = open("insample_error_tanh.txt",'w')
     for i in range(len(insample_error)):
         file.write(str(i))
         file.write("   ")
         file.write(str(insample_error[i]))
         file.write('\n')
     file.close()
     
      ######################  TESTING   ###############################
     
     test_lines=[]
     count1=0
     files = open('test_data.txt', 'r')
     
     
     for line in files:
         test_lines.append(line)
         print('done')
         
     wordT=[]    
     for i in range(0,len(test_lines)):
          wordT.append(tokenizer(test_lines[i]))
     print(wordT)
          
     wordsT=[]
     for i in range(0,len(wordT)):
         wordsT.append(preprocessing_txt(wordT[i]))
         
     wordssT=[]       
     for k in range(0,len(wordsT)):
         wordssT.append(tokenizer(wordsT[k]))
     print(wordssT)
     
     
     input_vectorTEST=np.zeros(shape=(len(test_lines),len(list_term)))
     b=0
     for a in range(0,len(test_lines)):
         b=0
         for c in range(0,len(list_term)):
            
        
    
             
           
                  if list_term[c] == wordssT[a][b]:
                       input_vectorTEST[a][c]=1
                      # c=0
                      # b=b+1
                       if b == len(wordssT[a])-1:
                           b=0
                       else:
                           b=b+1
                       
                  else:
                       input_vectorTEST[a][c]=0
                       
                       
     print('done')
     
     ## assume ham = 0 and spam = 1
     Y_test = np.zeros(shape=(len(test_lines),1))
     for i in range(0,len(test_lines)):
         
         if input_vectorTEST[i][0] == 1:
         
            Y_test[i][0] = 0
         else:
            Y_test[i][0] = 1
     
   ###################################################  
     Z_test = input_vectorTEST.dot(D1)
     
     act_layer1_test = np.zeros(shape=(len(test_lines),100))
     
     for i in range(0,len(test_lines)):
         for j in range(0,100):
             act_layer1_test[i][j] = tanh(Z_test[i][j])
       
     
     
     Z1_test = act_layer1_test.dot(D2)
     
     act_layer2_test = np.zeros(shape=(len(test_lines),50))
     
     for i in range(0,len(test_lines)):
         for j in range(0,50):
             act_layer2_test[i][j] = tanh(Z1_test[i][j])
     
        
    # W3 = np.random.randn(50,1) * np.sqrt(2/100)  
     
     Z3_test = act_layer2_test.dot(D3)
     act_layer3_test = np.zeros(shape=(len(test_lines),1))
     for i in range(0,len(test_lines)):
         for j in range(0,1):
             act_layer3_test[i][j] = tanh(Z3_test[i][j])
     
            
             
    ########################## accuracy####################
    
     output_test = np.zeros(shape=(len(test_lines),1))
     
     for i in range(0,len(test_lines)):
         for j in range(0,1):
             
             if float(act_layer3_test[i]) < 0.1:
                 output_test[i][j] = 0
             else:
                 output_test[i][j] = 1

    ###################### test accurcy############
     outsample_error=[]
     total = 0    
     for i in range(0,len(test_lines)):
         
         if float(output_test[i]) == float(Y_test[i]):
             total = total + 1
             
     accuracy = (total/len(test_lines))*100
     print(accuracy)
     
     error = len(test_lines) - total
        
     outsample_error.append(error)
     
     
     file = open("outsample_error_tanh.txt",'w')
     for i in range(len(outsample_error)):
         file.write(str(i))
         file.write("   ")
         file.write(str(outsample_error[i]))
         file.write('\n')
     file.close()
     
