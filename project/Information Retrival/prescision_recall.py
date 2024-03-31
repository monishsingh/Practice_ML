# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 02:23:42 2018

@author: mona
"""

import pandas as pd
import numpy as np
              
def precision_and_recall(output_file,filename):
    
    """
    Args:
        output_file: file containing result for queries
    """
   # import pandas as pd
 
    output = open("output.txt",'r')
#query = open("query.txt","r")
    est_output = open("elastics_after_updates.txt",'r')

    query = open("query.txt","r")
    query_ID = []

    for line in query:
        query_ID.append(line.split()[0])
     

    query_ID     
     
    e_out = pd.DataFrame(columns = ["q_ID","Doc"])
    o_out = pd.DataFrame(columns=["q_ID","Doc"])
#e_out ={}
#o_out ={}

#query = []
    docs = []
    k = 0
   #import numpy as np
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
    
    prerec = open("precision_before_update.txt","a")
    prerec.write("queryID" + "     " + "precision" + "     " + "recall" + "\n")
       
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
        prerec.write(str(q_ID) + " " + str(precision) + " " + str(recall) +"    "+str(f_score)+""+ "\n")
    prerec.close()
    output.close()
    est_output.close()

    
if __name__ == "__main__":
	precision_and_recall("elastics_before_updates.txt","precision_before_update.txt")
	#precision_and_recall("grep_output.txt","grep_precision_and_recall.txt")
  