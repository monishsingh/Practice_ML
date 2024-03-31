# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 23:21:51 2018

@author: mona
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as pt

#data1 = pd.read_table('insample_error.txt', delim_whitespace=True, names=('A', 'B'))
data2 = pd.read_table('outsample_error_softmax.txt', delim_whitespace=True, names=('A', 'B'))     
p1=np.mat(data2['B'])  

p=np.transpose(p1)
     
q1=np.mat(data2['A'])  

q=np.transpose(q1)
     
#pt.plot(q,p,'rx','MarkerSize',10)
pt.plot(q,p,linewidth=2.0,label='outsample_error')
pt.ylabel('outsample_error')
pt.xlabel('iteration(epoch)')

pt.title('output_error vs itertion(softmax)')
pt.legend()
pt.pause(5)
