# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 17:29:10 2018

@author: mona
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as pt

data1 = pd.read_table('insample_error_tanh.txt', delim_whitespace=True, names=('A', 'B'))
#data2 = pd.read_table('outsample_error.txt', delim_whitespace=True, names=('C', 'D'))     
p1=np.mat(data1['B'])  

p=np.transpose(p1)
     
q1=np.mat(data1['A'])  

q=np.transpose(q1)
     
#pt.plot(q,p,'rx','MarkerSize',10)
pt.plot(q,p,linewidth=2.0,label='insample_error')
pt.ylabel('insample_error')
pt.xlabel('iteration(epoch)')

pt.title('insample_error vs itertion(tanh)')
pt.legend()
pt.pause(5)
