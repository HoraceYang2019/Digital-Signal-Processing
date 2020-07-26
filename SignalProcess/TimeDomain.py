# -*- coding: utf-8 -*-
'''
'''
import numpy as np
import scipy.stats as st

# In[1]: calculate time doamin features 
def getTimeFeature(s):
    absv = np.absolute(s)
    root = np.mean(np.sqrt(absv))
    rms = np.sqrt(np.mean(np.square(s)))# rms
    peak = np.max(absv) # peak
    sk = st.skew(s)
    ku = st.kurtosis(s)
    cr = peak/rms
    cl = peak/root # clearance
    shape = rms/np.mean(absv) # shape
    ip = peak/np.mean(absv) # impluse
    # mean, sd, root, rms,
    X = [s.mean(), s.std(), root, rms, peak, sk, ku, cr, cl, shape, ip]
    return (X)

# In[2]: inital data

a = np.array([1, -2, 3])
b = np.array([1, 4, 6])
    
print(getTimeFeature(a))
print(getTimeFeature(b))

# In[3]: Question
'''
1. What differences amnong std, sk, and ku?
2. What differences among mean, root, and rms?
3. How to define the time range for time-domain features?
'''