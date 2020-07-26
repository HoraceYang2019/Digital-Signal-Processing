# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 08:32:56 2020
https://pyhht.readthedocs.io/en/latest/apiref/pyhht.html#module-pyhht.emd
pip install pyhht
@author: ASUS
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyhht.emd import EMD
from pyhht.utils import get_envelops
from pyhht.visualization import plot_imfs
# In[]
# load data
t = np.arange(0,1, 0.01)
x = 2*np.sin(2*np.pi*15*t) +4*np.sin(2*np.pi*10*t)*np.sin(2*np.pi*t*0.1)+np.sin(2*np.pi*5*t)

upper, lower = get_envelops(x)
plt.plot(upper)
plt.plot(lower)
plt.show()

# In[] EMD decompose
decomposer = EMD(x)
imfs = decomposer.decompose() # Decompose the input signal into IMFs.

plot_imfs(x, imfs, t)
print('%.3f' % decomposer.io())

plot_imfs.show()

plt.plot(imfs[1,:].T)
# In[] save IMFs
arr = np.vstack((imfs,x))
dataframe = pd.DataFrame(arr.T)
