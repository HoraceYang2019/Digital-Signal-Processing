# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 16:27:27 2019
http://benalexkeen.com/correlation-in-python/
"""

import numpy as np

# In[1]: Positive Correlation
np.random.seed(1)

# 1000 random integers between 0 and 50
x = np.random.randint(0, 50, 1000)
# Positive Correlation with some noise
y = x + np.random.normal(0, 10, 1000)
np.corrcoef(x, y)

# In[]: plot
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

plt.scatter(x, y)
plt.show()

# In[2]: Negative Correlation
x = np.random.randint(0, 50, 1000)

# Negative Correlation with some noise
y = 100 - x + np.random.normal(0, 5, 1000)

np.corrcoef(x, y)
plt.scatter(x, y)
plt.show()

# In[3] No/Weak Correlation
x = np.random.randint(0, 50, 1000)
y = np.random.randint(0, 50, 1000)

np.corrcoef(x, y)

# In[4]: Correlation Matrix
import pandas as pd

df = pd.DataFrame({'a': np.random.randint(0, 50, 1000)})
df['b'] = df['a'] + np.random.normal(0, 10, 1000) # positively correlated with 'a'
df['c'] = 100 - df['a'] + np.random.normal(0, 5, 1000) # negatively correlated with 'a'
df['d'] = np.random.randint(0, 50, 1000) # not correlated with 'a'

from pandas.plotting import scatter_matrix

df.corr()
scatter_matrix(df, figsize=(6, 6))
plt.show()

# In[5]
# http://hamelg.blogspot.com/2015/11/python-for-data-analysis-part-25-chi.html
import numpy as np
import pandas as pd
import scipy.stats as stats

np.random.seed(10)

# Sample data randomly at fixed probabilities
voter_race = np.random.choice(a= ["C1","C2","C3","C4","C5"],
                              p = [0.05, 0.15 ,0.25, 0.05, 0.5],
                              size=1000)

# Sample data randomly at fixed probabilities
voter_party = np.random.choice(a= ["B1","B2","B3"],
                              p = [0.4, 0.2, 0.4],
                              size=1000)

voters = pd.DataFrame({"race":voter_race, 
                       "party":voter_party})

voter_tab = pd.crosstab(voters.race, voters.party, margins = True)

voter_tab.columns = ["B1","B2","B3","row_totals"]

voter_tab.index = [""C1","C2","C3","C4","C5","col_totals"]

observed = voter_tab.ix[0:5,0:3]   # Get table without totals for later use
voter_tab

# In[]
expected =  np.outer(voter_tab["row_totals"][0:5],
                     voter_tab.loc["col_totals"][0:3]) / 1000

expected = pd.DataFrame(expected)

expected.columns = ["B1","B2","B3"]
expected.index = ["C1","C2","C3","C4","C5"]

chi_squared_stat = (((observed-expected)**2)/expected).sum().sum()

crit = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 8)   # *

p_value = 1 - stats.chi2.cdf(x=chi_squared_stat,  # Find the p-value
                             df=8)
