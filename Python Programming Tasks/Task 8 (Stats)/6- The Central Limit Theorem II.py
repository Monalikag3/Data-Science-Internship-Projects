#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math


def cdf(x, mean, std):
    return 1/2*(1+math.erf((x-mean) / std / 2**(1/2)))

# inputs
mean = 2.4
std = 2
n = 100
target = 250

# characteristics of sum
mean_s = n * mean
std_s = std * n**(1/2)

# Find the probability that sum <= 250
print(round(cdf(target, mean_s, std_s), 4))


# In[ ]:




