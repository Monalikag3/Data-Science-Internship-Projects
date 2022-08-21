#!/usr/bin/env python
# coding: utf-8

# In[1]:


import statistics as st

n = 5
x = [95, 85, 80, 70, 60]
y = [85, 95, 70, 65, 70]
meanX = st.mean(x)
meanY = st.mean(y)
x_s = sum([x[i] ** 2 for i in range(5)])
xy = sum([x[i]*y[i] for i in range(5)])

b = (n * xy - sum(x) * sum(y)) / (n * x_s - (sum(x) ** 2))
a = meanY - b * meanX

print (round(a + 80 * b, 3))


# In[ ]:




