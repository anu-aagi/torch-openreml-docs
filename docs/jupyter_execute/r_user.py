#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('Rscript source/r_user/code.R > source/r_user/output.txt 2>/dev/null')

get_ipython().system('cat source/r_user/output.txt')

