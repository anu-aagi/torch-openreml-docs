#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch_openreml.covariance import DummyMatrix

rep = ["rep1", "rep2", "rep2"]
block = ["block1", "block2", "block1"]

mat = DummyMatrix(rep, block)
print(mat())
print(mat.colnames)


# In[2]:


mat = DummyMatrix(rep, block, drop_first=True)
print(mat())
print(mat.colnames)


# In[3]:


mat = DummyMatrix(rep, block, levels=[["rep1", "rep2", "rep3"], ["block1", "block2"]])
print(mat())
print(mat.colnames)


# In[4]:


mat = DummyMatrix(rep, block, levels=[["rep3", "rep1"], ["block1", "block2"]], lex_order=False)
print(mat())
print(mat.colnames)


# In[5]:


mat = DummyMatrix(rep, block, levels=[["rep2", "rep1"], ["block1", "block2"]], lex_order=False, drop_empty_cols=True)
print(mat())
print(mat.colnames)

