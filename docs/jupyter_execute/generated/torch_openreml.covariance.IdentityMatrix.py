#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import IdentityMatrix

mat = IdentityMatrix(3)
mat()


# In[2]:


import torch
from torch_openreml.covariance import IdentityMatrix

mat = IdentityMatrix(3)
mat()


# In[3]:


mat(torch.tensor([], dtype=torch.float64))


# In[4]:


import torch
from torch_openreml.covariance import IdentityMatrix

mat = IdentityMatrix(3)
mat.inverse()


# In[5]:


mat.inverse(torch.tensor([], dtype=torch.float64))

