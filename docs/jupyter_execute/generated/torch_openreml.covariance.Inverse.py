#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import ScalarMatrix, Inverse

op = Inverse(a=ScalarMatrix(3))
free_params = torch.tensor([0.5])
op(free_params)


# In[2]:


import torch
from torch_openreml.covariance import ScalarMatrix, Inverse

op = Inverse(a=ScalarMatrix(3))
free_params = torch.tensor([0.5])
grad, grad_names = op.manual_grad(free_params)
grad


# In[3]:


grad_names

