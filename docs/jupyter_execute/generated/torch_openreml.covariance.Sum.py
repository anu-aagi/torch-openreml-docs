#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import AR1Matrix, ScalarMatrix, Sum

op = Sum(time=AR1Matrix(4), noise=ScalarMatrix(4))
free_params = torch.tensor([0.5, 1.0, 1.0])
op(free_params)


# In[2]:


import torch
from torch_openreml.covariance import AR1Matrix, ScalarMatrix, Sum

op = Sum(time=AR1Matrix(4), noise=ScalarMatrix(4))
free_params = torch.tensor([0.5, 1.0, 1.0])
grad, grad_names = op.manual_grad(free_params)
grad


# In[3]:


grad_names

