#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import DummyMatrix, DiagonalMatrix, CovariancePropagation

z = DummyMatrix(["a", "b", "c", "a"])
z()


# In[2]:


g = DiagonalMatrix(3)
op = CovariancePropagation(z=z, g=g)
free_params = torch.tensor([0.0, 0.5, 1.0])
op(free_params)


# In[3]:


import torch
from torch_openreml.covariance import DummyMatrix, DiagonalMatrix, CovariancePropagation

z = DummyMatrix(["a", "b", "c", "a"])
z()


# In[4]:


g = DiagonalMatrix(3)
op = CovariancePropagation(z=z, g=g)
free_params = torch.tensor([0.0, 0.5, 1.0])
grad, grad_names = op.manual_grad(free_params)
grad


# In[5]:


grad_names

