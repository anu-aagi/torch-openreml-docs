#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import DummyMatrix, Transpose

op = Transpose(a=DummyMatrix(["a", "b", "c", "a"]))
op(torch.tensor([]))


# In[2]:


import torch
from torch_openreml.covariance import LowerTriangularMatrix, Transpose

op = Transpose(a=LowerTriangularMatrix(3, 2))
free_params = torch.tensor([0.0, 0.5, 1.0, 0.2, -0.3])
grad, grad_names = op.manual_grad(free_params)
grad


# In[3]:


grad_names

