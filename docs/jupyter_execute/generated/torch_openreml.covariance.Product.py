#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import DummyMatrix, Product, ScalarMatrix

op = Product(a=DummyMatrix(["a", "b", "c", "a"]), b=ScalarMatrix(3))
free_params = torch.tensor([1.0])
op(free_params)


# In[2]:


import torch
from torch_openreml.covariance import AR1Matrix, DiagonalMatrix, Product

op = Product(a=DiagonalMatrix(2), b=AR1Matrix(2))
free_params = torch.tensor([0.0, 0.5, 0.0, 0.5])
op.inverse(free_params)


# In[3]:


import torch
from torch_openreml.covariance import DummyMatrix, Product, ScalarMatrix

op = Product(a=DummyMatrix(["a", "b", "c", "a"]), b=ScalarMatrix(3))
free_params = torch.tensor([1.0])
grad, grad_names = op.manual_grad(free_params)
grad


# In[4]:


grad_names

