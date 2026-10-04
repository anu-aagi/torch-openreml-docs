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
from torch_openreml.covariance import DummyMatrix, Product, ScalarMatrix

op = Product(a=DummyMatrix(["a", "b", "c", "a"]), b=ScalarMatrix(3))
free_params = torch.tensor([1.0])
grad, grad_names = op.manual_grad(free_params)
grad


# In[3]:


grad_names

