#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import DummyMatrix, DiagonalMatrix, ScalarMatrix, MarginalCovariance

z = DummyMatrix(["a", "b", "c", "a"])
g = DiagonalMatrix(3)
r = ScalarMatrix(4)
op = MarginalCovariance(Z=z, G=g, R=r)
op(torch.zeros(4))


# In[2]:


import torch
from torch_openreml.covariance import DummyMatrix, DiagonalMatrix, ScalarMatrix, MarginalCovariance

z = DummyMatrix(["a", "b", "c", "a"])
g = DiagonalMatrix(3)
r = ScalarMatrix(4)
op = MarginalCovariance(Z=z, G=g, R=r)
op(torch.zeros(4))


# In[3]:


import torch
from torch_openreml.covariance import DummyMatrix, DiagonalMatrix, ScalarMatrix, MarginalCovariance

z = DummyMatrix(["a", "b", "c", "a"])
g = DiagonalMatrix(3)
r = ScalarMatrix(4)
op = MarginalCovariance(Z=z, G=g, R=r)
grad, grad_names = op.manual_grad(torch.zeros(4))
grad


# In[4]:


grad_names

