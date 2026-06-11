#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import ScalarMatrix, DiagonalMatrix, BlockDiagonal

block = BlockDiagonal(
    residual=ScalarMatrix(3),
    random=DiagonalMatrix(2)
)
free_params = torch.tensor([0.5, 0.0, 1.0])
block(free_params)


# In[2]:


import torch
from torch_openreml.covariance import ScalarMatrix, DiagonalMatrix, BlockDiagonal

block = BlockDiagonal(
    A=ScalarMatrix(3),
    B=DiagonalMatrix(2)
)
free_params = torch.tensor([0.5, 0.0, 1.0])
grad, grad_names = block.manual_grad(free_params)
grad


# In[3]:


grad_names

