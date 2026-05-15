#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import CompoundSymmetricMatrix

mat = CompoundSymmetricMatrix(3)
free_params = torch.tensor([0.5, 0.0])
print(mat(free_params))
print(mat.grad(free_params))

