#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(3)
free_params = torch.tensor([0.0, 0.5, 1.0])
sigma2 = mat.build_params(free_params)
mat.set_intermediates(free_params, {"sigma2": sigma2})
mat.get_intermediates(free_params)


# In[2]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(3)
free_params = torch.tensor([0.0, 0.5, 1.0])
sigma2 = mat.build_params(free_params)
mat.set_intermediates(free_params, {"sigma2": sigma2})
mat.get_intermediates(free_params)


# In[3]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(3)
free_params = torch.tensor([0.0, 0.5, 1.0])
sigma2 = mat.build_params(free_params)
mat.set_intermediates(free_params, {"sigma2": sigma2})
print(mat.get_intermediates(free_params))
mat.reset_intermediates()
print(mat.get_intermediates(free_params))


# In[4]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(3)
free_params = torch.tensor([0.0, 0.5, 1.0])
mat.trans_grad(free_params)


# In[5]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(2)
free_params = torch.tensor([0.0, 0.5])
grad, grad_names = mat.auto_grad(free_params)
grad, grad_names


# In[6]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(2)
free_params = torch.tensor([0.0, 0.5])
grad, grad_names = mat.grad(free_params)
grad, grad_names

