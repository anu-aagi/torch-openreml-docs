#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(3)
free_params = torch.tensor([0.0, 0.5, 1.0])
params = mat.build_params(free_params)
mat.set_intermediates(params, {"log(sigma^2)/2": torch.log(params) / 2})
mat.get_intermediates(params)


# In[2]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(3)
free_params = torch.tensor([0.0, 0.5, 1.0])
params = mat.build_params(free_params)
mat.set_intermediates(params, {"log(sigma^2)/2": torch.log(params) / 2})
mat.get_intermediates(params)


# In[3]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(3)
free_params = torch.tensor([0.0, 0.5, 1.0])
params = mat.build_params(free_params)
mat.set_intermediates(params, {"log(sigma^2)/2": torch.log(params) / 2})
print(mat.get_intermediates(params))


# In[4]:


mat.reset_intermediates()
print(mat.get_intermediates(free_params))


# In[5]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(3)
free_params = torch.tensor([0.0, 0.5, 1.0])
mat.build_params(free_params)


# In[6]:


mat.build_params()


# In[7]:


mat.param_specs["sigma^2_2"]["fixed"] = True
mat.build_params(free_params[0:2])


# In[8]:


mat.build_params(free_params[0:2], include_fixed=False)


# In[9]:


mat.build_params(free_params[0:2], include_fixed=False, trans=False)


# In[10]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(3)
free_params = torch.tensor([0.0, 0.5, 1.0])
mat.trans_grad(free_params)


# In[11]:


mat.trans_grad()


# In[12]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(2)
free_params = torch.tensor([0.0, 0.5])
grad, grad_names = mat.auto_grad(free_params)
grad, grad_names


# In[13]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(2)
free_params = torch.tensor([0.0, 0.5])
grad, grad_names = mat.grad(free_params)
grad, grad_names

