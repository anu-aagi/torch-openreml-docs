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


from torch_openreml.covariance import ScalarMatrix

mat = ScalarMatrix(3)
mat.disable_cache()
mat.enable_cache()
mat.cache


# In[6]:


import torch
from torch_openreml.covariance import ScalarMatrix

mat = ScalarMatrix(3)
params = mat.build_params(torch.tensor([0.5]))
mat.set_intermediates(params, "value")
mat.disable_cache()
print(mat.get_intermediates(params))


# In[7]:


from torch_openreml.covariance import ScalarMatrix

mat = ScalarMatrix(3)
mat.get_default_dtype_device()


# In[8]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(3)
free_params = torch.tensor([0.0, 0.5, 1.0])
mat.build_params(free_params)


# In[9]:


mat.build_params()


# In[10]:


mat.param_specs["sigma^2_2"]["fixed"] = True
mat.build_params(free_params[0:2])


# In[11]:


mat.build_params(free_params[0:2], include_fixed=False)


# In[12]:


mat.build_params(free_params[0:2], include_fixed=False, trans=False)


# In[13]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(3)
mat.set_param_specs("sigma^2_1", fixed=True, default=torch.tensor([7.5]))
mat.free_param_names


# In[14]:


mat.build_params(torch.tensor([1.0, 3.0]), trans=False)


# In[15]:


from torch_openreml.covariance import BlockDiagonal, ScalarMatrix

op = BlockDiagonal(subject=DiagonalMatrix(2), noise=ScalarMatrix(2))
op.set_param_specs("subject/", fixed=True)
op.free_param_names


# In[16]:


op.set_param_specs("/", fixed=True)
op.free_param_names


# In[17]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(3)
free_params = torch.tensor([0.0, 0.5, 1.0])
mat.trans_grad(free_params)


# In[18]:


mat.trans_grad()


# In[19]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(2)
free_params = torch.tensor([0.0, 0.5])
grad, grad_names = mat.auto_grad(free_params)
grad, grad_names


# In[20]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(2)
free_params = torch.tensor([0.0, 0.5])
grad, grad_names = mat.grad(free_params)
grad, grad_names


# In[21]:


from torch_openreml.covariance import ScalarMatrix

mat = ScalarMatrix(3)
mat.resolve_dim(["a", "b", "a", "c"])

