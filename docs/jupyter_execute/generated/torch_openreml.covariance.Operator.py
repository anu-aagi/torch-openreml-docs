#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import Sum, ScalarMatrix

x = Sum(ScalarMatrix(2), ScalarMatrix(2))
x


# In[2]:


x = Sum(A = ScalarMatrix(2), B = ScalarMatrix(2))
x


# In[3]:


x.param_names


# In[4]:


x(torch.zeros(2))


# In[5]:


x({"A/sigma^2": torch.zeros(1), "B/sigma^2": torch.zeros(1)})


# In[6]:


import torch
from torch_openreml.covariance import Sum, ScalarMatrix

x = Sum(ScalarMatrix(2), ScalarMatrix(2))
free_params = torch.tensor([0.0, 0.5])

x.build_params(free_params)


# In[7]:


x.build_params()


# In[8]:


x.build_params(free_params, trans=False)


# In[9]:


x.build_params(free_params, out_format="dict")


# In[10]:


import torch
from torch_openreml.covariance import Sum, ScalarMatrix

x = Sum(ScalarMatrix(2), ScalarMatrix(2))
v_groups = x.build_operands(torch.tensor([1.0, 2.0]))
print(v_groups[0])
print(v_groups[1])


# In[11]:


import torch
from torch_openreml.covariance import Sum, ScalarMatrix

x = Sum(ScalarMatrix(2), ScalarMatrix(2))
grad_groups, grad_name_groups = x.operands_grad(torch.tensor([1.0, 2.0]))
print(grad_groups[0])
print(grad_groups[1])
print(grad_name_groups[0])
print(grad_name_groups[1])


# In[12]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(2)
free_params = torch.tensor([0.0, 0.5])
grad, grad_names = mat.auto_grad(free_params)
grad, grad_names

