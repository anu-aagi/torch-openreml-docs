#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import OperatorGram, LowerTriangularMatrix

x = LowerTriangularMatrix(3, 2)
op = OperatorGram(x, gram_type="xtx")
op()


# In[2]:


op_xxt = OperatorGram(x, gram_type="xxt")
op_xxt()


# In[3]:


import torch
from torch_openreml.covariance import OperatorGram, LowerTriangularMatrix

x = LowerTriangularMatrix(3, 2)
op = OperatorGram(x, gram_type="xtx")
free_params = torch.tensor([0.0, 0.5, 1.0, 0.2, -0.3])
grad, grad_names = op.manual_grad(free_params)
grad


# In[4]:


grad_names


# In[5]:


op_xxt = OperatorGram(x, gram_type="xxt")
grad_xxt, grad_names_xxt  = op_xxt.manual_grad(free_params)
grad_xxt

