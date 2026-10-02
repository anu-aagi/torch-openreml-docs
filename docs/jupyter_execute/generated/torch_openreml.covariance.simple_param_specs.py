#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch_openreml.covariance.param import simple_param_specs

simple_param_specs(3)


# In[2]:


import torch

simple_param_specs(2, default=torch.tensor([1.5]))

