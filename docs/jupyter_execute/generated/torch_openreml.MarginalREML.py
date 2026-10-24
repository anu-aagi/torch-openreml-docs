#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml import MarginalREML
from torch_openreml.covariance import ScalarMatrix

n, p = 50, 2
y = torch.randn(n)
x = torch.randn(n, p)
init_values = torch.tensor([0.0])

mat = ScalarMatrix(n)
reml = MarginalREML(mat)
theta_hat, beta_hat, n_iter = reml.optimize(y, x, init_values, verbose=1)
theta_hat, beta_hat


# In[2]:


import torch
from torch_openreml import MarginalREML
from torch_openreml.covariance import ScalarMatrix

n = 50
y = torch.randn(n)
x = torch.randn(n, 2)

reml = MarginalREML(ScalarMatrix(n))
reml.optimize(y, x, torch.tensor([0.0]), verbose=0)
sorted(reml.tree())

