#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch_openreml
print(torch_openreml.example_data.john_alpha)


# In[2]:


import torch
import pandas as pd
from torch_openreml import MixedModelREML
from torch_openreml.covariance import DummyMatrix, IdentityMatrix, ScalarMatrix, KroneckerProduct, Augment
from torch_openreml.example_data import john_alpha


# In[3]:


rep = john_alpha["rep"]
block = john_alpha["block"]
gen = john_alpha["gen"]
n = len(john_alpha)

y = torch.tensor(john_alpha["yield"].values)
X = Augment(torch.ones(n, 1), DummyMatrix(rep, drop_first=True))()


# In[4]:


reml = MixedModelREML(
    genotype = (DummyMatrix(gen), ScalarMatrix(gen)),
    replicate_block = (DummyMatrix(rep, block), KroneckerProduct(
        G_rep = IdentityMatrix(rep),
        G_block = ScalarMatrix(block),
    )),
    R = ScalarMatrix(n)
)


# In[5]:


print(reml)


# In[6]:


free_param_names, free_params = reml.optimize(y, X, torch.zeros(3), verbose=1)
print(free_param_names)
print(free_params)
print(reml.build_params(free_params))


# In[7]:


beta_hat = reml.blue()
print(beta_hat)


# In[8]:


b_hat = reml.blup()
b_hat


# In[9]:


gen_blup = reml.blup_per_term()
print(gen_blup["genotype"])

