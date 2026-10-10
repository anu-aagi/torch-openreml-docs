#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch_openreml
print(torch_openreml.example_data.john_alpha)


# In[2]:


import torch
import pandas as pd
from torch_openreml import MarginalREML, blup
from torch_openreml.covariance import DummyMatrix, IdentityMatrix, ScalarMatrix, Sum, CovariancePropagation, KroneckerProduct, Augment, BlockDiagonal
from torch_openreml.example_data import john_alpha


# In[3]:


rep = john_alpha["rep"]
block = john_alpha["block"]
gen = john_alpha["gen"]
n = len(john_alpha)

y = torch.tensor(john_alpha["yield"].values)
X = Augment(torch.ones(n, 1), DummyMatrix(rep, drop_first=True))()

V = Sum(
    random = CovariancePropagation(
        Z = Augment(
            Z_gen = DummyMatrix(gen),
            Z_rep_block = DummyMatrix(rep, block),
        ),
        G = BlockDiagonal(
            G_gen = ScalarMatrix(gen),
            G_rep_block = KroneckerProduct(
                G_rep = IdentityMatrix(rep),
                G_block = ScalarMatrix(block),
            ),
        ),
    ),
    residual = ScalarMatrix(n),
)

print(V)


# In[4]:


reml = MarginalREML(V)
theta_hat, beta_hat, n_iter = reml.optimize(y, X, torch.zeros(3), verbose=2)
print(theta_hat, V.build_params(theta_hat))
print(V.free_param_names)
print(beta_hat)


# In[5]:


tree, _ = V.tree(theta_hat)


# In[6]:


b_hat = blup(y, X, tree["random/Z"], tree["random/G"], tree["/"])
b_hat


# In[7]:


gen_blup = blup(y, X, tree["random/Z/Z_gen"], tree["random/G/G_gen"], tree["/"])
pd.Series(gen_blup, index=sorted(set(gen)))

