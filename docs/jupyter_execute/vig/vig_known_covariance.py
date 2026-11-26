#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import torch

from torch_openreml import MixedModelREML
from torch_openreml.covariance import DummyMatrix, Product, ScalarMatrix

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=3, sci_mode=False)
pd.set_option("display.precision", 3)

n_sites = 6
site_index = torch.arange(n_sites)
A = 0.3 ** (site_index[:, None] - site_index[None, :]).abs()

A


# In[2]:


generator = torch.Generator().manual_seed(2026)
n_per_site = 20
n_obs = n_sites * n_per_site

site = torch.arange(n_sites).repeat_interleave(n_per_site)
day = torch.arange(n_per_site).repeat(n_sites)

L = torch.linalg.cholesky(A)
z = torch.randn(n_sites, generator=generator)
u = 10.0 * (L @ z)
e = 6.0 * torch.randn(n_obs, generator=generator)
y = 50.0 + 3.0 * day + u[site] + e

data = pd.DataFrame(
    {
        "site": [f"site_{s:02d}" for s in site.tolist()],
        "day": day.numpy(),
        "y": y.numpy(),
    }
)

y = torch.tensor(data["y"].values)
X = torch.stack(
    [
        torch.ones(len(data)),
        torch.tensor(data["day"].values, dtype=torch.float64),
    ],
    dim=1,
)
data.head()


# In[3]:


data.describe(include="all").T[
    ["count", "unique", "top", "freq", "mean", "std", "min", "max"]
]


# In[4]:


Z = DummyMatrix(data["site"])
G = Product(scale=ScalarMatrix(data["site"]), known=A)

reml = MixedModelREML(site=(Z, G))
reml.optimize(y, X)

reml.build_params(out_format="dict")


# In[5]:


reml.tree()["G"]


# In[6]:


print("converged:", reml.is_converged())
print("loglik:", reml.loglik()[0])
print("fixed effects:", reml.blue())
print("fixed-effect SEs:", torch.sqrt(reml.var_blue()))
print("first five site BLUPs:")
print(reml.blup_per_term()["site"][:5])

