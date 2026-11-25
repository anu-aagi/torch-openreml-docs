#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import torch

from torch_openreml import MixedModelREML
from torch_openreml.covariance import DummyMatrix, ScalarMatrix

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=3, sci_mode=False)
pd.set_option("display.precision", 3)

generator = torch.Generator().manual_seed(2026)
n_sites, n_per_site, n_days = 6, 6, 6
n_subjects = n_sites * n_per_site

site_of_subject = torch.arange(n_subjects) // n_per_site
subject = torch.arange(n_subjects).repeat_interleave(n_days)
site = site_of_subject[subject]
day = torch.arange(n_days).repeat(n_subjects)

b_site = 20.0 * torch.randn(n_sites, generator=generator)
b_subject = 25.0 * torch.randn(n_subjects, generator=generator)
y = (
    250.0
    + 10.0 * day
    + b_site[site]
    + b_subject[subject]
    + 25.0 * torch.randn(len(day), generator=generator)
)

data = pd.DataFrame(
    {
        "site": [f"site_{s:02d}" for s in site.tolist()],
        "subject": [f"subject_{i:02d}" for i in subject.tolist()],
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


# In[2]:


data.describe(include="all").T[
    ["count", "unique", "top", "freq", "mean", "std", "min", "max"]
]


# In[3]:


Z_subject = DummyMatrix(data["subject"])
G_subject = ScalarMatrix(data["subject"])

reml_subject = MixedModelREML(subject=(Z_subject, G_subject))
reml_subject.optimize(y, X)

reml_subject.build_params(out_format="dict")


# In[4]:


Z_site = DummyMatrix(data["site"])
G_site = ScalarMatrix(data["site"])

reml_site_subject = MixedModelREML(
    site=(Z_site, G_site),
    subject=(Z_subject, G_subject),
)
reml_site_subject.optimize(y, X)

reml_site_subject.build_params(out_format="dict")


# In[5]:


print("converged:", reml_site_subject.is_converged())
print("loglik:", reml_site_subject.loglik()[0])
print("fixed effects:", reml_site_subject.blue())
print("fixed-effect SEs:", torch.sqrt(reml_site_subject.var_blue()))

blups = reml_site_subject.blup_per_term()
print("site BLUPs:", blups["site"])
print("subject BLUPs:", blups["subject"])

