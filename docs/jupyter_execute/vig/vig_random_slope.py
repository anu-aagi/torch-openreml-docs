#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import torch

from torch_openreml import MixedModelREML
from torch_openreml.covariance import (
    Augment,
    DummyMatrix,
    Gram,
    HadamardProduct,
    IdentityMatrix,
    KroneckerProduct,
    LowerTriangularMatrix,
    ScalarMatrix,
)

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=3, sci_mode=False)
pd.set_option("display.precision", 3)

generator = torch.Generator().manual_seed(2026)
n_subjects, n_days = 72, 8

subject = torch.arange(n_subjects).repeat_interleave(n_days)
day = torch.arange(n_days).repeat(n_subjects)

z_intercept = torch.randn(n_subjects, generator=generator)
z_slope = torch.randn(n_subjects, generator=generator)
b_intercept = 30.0 * z_intercept
b_slope = 10.0 * (-0.5 * z_intercept + 0.75**0.5 * z_slope)

y = (
    250.0
    + 10.0 * day
    + b_intercept[subject]
    + b_slope[subject] * day
    + 25.0 * torch.randn(len(day), generator=generator)
)

data = pd.DataFrame(
    {
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


D = DummyMatrix(data["subject"])
day_values = torch.tensor(data["day"].values).to(torch.float64).unsqueeze(-1)
Z_slope = HadamardProduct(group=D, x=day_values)

reml_slope = MixedModelREML(subject=(Z_slope, ScalarMatrix(data["subject"])))
reml_slope.optimize(y, X)

reml_slope.build_params(out_format="dict")


# In[4]:


Z_subject = Augment(
    intercept=D,
    slope=HadamardProduct(group=D, x=day_values),
)

G_subject = KroneckerProduct(
    covariance=Gram(
        LowerTriangularMatrix(2, 2),
        gram_type="xxt",
    ),
    subject=IdentityMatrix(n_subjects),
)

reml_subject = MixedModelREML(subject=(Z_subject, G_subject))
reml_subject.optimize(y, X)

reml_subject.build_params(out_format="dict")


# In[5]:


reml_subject.tree()["G/covariance"]


# In[6]:


reml_independent = MixedModelREML(
    subject_intercept=(D, ScalarMatrix(data["subject"])),
    subject_slope=(Z_slope, ScalarMatrix(data["subject"])),
)
reml_independent.optimize(y, X)

reml_independent.build_params(out_format="dict")


# In[7]:


print("converged:", reml_subject.is_converged())
print("loglik:", reml_subject.loglik()[0])
print("fixed effects:", reml_subject.blue())
print("fixed-effect SEs:", torch.sqrt(reml_subject.var_blue()))

blups = reml_subject.blup_per_term()["subject"]
print("first three subject BLUPs (intercept, slope):")
print(blups.reshape(2, n_subjects).T[:3])

