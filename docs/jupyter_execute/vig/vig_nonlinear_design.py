#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import torch

from torch_openreml import MixedModelREML
from torch_openreml.covariance import ScalarMatrix, SimpleMatrix

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=3, sci_mode=False)
pd.set_option("display.precision", 3)

def simulate_nonlinear_design():
    n_subject = 40
    n_time = 10
    beta = torch.tensor([2.0, 0.4])
    sigma2_random = 0.8
    sigma2_residual = 0.3
    tau = 2.5

    generator = torch.Generator().manual_seed(2026)

    subject = torch.arange(n_subject).repeat_interleave(n_time)
    time = torch.arange(n_time).repeat(n_subject)

    u = torch.randn(n_subject, generator=generator)
    u = u * sigma2_random**0.5

    y = (
        beta[0]
        + beta[1] * time
        + u[subject] * torch.exp(-time / tau)
        + torch.randn(len(time), generator=generator)
        * sigma2_residual**0.5
    )

    return {
        "y": y,
        "subject": subject,
        "time": time,
        "u": u,
    }

sim = simulate_nonlinear_design()
data = pd.DataFrame(
    {
        "y": sim["y"],
        "subject": sim["subject"],
        "time": sim["time"],
    }
)

data.head()


# In[2]:


y = sim["y"]
subject = sim["subject"]
time = sim["time"]
n_subject = int(subject.max()) + 1

X = torch.stack([torch.ones_like(time), time], dim=1)

def decay_design(params):
    tau = params[0].exp()
    loadings = torch.exp(-time / tau)
    return torch.eye(n_subject)[subject] * loadings.unsqueeze(-1)

Z = SimpleMatrix(num_free_params=1, call=decay_design)

print("Z parameters:", Z.param_names)
print("Z shape:", Z(torch.tensor([0.0])).shape)


# In[3]:


model = MixedModelREML(random=(Z, ScalarMatrix(n_subject)))

free_names, theta_hat = model.optimize(y, X)

print("free parameters:", free_names)
print("converged:", model.is_converged())


# In[4]:


params = model.build_params(out_format="dict")

tau_hat = params["Z/theta_0"].exp()
sigma2_random_hat = params["G/sigma^2"]
sigma2_residual_hat = params["R/sigma^2"]

print("tau:", tau_hat)
print("subject variance:", sigma2_random_hat)
print("residual variance:", sigma2_residual_hat)


# In[5]:


print("fixed effects:", model.blue())
print("standard errors:", torch.sqrt(model.var_blue()))


# In[6]:


tree = model.tree()

print("tree paths:", list(tree))
print("design shape:", tree["Z"].shape)
print(tree["Z"][:3, :3])


# In[7]:


b_hat = model.blup_per_term()["random"]

print("BLUP shape:", b_hat.shape)
print(
    "BLUP correlation:",
    torch.corrcoef(torch.stack([sim["u"], b_hat]))[0, 1],
)

