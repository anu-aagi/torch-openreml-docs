#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch

from torch_openreml import MarginalREML
from torch_openreml.covariance import DummyMatrix, ScalarMatrix, CovariancePropagation, Sum

n, p = 50, 2

y = torch.randn(n)
X = torch.randn(n, p)

Z = DummyMatrix(["a", "b"] * 25)

V = Sum(
    CovariancePropagation(
        Z,
        ScalarMatrix(2),
    ),
    ScalarMatrix(n),
)

reml = MarginalREML(V)

theta_start = torch.zeros(V.num_free_params)

theta_hat, beta_hat, n_iter = reml.optimize(
    y,
    X,
    theta_start,
    verbose=2,
)


# In[2]:


theta_last = reml.get_theta(select="last")
theta_best = reml.get_theta(select="best")

beta_last = reml.get_beta(select="last")
beta_best = reml.get_beta(select="best")


# In[3]:


beta_hat = reml.blue(y, X, theta_hat)


# In[4]:


y_hat = reml.predict(
    y,
    X,
    theta_hat,
)


# In[5]:


e = reml.residual(
    y,
    X,
    theta_hat,
)


# In[6]:


loglik = reml.loglik(y, X, theta_hat)


# In[7]:


import torch

from torch_openreml import MarginalREML
from torch_openreml.utils import augment, n_distinct

from torch_openreml.covariance import (
    DummyMatrix,
    IdentityMatrix,
    ScalarMatrix,
    Sum,
    CovariancePropagation,
    KroneckerProduct,
)

from torch_openreml.example_data import john_alpha

# --- response ---
y = torch.tensor(john_alpha["yield"].values)

# --- fixed effects ---
X = augment(
    torch.ones(len(john_alpha), 1),
    DummyMatrix(john_alpha["rep"], drop_first=True)()
)

# --- random effect design matrices ---
Z_gen = DummyMatrix(john_alpha["gen"])
Z_rep_block = DummyMatrix(john_alpha["rep"], john_alpha["block"])

# --- covariance components ---
G_gen = ScalarMatrix(n_distinct(john_alpha["gen"]))
G_rep = IdentityMatrix(n_distinct(john_alpha["rep"]))
G_block = ScalarMatrix(n_distinct(john_alpha["block"]))

R = ScalarMatrix(len(john_alpha))

# --- marginal covariance ---
V = Sum(
    CovariancePropagation(Z_gen, G_gen),
    CovariancePropagation(
        Z_rep_block,
        KroneckerProduct(G_rep, G_block)
    ),
    R
)

# --- REML fit ---
reml = MarginalREML(V)

theta_start = torch.zeros(V.num_free_params)

theta_hat, beta_hat, n_iter = reml.optimize(
    y,
    X,
    theta_start,
    verbose=2,
)

# --- results ---
print("theta:", theta_hat)

print("variance components:", V.build_params(theta_hat))

print("fixed effects:", beta_hat)

print("loglik:", reml.loglik(y, X, theta_hat))


# In[8]:


scores = [
    torch.norm(s).item()
    for s in reml.history["score"]
]

logliks = [
    ll.item()
    for ll in reml.history["loglik"]
]

print(
    "Score norms:",
    [f"{s:.6f}" for s in scores],
)

print(
    "Log-likelihoods:",
    [f"{ll:.4f}" for ll in logliks],
)

