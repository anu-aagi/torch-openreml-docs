#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch

torch.set_default_dtype(torch.float64)

from torch_openreml import MixedModelREML
from torch_openreml.covariance import (
    DiagonalMatrix,
    DummyMatrix,
    Gram,
    IdentityMatrix,
    KroneckerProduct,
    LowerTriangularMatrix,
    ScalarMatrix,
    Sum,
)
from torch_openreml.example_data import fa_met, fa_met_params

fa_met.head()


# In[2]:


means = torch.tensor(fa_met_params["means"])
loadings = torch.tensor(fa_met_params["loadings"])
uniqueness = torch.tensor(fa_met_params["uniqueness"])
sigma2 = fa_met_params["sigma2"]

sigma_g = loadings @ loadings.T + torch.diag(uniqueness)
means, sigma2, sigma_g.round(decimals=4)


# In[3]:


cells = fa_met.drop_duplicates(["genotype", "environment"])
genetic = torch.tensor(cells["genetic"].to_numpy())

genotype_labels = fa_met["genotype"]
environment_labels = fa_met["environment"]

y = torch.tensor(fa_met["yield"].to_numpy())
X = DummyMatrix(environment_labels)()

y.shape, X.shape, len(genotype_labels)


# In[4]:


Z = DummyMatrix(genotype_labels, environment_labels)

k = 2

fa = Sum(
    factors=Gram(
        LowerTriangularMatrix(environment_labels, k),
        gram_type="xxt",
    ),
    uniqueness=DiagonalMatrix(environment_labels),
)

G = KroneckerProduct(
    genotype=IdentityMatrix(genotype_labels),
    environment=fa,
)

reml = MixedModelREML(genotype=(Z, G), R=ScalarMatrix(len(y)))


# In[5]:


names = reml.V.free_param_names
len(names), names


# In[6]:


names, theta_hat = reml.optimize(y, X, verbose=1)


# In[7]:


reml.is_converged(), reml.loglik()


# In[8]:


theta_hat.round(decimals=4)


# In[9]:


params = reml.build_params(out_format="dict")
params


# In[10]:


tree = reml.tree()

Sigma_hat = tree["G/environment"]
Lambda_hat = tree["G/environment/factors/op_0"]
Psi_hat = tree["G/environment/uniqueness"]

Sigma_hat.round(decimals=4)


# In[11]:


Lambda_hat.round(decimals=4)


# In[12]:


psi_hat = Psi_hat.diagonal()
uniqueness.round(decimals=4), psi_hat.round(decimals=4)


# In[13]:


sigma2_hat = params["R/sigma^2"]
sigma2_hat


# In[14]:


sign = torch.sign(torch.diagonal(Lambda_hat))
Lambda_aligned = Lambda_hat * sign

Lambda_aligned.round(decimals=4)


# In[15]:


loadings


# In[16]:


(Sigma_hat - sigma_g).abs().max()


# In[17]:


torch.corrcoef(torch.stack([Sigma_hat.flatten(), sigma_g.flatten()]))[0, 1]


# In[18]:


blup = reml.blup_per_term()["genotype"]

torch.corrcoef(torch.stack([genetic, blup]))[0, 1]


# In[19]:


(blup - genetic).pow(2).mean().sqrt()


# In[20]:


beta_hat = reml.blue()
se_beta = torch.sqrt(reml.var_blue())

beta_hat.round(decimals=4), se_beta.round(decimals=4)


# In[21]:


def fit_fa(k):
    fa = Sum(
        factors=Gram(
            LowerTriangularMatrix(environment_labels, k),
            gram_type="xxt",
        ),
        uniqueness=DiagonalMatrix(environment_labels),
    )
    G = KroneckerProduct(
        genotype=IdentityMatrix(genotype_labels),
        environment=fa,
    )
    model = MixedModelREML(genotype=(Z, G), R=ScalarMatrix(len(y)))
    names_k, _ = model.optimize(y, X)
    loglik_k = model.loglik()[0].item()
    p_k = len(names_k)
    return p_k, loglik_k, -2 * loglik_k + 2 * p_k

for k_k in (1, 2):
    p_k, loglik_k, aic_k = fit_fa(k_k)
    print(f"k = {k_k}: p = {p_k}, loglik = {loglik_k:.3f}, AIC = {aic_k:.3f}")


# In[22]:


env_levels = sorted(fa_met["environment"].unique())
geno_levels = sorted(fa_met["genotype"].unique())

generator = torch.Generator().manual_seed(20261120)
missing = {}
for genotype in geno_levels:
    picks = torch.randperm(len(env_levels), generator=generator)[:3]
    missing[genotype] = {env_levels[i] for i in picks.tolist()}

keep = [
    environment not in missing[genotype]
    for genotype, environment in zip(fa_met["genotype"], fa_met["environment"])
]
unbalanced = fa_met[keep].reset_index(drop=True)

unbalanced.head()


# In[23]:


unbalanced_cells = unbalanced.drop_duplicates(["genotype", "environment"])
len(unbalanced), len(unbalanced_cells), unbalanced_cells.groupby("environment").size()


# In[24]:


y_u = torch.tensor(unbalanced["yield"].to_numpy())
X_u = DummyMatrix(unbalanced["environment"], levels=[env_levels])()
Z_u = DummyMatrix(
    unbalanced["genotype"],
    unbalanced["environment"],
    levels=[geno_levels, env_levels],
)

Z_u().shape, (Z_u() == 0).all(dim=0).sum()


# In[25]:


fa_u = Sum(
    factors=Gram(LowerTriangularMatrix(env_levels, 2), gram_type="xxt"),
    uniqueness=DiagonalMatrix(env_levels),
)
G_u = KroneckerProduct(
    genotype=IdentityMatrix(geno_levels),
    environment=fa_u,
)

model_u = MixedModelREML(genotype=(Z_u, G_u), R=ScalarMatrix(len(y_u)))
len(model_u.V.free_param_names)


# In[26]:


names_u, theta_u = model_u.optimize(y_u, X_u)

model_u.is_converged(), model_u.loglik()


# In[27]:


sigma2_u = model_u.build_params(out_format="dict")["R/sigma^2"]
sigma2_u


# In[28]:


tree_u = model_u.tree()
Sigma_u = tree_u["G/environment"]
psi_u = tree_u["G/environment/uniqueness"].diagonal()


# In[29]:


torch.corrcoef(torch.stack([Sigma_u.flatten(), sigma_g.flatten()]))[0, 1]


# In[30]:


(Sigma_u - sigma_g).abs().max()


# In[31]:


psi_u.round(decimals=4), uniqueness.round(decimals=4)


# In[32]:


blup_u = model_u.blup_per_term()["genotype"]
seen = (Z_u() != 0).any(dim=0)
seen.sum()


# In[33]:


def corr(x, y):
    return torch.corrcoef(torch.stack([x, y]))[0, 1]

def rmse(x, y):
    return (x - y).pow(2).mean().sqrt()


# In[34]:


corr(genetic[seen], blup_u[seen])


# In[35]:


rmse(genetic[seen], blup_u[seen])


# In[36]:


corr(genetic[~seen], blup_u[~seen])


# In[37]:


rmse(genetic[~seen], blup_u[~seen])


# In[38]:


blup_u[~seen].std(), genetic[~seen].std()


# In[39]:


def fit_fa_u(k):
    fa = Sum(
        factors=Gram(
            LowerTriangularMatrix(env_levels, k),
            gram_type="xxt",
        ),
        uniqueness=DiagonalMatrix(env_levels),
    )
    G = KroneckerProduct(
        genotype=IdentityMatrix(geno_levels),
        environment=fa,
    )
    model = MixedModelREML(genotype=(Z_u, G), R=ScalarMatrix(len(y_u)))
    names_k, _ = model.optimize(y_u, X_u)
    loglik_k = model.loglik()[0].item()
    p_k = len(names_k)
    return p_k, loglik_k, -2 * loglik_k + 2 * p_k

p_1, loglik_1, aic_1 = fit_fa_u(1)
print(f"k = 1: p = {p_1}, loglik = {loglik_1:.3f}, AIC = {aic_1:.3f}")
print(f"k = 2: p = {len(names_u)}, loglik = {model_u.loglik()[0].item():.3f}, "
      f"AIC = {-2 * model_u.loglik()[0].item() + 2 * len(names_u):.3f}")

