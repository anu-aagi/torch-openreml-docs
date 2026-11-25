#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=4, sci_mode=False)

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

Sigma_g = loadings @ loadings.T + torch.diag(uniqueness)
print(f"{means=}")
print(f"{loadings=}")
print(f"{sigma2=}")
print(f"{Sigma_g=}")


# In[3]:


genotype = fa_met["genotype"]
environment = fa_met["environment"]

y = torch.tensor(fa_met["yield"].values)
X = DummyMatrix(environment)()


# In[4]:


Z = DummyMatrix(genotype, environment)

k = 2

G = KroneckerProduct(
    genotype=IdentityMatrix(genotype),
    environment=Sum(
        factors=Gram(
            loadings=LowerTriangularMatrix(environment, k),
            gram_type="xxt",
        ),
        uniqueness=DiagonalMatrix(environment),
    ),
)

reml = MixedModelREML(g_by_e=(Z, G), R=ScalarMatrix(len(y)))


# In[5]:


names = reml.V.free_param_names
names


# In[6]:


names, theta_hat = reml.optimize(y, X, verbose=1)


# In[7]:


theta_hat


# In[8]:


params = reml.build_params(out_format="dict")
params


# In[9]:


tree = reml.tree()

Sigma_hat = tree["G/environment"]
Lambda_hat = tree["G/environment/factors/loadings"]
Psi_hat = tree["G/environment/uniqueness"]


# In[10]:


Psi_hat


# In[11]:


sign = torch.sign(torch.diagonal(Lambda_hat))
Lambda_aligned = Lambda_hat * sign

Lambda_aligned


# In[12]:


loadings


# In[13]:


(Sigma_hat - Sigma_g).abs().max()


# In[14]:


torch.corrcoef(torch.stack([Sigma_hat.flatten(), Sigma_g.flatten()]))[0, 1]


# In[15]:


genetic = torch.tensor(fa_met.drop_duplicates(["genotype", "environment"])["genetic"].values)

blup = reml.blup_per_term()["g_by_e"]

torch.corrcoef(torch.stack([genetic, blup]))[0, 1]


# In[16]:


(blup - genetic).pow(2).mean().sqrt()


# In[17]:


beta_hat = reml.blue()
se_beta = torch.sqrt(reml.var_blue())

beta_hat, se_beta


# In[18]:


def fit_fa(k):
    G = KroneckerProduct(
        genotype=IdentityMatrix(genotype),
        environment=Sum(
            factors=Gram(
                LowerTriangularMatrix(environment, k),
                gram_type="xxt",
            ),
            uniqueness=DiagonalMatrix(environment),
        ),
    )
    model = MixedModelREML(g_by_e=(Z, G), R=ScalarMatrix(len(y)))
    names_k, _ = model.optimize(y, X, armijo_trials=5, fallback_eta=0.1)
    loglik_k = model.loglik()[0].item()
    p_k = len(names_k)
    return p_k, loglik_k, -2 * loglik_k + 2 * p_k

for k_k in (1, 2, 3):
    p_k, loglik_k, aic_k = fit_fa(k_k)
    print(f"k = {k_k}: p = {p_k}, loglik = {loglik_k:.3f}, AIC = {aic_k:.3f}")


# In[19]:


env_levels = sorted(fa_met["environment"].unique())
gen_levels = sorted(fa_met["genotype"].unique())

generator = torch.Generator().manual_seed(20261120)
missing = {}
for genotype in gen_levels:
    picks = torch.randperm(len(env_levels), generator=generator)[:3]
    missing[genotype] = {env_levels[i] for i in picks.tolist()}

keep = [
    environment not in missing[genotype]
    for genotype, environment in zip(fa_met["genotype"], fa_met["environment"])
]
unbalanced = fa_met[keep].reset_index(drop=True)

unbalanced.head()


# In[20]:


unbalanced.drop_duplicates(["genotype", "environment"]).groupby("environment").size()


# In[21]:


y_u = torch.tensor(unbalanced["yield"].values)
X_u = DummyMatrix(unbalanced["environment"], levels=[env_levels])()
Z_u = DummyMatrix(unbalanced["genotype"], unbalanced["environment"])


# In[22]:


G_u = KroneckerProduct(
    genotype=IdentityMatrix(gen_levels),
    environment=Sum(
        factors=Gram(LowerTriangularMatrix(env_levels, 2), gram_type="xxt"),
        uniqueness=DiagonalMatrix(env_levels),
    ),
)

model_u = MixedModelREML(genotype=(Z_u, G_u), R=ScalarMatrix(len(y_u)))


# In[23]:


names_u, theta_u = model_u.optimize(y_u, X_u, verbose=1)


# In[24]:


model_u.build_params(out_format="dict")["R/sigma^2"]


# In[25]:


tree_u = model_u.tree()
Sigma_u = tree_u["G/environment"]
Psi_u = tree_u["G/environment/uniqueness"]


# In[26]:


Psi_u


# In[27]:


torch.corrcoef(torch.stack([Sigma_u.flatten(), Sigma_g.flatten()]))[0, 1]


# In[28]:


(Sigma_u - Sigma_g).abs().max()


# In[29]:


blup_u = model_u.blup_per_term()["genotype"]
seen = (Z_u() != 0).any(dim=0)


# In[30]:


def corr(x, y):
    return torch.corrcoef(torch.stack([x, y]))[0, 1]

def rmse(x, y):
    return (x - y).pow(2).mean().sqrt()


# In[31]:


corr(genetic[seen], blup_u[seen])


# In[32]:


rmse(genetic[seen], blup_u[seen])


# In[33]:


corr(genetic[~seen], blup_u[~seen])


# In[34]:


rmse(genetic[~seen], blup_u[~seen])


# In[35]:


blup_u[~seen].std(), genetic[~seen].std()


# In[36]:


def fit_fa_u(k):
    G = KroneckerProduct(
        genotype=IdentityMatrix(gen_levels),
        environment=Sum(
            factors=Gram(
                LowerTriangularMatrix(env_levels, k),
                gram_type="xxt",
            ),
            uniqueness=DiagonalMatrix(env_levels),
        ),
    )
    model = MixedModelREML(genotype=(Z_u, G), R=ScalarMatrix(len(y_u)))
    names_k, _ = model.optimize(y_u, X_u, armijo_trials=5, fallback_eta=0.1)
    loglik_k = model.loglik()[0].item()
    p_k = len(names_k)
    return p_k, loglik_k, -2 * loglik_k + 2 * p_k

for k_k in (1, 2, 3):
    p_k, loglik_k, aic_k = fit_fa_u(k_k)
    print(f"k = {k_k}: p = {p_k}, loglik = {loglik_k:.3f}, AIC = {aic_k:.3f}")

