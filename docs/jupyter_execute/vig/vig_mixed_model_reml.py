#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch

from torch_openreml import MixedModelREML
from torch_openreml.covariance import DummyMatrix, ScalarMatrix

n, p = 10, 2

group = ["a", "b"] * 5
effect = torch.tensor([1.0, -1.0]).repeat(n // 2)

X = torch.randn(n, p)
y = X @ torch.tensor([0.5, -0.5]) + effect + 0.5 * torch.randn(n)

reml = MixedModelREML(
    genotype=(DummyMatrix(group), ScalarMatrix(group)),
)

names, free_params = reml.optimize(y, X, verbose=1)
names, free_params


# In[2]:


history = reml.history
history["loglik"]


# In[3]:


torch.stack(history["loglik"])


# In[4]:


torch.stack(history["free_params"])


# In[5]:


names, theta_hat = reml.get_free_params("last")
names, theta_hat


# In[6]:


params = reml.build_params("last", out_format="dict")
params


# In[7]:


tree = reml.tree("last")
tree


# In[8]:


beta_hat = reml.blue(y, X, theta_hat)
beta_hat


# In[9]:


cov_beta = reml.cov_blue(X, theta_hat)
cov_beta


# In[10]:


se_beta = torch.sqrt(reml.var_blue(X, theta_hat))
se_beta


# In[11]:


b_hat = reml.blup()
b_hat


# In[12]:


b_cov = reml.cov_blup()
b_cov


# In[13]:


b_var = reml.var_blup()
b_var


# In[14]:


b_by_term = reml.blup_per_term()
b_by_term


# In[15]:


b_cov_by_term = reml.cov_blup_per_term()
b_cov_by_term


# In[16]:


b_var_by_term = reml.var_blup_per_term()
b_var_by_term


# In[17]:


y_marginal = reml.marginal_predict()
y_marginal


# In[18]:


y_conditional = reml.predict()
y_conditional


# In[19]:


y_new = reml.predict_new(
    X[:2],
    {"genotype": torch.tensor([[1.0, 0.0], [0.0, 1.0]])},
)
y_new


# In[20]:


z_new = DummyMatrix(group).transform(["a", "b"])

y_new = reml.predict_new(
    X[:2],
    z_new,
)
y_new


# In[21]:


e_marginal = reml.marginal_residual(y, X, theta_hat)
e_marginal


# In[22]:


e_conditional = reml.residual(y, X, theta_hat)
e_conditional


# In[23]:


c11, c12, c21, c22 = reml.c_inverse_blocks(X, theta_hat)
c11, c12, c21, c22


# In[24]:


loglik_value, loglik_method = reml.loglik(y, X, theta_hat)
loglik_value, loglik_method


# In[25]:


reml.is_converged()

