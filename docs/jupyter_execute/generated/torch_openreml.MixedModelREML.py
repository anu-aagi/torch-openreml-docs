#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml import MixedModelREML
from torch_openreml.covariance import ScalarMatrix, DummyMatrix

n, p = 100, 3
y = torch.randn(n)
x = torch.randn(n, p)
group = [f"g{i % 20}" for i in range(n)]
z = DummyMatrix(group)
reml = MixedModelREML((z, ScalarMatrix(20)), R=ScalarMatrix(n))
free_param_names, free_params_hat = reml.optimize(y, x, torch.tensor([0.0, 0.0]), verbose=1)


# In[2]:


import torch
from torch_openreml import MixedModelREML
from torch_openreml.covariance import DummyMatrix, ScalarMatrix

torch.manual_seed(0)
n = 30
y = torch.randn(n)
x = torch.randn(n, 2)
z = DummyMatrix([f"g{i % 5}" for i in range(n)])
reml = MixedModelREML((z, ScalarMatrix(5)), R=ScalarMatrix(n))
reml.blue(y, x, torch.zeros(2))


# In[3]:


import torch
from torch_openreml import MixedModelREML
from torch_openreml.covariance import DummyMatrix, ScalarMatrix

torch.manual_seed(0)
n = 30
x = torch.randn(n, 2)
z = DummyMatrix([f"g{i % 5}" for i in range(n)])
reml = MixedModelREML((z, ScalarMatrix(5)), R=ScalarMatrix(n))
reml.cov_blue(x, torch.zeros(2))


# In[4]:


import torch
from torch_openreml import MixedModelREML
from torch_openreml.covariance import DummyMatrix, ScalarMatrix

torch.manual_seed(0)
n = 30
x = torch.randn(n, 2)
z = DummyMatrix([f"g{i % 5}" for i in range(n)])
reml = MixedModelREML((z, ScalarMatrix(5)), R=ScalarMatrix(n))
reml.var_blue(x, torch.zeros(2))


# In[5]:


import torch
from torch_openreml import MixedModelREML
from torch_openreml.covariance import DummyMatrix, ScalarMatrix

torch.manual_seed(0)
n = 30
x = torch.randn(n, 2)
z = DummyMatrix([f"g{i % 5}" for i in range(n)])
reml = MixedModelREML((z, ScalarMatrix(5)), R=ScalarMatrix(n))
c11, c12, c21, c22 = reml.c_inverse_blocks(x, torch.zeros(2))
print(c11.shape, c12.shape, c21.shape, c22.shape)


# In[6]:


import torch
from torch_openreml import MixedModelREML
from torch_openreml.covariance import DummyMatrix, ScalarMatrix

torch.manual_seed(0)
n = 30
y = torch.randn(n)
x = torch.randn(n, 2)
z = DummyMatrix([f"g{i % 5}" for i in range(n)])
reml = MixedModelREML((z, ScalarMatrix(5)), R=ScalarMatrix(n))
reml.blup(y, x, torch.zeros(2))


# In[7]:


import torch
from torch_openreml import MixedModelREML
from torch_openreml.covariance import DummyMatrix, ScalarMatrix

torch.manual_seed(0)
n = 30
x = torch.randn(n, 2)
z = DummyMatrix([f"g{i % 5}" for i in range(n)])
reml = MixedModelREML((z, ScalarMatrix(5)), R=ScalarMatrix(n))
reml.cov_blup(x, torch.zeros(2))


# In[8]:


import torch
from torch_openreml import MixedModelREML
from torch_openreml.covariance import DummyMatrix, ScalarMatrix

torch.manual_seed(0)
n = 30
x = torch.randn(n, 2)
z = DummyMatrix([f"g{i % 5}" for i in range(n)])
reml = MixedModelREML((z, ScalarMatrix(5)), R=ScalarMatrix(n))
reml.var_blup(x, torch.zeros(2))


# In[9]:


import torch
from torch_openreml import MixedModelREML
from torch_openreml.covariance import DummyMatrix, ScalarMatrix

torch.manual_seed(0)
n = 30
y = torch.randn(n)
x = torch.randn(n, 2)
z1 = DummyMatrix([f"a{i % 5}" for i in range(n)])
z2 = DummyMatrix([f"b{i % 3}" for i in range(n)])
reml = MixedModelREML(
    (z1, ScalarMatrix(5)),
    (z2, ScalarMatrix(3)),
    R=ScalarMatrix(n),
)
reml.blup_per_term(y, x, torch.zeros(3))


# In[10]:


import torch
from torch_openreml import MixedModelREML
from torch_openreml.covariance import DummyMatrix, ScalarMatrix

torch.manual_seed(0)
n = 30
x = torch.randn(n, 2)
z1 = DummyMatrix([f"a{i % 5}" for i in range(n)])
z2 = DummyMatrix([f"b{i % 3}" for i in range(n)])
reml = MixedModelREML(
    (z1, ScalarMatrix(5)),
    (z2, ScalarMatrix(3)),
    R=ScalarMatrix(n),
)
reml.cov_blup_per_term(x, torch.zeros(3))


# In[11]:


import torch
from torch_openreml import MixedModelREML
from torch_openreml.covariance import DummyMatrix, ScalarMatrix

torch.manual_seed(0)
n = 30
x = torch.randn(n, 2)
z1 = DummyMatrix([f"a{i % 5}" for i in range(n)])
z2 = DummyMatrix([f"b{i % 3}" for i in range(n)])
reml = MixedModelREML(
    (z1, ScalarMatrix(5)),
    (z2, ScalarMatrix(3)),
    R=ScalarMatrix(n),
)
reml.var_blup_per_term(x, torch.zeros(3))


# In[12]:


import torch
from torch_openreml import MixedModelREML
from torch_openreml.covariance import DummyMatrix, ScalarMatrix

torch.manual_seed(0)
n = 30
y = torch.randn(n)
x = torch.randn(n, 2)
z = DummyMatrix([f"g{i % 5}" for i in range(n)])
reml = MixedModelREML((z, ScalarMatrix(5)), R=ScalarMatrix(n))
reml.predict(y, x, torch.zeros(2))


# In[13]:


import torch
from torch_openreml import MixedModelREML
from torch_openreml.covariance import DummyMatrix, ScalarMatrix

torch.manual_seed(0)
n = 30
y = torch.randn(n)
x = torch.randn(n, 2)
z = DummyMatrix([f"g{i % 5}" for i in range(n)])
reml = MixedModelREML((z, ScalarMatrix(5)), R=ScalarMatrix(n))
reml.optimize(y, x, torch.zeros(2))

x_new = torch.zeros(2, 2)
z_new = torch.zeros(2, 5)
z_new[0, 0] = 1.0
reml.predict_new(x_new, z_new)


# In[14]:


import torch
from torch_openreml import MixedModelREML
from torch_openreml.covariance import DummyMatrix, ScalarMatrix

torch.manual_seed(0)
n = 30
y = torch.randn(n)
x = torch.randn(n, 2)
z = DummyMatrix([f"g{i % 5}" for i in range(n)])
reml = MixedModelREML((z, ScalarMatrix(5)), R=ScalarMatrix(n))
reml.predict(y, x, torch.zeros(2))
reml.residual(y, x, torch.zeros(2))


# In[15]:


import torch
from torch_openreml import MixedModelREML
from torch_openreml.covariance import DummyMatrix, ScalarMatrix

torch.manual_seed(0)
n = 30
y = torch.randn(n)
x = torch.randn(n, 2)
z = DummyMatrix([f"g{i % 5}" for i in range(n)])
reml = MixedModelREML((z, ScalarMatrix(5)), R=ScalarMatrix(n))
reml.optimize(y, x, torch.zeros(2), max_iter=20)
names, values = reml.get_free_params("best")
names, values


# In[16]:


import torch
from torch_openreml import MixedModelREML
from torch_openreml.covariance import DummyMatrix, ScalarMatrix

torch.manual_seed(0)
n = 30
y = torch.randn(n)
x = torch.randn(n, 2)
z = DummyMatrix([f"g{i % 5}" for i in range(n)])
reml = MixedModelREML((z, ScalarMatrix(5)), R=ScalarMatrix(n))
raw = reml.build_params(torch.zeros(2), trans=False)
model_scale = reml.build_params(torch.zeros(2), trans=True)
raw, model_scale


# In[17]:


import torch
from torch_openreml import MixedModelREML
from torch_openreml.covariance import DummyMatrix, ScalarMatrix

torch.manual_seed(0)
n = 8
y = torch.randn(n)
x = torch.randn(n, 2)
z = DummyMatrix([f"g{i % 3}" for i in range(n)])
reml = MixedModelREML((z, ScalarMatrix(3)), R=ScalarMatrix(n))
reml.optimize(y, x, torch.zeros(2), max_iter=20)
reml.tree()

