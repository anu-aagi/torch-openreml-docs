#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch

from torch_openreml.covariance import (
    DiagonalMatrix,
    ScalarMatrix,
    Sum,
)

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=3, sci_mode=False)

op = Sum(
    genetic=ScalarMatrix(3),
    environment=DiagonalMatrix(3),
)
theta = torch.zeros(4)

print("names:", op.param_names)
print(op(theta))
print("shape:", op.shape)


# In[2]:


op.build_params(theta, out_format="dict")


# In[3]:


positional = Sum(ScalarMatrix(2), DiagonalMatrix(2))
print(positional.param_names)


# In[4]:


named = Sum({"left": ScalarMatrix(2), "right": DiagonalMatrix(2)})
print(named.param_names)


# In[5]:


fixed = Sum(
    signal=ScalarMatrix(2),
    baseline=torch.eye(2),
)

print("names:", fixed.param_names)
print(fixed(torch.tensor([0.0])))


# In[6]:


mixed = Sum(
    signal=ScalarMatrix(2),
    baseline=torch.eye(2, dtype=torch.float32),
)

print("default build:", mixed(torch.tensor([0.0])).dtype)
print(
    "float32 input:",
    mixed(torch.tensor([0.0], dtype=torch.float32)).dtype,
)


# In[7]:


from torch_openreml.covariance import BlockDiagonal

nested = Sum(
    inner=BlockDiagonal(
        DiagonalMatrix(2),
        ScalarMatrix(2),
    ),
    extra=ScalarMatrix(4),
)
theta = torch.tensor([0.0, 0.5, 1.0, 0.5])

for path, node, params in nested.walk(theta):
    print(f"{path:16} {type(node).__name__:22} {params}")


# In[8]:


results = nested.tree(theta)

print("paths:", list(results))
print(results["inner/op_0"])


# In[9]:


grad, names = op.grad(theta)

print("gradient shape:", grad.shape)
print("gradient names:", names)

