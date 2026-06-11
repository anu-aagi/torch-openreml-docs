#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import DiagonalMatrix, Adapter
from torch_openreml.covariance.transform import TransformIdentity

# Wrap a 2x2 diagonal matrix so that both variances share a
# single parameter (sum-to-one constraint).
adaptee = DiagonalMatrix(2)

def param_map(params):
    # params[0] drives one variance; the other is 1 - params[0]
    p = torch.sigmoid(params[0])
    return torch.stack([p, 1 - p])

param_specs = {
    "logit": {
        "fixed": False,
        "default": torch.tensor([0.0]),
        "trans": TransformIdentity(),
    }
}

adapter = Adapter(adaptee, param_specs, param_map)
adapter(torch.tensor([0.0]))

