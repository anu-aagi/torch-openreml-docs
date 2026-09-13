#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch_openreml.config import (
    jacobian_method,
    get_default_jacobian_method,
    get_default_chunk_size,
)

with jacobian_method("jacrev", chunk_size=16):
    print(get_default_jacobian_method(), get_default_chunk_size())
print(get_default_jacobian_method(), get_default_chunk_size())

