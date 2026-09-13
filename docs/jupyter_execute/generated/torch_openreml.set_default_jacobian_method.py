#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch_openreml.config import (
    set_default_jacobian_method,
    get_default_jacobian_method,
)

set_default_jacobian_method("jacfwd")
get_default_jacobian_method()

