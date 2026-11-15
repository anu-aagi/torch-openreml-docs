#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import Sum, ScalarMatrix

x = Sum(ScalarMatrix(2), ScalarMatrix(2))
x


# In[2]:


x = Sum(A = ScalarMatrix(2), B = ScalarMatrix(2))
x


# In[3]:


x.param_names


# In[4]:


x(torch.zeros(2))


# In[5]:


x({"A/sigma^2": torch.zeros(1), "B/sigma^2": torch.zeros(1)})


# In[6]:


import torch
from torch_openreml.covariance import Sum, ScalarMatrix

x = Sum(ScalarMatrix(2), ScalarMatrix(2))
free_params = torch.tensor([0.0, 0.5])

x.build_params(free_params)


# In[7]:


x.build_params()


# In[8]:


x.build_params(free_params, trans=False)


# In[9]:


x.build_params(free_params, out_format="dict")


# In[10]:


import torch
from torch_openreml.covariance import Sum, ScalarMatrix

x = Sum(ScalarMatrix(2), ScalarMatrix(2))
v_groups = x.build_operands(torch.tensor([1.0, 2.0]))
print(v_groups[0])
print(v_groups[1])


# In[11]:


import torch
from torch_openreml.covariance import BlockDiagonal, DiagonalMatrix, ScalarMatrix, Sum

op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)),
         extra=ScalarMatrix(4))
nodes = op.walk(torch.tensor([0.0, 0.5, 1.0, 0.5]))


# In[12]:


[(path, operand, free_params) for path, operand, free_params in nodes]


# In[13]:


import torch
from torch_openreml.covariance import BlockDiagonal, DiagonalMatrix, ScalarMatrix, Sum

op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)),
         extra=ScalarMatrix(4))
results = op.tree(torch.tensor([0.0, 0.5, 1.0, 0.5]))


# In[14]:


results


# In[15]:


results["inner/op_0"]


# In[16]:


op.free_param_tree(torch.tensor([0.0, 0.5, 1.0, 0.5]))


# In[17]:


import torch
from torch_openreml.covariance import BlockDiagonal, DiagonalMatrix, ScalarMatrix, Sum

op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)),
         extra=ScalarMatrix(4))
free_params_by_path = op.free_param_tree(torch.tensor([0.0, 0.5, 1.0, 0.5]))


# In[18]:


free_params_by_path["inner"]


# In[19]:


import torch
from torch_openreml.covariance import BlockDiagonal, DiagonalMatrix, ScalarMatrix, Sum

op = Sum(inner=BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)),
         extra=ScalarMatrix(4))
grads = op.grad_tree(torch.tensor([0.0, 0.5, 1.0, 0.5]))


# In[20]:


grads["inner/op_0"]


# In[21]:


grads["inner"][1]


# In[22]:


import torch
from torch_openreml.covariance import Sum, ScalarMatrix

x = Sum(ScalarMatrix(2), ScalarMatrix(2))
grad_groups, grad_name_groups = x.operands_grad(torch.tensor([1.0, 2.0]))
print(grad_groups[0])
print(grad_groups[1])
print(grad_name_groups[0])
print(grad_name_groups[1])


# In[23]:


import torch
from torch_openreml.covariance import ScalarMatrix, Sum

x = Sum(ScalarMatrix(2), ScalarMatrix(2))
inverses = x.operands_manual_inverse(torch.tensor([1.0, 2.0]))
print(inverses[0])
print(inverses[1])

