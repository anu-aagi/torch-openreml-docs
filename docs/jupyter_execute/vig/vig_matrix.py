#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch

from torch_openreml.covariance import ScalarMatrix

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=3, sci_mode=False)

mat = ScalarMatrix(3)

print("shape:", mat.shape)
print("parameters:", mat.param_names)
print("free parameters:", mat.free_param_names)
print("number of parameters:", mat.num_params)
print("number of free parameters:", mat.num_free_params)


# In[2]:


mat.param_specs["sigma^2"]


# In[3]:


mat()


# In[4]:


theta = torch.tensor([0.5])
mat(theta)


# In[5]:


print("transformed:", mat.build_params(theta))
print("untransformed:", mat.build_params(theta, trans=False))
print("named:", mat.build_params(theta, out_format="dict"))


# In[6]:


named_theta = {"sigma^2": torch.tensor([0.5])}

print("matrix:")
print(mat(named_theta))
print("named values:", mat.build_params(named_theta))

named_grad, _ = mat.grad(named_theta)

print("named gradient shape:", named_grad.shape)


# In[7]:


fixed = ScalarMatrix(3)
fixed.set_param_specs("sigma^2", fixed=True)

print("free parameters:", fixed.free_param_names)
print("built:", fixed.build_params(out_format="dict"))
print("without fixed:", fixed.build_params(include_fixed=False))
print("gradient:", fixed.grad())


# In[8]:


from torch_openreml.covariance import DiagonalMatrix
from torch_openreml.covariance.transform import TransformExp

diag = DiagonalMatrix(3)

diag.set_param_specs(
    "/",
    default=torch.tensor([0.0, 0.0, 0.5]),
    fixed=[True, False, True],
)
diag.set_param_specs("sigma^2_1", trans=TransformExp())

print("free parameters:", diag.free_param_names)
print("built:", diag.build_params(torch.tensor([0.5])))
print(
    "named:",
    diag.build_params(torch.tensor([0.5]), out_format="dict"),
)


# In[9]:


grad, names = mat.grad(theta)

print("shape:", grad.shape)
print("names:", names)
print("diagonal of the first Jacobian:", torch.diagonal(grad[0]))


# In[10]:


mat.grad_mode = "auto"
grad_auto, _ = mat.grad(theta)
mat.grad_mode = "default"

torch.allclose(grad, grad_auto)


# In[11]:


inverse = mat.manual_inverse(theta)

print(inverse)
print(
    "matches numerical inverse:",
    torch.allclose(inverse, torch.linalg.inv(mat(theta))),
)


# In[12]:


from torch_openreml.covariance import AR1Matrix

ar = AR1Matrix(4)

print("parameters:", ar.param_names)
print("free parameters:", ar.free_param_names)
print("default matrix:")
print(ar())


# In[13]:


ar_params = torch.tensor([0.0, 0.5])

print(ar.build_params(ar_params, out_format="dict"))
print(ar(ar_params))


# In[14]:


ar_grad, ar_names = ar.grad(ar_params)

print("shape:", ar_grad.shape)
print("names:", ar_names)


# In[15]:


from torch_openreml.covariance import DummyMatrix

design = DummyMatrix(["a", "b", "a", "c"])

print("shape:", design.shape)
print("columns:", design.colnames)
print("free parameters:", design.free_param_names)
print(design())


# In[16]:


print(design.grad())
print(design.transform(["a", "c", "z"]))


# In[17]:


from torch_openreml.covariance import KnownMatrix

known = KnownMatrix(torch.tensor([[1.0, 0.25], [0.25, 1.0]]))

print(known())
print(known.manual_inverse())
print(known.grad())


# In[18]:


from torch_openreml.covariance import SimpleMatrix

def equicorrelation(theta):
    rho = torch.sigmoid(theta[0])
    return (1 - rho) * torch.eye(3) + rho * torch.ones(3, 3)

custom = SimpleMatrix(1, equicorrelation)

print("parameters:", custom.param_names)
print(custom())


# In[19]:


custom_grad, _ = custom.grad()

print("gradient shape:", custom_grad.shape)

