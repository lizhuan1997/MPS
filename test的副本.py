import numpy as np
import torch

a=torch.rand([3,3,3])
sa=a.size()
sb=(list(sa)+list([3]))
b=torch.rand(sb)
print b.size()