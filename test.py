import numpy as np
import torch
from tensor_method import contra,svd_update
import time
a=torch.rand([10,10],dtype=torch.float32)
t1=time.time()
s,v,d=torch.svd(a)
t2=time.time()
s,v,d=torch.svd(a)
t3=time.time()
print t2-t1,t3-t2