import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0)) 