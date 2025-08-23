import torch
print(torch.version.cuda)       # Should print 11.x
print(torch.cuda.is_available()) # Should print True
print(torch.cuda.get_device_name(0))  # Should show RTX 4070 Ti