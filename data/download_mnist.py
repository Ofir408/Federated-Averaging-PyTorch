import torch

print(torch.cuda.get_device_name(0))

tensor = torch.randn((2,3))
cuda0 = torch.device('cuda:0')
tensor.to(cuda0)
tensor.split(split_size=5)