import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Available device",device)
tensor_on_gpu = torch.tensor([1, 2, 3]).to(device)
print(tensor_on_gpu * 5)