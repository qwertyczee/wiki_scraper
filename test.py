import torch
print(torch.cuda.is_available())  # Mělo by to vrátit True
print(torch.cuda.get_device_name(0))  # Zobrazí název GPU
