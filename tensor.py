import torch
import numpy as np

python_list = [1, 2, 3, 4, 5]

np_array = np.array(python_list)

np_array_new = np_array * 2 + 5

tensor_from__list = torch.from_numpy(np_array_new)

print(np_array)
print(np_array_new)
print(tensor_from__list)
print(tensor_from__list.shape)
print(tensor_from__list.dtype)


tensor_from__list.add_(100)
print("-------------------------")


print(np_array)
print(np_array_new)
print(tensor_from__list)
print(tensor_from__list.shape)
print(tensor_from__list.dtype)
