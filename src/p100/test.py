print("hello_world")

import torch
print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0))

import tensorflow as tf
print("GPU available:", tf.config.list_physical_devices('GPU'))
