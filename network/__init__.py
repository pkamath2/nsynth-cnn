from network.NSynthCNN import NSynthCNN
import torch 

def load_network(input_shape, num_classes, filters, kernels, padding, pooling, dense):
    return NSynthCNN(input_shape, num_classes, filters, kernels, padding, pooling, dense)
