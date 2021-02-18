import math
import torch.nn as nn
import torch.nn.functional as F


class NSynthCNN(nn.Module):
    def __init__(self, input_shape, num_classes, filters, kernels, padding, pooling, dense):
        super(NSynthCNN, self).__init__()
        self.input_shape = input_shape
        self.filters = filters
        self.kernels = kernels
        self.padding = padding
        self.pooling = pooling
        self.dense = dense
        
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=filters[0], kernel_size=kernels[0], padding=(padding[0],padding[0]))
        
        self.conv2 = nn.Conv2d(in_channels=filters[0], out_channels=filters[1], kernel_size=kernels[1], padding=(padding[1],padding[1])) #Same padding
        self.dropout2 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv2d(in_channels=filters[1], out_channels=filters[2], kernel_size=kernels[2], padding=(padding[2],padding[2])) #Same padding

        self.conv4 = nn.Conv2d(in_channels=filters[2], out_channels=filters[3], kernel_size=kernels[3], padding=(padding[3],padding[3])) #Same padding
        self.dropout4 = nn.Dropout(p=0.2)

        self.conv5 = nn.Conv2d(in_channels=filters[3], out_channels=filters[4], kernel_size=kernels[4], padding=(padding[4],padding[4])) #Same padding

        self.flatten = nn.Flatten(1)

        flat_shape = filters[4] * math.floor(input_shape[1]/(pooling[0] * pooling[1] * pooling[2] * pooling[3])) * math.floor(input_shape[2]/(pooling[0] * pooling[1] * pooling[2] * pooling[3]))
        print(flat_shape)
        self.fc1 = nn.Linear(int(flat_shape), dense[0])
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(dense[0], num_classes)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(self.pooling[0],self.pooling[0]))
        
        x = self.conv2(x)
        x = self.dropout2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(self.pooling[1],self.pooling[1]))
        
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(self.pooling[2],self.pooling[2]))

        x = self.conv4(x)
        x = self.dropout4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(self.pooling[3],self.pooling[3]))

        x = self.conv5(x)
        x = F.relu(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        
        return x