import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.2)
        self.fc1 = nn.Linear(num_classes, num_classes*2)
        self.fc2 = nn.Linear(num_classes*2, num_classes*4)
        self.fc3 = nn.Linear(num_classes*4, num_classes*2)

        self.cnn1 = nn.Conv2d(in_channels = 1, out_channels = 3, kernel_size = 3, stride = 1, padding = 1)
        self.batch_norm1 = nn.BatchNorm2d(3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        self.cnn2 = nn.Conv2d(in_channels = 3, out_channels = 9, kernel_size = 3, stride = 1, padding = 1)
        self.batch_norm2 = nn.BatchNorm2d(9)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.cnn3 = nn.Conv2d(in_channels = 9, out_channels = 27, kernel_size = 3, stride = 1, padding = 1)
        self.batch_norm3 = nn.BatchNorm2d(27)

        self.cnn4 = nn.Conv2d(in_channels = 27, out_channels = 9, kernel_size = 3, stride = 1, padding = 1)
        self.batch_norm4 = nn.BatchNorm2d(9)

    def forward(self, input_): 
        out = self.cnn1(input_)
        out = self.batch_norm1(out) 
        out = self.cnn2(out)
        out = self.batch_norm2(out) 
        out = self.cnn3(out)
        out = self.batch_norm3(out)
        out = self.cnn4(out)
        out = self.batch_norm4(out)
        out = out.reshape(-1, self.num_classes)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out 

