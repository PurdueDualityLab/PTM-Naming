import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger

class CNN_DARA(nn.Module):
    def __init__(self, input_size=4296, output_size=22):
        super(CNN_DARA, self).__init__()
        # CNN backbone layers
        self.input_size = input_size
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)  # Max pooling layer
        self.dropout = nn.Dropout(0.5) 
        conv_output_size = 128 * (input_size // 4)
        # print(f"conv_output_size: {conv_output_size}")
        self.fc1 = nn.Linear(conv_output_size, 1024)  # First hidden layer
        self.fc2 = nn.Linear(1024, 512)         # Second hidden layer
        self.fc3 = nn.Linear(512, 256)         # Third hidden layer
        self.fc4 = nn.Linear(256, output_size) # Output layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # logger.debug(x.shape)
        # logger.debug(self.input_size)
        # x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))  # Activation function for hidden layer
        x = self.dropout(x)      # Apply dropout
        x = F.relu(self.fc2(x))  # Activation function for hidden layer
        x = self.dropout(x)      # Apply dropout
        x = F.relu(self.fc3(x))  # Activation function for hidden layer
        x = self.fc4(x)          # No activation function is applied to the output layer
        # x = x.view(-1, self.output_size)
        # logger.debug(f"output shape: {x.shape}")
        return x

    def save_model(self, path='model.pth'):
        # Save the model state
        torch.save(self.state_dict(), path)

    def load_model(self, path='model.pth'):
        # Load the model state
        self.load_state_dict(torch.load(path))
        self.eval()  # Set the model to evaluation mode
