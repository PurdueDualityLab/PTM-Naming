import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger

class DARA_classifier(nn.Module):
    def __init__(self, input_size=1036, output_size=32):
        super(DARA_classifier, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, 512)  # First hidden layer
        self.fc2 = nn.Linear(512, 256)         # Second hidden layer
        self.fc3 = nn.Linear(256, 128)         # Third hidden layer
        self.fc4 = nn.Linear(128, output_size) # Output layer
        self.downsample = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # logger.debug(x.shape)
        # logger.debug(self.input_size)
        x = F.relu(self.fc1(x))  # Activation function for hidden layer
        x = F.relu(self.fc2(x))  # Activation function for hidden layer
        x = self.dropout(x)      # Apply dropout
        x = F.relu(self.fc3(x))  # Activation function for hidden layer
        x = self.dropout(x)      # Apply dropout
        x = self.fc4(x)          # No activation function is applied to the output layer
        x = x.view(-1, self.output_size)
        # logger.debug(f"output shape: {x.shape}")
        return x

    def save_model(self, path='model.pth'):
        # Save the model state
        torch.save(self.state_dict(), path)

    def load_model(self, path='model.pth'):
        # Load the model state
        self.load_state_dict(torch.load(path))
        self.eval()  # Set the model to evaluation mode
