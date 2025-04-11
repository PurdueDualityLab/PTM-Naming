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
        self.downsample = nn.MaxPool1d(2)
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


class WiderFirstConvCNN_DARA(nn.Module):
    def __init__(self, input_size=4296, output_size=22):
        super(WiderFirstConvCNN_DARA, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.dropout_conv = nn.Dropout(0.3)

        # Calculate flattened size
        conv_output_size = 128 * (input_size // 2)
        print(f"conv_output_size: {conv_output_size}")

        self.fc1 = nn.Linear(conv_output_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc_out = nn.Linear(512, output_size)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout_conv(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc(x)
        x = self.fc_out(x)

        return x
    
class DeeperSmallerCNN_DARA(nn.Module):
    def __init__(self, input_size=4296, output_size=22):
        super(DeeperSmallerCNN_DARA, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)

        # Calculate flattened size
        conv_output_size = 128 * (input_size // 8) # Three pooling layers
        print(f"conv_output_size: {conv_output_size}")

        self.fc1 = nn.Linear(conv_output_size, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc_out = nn.Linear(256, output_size)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc(x)
        x = self.fc_out(x)

        return x
    
class KernelStrideCNN_DARA(nn.Module):
    def __init__(self, input_size=4296, output_size=22):
        super(KernelStrideCNN_DARA, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout_conv1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout_conv2 = nn.Dropout(0.25)

        # Calculate flattened size
        conv1_out_len = (input_size - 7 + 2 * 3) // 2 + 1
        conv2_out_len = (conv1_out_len - 5 + 2 * 2) // 2 + 1
        conv_output_size = 128 * conv2_out_len
        print(f"conv_output_size: {conv_output_size}")

        self.fc1 = nn.Linear(conv_output_size, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc_out = nn.Linear(256, output_size)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout_conv1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout_conv2(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc(x)
        x = self.fc_out(x)

        return x
    
class DeeperCNN_DARA(nn.Module):
    def __init__(self, input_size=4296, output_size=22):
        super(DeeperCNN_DARA, self).__init__()
        # CNN backbone layers
        self.input_size = input_size
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool = nn.MaxPool1d(2)  # Max pooling layer
        self.dropout = nn.Dropout(0.3)

        # Calculate flattened size
        # Four pooling layers (assuming one after each conv block)
        conv_output_size = 512 * (input_size // 16)
        print(f"conv_output_size: {conv_output_size}")

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 1024)  # First hidden layer
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)         # Second hidden layer
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.dropout_fc2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 256)         # Third hidden layer
        self.bn_fc3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, output_size) # Output layer
        self.downsample = nn.MaxPool1d(2) # Consider removing if already pooling
        self.dropout_final = nn.Dropout(0.3)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc(x)

        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc2(x)

        x = F.relu(self.fc3(x))
        x = self.fc4(x) # No activation on the output layer for CrossEntropyLoss
        # x = self.dropout_final(x) # Consider adding dropout to the output if needed

        return x