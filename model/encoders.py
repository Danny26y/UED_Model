import torch
import torch.nn as nn

class MagEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool2d((3, 3))

        self.fc1 = nn.Linear(32 * 3 * 3, 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu3(self.fc1(x)))
        return x

class ThermalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.relu4(self.fc2(x))
        return x

class GasEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=32, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.fc = nn.Linear(64, 64)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        # hn shape: (num_layers * num_directions, batch, hidden_size)
        # last hidden state from both directions:
        h_forward = hn[-2, :, :]
        h_backward = hn[-1, :, :]
        x = torch.cat((h_forward, h_backward), dim=1)
        x = self.relu(self.fc(x))
        return x
