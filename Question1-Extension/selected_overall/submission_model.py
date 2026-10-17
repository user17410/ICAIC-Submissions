from torch import nn

class Model(nn.Module):
    def __init__(self, input_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 8)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(8, 8)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.sigmoid(self.fc3(x))
