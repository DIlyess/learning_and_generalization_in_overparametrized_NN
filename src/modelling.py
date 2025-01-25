import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# ------ Fonctions principales -----
def train_network(
    model, x_train, y_train, lr=0.01, weight_decay=1e-4, epochs=1000, batch_size=50
):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    for epoch in tqdm(range(epochs)):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            logging.debug(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")


# NTK : here we are simply training a linear model on the feature map
def train_ntk_model(model, x_train, y_train, lr=0.01, weight_decay=1e-4, epochs=1000):
    model.eval()
    with torch.no_grad():
        feature_map = model(x_train)

    ntk_model = nn.Linear(feature_map.shape[1], 1)
    optimizer = optim.SGD(ntk_model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        output = ntk_model(feature_map)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            logging.debug(f"NTK Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")

    return ntk_model


def evaluate_model(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test).detach().numpy()
    test_error = np.mean((y_pred - y_test.numpy()) ** 2)
    return test_error


# ----- Fonctions secondaires -----
class TwoLayerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class ThreeLayerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(ThreeLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))


class ThreeLayerLastOnly(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(ThreeLayerLastOnly, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)
        self.relu = nn.ReLU()

        # Freeze first and second layers
        for param in self.fc1.parameters():
            param.requires_grad = False
        for param in self.fc2.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
        return self.fc3(x)