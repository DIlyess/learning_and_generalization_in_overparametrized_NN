import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Target function to approximate
def target_function(x):
    return (
        np.sin(3 * x[:, 0]) + np.sin(3 * x[:, 1]) + np.sin(3 * x[:, 2]) - 2
    ) ** 2 * np.cos(7 * x[:, 3])


# random gaussian vector of dim d
def generate_data(N, d=4, seed=42):
    np.random.seed(seed)
    x = np.random.randn(N, d)  # Generate Gaussian features
    x /= np.linalg.norm(x, axis=1, keepdims=True)  # Normalize
    y = target_function(x)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(
        y, dtype=torch.float32
    ).view(-1, 1)


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


def train_network(
    model, x_train, y_train, lr=0.01, weight_decay=1e-4, epochs=1000, batch_size=50
):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")


# NTK : here we are simply training a linear model on the feature map
def train_ntk_model(model, x_train, y_train, lr=0.01, weight_decay=1e-4, epochs=1000):
    model.eval()
    with torch.no_grad():
        feature_map = model(x_train)

    ntk_model = nn.Linear(feature_map.shape[1], 1)
    optimizer = optim.SGD(ntk_model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = ntk_model(feature_map)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"NTK Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")

    return ntk_model


def evaluate_model(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test).detach().numpy()
    test_error = np.mean((y_pred - y_test.numpy()) ** 2)
    return test_error


def main_experiment(N=1000, hidden_dim1=200, hidden_dim2=100, epochs=1000):
    x_train, y_train = generate_data(N)
    x_test, y_test = generate_data(N=200)

    # 2-Layer
    model_two_layer = TwoLayerNN(input_dim=4, hidden_dim=hidden_dim1)
    print("\nTraining Two-Layer Network...")
    train_network(model_two_layer, x_train, y_train, epochs=epochs)
    error_two_layer = evaluate_model(model_two_layer, x_test, y_test)

    # 3-Layer (full)
    model_three_layer = ThreeLayerNN(
        input_dim=4, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2
    )
    print("\nTraining Three-Layer Network...")
    train_network(model_three_layer, x_train, y_train, epochs=epochs)
    error_three_layer = evaluate_model(model_three_layer, x_test, y_test)

    # 3-Layer ( last)
    model_three_layer_last = ThreeLayerLastOnly(
        input_dim=4, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2
    )
    print("\nTraining Three-Layer (Last Layer Only)...")
    train_network(model_three_layer_last, x_train, y_train, epochs=epochs)
    error_three_layer_last = evaluate_model(model_three_layer_last, x_test, y_test)

    # NTK
    # print("\nTraining NTK Approximation Model...")
    # ntk_model = train_ntk_model(model_three_layer, x_train, y_train, epochs=epochs)
    # error_ntk = evaluate_model(
    #     lambda x: ntk_model(model_three_layer(x)), x_test, y_test
    # )
    error_ntk = 0

    results = {
        "Two-Layer NN": error_two_layer,
        "Three-Layer NN": error_three_layer,
        "Three-Layer (Last Layer)": error_three_layer_last,
        "NTK Approximation": error_ntk,
    }

    for model, err in results.items():
        print(f"{model}: Test Error = {err:.4f}")

    plt.bar(results.keys(), results.values(), color=["blue", "green", "red", "purple"])
    plt.ylabel("Test Error")
    plt.title("Comparison of Model Performance")
    plt.show()


main_experiment()
