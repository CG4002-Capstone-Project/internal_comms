import warnings

import numpy as np
import torch
import torch.nn as nn
from joblib import load

warnings.filterwarnings("ignore")

activities = ["hair", "listen", "sidepump", "dab", "wipe", "gun", "elbow", "pointhigh"]
n_labels = len(activities)


def scale_data(data, scaler, is_train=False):
    """
        data: inputs of shape (num_instances, num_features, num_time_steps)
        scaler: standard scalar to scale data
    """
    num_instances, num_time_steps, num_features = data.shape
    data = np.reshape(data, newshape=(-1, num_features))
    if is_train:
        data = scaler.fit_transform(data)
    else:
        data = scaler.transform(data)
    data = np.reshape(data, newshape=(num_instances, num_time_steps, num_features))
    return data


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(6, 32, 3)
        self.relu1 = nn.ReLU()
        self.dp1 = nn.Dropout(0.6)

        self.pool1 = nn.MaxPool1d(2)
        self.flat1 = nn.Flatten()
        self.dp2 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(928, 256)
        self.relu2 = nn.ReLU()
        self.dp3 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(256, 128)
        self.dp4 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(128, n_labels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dp1(x)

        x = self.pool1(x)
        x = self.flat1(x)
        x = self.dp2(x)

        x = self.fc1(x)
        x = self.relu2(x)
        x = self.dp3(x)

        x = self.fc2(x)
        x = self.dp4(x)

        x = self.fc3(x)

        return x


class Dataset(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        data = self.X[idx]
        target = self.y[idx][0]
        return data, target

    def __len__(self):
        return len(self.X)


def load_dataloader(X, y):
    dataset = Dataset(X, y)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=True, num_workers=4,
    )

    return dataloader


if __name__ == "__main__":
    model_path = "cnn_model.pth"
    scaler_path = "cnn_std_scaler.bin"
    inputs_path = "inputs.npy"
    labels_path = "labels.npy"

    # Load data
    X, y = np.load(inputs_path), np.load(labels_path)

    # Load model
    model = CNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load scaler
    scaler = load(scaler_path)

    # Prepare data
    dataloader = load_dataloader(X, y)

    # Run inference
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs = inputs.numpy()  # convert to numpy
        inputs = scale_data(inputs, scaler)  # scale features
        inputs = torch.tensor(inputs)  # convert to tensor

        outputs = model(inputs.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print("Accuracy: ", correct / total)
