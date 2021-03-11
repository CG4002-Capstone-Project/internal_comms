import warnings

import numpy as np
import torch
import torch.nn as nn
from joblib import load
from scipy import signal, stats

warnings.filterwarnings("ignore")

activities = ["dab", "gun", "elbow"]
n_labels = len(activities)


def scale_data(data, scaler, is_train=False):
    """
        data: inputs of shape (num_instances, num_features, num_time_steps)
        scaler: standard scalar to scale data
    """
    if is_train:
        data = scaler.fit_transform(data)
    else:
        data = scaler.transform(data)
    return data


def compute_mean(data):
    return np.mean(data)


def compute_variance(data):
    return np.var(data)


def compute_median_absolute_deviation(data):
    return stats.median_absolute_deviation(data)


def compute_root_mean_square(data):
    def compose(*fs):
        def wrapped(x):
            for f in fs[::-1]:
                x = f(x)
            return x

        return wrapped

    rms = compose(np.sqrt, np.mean, np.square)
    return rms(data)


def compute_interquartile_range(data):
    return stats.iqr(data)


def compute_percentile_75(data):
    return np.percentile(data, 75)


def compute_kurtosis(data):
    return stats.kurtosis(data)


def compute_min_max(data):
    return np.max(data) - np.min(data)


def compute_signal_magnitude_area(data):
    return np.sum(data) / len(data)


def compute_zero_crossing_rate(data):
    return ((data[:-1] * data[1:]) < 0).sum()


def compute_spectral_centroid(data):
    spectrum = np.abs(np.fft.rfft(data))
    normalized_spectrum = spectrum / np.sum(spectrum)
    normalized_frequencies = np.linspace(0, 1, len(spectrum))
    spectral_centroid = np.sum(normalized_frequencies * normalized_spectrum)
    return spectral_centroid


def compute_spectral_entropy(data):
    freqs, power_density = signal.welch(data)
    return stats.entropy(power_density)


def compute_spectral_energy(data):
    freqs, power_density = signal.welch(data)
    return np.sum(np.square(power_density))


def compute_principle_frequency(data):
    freqs, power_density = signal.welch(data)
    return freqs[np.argmax(np.square(power_density))]


def extract_raw_data_features_per_row(f_n):
    f1_mean = compute_mean(f_n)
    f1_var = compute_variance(f_n)
    f1_mad = compute_median_absolute_deviation(f_n)
    f1_rms = compute_root_mean_square(f_n)
    f1_iqr = compute_interquartile_range(f_n)
    f1_per75 = compute_percentile_75(f_n)
    f1_kurtosis = compute_kurtosis(f_n)
    f1_min_max = compute_min_max(f_n)
    f1_sma = compute_signal_magnitude_area(f_n)
    f1_zcr = compute_zero_crossing_rate(f_n)
    f1_sc = compute_spectral_centroid(f_n)
    f1_entropy = compute_spectral_entropy(f_n)
    f1_energy = compute_spectral_energy(f_n)
    f1_pfreq = compute_principle_frequency(f_n)
    return (
        f1_mean,
        f1_var,
        f1_mad,
        f1_rms,
        f1_iqr,
        f1_per75,
        f1_kurtosis,
        f1_min_max,
        f1_sma,
        f1_zcr,
        f1_sc,
        f1_entropy,
        f1_energy,
        f1_pfreq,
    )


def extract_raw_data_features(X, n_features=84):
    new_features = np.ones((X.shape[0], n_features))
    rows = X.shape[0]
    cols = X.shape[1]
    for row in range(rows):
        features = []
        for col in range(cols):
            f_n = X[row][col]
            feature = extract_raw_data_features_per_row(f_n)
            features.extend(feature)
        new_features[row] = np.array(features)
    return new_features


class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(84, 64)
        self.dp1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(64, 16)
        self.dp2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(16, n_labels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dp1(x)

        x = self.fc2(x)
        x = self.dp2(x)

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
    model_path = "dnn_model.pth"
    scaler_path = "dnn_std_scaler.bin"
    inputs_path = "inputs.npy"
    labels_path = "labels.npy"

    # Load data
    X, y = np.load(inputs_path), np.load(labels_path)

    # Load model
    model = DNN()
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
        inputs = extract_raw_data_features(inputs)  # extract features
        inputs = scale_data(inputs, scaler)  # scale features
        inputs = torch.tensor(inputs)  # convert to tensor

        outputs = model(inputs.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print("Accuracy: ", correct / total)
