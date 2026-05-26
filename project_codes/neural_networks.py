## we referred to the class architecture from the assignments, but rewrote most of the code.

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import random
import numpy as np
import pandas as pd

import copy

from sklearn.metrics import f1_score, confusion_matrix

import seaborn as sns
import matplotlib.pylab as plt

from data_processing import ProcessedData

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)

BASE_DIR = Path(__file__).resolve().parent.parent
PATH = BASE_DIR / "final_report" / "plots"

BATCH_SIZE = [50, 100, 200]
HIDDEN_LAYER = [2, 5, 10]
HIDDEN_DIM = [31, 62]
LEARNING_RATE = [0.01, 0.001, 0.0001]
PENALTY = [0.01, 0.001, 0.0001]


class TorchDataset():
    """
    Convert data into tensors and creates mini-batch loaders.
    """

    def __init__(self, dataset_handler, batch_size):
        self.X_train = torch.from_numpy(dataset_handler.train_x).float()
        self.y_train = torch.from_numpy(dataset_handler.train_y).float().view(-1, 1)

        self.X_val = torch.from_numpy(dataset_handler.val_x).float()
        self.y_val = torch.from_numpy(dataset_handler.val_y).float().view(-1, 1)

        self.X_test = torch.from_numpy(dataset_handler.test_x).float()
        self.y_test = torch.from_numpy(dataset_handler.test_y).float().view(-1, 1)

        ## Pure SGD takes too long to train the model. Therefore, mini-batch training is introduced.
        self.train_loader = DataLoader(TensorDataset(self.X_train, self.y_train), batch_size=batch_size, shuffle=True)


class TorchNetwork(nn.Module):
    """
    Define the neural network architecture with configurable layers.
    """

    def __init__(self, num_features, hidden_layer, hidden_dim):
        super().__init__()
        layers = []
        ## Input Layer
        layers.append(nn.Linear(num_features, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(hidden_layer-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.n = nn.Sequential(*layers)

    def forward(self, x):
        return self.n(x)


class Model():
    """
    Combine network, optimizer, and loss function for training and evaluation.
    """
    
    def __init__(self, dataset, BATCH_SIZE, hidden_layer, hidden_dim, lr, penalty):
        self.torch_dataset = TorchDataset(dataset, BATCH_SIZE)
        self.model = TorchNetwork(self.torch_dataset.X_train.shape[1], hidden_layer, hidden_dim)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=penalty)
        self.pos_weight = torch.tensor([(self.torch_dataset.y_train == 0).sum()/(self.torch_dataset.y_train == 1).sum()]).float()
        self.loss_func = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def train(self, epochs=50, threshold=0.5):

        f1_history = []
        initial_loss = None

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for x, y in self.torch_dataset.train_loader:
                self.optim.zero_grad()
            
                output = self.model(x)
                batch_loss = self.loss_func(output, y)

                batch_loss.backward()
                self.optim.step()
                total_loss += batch_loss.item() * x.size(0)

            avg_loss = total_loss/len(self.torch_dataset.X_train)

            if epoch == 0 :
                initial_loss = avg_loss
            elif epoch > 5 and avg_loss > initial_loss:
                return []
    
            self.model.eval()
            with torch.no_grad():
                result = self.model(self.torch_dataset.X_val)
                y_hat = (torch.sigmoid(result) > threshold).float()
                val_f1 = f1_score(self.torch_dataset.y_val.numpy(), y_hat.numpy())
                f1_history.append(val_f1)

        return f1_history

    def test(self, threshold=0.5):
        self.model.eval()
        with torch.no_grad():
            result = self.model(self.torch_dataset.X_test)
            y_hat = (torch.sigmoid(result) > threshold).float()
            f1 = f1_score(self.torch_dataset.y_test.numpy(), y_hat.numpy())
            conf_matrix = confusion_matrix(self.torch_dataset.y_test.numpy(), y_hat.numpy())
            tn, fp, fn, tp = conf_matrix[0,0], conf_matrix[0,1], conf_matrix[1,0], conf_matrix[1,1]
        print("F1:", f1, "\n")
        print(conf_matrix)
        print("False negative rate:", fn / (fn + tp), "\n")
        print("False positive rate:", fp / (fp + tn), "\n")


def train_loop(dataset_handler):
    """
    Perform hyperparameter tuning and selects the best model based.
    """

    best_score = 0
    best_f1_mean = 0
    best_hyperparameter = None
    best_model = None
    results = []

    for bs in BATCH_SIZE:
        for hl in HIDDEN_LAYER:
            for hd in HIDDEN_DIM:
                for lr in LEARNING_RATE:
                    for p in PENALTY:
                        model = Model(dataset_handler, bs, hl, hd, lr, p)
                        f1_history = model.train(epochs=20)
                        result = {"batch_size": bs, "hidden_layer": hl,
                                  "hidden_dim": hd, "learning_rate": lr,
                                  "penalty": p}

                        if not f1_history or len(f1_history) < 7:
                            result["f1"] = 0
                            results.append(result)
                            continue

                        f1_std = np.std(f1_history)
                        if f1_std > 0.1:
                            result["f1"] = 0
                            results.append(result)
                            continue

                        current_f1_mean = np.mean(f1_history[-5:])
                        result["f1"] = current_f1_mean
                        results.append(result)
                        score = current_f1_mean - f1_std

                        if score > best_score:
                            best_score = score
                            best_f1_mean = current_f1_mean
                            best_hyperparameter = {"Batch size": bs, "Number of hidden layers": hl,
                                                   "Hidden dimension": hd, "Learning rate": lr, "Penalty rate": p}
                            best_model = copy.deepcopy(model.model.state_dict())
    return best_f1_mean, best_hyperparameter, best_model, results

def final_result(dataset_handler):
    """
    Run training, evaluation, and generates performance visualizations.
    """

    best_f1_mean, best_hyperparameter, best_model, results = train_loop(dataset_handler)

    final_model = Model(dataset_handler, best_hyperparameter["Batch size"],
                                         best_hyperparameter["Number of hidden layers"],
                                         best_hyperparameter["Hidden dimension"],
                                         best_hyperparameter["Learning rate"],
                                         best_hyperparameter["Penalty rate"])

    print("Best F1:", best_f1_mean)
    print("Best Model", best_hyperparameter)
    final_model.model.load_state_dict(best_model)
    final_model.test()

    df_results = pd.DataFrame(results)

    table = df_results.pivot_table(index="hidden_layer", columns="hidden_dim", values="f1", aggfunc="mean")
    plt.figure(figsize=(7, 4))
    sns.heatmap(table, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title("Mean F1 Score by Number of Hidden Layers and Hidden Dimensions")
    plt.ylabel("Number of Hidden Layers")
    plt.xlabel("Number of Hidden Dimension")
    plt.savefig(PATH / "heatmap_hl_hd")
    plt.close()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    for hl in HIDDEN_LAYER:
        model = Model(dataset_handler, best_hyperparameter["Batch size"],
                                       hl,
                                       best_hyperparameter["Hidden dimension"],
                                       best_hyperparameter["Learning rate"],
                                       best_hyperparameter["Penalty rate"])
        train_f1 = model.train()
        ax.plot(range(len(train_f1)), train_f1, label=f"Train F1 (Hidden Layer = {hl})")

    ax.set_ylim([0.4, 0.6]) 
    ax.legend(loc="lower right", fontsize=14)
    ax.set_title("F1 Score over Epochs by Hidden Layers")
    ax.set_ylabel("f1 score")
    ax.set_xlabel("epochs")
    plt.savefig(PATH / "f1_hidden_layers.png")
    plt.close()

def main():
    dataset_handler = ProcessedData(random_state=1234)
    final_result(dataset_handler)

if __name__ == "__main__":
    main()