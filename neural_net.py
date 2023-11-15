import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

class NeuralNet(nn.Module):
    
    def __init__(self, n_in_feats: int, n_classes: int, hidden_config: list):
        super(NeuralNet, self).__init__()
        self.n_in_feats = n_in_feats
        self.n_classes = n_classes
        self.hidden_config = hidden_config
        self.n_hidden_layers = len(hidden_config)
        self.layers = nn.ModuleList()
        self.build_model()

    def build_model(self):
        self.layers.append(nn.Linear(self.n_in_feats, self.hidden_config[0]))
        for i in range(self.n_hidden_layers - 1):
            self.layers.append(nn.Linear(self.hidden_config[i], self.hidden_config[i+1]))
        self.layers.append(nn.Linear(self.hidden_config[-1], self.n_classes))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x
    
    def train(self, x, y, epochs=100, lr=0.01, iter_step=10):
        train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.2)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        train_loss = []
        val_loss = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.forward(train_x)
            loss = criterion(y_pred, train_y)
            loss.backward()
            optimizer.step()

            val_pred = self.forward(val_x)
            v_loss = criterion(val_pred, val_y)
            if epoch % iter_step == 0:
                print(f'Epoch {epoch: 5} | Train loss: {loss.item(): 4.4f} | Val loss: {v_loss.item(): 4.4f}')
            train_loss.append(loss.item())
            val_loss.append(v_loss.item())
        return train_loss, val_loss

    def predict(self, X):
        y_pred = self.forward(X)
        return torch.argmax(y_pred, dim=1)