import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    
    def __init__(self, n_in_feats: int, n_hid_out_feats: int, hidden_config: list):
        super(NeuralNet, self).__init__()
        self.n_in_feats = n_in_feats
        self.n_hid_out_feats = n_hid_out_feats
        self.hidden_config = hidden_config
        self.n_hidden_layers = len(hidden_config)
        self.layers = nn.ModuleList()
        self.build_model()

    def build_model(self):
        self.layers.append(nn.Linear(self.n_in_feats, self.hidden_config[0]))
        for i in range(1, self.n_hidden_layers):
            self.layers.append(nn.Linear(self.hidden_config[i-1], self.hidden_config[i]))
        self.layers.append(nn.Linear(self.hidden_config[-1], self.n_out_feats))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = F.softmax(self.layers[-1](x))
        return x
    
    def predict(self, x):
        with torch.no_grad():
            x = torch.Tensor(x)
            return self.forward(x).argmax().item()
        
    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def save(self, path):
        torch.save(self.state_dict(), path)