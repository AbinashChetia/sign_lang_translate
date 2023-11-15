import torch
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from neural_net import NeuralNet

import matplotlib.pyplot as plt


# initializing model directory

DATA_LOC = 'prep_data/asl_data_2023_11_15_15_30_00.pickle'
model_prefix = f'trained_models/{DATA_LOC.split("/")[-1].split("_")[0]}_model'

# loading labelled dataset

data_dict = pickle.load(open(DATA_LOC, 'rb'))

data = torch.tensor(data_dict['data'], device='mps')
labels = data_dict['labels']
lab_encdr = LabelEncoder()
labels = lab_encdr.fit_transform(labels)
labels = torch.tensor(labels, device='mps')

def nn_train_test_pipeline(data, labels):
    
    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # training Random Forest Classifier model
    nn = NeuralNet(n_in_feats=42, n_classes=29, hidden_config=[64, 128])
    nn.to('mps')
    print('\nTraining...')
    train_loss, val_loss = nn.train(X_train, y_train, epochs=1000, lr=0.01, iter_step=100)
    plt.plot(train_loss, label='train loss')
    plt.plot(val_loss, label='validation loss')
    plt.legend()
    plt.show()

    # testing the trained model

    print('\nTesting...')
    y_pred = nn.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

    return nn

print('\nModel for all signs:')
model = nn_train_test_pipeline(data=data, labels=labels)

print(model)