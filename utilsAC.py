import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

def plot_cf(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(50,50))
    ax = sn.heatmap(cm, annot=True, annot_kws={'size': 16}, square=True, cbar=False, fmt='g')
    plt.xlabel('Predicted') 
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def print_metrics(y_true, y_pred):
    print('Accuracy: ', accuracy_score(y_true, y_pred))
    print('Precision: ', precision_score(y_true, y_pred, average='macro', zero_division=0))
    print('Recall: ', recall_score(y_true, y_pred, average='macro'))
    print('F1: ', f1_score(y_true, y_pred, average='macro'))
    # print('Classification Report:')
    # print(classification_report(y_true, y_pred))