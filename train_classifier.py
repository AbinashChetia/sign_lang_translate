import pickle
import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np

DATA_LOC = 'prep_data/gen_data_2023_10_13_16_04_04.pickle'
model_prefix = DATA_LOC.split('/')[-1].split('_')[0]

data_dict = pickle.load(open(DATA_LOC, 'rb'))


data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
print('Training...')
clf.fit(X_train, y_train)

print('Testing...')
y_pred = clf.predict(X_test)
print(f'{accuracy_score(y_test, y_pred) * 100}% of samples classified correctly')

now = dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
f = open(f'trained_models/{model_prefix}_model_{now}.pickle', 'wb')
pickle.dump({'model': clf}, f)
f.close()
print('Model saved!')