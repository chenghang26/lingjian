from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
import numpy as np
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, default=['./input_train.txt'])
parser.add_argument('--test_dir', type=str, default=['./input_test.txt'])
args = parser.parse_args()
train_filename = args.train_dir
test_filename = args.test_dir

train_data = []
train_label = []
test_data = []
with open(train_filename[0], 'r') as f1:
    lines = f1.readlines()
    for line in lines:
        line = list(line.strip().split())
        for item in line:
            item = item.split('_')
            s = []
            for i, val in enumerate(item):
                if i<2 or i==3:
                    s.append(val)
                elif i==2:
                    s.append(eval(val))
            train_label.append(item[4])
            train_data.append(s)

with open(test_filename[0], 'r') as f2:
    lines = f2.readlines()
    for line in lines:
        line = list(line.strip().split())
        for item in line:
            item = item.split('_')
            s = []
            for i, val in enumerate(item):
                if i<2 or i==3:
                    s.append(val)
                elif i==2:
                    s.append(eval(val))
            test_data.append(s)


train_data = pd.DataFrame(train_data, columns=['原材料','工人编号','环境','设备'])
train_data = pd.get_dummies(train_data)
test_data = pd.DataFrame(test_data, columns=['原材料','工人编号','环境','设备'])
test_data = pd.get_dummies(test_data)

train_data, test_data = train_data.align(test_data, join='inner', axis=1)
train_data = train_data.values
test_data = test_data.values

train_data = train_data.astype(np.float64)
test_data = test_data.astype(np.float64)
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

train_labels = []
for label in  train_label:
    if label=='material' :
        train_labels.append(0)
    elif label=='worker':
        train_labels.append(1)
    elif label=='environment':
        train_labels.append(2)
    elif label=='device':
        train_labels.append(3)
train_labels = to_categorical(train_labels)

model = Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4, activation='softmax'))

model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=20, batch_size=512)

y = model.predict_classes(test_data)
with open('output.txt', 'a') as f3:
    res = ''
    for i in y:
        if i==0:
            res += 'material '
        elif i==1:
            res += 'worker '
        elif i==2:
            res += 'environment '
        elif i==3:
            res += 'device '
    f3.write(res)