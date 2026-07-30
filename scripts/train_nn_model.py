import pandas as pd
import numpy as np
import random
import pickle

from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

# Loading in full generated datasets, and splitting into testing and training
if False:
    with open('../data/all_cluster_predictions.pckl', 'rb') as fil:
        all_cluster_predictions = pickle.load(fil)

    with open('../data/all_cluster_data.pckl', 'rb') as fil:
        all_cluster_data = pickle.load(fil)

    all_cluster_data_predictions = pd.merge(all_cluster_data, all_cluster_predictions,
                                            on=['cluster_id', 'forecast_year'], how='inner')

    all_cluster_ids = list(set(all_cluster_data_predictions['cluster_id']))
    random.seed(12)
    random.shuffle(all_cluster_ids)
    cutoff = int(len(all_cluster_ids) * 0.2)  # 20% testing
    test_ids = all_cluster_ids[:cutoff]
    train_ids = all_cluster_ids[cutoff:]

    training_data = all_cluster_data_predictions[all_cluster_data_predictions['cluster_id'].isin(train_ids)]
    testing_data = all_cluster_data_predictions[all_cluster_data_predictions['cluster_id'].isin(test_ids)]

    with open('../data/training_data.pckl', 'wb') as fil:
        pickle.dump(training_data, fil)
    with open('../data/testing_data.pckl', 'wb') as fil:
        pickle.dump(testing_data, fil)

# Loading in testing and training data
with open('training_data.pckl','rb') as fil:
    training_data = pickle.load(fil)
with open('testing_data.pckl','rb') as fil:
    testing_data = pickle.load(fil)

def zscore(column):
    zcolumn = (column - column.mean()) / column.std()
    return zcolumn


def zcut(column):
    zcolumn = zscore(column)
    return zcolumn.clip(upper=4, lower=-4)

def zscore(column):
    zcolumn = (column - column.mean())/column.std()
    return zcolumn
def zcut(column):
    zcolumn = zscore(column)
    return zcolumn.clip(upper=4,lower=-4)

feature_list = [
    'L0',
    'L1',
    'L2',
    'L3',
    'L4',
    'L5',
    'L6',
    'L7',
    'L8',
    'L9',
    'L10',
    'L11',
    'L12',
    'L13',
    'L14',
    'L15',
    'I0',
    'I1',
    'D0',
    'D1',
    'D2',
    'D3',
    'F0',
    'F1',
    'F2',
    'F3',
    'F4',
    'F5',
    'F6',
    'F7',
    'F8',
    'F9',
    'F10',
    'F11',
    'N0',
    'N1',
    'N2',
    'N3',
    'N4',
    'N5',
    'N6',
    'P0',
    'C0',
    'C1',
    'C2',
    'CC0',
    'CC1',
    'CC2',
    'S0',
    'S1',
    'S2',
    'S3',
    'S4',
    'S5',
    'S6',
    'S7',
    'S8',
    'S9',
    'S10',
    'S11',
    'S12',
    'S13',
    'S14',
    'S15',
    'dS0',
    'dS1',
    'dS2',
    'dS3',
    'dS4',
    'dS5',
    'dS6',
    'dS7',
    'dS8',
    'dS9',
    'dS10',
    'dS11',
    'dS12',
    'dS13',
    'dS14',
    'CS0',
    'CS1',
    'CS2',
    'CS3',
    'CS4',
    'CS5',
    'CS6',
    'CS7',
    'CS8',
    'CS9',
    'CS10',
    'CS11',
    'CS12',
    'CS13',
    'CS14',
    'CS15',
    'dCS0',
    'dCS1',
    'dCS2',
    'dCS3',
    'dCS4',
    'dCS5',
    'dCS6',
    'dCS7',
    'dCS8',
    'dCS9',
    'dCS10',
    'dCS11',
    'dCS12',
    'dCS13',
    'dCS14',
    'RS0',
    'RS1',
    'RS2',
    'RS3',
    'RS4',
    'RS5',
    'RS6',
    'RS7',
    'RS8',
    'RS9',
    'RS10',
    'RS11',
    'RS12',
    'RS13',
    'RS14',
    'RS15',
    'dRS0',
    'dRS1',
    'dRS2',
    'dRS3',
    'dRS4',
    'dRS5',
    'dRS6',
    'dRS7',
    'dRS8',
    'dRS9',
    'dRS10',
    'dRS11',
    'dRS12',
    'dRS13',
    'dRS14',
]
transforms = {k: zcut for k in feature_list}

FY_max = 2022
growth_year = 'EG_3Yr'

train_set = [training_data[training_data['forecast_year'] == FY] for FY in range(2015, FY_max + 1,1)]
test_set = [testing_data[testing_data['forecast_year'] == FY] for FY in range(2015, FY_max + 1,1)]

x_test = pd.concat([pd.concat([transforms[k](x[k]) for k in transforms.keys()], axis=1) for x in test_set])
x_train = pd.concat([pd.concat([transforms[k](x[k]) for k in transforms.keys()], axis=1) for x in train_set])

y_test = pd.concat([x[[growth_year]] for x in test_set]).rename(columns={growth_year:'EG'})
y_train = pd.concat([x[[growth_year]] for x in train_set]).rename(columns={growth_year:'EG'})

print(f"Length of test data is: {len(x_test):,.0f}")
print(f"Length of train data is: {len(x_train):,.0f}")

# Oversampling testing data
os = RandomOverSampler(random_state=0)
os_datax, os_datay = os.fit_resample(x_train, y_train)

print(f"Length of oversampled training data is: {len(os_datax):,.0f}")
print(f"N extreme growth in oversampled training data is: {sum(os_datay['EG'] == 1):,.0f}")
print(f"N no extreme growth in oversampled training data is: {sum(os_datay['EG'] == 0):,.0f}")

trainset = [(torch.Tensor(feature), int(label[0])) for feature, label in zip(list(os_datax.values), list(os_datay.values))]
testset = [(torch.Tensor(feature), int(label[0])) for feature, label in zip(list(x_test.values), list(y_test.values))]
print(f"Length of trainset = {len(trainset):,.0f}")
print(f"Length of testset = {len(testset):,.0f}")


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        # Define fully connected layers (nn.Linear)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Apply ReLU activation after the first layer
        out = self.fc1(x)
        out = F.relu(out)
        # No activation function needed for the output layer here,
        # as the loss function (e.g., CrossEntropyLoss) will handle Softmax internally
        out = self.fc2(out)
        return out

# Model hyperparams:
n_classes = 2
n_features = len(transforms)
input_size = n_features
hidden_size = 128*2
num_classes = n_classes

# create the model
model = SimpleNN(input_size, hidden_size, num_classes)
# Loss function using cross entropy
loss_fn = nn.CrossEntropyLoss()
# optimize with stochastic gradient descent
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# set hyperparams and things
EPOCHS = 200
print_num = 10
batch_size = 16
n_workers = 0

# loaders for batching
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

# savefile and eval list
model_file = f'../ML_models/5yr/simpleNN_{len(transforms)}features_{hidden_size}hiddensize_{growth_year}_epoch'
training_stats = []


def train_one_epoch(epoch_number, model, print_num=10, EPOCHS=11):
    running_loss = 0.0

    print('.... Training Stage')
    model.train(True)  # Set model to train mode
    for i, data in tqdm(enumerate(trainloader)):  # Do batch stuffs
        inputs, labels = data  # get batch info
        optimizer.zero_grad()  # zero gradients between batches
        outputs = model(inputs)  # get predictions for this batch
        loss = loss_fn(outputs, labels)  # get the loss for this batch
        loss.backward()  # get gradients
        optimizer.step()  # adjust learning weights
        running_loss += loss.item()

    if (epoch_number % print_num == 0) or (epoch_number == EPOCHS) or (print_num == 1):
        # Calculate some evaluation stats if
        print('.... Evaluation Stage')
        model.eval()  # Set model to eval mode to not corrupt training
        with torch.no_grad():
            print('........ Evaluating Training')
            y_training = []
            y_pred_train = []
            for i, data in tqdm(enumerate(trainloader)):
                inputs, labels = data
                y_training.extend(list(labels))
                pred_tmp = F.softmax(model(inputs), dim=1)
                y_pred_train.extend([np.argmax(pred.detach().numpy()) for pred in pred_tmp])

            print('........ Evaluating Testing')
            y_testing = []
            y_pred_test = []
            for i, data in tqdm(enumerate(testloader)):
                inputs, labels = data
                y_testing.extend(list(labels))
                pred_tmp = F.softmax(model(inputs), dim=1)
                y_pred_test.extend([np.argmax(pred.detach().numpy()) for pred in pred_tmp])

            accuracy_test = accuracy_score(y_testing, y_pred_test)
            precision_test = precision_score(y_testing, y_pred_test)
            recall_test = recall_score(y_testing, y_pred_test)
            f1_test = f1_score(y_testing, y_pred_test)
            report_test = classification_report(y_testing, y_pred_test)

            accuracy_train = accuracy_score(y_training, y_pred_train)
            precision_train = precision_score(y_training, y_pred_train)
            recall_train = recall_score(y_training, y_pred_train)
            f1_train = f1_score(y_training, y_pred_train)
            report_train = classification_report(y_training, y_pred_train)

        dict_evals = {
            'epoch': epoch_number,
            'accuracy_test': accuracy_test,
            'precision_test': precision_test,
            'recall_test': recall_test,
            'f1_test': f1_test,
            'report_test': report_train,
            'accuracy_train': accuracy_train,
            'precision_train': precision_train,
            'recall_train': recall_train,
            'f1_train': f1_train,
            'report_train': report_train,
            'loss': running_loss,
            'length of test set': len(x_test),
            'length of training set': len(x_train),
            'length of oversampled training set': len(os_datax)
        }

        return dict_evals
    return {'epoch': epoch_number}

print(f'Running training and saving off info to: {model_file}')
for epoch in range(EPOCHS+1):
    print(f"Epoch {epoch}")

    #Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    dict_evals = train_one_epoch(epoch, model, print_num=print_num, EPOCHS=EPOCHS)

    #saving off epoch info
    print('Saving Model')
    training_stats.append(dict_evals)
    history = pd.DataFrame(training_stats).dropna()
    with open(model_file+'.pckl', 'wb') as fil:
        pickle.dump(history, fil)

    torch.save(model.state_dict(),model_file+str(epoch)+'.pt')
    print(f"Done with Epoch {epoch + 1}")