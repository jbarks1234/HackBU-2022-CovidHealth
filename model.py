from numpy import vstack
from numpy import argmax
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch import Tensor
from torchvision import models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import Module
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
import torch
import torch.nn as nn

import time
import copy

np.set_printoptions(threshold=40)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_radius, feature_extract=False, use_pretrained=False):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(20, 1)
        input_size = 20

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = pd.read_csv(path, usecols=['STATEFIP','METAREA','OWNERSHP','ASECWT','AGE','SEX','RACE','MARST','POPSTAT','ASIAN','VETSTAT','CITIZEN','HISPAN','NATIVITY', 'OCC2010','CLASSWKR','UHRSWORK1','PROFCERT','EDUC99','DIFFANY','INCWAGE'])
        # df[['PAIDGH','HIMCARELY','HIMCAIDLY','HICHAMP','PHINSUR', 'POPSTAT']] = df[['PAIDGH','HIMCARELY','HIMCAIDLY','HICHAMP','PHINSUR', 'POPSTAT']].fillna(value=df[['PAIDGH','HIMCARELY','HIMCAIDLY','HICHAMP','PHINSUR', 'POPSTAT']].mean())
        print(df.head(10))

        # df.to_csv('./output_final.csv')
        # store the inputs and outputs
        # self.X = df.drop("INCWAGE", axis=1).values()
        # print(self.X)
        # self.y = df.values['INCWAGE']
        # self.y = df.values[17:18]
        # print(self.y)
        # print("\n\n")
        # self.X = df.loc[:, df.columns != "INCWAGE"].values
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]
        # preprocessing.normalize(
        print(self.X)
        print("\n\n")
        print(self.y[0])
        # ensure input data is floats
        self.X = self.X.astype('float32')
        # label encode target and ensure the values are floats
        # min_max_scaler = preprocessing.MinMaxScaler()
        #
        # norm = min_max_scaler.fit_transform(self.y)
        # self.y = pd.DataFrame(norm)
        # self.y = LabelEncoder().fit_transform(self.y[0])

        print(self.y)

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])
    
def train_model2(model, trainloader, validloader, criterion, optimizer, scheduler, epochs, diff_lr=False, device='cuda'):
    """
    Train the model and run inference on the validation dataset. Capture the loss
    of the trained model and validation model. Also display the accuracy of the
    validation model
    :param model - a pretrain model object
    :param trainloader - a generator object representing the train dataset
    :param validloader - a generator object representing the validation dataset
    :param criterion - a loss object
    :param optimizer - an optimizer object
    :param scheduler - a scheduler object that varies the learning rate every n epochs
    :param epochs - an integer specifying the number of epochs to train the model
    :param diff_lr - a boolean specifying whether to use differential learning rate
    :param device - a string specifying whether to use cuda or cpu
    return a trained model with the best weights
    """
    start = time.time()
    print_every = 50
    steps = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    valid_loss_min = np.Inf
    training_loss, validation_loss = [], []
    for epoch in range(epochs):
        lr_used = 0
        if diff_lr:
            for param in optimizer.param_groups:
                if param['lr'] > lr_used:
                    lr_used = param['lr']
            print('learning rate being used {}'.format(lr_used))
        running_loss = 0
        # train_acc = 0
        # \scheduler.step()
        model.train()
        for idx, (inputs, target) in enumerate(trainloader):
            steps += 1
            inputs, target = inputs.type(torch.FloatTensor).to(device), target.type(torch.FloatTensor).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            print(inputs)
            print("\n\n")
            print(target)
            # forward pass and backward pass
            output = model(inputs[..., None, None].float())
            output = np.squeeze(output)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            # ps = torch.exp(output)
            # train_acc += (ps.max(dim=1)[1] == labels.data).type(torch.FloatTensor).mean()

        #            if steps % print_every == 0:
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            valid_loss = validation(model, validloader, criterion, device)

        # if test_accuracy > best_acc:
        if valid_loss < valid_loss_min:
            valid_loss_min = valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        print("Epoch: {}/{}... ".format(epoch + 1, epochs),
              "Train MSE loss: {:.4f}".format(running_loss / len(trainloader.dataset)),
              "Validation MSE loss: {:.4f}".format(valid_loss / len(validloader.dataset)),
              )
        # save the losses
        training_loss.append(running_loss / len(trainloader.dataset))
        validation_loss.append(valid_loss / len(validloader.dataset))
        running_loss = 0

    print('Best validation MSE loss is {:.4f}'.format(valid_loss_min / len(validloader.dataset)))
    print('Time to complete training {} minutes'.format((time.time() - start) / 60))
    model.load_state_dict(best_model_wts)
    return model, training_loss, validation_loss

def validation(model, validloader, criterion=None, device='cuda'):
    """
    Compute loss on the validation dataset
    :param model - a pretrained model object
    :param validloader - a generator object representing the validataion dataset
    :param criterion - a loss object
    :param device - a string specifying whether to use cuda or cpu
    return a tuple of loss and accuracy
    """
    valid_loss = 0
    model.eval()
    for images, target in validloader:
        images, target = images.to(device), target.to(device)
        output = model(images)

        output = np.squeeze(output)
        target = np.squeeze(target)
        valid_loss += criterion(output, target).item() * images.size(0)  # used to be target, output

    return valid_loss


# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 32)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(32, 16)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        self.hidden3 = Linear(16, 12)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()
        self.hidden4 = Linear(12, 8)
        kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        self.act4 = ReLU()
        # third hidden layer and output
        self.hidden5 = Linear(8, 3)
        xavier_uniform_(self.hidden5.weight)
        self.act5 = Softmax(dim=1)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # output layer
        X = self.hidden3(X)
        X = self.act3(X)
        X = self.hidden4(X)
        X = self.act4(X)
        X = self.hidden5(X)
        X = self.act5(X)
        return X

def optimizer(model, lr=0.001, weight_decay=1e-3 / 200):
    """
    Define the optimizer used to reduce the loss
    :param model - a pretrained model object
    :param lr - a floating point value defining the learning rate
    :param weight_decay - apply L2 regularization
    return an optimizer object
    """
    if model.__dict__['_modules'].get('fc', None):
        return optim.Adam(model.fc.parameters(), lr=lr, weight_decay=weight_decay)
    return optim.Adam(model.classifier.parameters(), lr=lr, weight_decay=weight_decay)


# prepare the dataset
def prepare_data(path):
    # load the dataset
    dataset = CSVDataset(path)
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=64, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl


# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # enumerate epochs
    for epoch in range(20):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            inputs, targets = inputs.type(torch.FloatTensor).to(device), targets.type(torch.FloatTensor).to(device)

            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()


# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        # convert to class labels
        yhat = argmax(yhat, axis=1)
        # reshape for stacking
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc


# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat


# prepare the data
if __name__ == '__main__':
    device = 'cuda'
    model_name = "resnet"
    num_inputs = 20
    num_classes = 1

    batch_size = 64
    num_epochs = 20
    lr = 0.0001
    path = './output_final.csv'
    train_dl, test_dl = prepare_data(path)
    print(len(train_dl.dataset), len(test_dl.dataset))
    # define the network
    model_ft = MLP(20)
    # train the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_ft = model_ft.to(device)
    # evaluate the model
    # acc = evaluate_model(test_dl, model)
    # model_ft, input_size = initialize_model(model_name, num_inputs, False, use_pretrained=False)

    train_model(train_dl, model_ft)
    acc = evaluate_model(test_dl, model_ft)
    # optim_ = optimizer(model_ft, lr)
    #
    # exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_, T_max=num_epochs, eta_min=0)
    #
    # # model_ft.load_state_dict(torch.load('alexnet_f_model.pt'))
    # #
    # # model_ft, training_loss, validation_loss = train_model2(model_ft, train_dl,
    #                                                         test_dl, criterion,
    #                                                         optim_, exp_lr_scheduler, num_epochs, False, device='cuda')

    torch.save(model_ft.state_dict(), './f_model.pt')


    print('Accuracy: %.3f' % acc)
    # make a single prediction
    row = [5.1, 3.5, 1.4, 0.2]
    yhat = predict(row, model_ft)
    print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))