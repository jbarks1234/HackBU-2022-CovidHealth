from numpy import vstack
from numpy import sqrt
from pandas import read_csv
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn.init import xavier_uniform_
from sklearn.preprocessing import MinMaxScaler

# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = read_csv(path,
                      usecols=['STATEFIP', 'AGE', 'SEX', 'RACE', 'PAIDGH', 'HICHAMP',
                               'VETSTAT', 'NATIVITY', 'HISPAN', 'OCC2010', 'EMPSAME',
                               'UHRSWORK1', 'PROFCERT', 'EDUC99', 'PHINSUR', 'INCWAGE'])
        # store the inputs and outputs
        df = df[['STATEFIP', 'AGE', 'SEX', 'RACE', 'PAIDGH', 'HICHAMP',
                               'VETSTAT', 'NATIVITY', 'HISPAN', 'OCC2010', 'EMPSAME',
                               'UHRSWORK1', 'PROFCERT', 'EDUC99', 'PHINSUR', 'INCWAGE']]
        # x = df['INCWAGE'].values.reshape(-1, 1)
        # x_scaled = scaler.fit_transform(x)
        # df_temp = pd.DataFrame(x_scaled, columns='INCWAGE', index=df.index)
        # df['INCWAGE'] = df_temp
        x_scaled = scaler.fit_transform(df['INCWAGE'].values.reshape(-1, 1))
        df['INCWAGE'] = pd.DataFrame(x_scaled)

        self.X = df.values[:, :-1].astype('float32')
        self.y = df.values[:, -1].astype('float32')
        print(self.X)
        # ensure target has the right shape
        self.y = self.y.reshape((len(self.y), 1))
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


# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 10)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = Sigmoid()
        # second hidden layer
        self.hidden2 = Linear(10, 8)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = Sigmoid()
        # third hidden layer and output
        self.hidden3 = Linear(8, 1)
        xavier_uniform_(self.hidden3.weight)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        return X


# prepare the dataset
def prepare_data(path):
    # load the dataset
    dataset = CSVDataset(path)
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl


# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # enumerate epochs
    for epoch in range(20):
        print(epoch, " : epoch completed")
        running_loss = 0

        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
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

            running_loss += loss.item() * inputs.size(0)

        print("Train MSE loss: {:.4f}".format(running_loss / len(train_dl.dataset)))

# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate mse
    mse = mean_squared_error(actuals, predictions)
    return mse


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
scaler = MinMaxScaler()
path = './output_final_v3.csv'
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))
# define the network
model = MLP(15)
# train the model
# train_model(train_dl, model)
# # evaluate the model
# mse = evaluate_model(test_dl, model)
# torch.save(model.state_dict(), './f_model_v3.pt')
# print('MSE: %.3f, RMSE: %.3f' % (mse, sqrt(mse)))
# model.load_state_dict(torch.load('./f_model_v3.pt'))
#
# # make a single prediction (expect class=1)
# row = [34, 50.00, 2, 1, 0.0, 0.0, 0.0, 1, 0.0, 1010, 1, 40.0, 1, 16, 0.0]
# yhat = predict(row, model)
# print('Predicted: %.3f' % yhat)
# print(scaler.inverse_transform(yhat)[0][0])
# print('Predicted: %.3f' % yhat)
