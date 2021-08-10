from torch import nn
from sysidentpy.neural_network import NARXNN
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class NARX(nn.Module):
        def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 10)
                self.lin2 = nn.Linear(10, 10)
                self.lin3 = nn.Linear(10, 1)
                self.tanh = nn.Tanh()

        def forward(self, xb):
                z = self.lin(xb)
                z = self.tanh(z)
                z = self.lin2(z)
                z = self.tanh(z)
                z = self.lin3(z)
                return z

narx_net = NARXNN(
        net=NARX(),
        ylag=2,
        xlag=2,
        loss_func='mse_loss',
        optimizer='Adam',
        epochs=50,
        verbose=False,
        optim_params={'betas': (0.9, 0.999), 'eps': 1e-05} # optional parameters of the optimizer
)


dataframe = 'E:/UniversityFiles/Control-20210210T123743Z-001/Control/homework/MainModel_Version1/Mainmodel_vnew/Motion_Control/model_six_train/dataframe1/dataframe.txt'

df = pd.read_csv(dataframe)
dataset = df.values

min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(dataset)

X = X_scale[:,2:3]
Y = X_scale[:,4]

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X, Y, test_size=0.05)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)
X_train = X_train.reshape(-1, 1)
Y_train = Y_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)

train_dl = narx_net.data_transform(X_train, Y_train)
valid_dl = narx_net.data_transform(X_test, Y_test)
narx_net.fit(train_dl, valid_dl)
yhat = narx_net.predict(X_test, Y_test)
ee, ex, extras, lam = narx_net.residuals(X_test, Y_test, yhat)
narx_net.plot_result(Y_test, yhat, ee, ex)