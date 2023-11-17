from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint

wandb_logger = WandbLogger(log_model="all", project="traiding_lstm_01")
# wandb_logger.define_metric('val_loss', summary='min')
# wandb_logger.define_metric('train_loss', summary='min')

trainer = Trainer(logger=wandb_logger)


import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tqdm



# Here we define our model as a class
# class LSTM(nn.Module):
    
class LSTM(LightningModule):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.save_hyperparameters()


        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out
    
    # Prepare data for training
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length:i+seq_length+1]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)
df = pd.read_csv('../data/MNQ DEC23.Last-500-Volume-Action.txt')
print(df.head())


# Preprocess the data
# scaler = StandardScaler()
scaler = MinMaxScaler(feature_range=(-1, 1))
prices_scaled = scaler.fit_transform(df[['close']].values)


seq_length = 59
X, y = create_sequences(prices_scaled, seq_length)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)

# AAAAAAAAAAAAAAAAAAAA fcking shape!
y = torch.tensor(y, dtype=torch.float32).reshape((-1,1))


test_set_size = int(np.round(0.2*X.shape[0]));
train_set_size = X.shape[0] - (test_set_size);


X_train = X[:train_set_size]
y_train = y[:train_set_size]
X_test = X[train_set_size:]
y_test = y[train_set_size:]


print('X_train.shape = ',X_train.shape)
print('y_train.shape = ',y_train.shape)
print('X_test.shape = ',X_test.shape)
print('y_test.shape = ',y_test.shape)


# add one parameter
wandb_logger.experiment.config["data_shape"] = X.shape

# add multiple parameters
wandb_logger.experiment.config.update({
                                        "X_train":X_train.shape,
                                        "y_train":y_train.shape,
                                        "X_test":X_test.shape,
                                        "y_test":y_test.shape})




# Configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)



# Hyperparameters
input_size = 1 # number of features
hidden_size = 40 #
num_layers = 3
output_size = 1
num_epochs = 10
learning_rate = 0.01
batch_size = 128
# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)


losses = []
losses_after = []
# Training loop
i = 0
step = 0
# N split cross validation
for idx, (train_index, test_index) in enumerate(tqdm.tqdm(tscv.split(X_train))):    
    model = LSTM(input_dim=input_size, 
             hidden_dim=hidden_size,
             output_dim=output_size,
             num_layers=num_layers).to(device)
    wandb_logger.watch(model)
    #     model = LSTM(input_size=input_size,
    #                  hidden_size=hidden_size,
    #                  num_layers=num_layers,
    #                  output_size=output_size).to(device)
    #      input_dim, hidden_dim, num_layers, output_dim
    criterion = torch.nn.MSELoss()

    # https://discuss.pytorch.org/t/how-to-implement-weighted-mean-square-error/2547/6
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    print(f'Train Idx [{min(train_index)}:{max(train_index)}] ')
    print(f'Test  Idx [{min(test_index)}:{max(test_index)}]')
    fold_X_train, fold_X_test = X_train[train_index], X_train[test_index]
    fold_y_train, fold_y_test = y_train[train_index], y_train[test_index]

    # Convert to DataLoader for easy batching
    fold_train_data   = torch.utils.data.TensorDataset(fold_X_train, fold_y_train)
    fold_train_loader = torch.utils.data.DataLoader(dataset=fold_train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    
    for epoch in tqdm.tqdm_notebook(range(num_epochs), leave=False):
        for fold_batch_X, fold_batch_y in fold_train_loader:
            step+=1
            i+=1
            fold_batch_X, fold_batch_y = fold_batch_X.to(device), fold_batch_y.to(device)

            # Backward and optimize
            # Forward pass
            fold_batch_X_pred = model(fold_batch_X)
#             if i%10==0:
#                 print(fold_batch_X_pred.reshape(-1).cpu().detach().numpy()[:10])
            loss = criterion(fold_batch_X_pred, fold_batch_y)
            losses.append([loss.item(), 0, idx])
            wandb_logger.log("train_loss",loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

#         if (epoch+1) % 50 == 0:
#             print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        fold_val_test_data   = torch.utils.data.TensorDataset(fold_X_test, fold_y_test)
        fold_val_test_loader = torch.utils.data.DataLoader(dataset=fold_val_test_data,
                                                  batch_size=batch_size,
                                                  shuffle=False)

        predictions = []
        for fold_val_batch_X, fold_val_batch_y in fold_val_test_loader:
            step+=1
            fold_val_batch_X, fold_val_batch_y = fold_val_batch_X.to(device), fold_val_batch_y.to(device)
            fold_val_batch_X_pred = model(fold_val_batch_X)
            # predictions.extend(outputs.cpu().numpy())
            loss = criterion(fold_val_batch_X_pred, fold_val_batch_y)
#             print(loss.item())
            losses.append([loss.item(), 1, idx])
            wandb_logger.log("val_loss",loss.item())
    
#             checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode="max")


#     model.train()

    for iidx in range(idx+1):
        plt.plot(list(map(lambda x: x[0],filter(lambda x: (x[2]==iidx) &(x[1]==0), losses) )),
                 label=f'train {iidx}', alpha=0.2)
    plt.legend()
    plt.title(f'train')
    plt.savefig(f'./train_{idx}.jpg')
    plt.show()

    for iidx in range(idx+1):
        plt.plot(list(map(lambda x: x[0],filter(lambda x: (x[2]==iidx) &(x[1]==1), losses) )),
                    label=f'validate {iidx}', alpha=0.2)
    plt.legend()
    plt.title(f'validate')
    plt.savefig(f'./validate_{idx}.jpg')
#     plt.show()
#     break
