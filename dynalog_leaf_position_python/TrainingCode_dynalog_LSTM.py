# ENV: mlclog
import sys
print('Python %s on %s' % (sys.version, sys.platform))
import numpy as np
import scipy.io as sio
import os
import mat73
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
torch.cuda.set_device(0)
torch.cuda.current_device()
## data load
mat_result_path = r"C:\Users\user\Downloads\dynalog_leaf_position\dynalog_leaf_position_matlab\deeplearningData-220812-150745"
temp_train_input = mat73.loadmat(os.path.join(mat_result_path, 'train_input_data_1st.mat'))
temp_train_input = temp_train_input['iter00_input_data']
temp_train_input =  np.moveaxis(temp_train_input, 2,0)
temp_train_output = sio.loadmat(os.path.join(mat_result_path, 'train_output_data_1st.mat'))
temp_train_output = temp_train_output['iter00_output_data']

# fix parameters
input_size = 6  # batch size
num_classes = 1
squence_length = 10 # pre-determinded parameter

# Model hyper-parameters
hidden_size = 20 # 30
num_layers = 3
batch_size = 1500 #
learning_rate = 0.001


class ION_Dataset_Sequential(Dataset):
    def __init__(self, input, output):
        self.input = input
        self.output = output

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        x = self.input[idx]
        y = self.output[idx]
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        return x, y


train_load = ION_Dataset_Sequential(temp_train_input, temp_train_output)

train_loader = torch.utils.data.DataLoader(dataset=train_load,
                                           batch_size=batch_size,
                                           shuffle=False)

class LSTM_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes): #dropout
        super(LSTM_model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) # , dropout=1
        self.fc = nn.Linear(hidden_size, num_classes)  # 2 for bidirection
       # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        #out = self.dropout(out) ####
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


model = LSTM_model(input_size, hidden_size, num_layers, num_classes).to(device)

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss

# Loss and optimizer
criterion = RMSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)

num_epochs =200
loss_set = []
for epoch in range(num_epochs):
    train_labels_result = 0
    train_outputs_result = 0
    train_loss_result = 0
    test_rmse_result = []
    test_corr_result = []
    test_result_data = 0

    for i, (train_data, train_labels) in enumerate(train_loader):
        model.train()

        train_data = train_data.to(device)
        train_labels = train_labels.to(device)

        # Forward pass
        train_predict_outputs = model(train_data)
        loss = criterion(train_predict_outputs, train_labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # loss
        train_loss = loss.cpu().detach().numpy()
        train_loss_result = np.append(train_loss_result, train_loss)

        train_predict_outputs = train_predict_outputs.cpu().detach().numpy()
        train_labels = train_labels.cpu().detach().numpy()

        train_labels_result = np.append(train_labels_result, train_labels)
        train_outputs_result = np.append(train_outputs_result, train_predict_outputs)

    train_loss_mean = np.sum(train_loss_result[1:]) / len(train_loss_result[1:])
    loss_set.append(train_loss_mean)
    print('Epoch [{}/{}], train Loss: {:.6f}'.format(epoch+1, num_epochs, train_loss_mean ))


save_path = "./model_save/model_epoch_{}_loss_{}.pth".format(num_epochs,train_loss_mean)
torch.save(model, save_path)
print('model save')














