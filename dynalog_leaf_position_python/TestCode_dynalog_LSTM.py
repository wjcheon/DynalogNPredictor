import sys
print('Python %s on %s' % (sys.version, sys.platform))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import scipy.io as sio
trained_model_path ="./model_save/first_test_0.175_train_0.3458.pth"

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
torch.cuda.set_device(0)
torch.cuda.current_device()

mat_result_path = r"C:\Users\user\Downloads\dynalog_leaf_position\dynalog_leaf_position_matlab\deeplearningData-220812-150745"
temp_test_input = sio.loadmat(os.path.join(mat_result_path, 'test_input_data_1st.mat'))
temp_test_input = temp_test_input['iter00_input_data']
temp_test_input =  np.moveaxis(temp_test_input, 2,0)
temp_test_output = sio.loadmat(os.path.join(mat_result_path, 'test_output_data_1st.mat'))
temp_test_output = temp_test_output['iter00_output_data']
##
input_size = 6
num_classes = 1
squence_length = 10

# Hyper-parameters
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


test_load = ION_Dataset_Sequential(temp_test_input, temp_test_output)

test_loader = torch.utils.data.DataLoader(dataset=test_load,
                                           batch_size=batch_size,
                                           shuffle=False)
##

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

save_path = trained_model_path

save_model = torch.load(save_path)
model = save_model.to(device)

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

test_rmse_result_mean_result = []
test_rmse_result = []
test_corr_result = []
test_result_data = []

for i, (test_data, test_labels) in enumerate(test_loader):
    model.eval()
    torch.no_grad()
    test_data = test_data.to(device)
    test_labels = test_labels.to(device)

    test_predict_outputs = model(test_data)

    test_result_data = np.append(test_result_data, test_predict_outputs.cpu().detach().numpy())

    test_rmse = criterion(test_predict_outputs, test_labels)
    test_rmse_result.append(test_rmse.cpu().detach().numpy())

test_rmse_result_mean = np.mean(test_rmse_result)
test_rmse_result_mean_result.append(test_rmse_result_mean)

print('test Loss: {:.6f}'.format(test_rmse_result_mean ))


from scipy.io import savemat
test_result_data_save = {"test_result_data":test_result_data}
savemat(os.path.join(".","result","test_result_data.mat"), test_result_data_save)








