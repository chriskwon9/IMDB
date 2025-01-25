import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt

import datetime



#####
# LSTM 층이 1개임을 알 수 있다 : num_layers=1
#####lstm = nn.LSTM(input_size=10, hidden_size = 20, num_layers=1, bidirectional=False, batch_first=True)
#####print(lstm)


# batch_size, seq_length, input_size
#####inputs = torch.zeros(1, 35, 10)
#####print(inputs.shape)


#####outputs, (hidden_state, cell_state) = lstm(inputs)
#####print(outputs.shape, hidden_state.shape, cell_state.shape)

import random



device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.backends.mps.is_available():
        torch.manual_seed(seed_value)
        torch.use_deterministic_algorithms(True)


def build_sequence_dataset(df, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(df) - seq_length):
        _x = df.iloc[i:i + seq_length].values
        _y = df.iloc[i + seq_length]['temperature']
        dataX.append(_x)
        dataY.append(_y)
    return np.array(dataX), np.array(dataY)




class LSTMModel(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, output_size)   # 출력 크기 조정

    # 입력 x의 크기가 (batch_size, seq_length, input_size)
    def forward(self, x):
        batch_size = x.size(0)

        # 초기 은닉 상태와 셀 상태를 0으로 초기화
        h0, c0 = self.init_hidden(batch_size, x.device)
        # LSTM 레이어를 통과
        out, _ = self.lstm(x, (h0, c0))
        # (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return h0, c0
    

def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def validate_model(model, test_loader, criterion):
    model.eval()  # 모델을 평가 모드로 설정
    total_loss = 0
    actuals = []
    predictions = []

    with torch.no_grad():  # 그라디언트 계산을 비활성화
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target.view(-1, 1))
            total_loss += loss.item()

            # 예측값과 실제값을 리스트에 저장
            actuals.extend(target.squeeze(1).tolist())
            predictions.extend(output.squeeze(1).tolist())

    # 손실 계산
    avg_loss = total_loss / len(test_loader)

    return avg_loss, actuals, predictions


