import torch
import torch.nn as nn
import torch.optim as optim


import torchdata
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import random
import numpy as np
    

################################ 데이터 전처리

#### IMDBDataset 클래스 내부에서 tokenize_and_pad 함수를 사용해서 텍스트 데이터를 시퀀스로 변환하고,
#### 시퀀스 길이를 패딩 처리

#### IMDBDataset 클래스는 IMDB 데이터셋을 Pytorch 모델 학습에 적합한 방식으로 변환하고 
#### prepare_imdb_data 함수에서 사용된다

#### prepare_imdb_data 함수는 IMDB 데이터셋 로드 --> 레이블을 변환 --> IMDBDataset이용해서 DataLoader 객체 생성

################################



# 설정값
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.backends.mps.is_available():
        torch.manual_seed(seed_value)
        torch.use_deterministic_algorithms(True)



# 텍스트 데이터를 시퀀스로 변환
def tokenize_and_pad(texts, vocab, seq_length):
    # 텍스트를 단어 단위로 분리
    tokenizer = get_tokenizer("basic_english")
    # 각 텍스트를 숫자 시퀀스로 변환
    sequences = [
        torch.tensor([vocab[token] for token in tokenizer(text)], dtype=torch.long) for text in texts]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)

    return padded_sequences[:, :seq_length]



class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, seq_length):
        self.texts = texts 
        self.labels = labels
        self.vocab = vocab
        self.seq_length = seq_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        text_tensor = tokenize_and_pad([text], self.vocab, self.seq_length)[0]
        return text_tensor, torch.tensor(label, dtype=torch.float)
    

# 데이터 전처리
def prepare_imdb_data(seq_length):
    train_iter, test_iter = IMDB(split=('train', 'test'))
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, (label_text[1] for label_text in train_iter)),
                                      specials=["<pad>"])
    vocab.set_default_index(vocab["<pad>"])

    train_iter, test_iter = IMDB(split=('train', 'test'))

    # 텍스트와 레이블을 분리
    train_texts, train_labels = zip(*[(label_text[1], 1 if label_text[0] == 'pos' else 0) for label_text in train_iter])
    test_texts, test_labels = zip(*[(label_text[1], 1 if label_text[0] == 'pos' else 0) for label_text in test_iter])


    # Dataset & DataLoader 생성
    train_dataset = IMDBDataset(train_texts, train_labels, vocab, seq_length)
    test_dataset = IMDBDataset(test_texts, test_labels, vocab, seq_length)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    return train_loader, test_loader, len(vocab)
    

# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.squeeze(1), target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def validate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    actuals = []
    predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output.squeeze(1), target)
            total_loss += loss.item()
            actuals.extend(target.tolist())
            predictions.extend(torch.round(torch.sigmoid(output.squeeze(1))).tolist())
    accuracy = sum(1 for a, p in zip(actuals, predictions) if a == p) / len(actuals)
    return total_loss / len(test_loader), accuracy


# 메인 코드 
if __name__ == "__main__":
    set_seed(42)
    seq_length = 100  # 시퀀스 길이 설정
    train_loader, test_loader, vocab_size = prepare_imdb_data(seq_length)

    # 모델 및 하이퍼파라미터 설정
    model = LSTMModel(input_size=1, hidden_size=128, num_layers=2, output_size=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()  # 이진 분류를 위한 손실 함수

    # 학습
    num_epochs = 5
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss, accuracy = validate_model(model, test_loader, criterion)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
