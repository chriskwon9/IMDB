import os
import tarfile
import urllib.request
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 데이터 다운로드 및 압축 해제
def download_imdb_data():
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    filename = "aclImdb_v1.tar.gz"
    if not os.path.exists(filename):
        print("Downloading IMDB dataset...")
        urllib.request.urlretrieve(url, filename)

    # 압축 해제
    if not os.path.exists("aclImdb"):
        print("Extracting IMDB dataset...")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(members=tar.getmembers())


# 텍스트와 레이블 로드
def load_imdb_data(split="train"):
    data_path = f"aclImdb/{split}"
    texts = []
    labels = []
    for label in ["pos", "neg"]:
        label_path = os.path.join(data_path, label)
        for filename in os.listdir(label_path):
            if filename.endswith(".txt"):
                with open(os.path.join(label_path, filename), "r", encoding="utf-8") as f:
                    texts.append(f.read())
                    labels.append(1 if label == "pos" else 0)
    return texts, labels


# 데이터 전처리: 토큰화 및 패딩
def tokenize_and_pad(texts, vocab, seq_length):
    tokenizer = get_tokenizer("basic_english")
    processed_texts = []

    for text in texts:
        tokenized = tokenizer(text)
        indexed = [vocab[token] for token in tokenized if token in vocab]
        if len(indexed) < seq_length:
            indexed += [0] * (seq_length - len(indexed))  # Padding
        else:
            indexed = indexed[:seq_length]  # Truncate
        processed_texts.append(indexed)

    return processed_texts


# 어휘집 생성 --> 단어를 정수로 매핑하는 vocab객체 생성
def build_vocab(texts):
    tokenizer = get_tokenizer("basic_english")
    return build_vocab_from_iterator((tokenizer(text) for text in texts), specials=["<pad>"])


# 데이터 준비 함수
def prepare_imdb_data(seq_length):
    # 데이터 다운로드
    download_imdb_data()

    # 텍스트와 레이블 로드
    train_texts, train_labels = load_imdb_data("train")
    test_texts, test_labels = load_imdb_data("test")

    # 어휘집 생성
    vocab = build_vocab(train_texts)
    vocab.set_default_index(vocab["<pad>"])

    # 텍스트를 토크나이즈하고 패딩
    train_data = tokenize_and_pad(train_texts, vocab, seq_length)
    test_data = tokenize_and_pad(test_texts, vocab, seq_length)

    # 레이블을 텐서로 변환
    train_labels = torch.tensor(train_labels)
    test_labels = torch.tensor(test_labels)

    # PyTorch Dataset 생성
    train_dataset = TextDataset(train_data, train_labels)
    test_dataset = TextDataset(test_data, test_labels)

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader, len(vocab)


# PyTorch Dataset 정의
class TextDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long).clone().detach()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # 출력 크기 조정

    def forward(self, x):
        batch_size = x.size(0)

        # 초기 은닉 상태와 셀 상태를 0으로 초기화
        h0, c0 = self.init_hidden(batch_size, x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 마지막 time step의 출력 사용
        return out

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return h0, c0


# 모델 학습 함수
def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        data = data.float()
        data = data.unsqueeze(-1)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# 모델 평가 함수
def validate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    actuals = []
    predictions = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            data = data.float()
            data = data.unsqueeze(-1)

            output = model(data)
            loss = criterion(output, target.unsqueeze(1).float())
            total_loss += loss.item()

            # 예측값 저장
            actuals.extend(target.cpu().tolist())
            predictions.extend(torch.round(torch.sigmoid(output)).cpu().tolist())

    accuracy = (np.array(actuals) == np.array(predictions).flatten()).mean()
    return total_loss / len(test_loader), accuracy


# 주요 실행 코드
if __name__ == "__main__":
    # 하이퍼파라미터 설정
    seq_length = 100
    input_size = 1
    hidden_size = 256
    num_layers = 3
    output_size = 1
    num_epochs = 10
    learning_rate = 0.001

    # 데이터 로드
    train_loader, test_loader, vocab_size = prepare_imdb_data(seq_length)

    # 모델 초기화
    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 학습
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss, val_accuracy = validate_model(model, test_loader, criterion)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
