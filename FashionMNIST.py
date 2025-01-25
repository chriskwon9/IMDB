import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader


###################################################
## 에포크 & 배치 사이즈 정의
## 애플 GPU 사용 여부


epochs = 10
batch_size = 512


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt'
               'Sneaker', 'Bag', 'Ankle boot']

print(torch.__version__)
print(device)

###################################################
## Fashion MNIST는 28 x 28인데 이를 227 x 227로 바꿔준다
## Pytorch Tensor로도 변환시켜준다

## 데이터 로드와 전처리 과정



transform = transforms.Compose([
    transforms.Resize(227),
    transforms.ToTensor()
])

training_data = datasets.FashionMNIST(
    root = 'data',
    train = True,
    download = True,
    transform = transform
)

validation_data = datasets.FashionMNIST(
    root = 'data',
    train = False,
    download = True,
    transform = transform
)

###################################################
## DataLoader로 데이터셋을 배치 단위로 나눈다


training_loader = DataLoader(training_data, batch_size = 64, shuffle = True)
validation_loader = DataLoader(validation_data, batch_size = 64, shuffle = True)


def matplotlib_imshow(img):
    img = img.mean(dim=0)
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(npimg, cmap="Greys")


dataiter = iter(training_loader)
images, labels = dataiter.__next__()

img_grid = torchvision.utils.make_grid(images[0])

matplotlib_imshow(img_grid)
print(class_names[labels[0]])

###################################################
## alexnet 클래스 구현
## 5개의 Convolution Layer, 3개의 FFN Layer


class fashion_mnist_alexnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096,10)

    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.view(out.size(0), -1)

        out = F.relu(self.fc1(out))
        out = F.dropout(out, 0.5)
        out = F.relu(self.fc2(out))
        out = F.dropout(out, 0.5)
        out = self.fc3(out)
        out = F.log_softmax(out, dim=1)

        return out
    

###################################################
## 모델, 손실함수, 최적화 기법 지정


model = fashion_mnist_alexnet().to("mps")
criterion = F.nll_loss
optimizer = optim.Adam(model.parameters())


###################################################

from torchinfo import summary

# 모델과 디바이스 정의
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

# torchinfo의 summary 호출
summary(model, input_size=(batch_size, 1, 227, 227), device=device)



###################################################

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()    # parameter 업데이트
        if (batch_idx + 1) % 30 == 0:
            print("Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
        print('='*50)



###################################################

for epoch in range(1, epochs+1):
    train(model, device, training_loader, optimizer, epoch)
    test(model, device, validation_loader)


