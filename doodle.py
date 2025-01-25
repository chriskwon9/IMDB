import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



z = torch.FloatTensor([1.3, 5.1, 2.2, 0.7, 1.1])

hypo = F.softmax(z, dim=0)
print(hypo)

z2 = torch.FloatTensor([2.6, 10.2, 4.4, 1.4, 2.2])

hypo2 = F.softmax(z2, dim=0)
print(hypo2)


import torch
import torchdata

from torchdata.datapipes.iter import IterableWrapper

data = [1, 2, 3, 4]
datapipe = IterableWrapper(data)
for item in datapipe:
    print(item)
print(torch.__version__)  # PyTorch 버전
print(torchdata.__version__)  # TorchData 버전


