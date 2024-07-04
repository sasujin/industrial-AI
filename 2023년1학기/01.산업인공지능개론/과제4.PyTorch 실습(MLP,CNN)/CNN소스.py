
#-*- coding: utf-8 -*-
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, cache=True)
X = mnist.data
y = mnist.target
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=1/7, random_state=0) 
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.LongTensor(list(map(int, y_train)))
y_test = torch.LongTensor(list(map(int, y_test)))
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
X_train = X_train.view(-1, 1,28,28).float() 
X_test = X_test.view(-1,1,28,28).float() 
print(X_train.shape)
print(X_test.shape)
train = TensorDataset(X_train, y_train) test = TensorDataset(X_test, y_test)
BATCH_SIZE = 32
loader_train = DataLoader(train, batch_size = BATCH_SIZE, shuffle = loader_test = DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)
False)


class CNN(nn.Module): def __init__(self): super(CNN, self).__init__( )
self.conv1 = nn.Conv2d(1, 32, kernel_size=5) self.conv2 = nn.Conv2d(32, 32, kernel_size=5) self.conv3 = nn.Conv2d(32,64, kernel_size=5) self.fc1 = nn.Linear(3*3*64, 256) self.fc2= nn.Linear(256, 10)
self.loss_fn= nn.Cross EntropyLoss( )
self.optimizer = optim.Adam(self.parameters( ), Ir=0.01)
def forward(self, x):
X = F.relu(self.conv1(x))
X = F.relu (F.max_pool2d(self.conv2(x), 2))
X = F.dropout(x, p=0.5, training=self.training) F.relu (F.max_pool2d(self.conv3(x),2))
X =
X = F.dropout(x, p=0.5, training=self.training)
X = x.view(-1,3*3*64 )
X = F.relu(self.fc1(x))
X = F.dropout(x, training=self.training)
X = self.fc2(x)
return F.log_softmax(x, dim=1)

torch.nn.Conv2d(in_channels, out_channels, kernel_size,
stride=1, padding=0, dilation=1, groups=1,
bias=True, padding_mode='zeros')

def fit(model, loader_train):
optimizer = torch.optim.Adam(model.parameters( )) error = nn.CrossEntropyLoss()
EPOCHS = 1 model.train()
for epoch in range(EPOCHS):
correct = 0
for batch_idx, (X_batch, y_batch) in enumerate(loader_train): var_X_batch = Variable(X_batch).float( ) var_y_batch = Variable(y_batch)
optimizer.zero_grad()
output = model(var_X_batch)
loss =
error(output, var_y_batch)
loss.backward()
optimizer.step()
predicted = torch.max(output.data, 1)[1]
correct += (predicted == var_y_batch).sum( )
if batch_idx % 50 == : 0:
print('에포크 || : {} [{}/{} ({:.Of}%)]Wt ✩::.6f}Wt Accuracy:{:.3f}%'.format( epoch, batch_idx*len(X_batch), len(loader_train), 100.*batch_idx / len(loader_train),
loss.data,
correct*100./ (BATCH_SIZE*(batch_idx+1))))


def evaluate(model):
correct = 0
for test_imgs, test_labels in loader_test: test_imgs = Variable(test_imgs).float() output model(test_imgs)
predicted = torch.max(output,1)[1]
correct += (predicted
==
test_labels).sum()
print("EO: {f} % ".format( float(correct) /
(len(loader_test)*BATCH_SIZE)))
cnn = CNN()
evaluate(cnn)
fit(cnn, loader_train)
cnn.eval() # 99 &^€ 952 g
evaluate(cnn)
index = 10 # 테스트 데이터 중에서 확인해볼 데이터의 인덱스 data = X_test[index].view(-1, 1,28,28).float( )
output = cnn(data) #
:
print('{} & {}'.format(index, output)) predicted = torch.max(output, 1)
print('Eo Œ⇒ : {}'.format(index, predicted.numpy())) print('{}'.format(y_test[index]))
