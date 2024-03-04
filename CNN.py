#%%
"""Import library"""
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# for reproducibility
torch.manual_seed(0)
if device == 'cuda':
    torch.cuda.manual_seed_all(0)


#%%
"""Hyperparameter Setting"""
config = {
    "lr" : 0.001,
    "epochs" : 15,
    "batch_size" : 128
}

#%%
"""Dataset"""
mnist_train = datasets.MNIST(
    root='MNIST_data/',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

mnist_test = datasets.MNIST(
    root='MNIST_data/',
    train=False,
    transform=transforms.ToTensor(),
    download=True
)
# loader 
data_loader = DataLoader(
    dataset=mnist_train,
    batch_size=config["batch_size"],
    shuffle=True, 
    drop_last=False  
) 


   #%%
"""Model""" 
class CNN(nn.Module):
    def __init__(self): # 모델의 구조를 정의 초기함수 중요!!
        super(CNN, self).__init__()
        # self.keep_prob = 0.5
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        self.fc1 = torch.nn.Linear(4 * 4 * 128, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5))

        self.fc2 = torch.nn.Linear(625, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight) 

    def forward(self, x): # 실제 연산. 우리가 직접 실행하는 것이 아니라 모델 인스턴스 인자로 넣어주면 자동으로 작동한다
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   
        out = self.layer4(out)
        out = self.fc2(out)
        return out

#%%
"""Train"""
model = CNN().to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)    
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

total_batch = len(data_loader)
model.train()    

for epoch in range(config["epochs"]):
    avg_cost = 0 

    for X, Y in data_loader:
        X = X.to(device)
        Y = Y.to(device)

        preds = model(X)
        cost = criterion(preds, Y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))


#%%
    """Evaluation"""
with torch.no_grad():
    model.eval()   

    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_testS
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())