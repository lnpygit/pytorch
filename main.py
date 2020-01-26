import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
           'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

'''
        增加命令行运行参数输入
        增加gpu运行选择
        增加训练、测试、保存网络函数
'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)

        # nn.Dropout2d 随机失活
        # W/H=[(输入大小-卷积核大小+2*P）/步长]  +1.

        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def main():
    tranform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transet = torchvision.datasets.MNIST(root='./data', train=True, transform=tranform, download=True)
    testset = torchvision.datasets.MNIST(root='./data', train=False, transform=tranform, download=False)

    tranloader = data.DataLoader(dataset=transet, batch_size=4,
                                 shuffle=True, num_workers=1)
    testloader = data.DataLoader(dataset=testset, batch_size=4,
                                 shuffle=False, num_workers=1)

    net = Net()


def train_net():
    pass


def test_net():
    pass


if __name__ == '__main__':
    main()
