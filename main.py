import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
           'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

'''
        增加命令行运行参数输入
        # 增加gpu运行选择
        # 增加训练、测试、保存网络函数
'''


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)

        # nn.Dropout2d 随机失活
        # W/H=[(输入大小-卷积核大小+2*P）/步长]  +1.
        self.dropout1 = nn.Dropout2d(0.35)

        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = self.dropout1(x)
        x = x.view(-1, 16 * 4 * 4)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def main():
    tranform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transet = torchvision.datasets.MNIST(root='./data', train=True, transform=tranform, download=False)
    testset = torchvision.datasets.MNIST(root='./data', train=False, transform=tranform, download=False)

    trainloader = torch.utils.data.DataLoader(dataset=transet, batch_size=4,
                                              shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=4,
                                             shuffle=False, num_workers=1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_net(net, criterion, optimizer, trainloader, device)
    test_net(net, testloader, device)
    net_save(net)


def train_net(net, criterion, optimizer, trainloader, device, epoch=2):
    for i in range(epoch):
        running_loss = 0
        for t, data in enumerate(trainloader):
            input, target = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            output = net(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if t % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (i + 1, t + 1, running_loss / 2000))
                running_loss = 0

    print('Finished Training!')


def test_net(net, testloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, target = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


def net_save(net, path='./mnist_net.pth'):
    torch.save(net.state_dict(), path)


if __name__ == '__main__':
    main()
