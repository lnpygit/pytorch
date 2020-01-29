import torch
import argparse
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
           'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


class Net_CNN(nn.Module):
    def __init__(self):
        super(Net_CNN, self).__init__()

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

    # 通过命令行传入网络运行参数
    parser = argparse.ArgumentParser("用于输入网络参数")
    parser.add_argument('--gpu', action='store_true', default=False, help='在GPU可用的前提下使用GPU加速')
    parser.add_argument('--epoch', type=int, default=2, help='网络训练次数(默认为2)')
    parser.add_argument('--save', action='store_true', default=False, help="网络参数保存选择")
    parser.add_argument('--lr', type=float, default=0.001, help="网络的学习速率(默认为0.001)")
    parser.add_argument('--momentum', type=float, default=0.9, help="网络的学习冲量(默认为0.9)")

    args = parser.parse_args()

    epoch = args.epoch
    save = args.save
    lr = args.lr
    momentum = args.momentum
    gpu = args.gpu and torch.cuda.is_available()

    print("epoch:{}, save:{}, lr:{}, momentum:{}, gpu:{}".format(epoch, save, lr, momentum, gpu))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=False)
    testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=False)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=4,
                                              shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=4,
                                             shuffle=False, num_workers=1)

    device = "cuda" if gpu else "cpu"
    print("device:{}".format(device))

    net = Net_CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    train_net(net, criterion, optimizer, trainloader, device, epoch)
    test_net(net, testloader, device)
    if save:
        save_net(net)


def train_net(net, criterion, optimizer, trainloader, device, epochs=2):
    for epoch in range(epochs):
        running_loss = 0
        for step, data in enumerate(trainloader):
            input, target = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            output = net(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, step + 1, running_loss / 2000))
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


def save_net(net, path='./CNN_mnist_net.pth'):
    torch.save(net.state_dict(), path)
    print("save success!")


if __name__ == '__main__':
    main()
