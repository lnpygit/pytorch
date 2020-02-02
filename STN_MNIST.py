import torch
import argparse
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms


class Net_STN(nn.Module):
    def __init__(self):
        super(Net_STN, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(16 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        x = self.stn(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop(x)

        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def stn(self, x):
        xc = self.localization(x)
        xc = xc.view(-1, 10 * 3 * 3)

        theta = self.fc_loc(xc)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        grid = F.grid_sample(x, grid)

        return grid


def train_net(net, criterion, optimizer, trainloader, device, epochs):
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

    print('Accuracy of the network on the {} test images: {}/{} {} %'
          .format(total, correct, total, 100 * correct / total))


def save_net(net, path='./STN_mnist_net.pth'):
    torch.save(net.state_dict(), path)
    print("save success!")


def main():
    # 通过命令行传入网络运行参数
    parser = argparse.ArgumentParser("用于输入LSTM网络参数")
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
    net = Net_STN().to(device)
    print(net)
    print(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    train_net(net, criterion, optimizer, trainloader, device, epoch)
    test_net(net, testloader, device)
    if save:
        save_net(net)


if __name__ == "__main__":
    main()
