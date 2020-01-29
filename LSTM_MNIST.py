import torch
import argparse
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
           'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


class Net_LSTM(nn.Module):
    def __init__(self):
        super(Net_LSTM, self).__init__()

        # input_size, hidden_size, num_layer, bias, batch_first, diopout......
        self.lstm = nn.LSTM(
            input_size=28, hidden_size=64,
            num_layers=1, batch_first=True,
        )
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        out, (h_n, h_c) = self.lstm(x, None)
        out = self.fc(out[:, -1, :])

        return out


def train_net(net, criterion, optimizer, trainloader, device, epochs=2):
    for epoch in range(epochs):
        running_loss = 0
        for step, (image, target) in enumerate(trainloader):
            image = image.view(-1, 28, 28)
            image, target = image.to(device), target.to(device)

            optimizer.zero_grad()
            output = net(image)
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
        for (image, target) in testloader:
            image = image.view(-1, 28, 28)
            image, target = image.to(device), target.to(device)
            outputs = net(image)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


def save_net(net, path='./LSTM_mnist_net.pth'):
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

    transform = transforms.Compose({
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    })

    trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=False)
    testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=False)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=4,
                                              shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=4,
                                             shuffle=False, num_workers=1)

    device = "cuda" if gpu else "cpu"
    net = Net_LSTM().to(device)
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
