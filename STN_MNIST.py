import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms


class Net_STN(nn.Module):
    def __init__(self):
        super(Net_STN, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)

        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(20 * 4 * 4, 64)
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

        x = x.view(-1, 20 * 4 * 4)
        x = F.relu(self.fc1(x))

        x = self.drop(x)
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
        loader_len = len(trainloader)
        loader_len_14 = loader_len // 4
        loader_len_temp = loader_len_14 - 1

        for step, data in enumerate(trainloader):
            input, target = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            output = net(input)
            # loss = F.nll_loss(output, target)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % loader_len_14 == loader_len_temp:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, step * len(data[0]), len(trainloader.dataset),
                    100. * step / loader_len, running_loss/500))
                running_loss = 0

    print('Finished Training!')


def test_net(net, criterion, testloader, device):
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, target = data[0].to(device), data[1].to(device)
            outputs = net(images)
            test_loss += criterion(outputs, target).item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_loss /= len(testloader.dataset)
    print('Accuracy of the network on the {} test images: {}/{} {} %, loss: {}'
          .format(total, correct, total, 100 * correct / total, test_loss))


def save_net(net, path='./STN_mnist_net.pth'):
    torch.save(net.state_dict(), path)
    print("save success!")


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def visualize_stn(net, testloader, device):
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(testloader))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = net.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')


def main():
    # 通过命令行传入网络运行参数
    parser = argparse.ArgumentParser("用于输入STN网络参数")
    parser.add_argument('--gpu', action='store_true', default=False, help='在GPU可用的前提下使用GPU加速')
    parser.add_argument('--epoch', type=int, default=2, help='网络训练次数(默认为2)')
    parser.add_argument('--save', action='store_true', default=False, help="网络参数保存选择")
    parser.add_argument('--lr', type=float, default=0.001, help="网络的学习速率(默认为0.001)")
    parser.add_argument('--momentum', type=float, default=0.9, help="网络的学习冲量(默认为0.9)")
    parser.add_argument('--show', action='store_true', default=False, help='显示STN层处理后的图像')
    parser.add_argument('--nums', type=int, default=1, help='dataloader中num_workers参数(默认为1)')
    parser.add_argument('--batch', type=int, default=64, help='dataloader中batch_size参数(默认为64)')

    args = parser.parse_args()

    epoch = args.epoch
    save = args.save
    lr = args.lr
    momentum = args.momentum
    gpu = args.gpu and torch.cuda.is_available()
    show = args.show
    nums = args.nums
    batch = args.batch

    print("epoch:{}, save:{}, lr:{}, momentum:{},"
          " gpu:{}, show:{}, nums:{}, batch: {}"
          .format(epoch, save, lr, momentum,
                  gpu, show, nums, batch))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=False)
    testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=False)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch,
                                              shuffle=True, num_workers=nums)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch,
                                             shuffle=False, num_workers=nums)

    device = "cuda" if gpu else "cpu"
    net = Net_STN().to(device)
    print(net)
    print(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    train_net(net, criterion, optimizer, trainloader, device, epoch)
    test_net(net, criterion, testloader, device)
    if save:
        save_net(net)

    if show:
        visualize_stn(net, testloader, device)
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
