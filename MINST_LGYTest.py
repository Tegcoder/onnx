import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

# 设置随机种子以确保结果可重复性
torch.manual_seed(0)

# 加载数据集并进行数据预处理
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
test_loader = DataLoader(test_set, batch_size=512, shuffle=False)

# 定义模型


class SimpleNet(nn.Module):
    """
    SimpleNet是一个简单的神经网络模型，用于处理图像分类任务。

    Args:
        None

    Attributes:
        fc_layers (Sequential): 包含多个线性层的神经网络模型，用于处理输入数据。

    Example:
        >>> model = SimpleNet()
    """

    def __init__(self):
        super(SimpleNet, self).__init__()
        # 创建一个包含多个线性层的神经网络模型，用于处理输入数据
        self.fc_layers = nn.Sequential(
            nn.Linear(28 * 28, 256),  # 输入层到隐藏层1，包括线性变换和ReLU激活函数
            nn.ReLU(),
            nn.Linear(256, 64),      # 隐藏层1到隐藏层2，包括线性变换和ReLU激活函数
            nn.ReLU(),
            nn.Linear(64, 10)        # 隐藏层2到输出层，包括线性变换
        )

    def forward(self, x):
        """
        前向传播函数用于处理输入数据并生成模型的输出。

        Args:
            x (Tensor): 输入数据张量，通常是形状为 (batch_size, input_features) 的数据。

        Returns:
            Tensor: 模型的输出张量，通常是形状为 (batch_size, output_features) 的数据。
        """
        # 将输入张量重新排列成(batch_size, 28 * 28)的形状
        x = x.view(-1, 28 * 28)
        # 调用模型的全连接层以生成输出
        return self.fc_layers(x)


# 检查GPU是否可用，如果可用则使用GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SimpleNet().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 计时器开始
startTime = time.time()

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

# 测试模型性能
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy}%")

# 保存模型
# torch.save(model, 'MNIST_model.pth')
model_filename = f'MNIST_model_{time.strftime("%Y%m%d%H%M%S")}.pth'
torch.save(model, model_filename)

# 计时器结束
endTime = time.time()
print(f"代码执行时间：{endTime - startTime}秒")

# 加载模型并显示部分测试集图像和预测结果
modelx = torch.load('MNIST_model.pth')
modelx.eval()

samples = 10
fig, axes = plt.subplots(1, samples, figsize=(samples * 2, 2))
for i in range(samples):
    image, label = test_set[i]
    image = image.unsqueeze(0).to(device)
    output = modelx(image)
    predicted = output.argmax(dim=1)

    axes[i].imshow(image.cpu().squeeze(0).numpy().squeeze(), cmap='gray')
    axes[i].set_title(f"Predicted: {predicted.item()}")
    axes[i].axis("off")
plt.show()
