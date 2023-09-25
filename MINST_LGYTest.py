import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

startTime = time.time()
# 加载数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
test_loader = DataLoader(test_set, batch_size=512, shuffle=False)

# 定义模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SimpleNet().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

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

# 测试模型
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

    print(f"Accuracy: {100 * correct / total}%")
torch.save(model, 'MINIST_model.pth')

# 加载模型并显示部分测试集图像和预测结果
modelx = torch.load('MINIST_model.pth')
modelx.eval()
# 显示部分测试集图像和预测结果
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
endTime = time.time()
print(endTime-startTime)

# def plot_image(img, label, name):
#     fig = plt.figure()
#     for i in range(6):
#         plt.subplot(2, 3, i + 1)
#         plt.tight_layout()
#         plt.imshow(img[i][0] * 0.3081 + 0.1307, cmap='gray', interpolation='none')
#         plt.title("{}: {}".format(name, label[i].item()))
#         plt.xticks([])
#         plt.yticks([])
#     plt.show()
#
# image, label = next(iter(test_loader))
# out = modelx(image.view(image.size(0), 28 * 28).to(device))
# pred = out.argmax(dim=1)
# plot_image(image, pred, 'test')