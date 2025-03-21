import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. 載入 MNIST 資料集（轉成 Tensor 並正規化）
transform = transforms.Compose([
    transforms.ToTensor(),  # 轉為 tensor，並把值從 [0, 255] 壓到 [0, 1]
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

train_losses = []
test_accuracies = []

# 2. 定義 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 輸入 channel=1，輸出=32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 降維：28x28 → 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 降維：14x14 → 7x7
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),             # 攤平：64 x 7 x 7 = 3136
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)        # 10 個數字分類
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

model = CNN()

# 3. Loss 與 Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses = []
test_accuracies = []

# 4. 訓練模型（記錄 loss 和 accuracy）
for epoch in range(5):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss)

    # 評估測試集準確率
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    test_acc = correct / total
    test_accuracies.append(test_acc)

    print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}, Test Acc: {100 * test_acc:.2f}%")

# 5. 畫出 loss & accuracy 曲線
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot([acc * 100 for acc in test_accuracies], label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy Curve')
plt.grid(True)

plt.tight_layout()
plt.show()
