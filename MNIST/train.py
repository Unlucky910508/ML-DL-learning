# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from cnn_model import CNN  # <-- 從 cnn_model.py 匯入 CNN 類別

def main():
    # 1. 載入 MNIST 資料集
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 2. 建立模型
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    test_accuracies = []

    # 3. 訓練模型
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

    # 4. 儲存模型
    torch.save(model.state_dict(), "mnist_cnn.pth")

    # 5. 畫圖
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

# ✅ 加入這行，避免其他地方 import 這個檔案時執行訓練
if __name__ == "__main__":
    main()
