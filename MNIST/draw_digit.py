import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import torch
import torchvision.transforms as transforms

# ======== 建立畫布 UI =========
class App(tk.Tk):
    def __init__(self, model):
        super().__init__()
        self.title("手寫數字辨識")
        self.canvas = tk.Canvas(self, width=280, height=280, bg="black")
        self.canvas.pack()

        self.button_frame = tk.Frame(self)
        self.button_frame.pack()
        tk.Button(self.button_frame, text="辨識", command=self.predict).pack(side=tk.LEFT)
        tk.Button(self.button_frame, text="清除", command=self.clear).pack(side=tk.LEFT)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.image = Image.new("L", (280, 280), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.model = model

    def paint(self, event):
        x, y = event.x, event.y
        r = 8  # 筆畫粗細
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), color=0)
        self.draw = ImageDraw.Draw(self.image)

    def predict(self):
        # 轉成 28x28 並反相顏色（白字黑底）
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # 如果你的訓練資料有 normalize，要加
        ])

        input_tensor = transform(img).unsqueeze(0)  # 增加 batch 維度

        with torch.no_grad():
            output = self.model(input_tensor)
            pred = torch.argmax(output, dim=1).item()

        self.title(f"辨識結果：{pred}")

# ======== 載入你訓練好的模型 =========
from CNN_train import CNN  # 假設你之前的模型類別叫 CNN

model = CNN()
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=torch.device('cpu')))
model.eval()

# ======== 啟動畫布 =========
app = App(model)
app.mainloop()
