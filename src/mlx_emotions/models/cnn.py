import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from mlx_emotions.emotions import HF_LABELS, IDX_TO_EMOTION

IMG_SIZE = 128
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm(cout)
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm(cout),
            )

    def __call__(self, x):
        h = self.c1(x)
        h = self.b1(h)
        h = nn.relu(h)
        h = self.c2(h)
        h = self.b2(h)
        if self.skip is not None:
            x = self.skip(x)
        return nn.relu(h + x)


class EmotionCNN(nn.Module):
    # small resnet, ~2.8M params. input (N,H,W,3) -> (N,7)
    def __init__(self, width=32, dropout=0.3):
        super().__init__()
        c1, c2, c3, c4 = width, width * 2, width * 4, width * 8
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1, bias=False),
            nn.BatchNorm(c1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.l1 = nn.Sequential(Block(c1, c1), Block(c1, c1))
        self.l2 = nn.Sequential(Block(c1, c2, stride=2), Block(c2, c2))
        self.l3 = nn.Sequential(Block(c2, c3, stride=2), Block(c3, c3))
        self.l4 = nn.Sequential(Block(c3, c4, stride=2), Block(c4, c4))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c4, 7)

    def __call__(self, x):
        x = self.stem(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = mx.mean(x, axis=(1, 2), keepdims=True)
        x = mx.reshape(x, (x.shape[0], -1))
        x = self.drop(x)
        return self.fc(x)


def to_model_input(img, size=IMG_SIZE):
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - MEAN) / STD
    return mx.array(a[None])


class LocalClassifier:
    def __init__(self, weights=None, size=IMG_SIZE):
        self.size = size
        self.model = EmotionCNN()
        mx.eval(self.model.parameters())
        if weights:
            self.load(weights)

    def load(self, path):
        self.model.load_weights(list(mx.load(path).items()), strict=False)
        mx.eval(self.model.parameters())

    def predict(self, img):
        self.model.eval()
        logits = self.model(to_model_input(img, self.size))
        mx.eval(logits)
        probs = mx.softmax(logits, axis=-1)
        mx.eval(probs)
        p = np.array(probs[0].tolist(), dtype=float)
        i = int(p.argmax())
        return IDX_TO_EMOTION[i], p.tolist(), HF_LABELS[i]
