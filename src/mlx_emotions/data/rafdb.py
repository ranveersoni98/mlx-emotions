import random
from collections import Counter

import numpy as np
from PIL import Image, ImageEnhance

from mlx_emotions.models.cnn import IMG_SIZE, MEAN, STD

DATASET_ID = "deanngkl/raf-db-7emotions"


def load_hf():
    from datasets import load_dataset

    return load_dataset(DATASET_ID, split="train")


def split_by_class(labels, val_frac=0.1, seed=0):
    rng = random.Random(seed)
    buckets = {}
    for i, y in enumerate(labels):
        buckets.setdefault(y, []).append(i)
    tr, va = [], []
    for idxs in buckets.values():
        rng.shuffle(idxs)
        k = max(1, int(len(idxs) * val_frac))
        va += idxs[:k]
        tr += idxs[k:]
    rng.shuffle(tr)
    rng.shuffle(va)
    return tr, va


def aug(img, size=IMG_SIZE):
    img = img.convert("RGB")
    w, h = img.size
    pad = int(0.08 * min(w, h))
    if pad:
        dx = random.randint(-pad, pad)
        dy = random.randint(-pad, pad)
        img = img.transform((w, h), Image.AFFINE, (1, 0, dx, 0, 1, dy))
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.3:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))
    return img.resize((size, size), Image.BILINEAR)


def pil_to_np(img, size=IMG_SIZE):
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32) / 255.0
    return ((a - MEAN) / STD).astype(np.float32)


def class_weights(labels):
    # fear/disgust barely show up so bump them a bit
    c = Counter(labels)
    tot = len(labels)
    w = np.zeros(7, dtype=np.float32)
    for k in range(7):
        w[k] = (tot / max(1, c.get(k, 1))) ** 0.5
    return (w / w.mean()).astype(np.float32)


class Batches:
    def __init__(self, ds, idxs, bs=64, size=IMG_SIZE, shuffle=True, do_aug=False, seed=0):
        self.ds = ds
        self.idxs = list(idxs)
        self.bs = bs
        self.size = size
        self.shuffle = shuffle
        self.do_aug = do_aug
        self.rng = random.Random(seed)

    def __len__(self):
        return (len(self.idxs) + self.bs - 1) // self.bs

    def __iter__(self):
        order = list(self.idxs)
        if self.shuffle:
            self.rng.shuffle(order)
        for i in range(0, len(order), self.bs):
            xs, ys = [], []
            for j in order[i : i + self.bs]:
                ex = self.ds[j]
                im = ex["image"]
                if self.do_aug:
                    im = aug(im, self.size)
                xs.append(pil_to_np(im, self.size))
                ys.append(int(ex["label"]))
            yield np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int32)
