import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mu
import numpy as np

from mlx_emotions.data.rafdb import Batches, class_weights, load_hf, split_by_class
from mlx_emotions.emotions import HF_LABELS
from mlx_emotions.models.cnn import EmotionCNN, IMG_SIZE


def eval_acc(model, ds, idxs, bs, size):
    model.eval()
    right, n = 0, 0
    per = {k: [0, 0] for k in range(7)}
    for xb, yb in Batches(ds, idxs, bs, size, shuffle=False):
        logits = model(mx.array(xb))
        mx.eval(logits)
        pred = np.array(mx.argmax(logits, axis=-1).tolist())
        right += int((pred == yb).sum())
        n += len(yb)
        for t, p in zip(yb.tolist(), pred.tolist()):
            per[t][1] += 1
            if t == p:
                per[t][0] += 1
    out = {HF_LABELS[k]: round(v[0] / max(1, v[1]), 4) for k, v in per.items()}
    return round(right / max(1, n), 4), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/emotion_cnn.npz")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    print("loading raf-db, first run downloads stuff so it takes a bit...")
    ds = load_hf()
    labels = [int(x) for x in ds["label"]]
    print(len(ds), dict(sorted(Counter(labels).items())))

    tr, va = split_by_class(labels, args.val_frac, args.seed)
    if args.smoke:
        tr, va = tr[: args.batch_size * 2], va[: args.batch_size]
        args.epochs = 1
    print(f"train {len(tr)} val {len(va)}")

    cw = mx.array(class_weights([labels[i] for i in tr]))
    mx.eval(cw)
    print("weights", np.round(np.array(cw.tolist()), 3).tolist())

    model = EmotionCNN()
    mx.eval(model.parameters())
    steps = args.epochs * max(1, len(tr) // args.batch_size)
    opt = optim.AdamW(learning_rate=optim.cosine_decay(args.lr, steps), weight_decay=1e-4)

    def loss_fn(model, xb, yb):
        logits = model(xb)
        ce = nn.losses.cross_entropy(logits, yb, reduction="none")
        return mx.mean(ce * mx.take(cw, yb)), logits

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def step(xb, yb):
        (loss, _), grads = loss_and_grad(model, xb, yb)
        opt.update(model, grads)
        mx.eval(loss, model.parameters())
        return loss

    best = 0.0
    out = Path(args.out)
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        loader = Batches(ds, tr, args.batch_size, args.img_size,
                         shuffle=True, do_aug=not args.smoke, seed=args.seed + ep)
        for xb, yb in loader:
            loss = step(mx.array(xb), mx.array(yb))
            mx.eval(loss)
            losses.append(float(loss.item()))
            if args.smoke:
                break
        acc, per = eval_acc(model, ds, va, args.batch_size * 2, args.img_size)
        print(f"epoch {ep}/{args.epochs} loss={np.mean(losses):.4f} val={acc:.4f} {time.time()-t0:.1f}s")
        print(" ", per)
        if acc >= best:
            best = acc
            out.parent.mkdir(parents=True, exist_ok=True)
            mx.savez(str(out), **dict(mu.tree_flatten(model.parameters())))
            (out.parent / "labels.json").write_text(json.dumps(HF_LABELS))
            print(f"  saved {out} ({best:.4f})")

    print(f"done, best={best:.4f} -> {out}")


if __name__ == "__main__":
    main()
