import argparse
from pathlib import Path

from PIL import Image

from mlx_emotions.emotions import EMOJI
from mlx_emotions.inference.webcam import run_live_cnn
from mlx_emotions.models import LocalClassifier


def build_parser():
    p = argparse.ArgumentParser(description="webcam emotion thing")
    p.add_argument("--weights", default="checkpoints/emotion_cnn.npz")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--image", default=None)
    p.add_argument("--no-face-crop", action="store_true")
    return p


def main():
    args = build_parser().parse_args()

    w = Path(args.weights)
    if not w.exists():
        raise SystemExit(f"no weights at {w}, run: uv run mlx-emotions-train --epochs 30")
    clf = LocalClassifier(str(w))
    if args.image:
        emo, probs, raw = clf.predict(Image.open(args.image))
        print(f"{EMOJI.get(emo, '')} {emo} ({raw}, {[round(x, 3) for x in probs]})")
        return
    run_live_cnn(clf, camera=args.camera, use_face_crop=not args.no_face_crop)
