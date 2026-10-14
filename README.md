# mlx-emotions

webcam emotion detection on apple silicon.

emotions: `angry, disgust, fear, happy, sad, surprise, neutral`

weights: [RanveerSoni98/mlx-emotions-cnn](https://huggingface.co/RanveerSoni98/mlx-emotions-cnn) (trained on raf-db, 4 epochs, ~61% val)

## setup

needs macos on apple silicon + camera perm for your terminal.

```bash
uv sync
```

grab the weights (or train your own below):

```bash
hf download RanveerSoni98/mlx-emotions-cnn --local-dir checkpoints
```

## use

live webcam (q to quit):

```bash
uv run mlx-emotions
```

single photo:

```bash
uv run mlx-emotions --image photo.jpg
```

## train your own

uses raf-db off huggingface (`deanngkl/raf-db-7emotions`, ~20k faces). first run downloads it so give it a min.

```bash
uv run mlx-emotions-train --epochs 30 --batch-size 64
# quick check it runs:
uv run mlx-emotions-train --smoke --batch-size 16
```

saves to `checkpoints/emotion_cnn.npz`. 2.8M param resnet thing, 128px input. fear/disgust get bumped up in the loss cause raf-db barely has any.

## how it works

1. opencv grabs frames, haar cascade finds your face, biggest one wins
2. face crop -> 128x128 -> cnn -> 7 probs, takes like ms
3. label + conf drawn on the frame

## layout

```
src/mlx_emotions/
  __init__.py       cli (mlx-emotions)
  emotions.py       labels
  models/
    cnn.py          cnn + predict wrapper
  data/
    rafdb.py        raf-db loading, aug, batches
  training/
    train.py        training loop (mlx-emotions-train)
  inference/
    webcam.py       webcam loop
```


## License

Code is MIT - see [LICENSE](LICENSE).

<p align="center">
  Built with <3 and a lot of bad decisions! by <a href="https://ranveersoni.me">Ranveer Soni</a>
</p>