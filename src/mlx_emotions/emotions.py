EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

EMOJI = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐",
}

# raf-db order, this is what the model spits out
HF_LABELS = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

# model idx -> our label
IDX_TO_EMOTION = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}

EMOTION_TO_IDX = {v: k for k, v in IDX_TO_EMOTION.items()}
