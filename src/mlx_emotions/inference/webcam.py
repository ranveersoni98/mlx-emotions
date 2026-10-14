import time

import cv2
from PIL import Image


def get_faces():
    try:
        c = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        return None if c.empty() else c
    except AttributeError:
        return None


def run_live_cnn(clf, camera=0, use_face_crop=True):
    faces = get_faces()
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"cant open camera {camera}, check terminal camera perms")

    print("live, press q to quit")
    emo, probs = "neutral", None
    last, fps, last_pred = time.time(), 0.0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        if now - last_pred > 0.2:
            last_pred = now
            crop = frame
            if use_face_crop and faces is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                found = faces.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
                if len(found):
                    x, y, w, h = max(found, key=lambda b: b[2] * b[3])
                    p = int(0.2 * w)
                    H, W = frame.shape[:2]
                    crop = frame[max(0, y - p):min(H, y + h + p), max(0, x - p):min(W, x + w + p)]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            emo, probs, _ = clf.predict(Image.fromarray(rgb))

        fps = 0.9 * fps + 0.1 / max(1e-6, now - last)
        last = now
        if faces is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for x, y, w, h in faces.detectMultiScale(gray, 1.1, 5, minSize=(60, 60)):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        conf = max(probs) if probs else 0.0
        cv2.putText(frame, f"{emo.upper()} {conf:.2f} ({fps:.0f} fps)",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow("mlx-emotions - webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
