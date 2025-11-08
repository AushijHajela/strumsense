import cv2, mediapipe as mp, numpy as np, joblib, time, csv, os
from pathlib import Path
from datetime import datetime

MODEL_PATH = "pose_tracking/hand_chord_model.pkl"
ENCODER_PATH = "pose_tracking/label_encoder.pkl"
LOG_PATH = "pose_tracking/reports/session_log.csv"

# Config
STABILITY_FRAMES = 8       # consistent frames to confirm a chord
CONF_THRESHOLD = 0.6        # min confidence for "correct"
FEEDBACK_DURATION = 2       # seconds to show feedback text

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def log_feedback(chord, conf, feedback):
    """Append feedback to CSV log file."""
    ensure_dir(LOG_PATH)
    file_exists = os.path.isfile(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Chord", "Confidence", "Feedback"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         chord, f"{conf:.2f}", feedback])

def landmarks_to_vec(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    origin = pts[0]
    pts -= origin
    min_xy = pts[:, :2].min(axis=0)
    max_xy = pts[:, :2].max(axis=0)
    scale = np.linalg.norm(max_xy - min_xy) + 1e-6
    pts /= scale
    return pts.flatten()

def main():
    if not (Path(MODEL_PATH).exists() and Path(ENCODER_PATH).exists()):
        raise FileNotFoundError("Run train_model.py first.")

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(model_complexity=1, max_num_hands=1,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
    draw = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    prev_label = None
    stable_label = None
    stability_count = 0
    last_feedback_time = 0
    feedback = ""

    session_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🎸 StrumSense session started: {session_start}")
    print("Press 'q' to quit...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        display_text, color = "Hand not detected", (0, 0, 255)
        current_label = None
        conf = 0.0

        if results.multi_hand_landmarks:
            hlm = results.multi_hand_landmarks[0]
            draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS,
                                styles.get_default_hand_landmarks_style(),
                                styles.get_default_hand_connections_style())

            feat = landmarks_to_vec(hlm.landmark).reshape(1, -1)
            try:
                proba = model.predict_proba(feat)[0]
                idx = int(np.argmax(proba))
                conf = float(proba[idx])
            except:
                idx = int(model.predict(feat)[0])
                conf = 0.0

            current_label = encoder.inverse_transform([idx])[0]
            display_text = f"{current_label} ({conf:.2f})"
            color = (0, 255, 0) if conf >= CONF_THRESHOLD else (0, 165, 255)

            # Stability check
            if current_label == prev_label:
                stability_count += 1
            else:
                stability_count = 0
            prev_label = current_label

            # Confirm stable chord
            if stability_count >= STABILITY_FRAMES:
                if stable_label != current_label:
                    stable_label = current_label
                    last_feedback_time = time.time()
                    feedback = "✅ Correct chord!" if conf >= CONF_THRESHOLD else "⚠️ Try adjusting fingers"
                    log_feedback(current_label, conf, feedback)
                    print(f"{feedback} — {current_label} ({conf:.2f})")
                stability_count = 0

        # Show feedback for a short duration
        if feedback and (time.time() - last_feedback_time < FEEDBACK_DURATION):
            cv2.putText(frame, feedback, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
        elif time.time() - last_feedback_time >= FEEDBACK_DURATION:
            feedback = ""

        # Overlay current chord
        cv2.putText(frame, display_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("StrumSense - Real-Time Feedback", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print(f"\n📁 Session log saved to: {LOG_PATH}")
    print("Session ended.")

if __name__ == "__main__":
    main()
