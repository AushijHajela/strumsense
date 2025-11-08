import cv2
import mediapipe as mp
import numpy as np
import os
import time

CHORDS = ["G", "C", "D", "E", "A", "Am", "Em"]
DATA_DIR = "pose_tracking/data"
CAPTURE_COUNT = 200          # number of samples to capture
CAPTURE_INTERVAL = 0.5       # seconds between frames

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

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
    chord = input(f"Enter chord name {CHORDS}: ").strip()
    if chord not in CHORDS:
        print("Invalid chord.")
        return

    save_dir = os.path.join(DATA_DIR, chord)
    ensure_dir(save_dir)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(model_complexity=1, max_num_hands=1,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
    draw = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print(f"\nCollecting {CAPTURE_COUNT} samples for '{chord}' every {CAPTURE_INTERVAL}s.")
    print("Press 'q' to stop early.\n")

    saved = 0
    last_time = time.time()

    while saved < CAPTURE_COUNT:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hlm = results.multi_hand_landmarks[0]
            draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS,
                                styles.get_default_hand_landmarks_style(),
                                styles.get_default_hand_connections_style())

            if time.time() - last_time >= CAPTURE_INTERVAL:
                vec = landmarks_to_vec(hlm.landmark)
                ts = int(time.time() * 1000)
                np.save(os.path.join(save_dir, f"{chord}_{ts}.npy"), vec)
                saved += 1
                last_time = time.time()

        cv2.putText(frame, f"Chord: {chord} ({saved}/{CAPTURE_COUNT})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Auto Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print(f"\n✅ Finished saving {saved} samples for '{chord}'")

if __name__ == "__main__":
    main()
