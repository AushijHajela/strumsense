import cv2
import os
import mediapipe as mp
import numpy as np
import time

CHORDS = ["G","C","D","E","A","Am","Em"]
DATA_DIR="pose_tracking/data"

def ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def landmarks_to_vec(landmarks):
    pts=np.array([[lm.x,lm.y,lm.z] for lm in landmarks],dtype=np.float32)
    origin=pts[0]
    pts= pts-origin
    
    min_xy=pts[:,:2].min(axis=0)
    max_xy=pts[:,:2].max(axis=0)
    scale=np.linalg.norm(max_xy-min_xy)+1e-6
    pts/=scale
    
    return pts.flatten()

def main():
    chord=input(f"Enter chord name {CHORDS}: ").strip()
    if chord not in CHORDS:
        print("Invalid chord name")
        return
    
    count=int(input("Enter number of samples to collect: "))
    save_dir=os.path.join(DATA_DIR,chord)
    ensure_path(save_dir)
    
    mp_hands=mp.solutions.hands
    hands=mp_hands.Hands(static_image_mode=False,
                         max_num_hands=1,
                         min_detection_confidence=0.5,
                         min_tracking_confidence=0.5)
    draw=mp.solutions.drawing_utils
    styles=mp.solutions.drawing_styles
    
    cap=cv2.VideoCapture(0)
    saved,overlay=0,True
    print("Press 's' to save a sample,'o' to toggle overlay,'q' to quit")
    while cap.isOpened() and saved<count:
        ret,frame=cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame=cv2.flip(frame,1)
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=hands.process(rgb)
        
        vec=None
        if results.multi_hand_landmarks:
            hlm = results.multi_hand_landmarks[0]
            if overlay:
                draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS,
                                    styles.get_default_hand_landmarks_style(),
                                    styles.get_default_hand_connections_style())
            vec = landmarks_to_vec(hlm.landmark)

        cv2.putText(frame, f"Chord: {chord}",(10, 30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        cv2.putText(frame, f"Saved: {saved}/{count}",(10, 60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        cv2.imshow("Collect Data", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('h'): overlay = not overlay
        elif key == ord('s') and vec is not None:
            ts = int(time.time() * 1000)
            np.save(os.path.join(save_dir, f"{chord}_{ts}.npy"), vec)
            saved += 1
            print(f"Saved {saved}/{count}")
            if saved >= count: break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print(f"Finished saving {saved} samples for {chord}")

if __name__ == "__main__":
    main()
    
    
    