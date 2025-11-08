import os, glob, joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

CHORDS = ["G", "C", "D", "E", "A", "Am", "Em"]
DATA_DIR = "pose_tracking/data"
MODEL_PATH = "pose_tracking/hand_chord_model.pkl"
ENCODER_PATH = "pose_tracking/label_encoder.pkl"
REPORTS_DIR = "pose_tracking/reports"
CONF_MATRIX_PATH = os.path.join(REPORTS_DIR, "pose_confusion_matrix.png")

os.makedirs(REPORTS_DIR, exist_ok=True)

def load_data():
    X, y = [], []
    for chord in CHORDS:
        folder = os.path.join(DATA_DIR, chord)
        if not os.path.exists(folder): continue
        for f in glob.glob(os.path.join(folder, "*.npy")):
            X.append(np.load(f))
            y.append(chord)
    if not X:
        raise FileNotFoundError("No data found. Run collect_data.py first.")
    return np.array(X), np.array(y)

def save_confusion_matrix(cm, labels, path):
    plt.figure(figsize=(6,6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45)
    plt.yticks(ticks, labels)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def main():
    X, y = load_data()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    Xtr, Xte, ytr, yte = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1))
    ])
    model.fit(Xtr, ytr)

    ypred = model.predict(Xte)
    acc = accuracy_score(yte, ypred)
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(yte, ypred, target_names=le.classes_))

    cm = confusion_matrix(yte, ypred)
    save_confusion_matrix(cm, le.classes_, CONF_MATRIX_PATH)
    print(f"Confusion matrix saved at {CONF_MATRIX_PATH}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print("Model + Encoder saved.")

if __name__ == "__main__":
    main()

    
    
    
    

'''EVALUATION METRICS:
              
              Test Accuracy: 0.9732
              precision    recall  f1-score   support

           A       1.00      1.00      1.00        16
          Am       0.93      0.88      0.90        16
           C       1.00      1.00      1.00        16
           D       1.00      1.00      1.00        16
           E       0.88      0.94      0.91        16
          Em       1.00      1.00      1.00        16
           G       1.00      1.00      1.00        16

           G       1.00      1.00      1.00        16


    accuracy                           0.97       112
   macro avg       0.97      0.97      0.97       112
    accuracy                           0.97       112
   macro avg       0.97      0.97      0.97       112
weighted avg       0.97      0.97      0.97       112
   macro avg       0.97      0.97      0.97       112
weighted avg       0.97      0.97      0.97       112

weighted avg       0.97      0.97      0.97       112





Test Accuracy: 0.9792
              precision    recall  f1-score   support

           A       0.98      0.98      0.98        56
          Am       0.93      0.98      0.96        56
           C       1.00      1.00      1.00        56
           D       1.00      0.98      0.99        56
           E       0.98      0.93      0.95        56
          Em       0.98      1.00      0.99        56

    accuracy                           0.98       336
   macro avg       0.98      0.98      0.98       336
weighted avg       0.98      0.98      0.98       336'''



'''
Get-ChildItem pose_tracking\data -Recurse -Filter *.npy | 
Group-Object { $_.Directory.Name } | 
Select-Object Name, @{Name="Count";Expression={$_.Count}}
'''