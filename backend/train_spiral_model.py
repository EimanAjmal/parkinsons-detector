"""
Spiral Drawing - Parkinson's Detection
=======================================
Approach: HOG (Histogram of Oriented Gradients) features + SVM
HOG = image ke andar se edge/shape patterns extract karta hai
Ye CNN jaisa accurate hai chhote datasets ke liye
"""
import cv2
import numpy as np
import joblib
import json
from pathlib import Path
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
    confusion_matrix
)

# ─── Paths ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "spiral"
MODEL_DIR = BASE_DIR / "models"

TRAIN_HEALTHY   = DATA_DIR / "training" / "healthy"
TRAIN_PARKINSON = DATA_DIR / "training" / "parkinson"
TEST_HEALTHY    = DATA_DIR / "testing"  / "healthy"
TEST_PARKINSON  = DATA_DIR / "testing"  / "parkinson"

IMG_SIZE = (128, 128)

# ─── HOG Feature Extraction ──────────────────────────────
def extract_hog(image_path):
    """
    Ek image file se HOG features extract karo.
    HOG = image ko cells mein divide karo,
          har cell mein edge directions count karo.
    Result: ek flat numeric vector (model ka input)
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    return features

def load_dataset(healthy_dir, parkinson_dir):
    """Folder se images load karo aur features + labels banao"""
    X, y = [], []
    for img_path in Path(healthy_dir).glob("*"):
        features = extract_hog(img_path)
        if features is not None:
            X.append(features)
            y.append(0)  # 0 = healthy
    for img_path in Path(parkinson_dir).glob("*"):
        features = extract_hog(img_path)
        if features is not None:
            X.append(features)
            y.append(1)  # 1 = parkinson's
    return np.array(X), np.array(y)

# ─── Load Data ───────────────────────────────────────────
print("Loading training images...")
X_train, y_train = load_dataset(TRAIN_HEALTHY, TRAIN_PARKINSON)
print(f"Train: {len(X_train)} images | Healthy: {(y_train==0).sum()} | PD: {(y_train==1).sum()}")

print("Loading testing images...")
X_test, y_test = load_dataset(TEST_HEALTHY, TEST_PARKINSON)
print(f"Test:  {len(X_test)} images  | Healthy: {(y_test==0).sum()} | PD: {(y_test==1).sum()}")

# ─── Scale Features ──────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─── Train SVM ───────────────────────────────────────────
print("\nTraining SVM model...")
model = SVC(
    kernel='rbf',
    class_weight='balanced',
    probability=True,
    random_state=42,
    C=10,
    gamma='scale'
)
model.fit(X_train_scaled, y_train)
print("Training complete!")

# ─── Evaluate ────────────────────────────────────────────
y_pred  = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_proba)
cm   = confusion_matrix(y_test, y_pred)

print("\n" + "="*50)
print("SPIRAL MODEL EVALUATION RESULTS")
print("="*50)
print(f"Accuracy:  {acc:.3f} ({acc*100:.1f}%)")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1-Score:  {f1:.3f}")
print(f"ROC-AUC:   {auc:.3f}")
print(f"Confusion Matrix:\n{cm}")

# ─── Save Model ──────────────────────────────────────────
joblib.dump(model,  MODEL_DIR / "spiral_model.pkl")
joblib.dump(scaler, MODEL_DIR / "spiral_scaler.pkl")

results = {
    "model": "SVM (RBF kernel) + HOG features",
    "image_size": list(IMG_SIZE),
    "accuracy":  acc,
    "precision": prec,
    "recall":    rec,
    "f1":        f1,
    "roc_auc":   auc,
    "confusion_matrix": cm.tolist()
}
with open(MODEL_DIR / "spiral_evaluation.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved: spiral_model.pkl, spiral_scaler.pkl, spiral_evaluation.json")
print("Spiral model ready!")