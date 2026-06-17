
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from skimage.feature import hog
import joblib

# ---------------- CONFIG ----------------
DATASET_DIR = "PetImages"      # folder containing Cat/ and Dog/ subfolders
IMG_SIZE = 128                 # images are resized to IMG_SIZE x IMG_SIZE (bumped up for more detail)
IMAGES_PER_CLASS = 5000        # increase further (or set to None for all ~12,500/class) once this runs fine
SVM_C = 15                     # higher C = fits training data more closely
RANDOM_STATE = 42
# -----------------------------------------


def extract_features(img_path):
    """Load an image, resize it, and extract a HOG feature vector."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )
    return features


def load_dataset(dataset_dir, images_per_class=None):
    """Walk Cat/ and Dog/ folders and build feature/label/path arrays."""
    classes = {"Cat": 0, "Dog": 1}
    X, y, paths = [], [], []

    for class_name, label in classes.items():
        folder = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Couldn't find folder: {folder}")

        files = sorted(os.listdir(folder))
        if images_per_class:
            files = files[:images_per_class]

        print(f"Loading {len(files)} images from {folder} ...")
        for i, fname in enumerate(files):
            path = os.path.join(folder, fname)
            features = extract_features(path)
            if features is None:
                continue  # skip corrupt/unreadable files
            X.append(features)
            y.append(label)
            paths.append(path)

            if (i + 1) % 200 == 0:
                print(f"  {class_name}: {i + 1}/{len(files)} processed")

    return np.array(X), np.array(y), np.array(paths)


def save_confusion_matrix_plot(y_test, y_pred, out_path="confusion_matrix.png"):
    """Save the confusion matrix as a labeled heatmap image."""
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cat", "Dog"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix image to {out_path}")


def save_sample_predictions_plot(paths_test, y_test, y_pred, n=12, out_path="sample_predictions.png"):
    """Save a grid of test images with their true vs predicted labels."""
    label_names = {0: "Cat", 1: "Dog"}
    n = min(n, len(paths_test))
    idx = np.random.choice(len(paths_test), size=n, replace=False)

    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for ax, i in zip(axes, idx):
        img = cv2.imread(paths_test[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        true_label = label_names[y_test[i]]
        pred_label = label_names[y_pred[i]]
        correct = true_label == pred_label
        color = "green" if correct else "red"

        ax.imshow(img)
        ax.set_title(f"True: {true_label} | Pred: {pred_label}", color=color, fontsize=10)
        ax.axis("off")

    # Hide any unused subplots
    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved sample predictions image to {out_path}")


def main():
    print("Loading and featurizing dataset (this can take a few minutes)...")
    X, y, paths = load_dataset(DATASET_DIR, IMAGES_PER_CLASS)
    print(f"Total samples: {len(y)}  |  Feature length: {X.shape[1]}")

    # Split into train/test (paths split the same way so they line up with y_test)
    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
        X, y, paths, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Scale features (important for SVMs)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train SVM
    print("Training SVM (this may take a while)...")
    clf = svm.SVC(kernel="rbf", C=SVM_C, gamma="scale")
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save visual outputs (actual image files)
    save_confusion_matrix_plot(y_test, y_pred)
    save_sample_predictions_plot(paths_test, y_test, y_pred)

    # Save the trained model and scaler for later use
    joblib.dump(clf, "svm_cats_dogs_model.joblib")
    joblib.dump(scaler, "svm_cats_dogs_scaler.joblib")
    print("\nSaved model to svm_cats_dogs_model.joblib")


def predict_single_image(image_path, model_path="svm_cats_dogs_model.joblib",
                          scaler_path="svm_cats_dogs_scaler.joblib"):
    """Helper to classify a single new image after training."""
    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    features = extract_features(image_path)
    if features is None:
        print("Couldn't read image.")
        return
    features = scaler.transform([features])
    pred = clf.predict(features)[0]
    label = "Dog" if pred == 1 else "Cat"
    print(f"Prediction: {label}")


if __name__ == "__main__":
    main()
    # Example usage after training:
    # predict_single_image("some_test_image.jpg")
