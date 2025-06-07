import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt

# ─── 1. Define the 1D CNN architecture ────────────────────────────────────────
def build_balanced_cnn(num_classes):
    """
    1D CNN for raw gamma spectra with dynamic input length:
      - Input(shape=(None, 1)): num_channels can vary per example.
      - Three Conv1D blocks, each followed by batch norm, dropout, and pooling.
      - GlobalAveragePooling1D collapses variable-length maps to a fixed 128-dim vector.
      - Dense(256, ReLU, L2) → Dropout(0.5) → Dense(num_classes, softmax).
    """
    inputs = layers.Input(shape=(None, 1))  # (num_channels, 1) variable length

    # Conv Block 1
    x = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout1D(0.1)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Conv Block 2
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout1D(0.1)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Conv Block 3
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Global pooling to get a fixed-length feature vector
    x = layers.GlobalAveragePooling1D()(x)  # → shape (128,)

    # Dense layers
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs, outputs)

# ─── 2. Set num_classes to match the trained model ───────────────────────────
# If you trained with 3 classes (e.g., Natural / Enriched / Depleted), set num_classes = 3.
# If you trained with 2 classes (e.g., Uranium vs. Plutonium), set num_classes = 2.
num_classes = 2

# ─── 3. Instantiate and compile the model exactly as during training ──────────
model_1d = build_balanced_cnn(num_classes)
model_1d.compile(
    optimizer=keras.optimizers.Adam(),  # optimizer settings not used during evaluation
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ─── 4. Load the best‐trained weights ─────────────────────────────────────────
model_1d.load_weights(r'C:\Users\Rodion\PycharmProjects\CNN_Model_Files\best_weights.weights.h5')  # adjust path if needed

# ─── 5. Reconstruct or import test_ds ─────────────────────────────────────────
# You must recreate the same 'test_ds' tf.data.Dataset used during training.
# Here is an example snippet for rebuilding test_ds—adjust data_dir and parsing logic to your environment:

import os
from pathlib import Path

# a) Define class_to_idx mapping exactly as in training:
#    For example, if your folders were: "Natural", "Enriched", "Depleted"
class_to_idx = {
    "Natural_Uranium": 0,
    "Enriched_Uranium": 1,

}

# b) Rebuild test_ds from filepaths (adjust data_dir path)
data_dir = Path(r"C:\Users\Rodion\Desktop\Spectra_Raw_Database")
all_files = list(data_dir.rglob("*.spe"))

# Random shuffle with the same seed used in training
import random
random.seed(42)
random.shuffle(all_files)

# Compute train/val/test splits exactly as training:
N = len(all_files)
n_train = int(0.65 * N)
n_val   = int(0.20 * N)
# Test starts at index (n_train + n_val)
test_files = all_files[n_train + n_val :]

# c) Define the same preprocess_spe and tf_parse as in training:

def preprocess_spe(path_bytes):
    """
    Read a .spe file and return (counts_array, label).
    """
    val = path_bytes.numpy() if hasattr(path_bytes, 'numpy') else path_bytes
    path_str = val.decode('utf-8', 'ignore') if isinstance(val, (bytes, bytearray)) else str(val)

    counts = []
    in_data = False
    with open(path_str, 'r', encoding='latin-1', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not in_data:
                if line.upper().startswith('$DATA'):
                    in_data = True
                continue
            if line.startswith('$'):
                break
            if line.isdigit():
                counts.append(int(line))

    if not counts:
        raise ValueError(f"No data found in {path_str}")

    arr = np.array(counts, dtype=np.float32).reshape(-1, 1)  # (num_channels, 1)
    lbl = np.int32(class_to_idx[Path(path_str).parent.name])
    return arr, lbl

def tf_parse(path_tensor):
    """
    Wrapper so we can use preprocess_spe in tf.data.
    """
    spec, lbl = tf.py_function(
        func=preprocess_spe,
        inp=[path_tensor],
        Tout=[tf.float32, tf.int32]
    )
    spec.set_shape([None, 1])
    lbl.set_shape([])
    return spec, lbl

# d) Build test_ds with padded batches:
paths_ds = tf.data.Dataset.from_tensor_slices([str(p) for p in test_files])
ds_all   = paths_ds.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
test_ds  = ds_all.padded_batch(batch_size=256, padded_shapes=([None, 1], [])).prefetch(tf.data.AUTOTUNE)

# ─── 6. Evaluate on test_ds ───────────────────────────────────────────────────
loss_1d, acc_1d = model_1d.evaluate(test_ds)
print(f"1D CNN Test Loss: {loss_1d:.4f}, Test Accuracy: {acc_1d:.4f}")

# ─── 7. Build confusion matrix and classification report ───────────────────────
y_true, y_pred = [], []
for spec_batch, label_batch in test_ds:
    logits = model_1d.predict(spec_batch)          # (batch_size, num_classes)
    preds = np.argmax(logits, axis=1)               # (batch_size,)
    y_true.extend(label_batch.numpy().tolist())
    y_pred.extend(preds.tolist())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
print("Confusion Matrix (rows=true, cols=predicted):")
print(cm)

# Classification report (update class_names to your actual labels)
class_names = ['Natural', 'Enriched']
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print("Classification Report:")
print(report)

# ─── 8. Plot & save confusion matrix ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('1D CNN Confusion Matrix')
plt.colorbar(im, ax=ax)

tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

thresh = cm.max() / 2
for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black')

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_1d_eval.png', dpi=150)
plt.close()
print("Saved confusion matrix to 'confusion_matrix_1d_eval.png'")
