# CNN model build with Tensorflow and keras as a part of gamma spectrum analysis project by A.AbuAli

# ─────────────────0. Redirect stdout & stderr to both console and a log file ──────────────────────────────────────────
# This class Tee allows you to log everything printed to console into a file too – helpful for record-keeping
import sys
import os

log_path = "training_console_PU_U.log"
log_file = open(log_path, "w", buffering=1)  # line‐buffered


class Tee:
    def __init__(self, *streams, encoding="utf-8", errors="ignore"):
        self.streams = streams
        self.encoding = encoding
        self.errors = errors

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


sys.stdout = Tee(sys.stdout, log_file, encoding="utf-8", errors="ignore")
sys.stderr = Tee(sys.stderr, log_file, encoding="utf-8", errors="ignore")

# ──────────────────────────────────────1. Imports ─────────────────────────────────────────────────────────────
# Core libraries for ML, data handling, visualization, and metrics
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

try:
    import seaborn as sns

    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False

# ─────────────────────────────────────────── 2. Settings ──────────────────────────────────────────────────────────────
# Core libraries for ML, data handling, visualization, and metrics
data_dir = Path(r"C:\Users\Rodion\Desktop\Spectra_Raw_Database")
batch_size = 128
num_epochs = 15
seed = 42
train_frac = 0.75
val_frac = 0.10
random.seed(seed)
tf.random.set_seed(seed)

# ────────────────────────────────────── 3. Gather & shuffle files ─────────────────────────────────────────────────────
all_files = list(data_dir.rglob("*.spe"))

# ──────────────────────────────────────────── 4. Infer classes ────────────────────────────────────────────────────────
# Dynamically extract class labels from folder names
classes = sorted({p.parent.name for p in all_files})
class_to_idx = {c: i for i, c in enumerate(classes)}
num_classes = len(classes)
print(f"{len(all_files)} spectra across classes = {classes}")


# ────────────────────────────────── 5. Dynamic preprocessing ──────────────────────────────────────────────────────────
# Grab all .spe files recursively and shuffle them for randomness
def preprocess_spe(path_bytes):
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

    arr = np.array(counts, dtype=np.float32)
    arr = arr / np.max(arr)  # Normalize spectrum to [0,1]
    arr = arr.reshape(-1, 1)

    lbl = np.int32(class_to_idx[Path(path_str).parent.name])
    return arr, lbl


def tf_parse(path_tensor):
    # Bridge between raw path strings and TensorFlow Dataset pipeline
    spec, lbl = tf.py_function(
        func=preprocess_spe,
        inp=[path_tensor],
        Tout=[tf.float32, tf.int32]
    )
    spec.set_shape([None, 1])
    lbl.set_shape([])
    return spec, lbl


# ─────────────────────── 6. Stratified split using sklearn, then tf.data pipeline ─────────────────────────────────────
# Create dataset, split into train/val/test with proper padding for variable-length inputs
file_paths = np.array([str(p) for p in all_files])
labels = np.array([class_to_idx[p.parent.name] for p in all_files])

train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
    file_paths, labels, test_size=1 - (train_frac + val_frac), random_state=seed, stratify=labels
)
train_paths, val_paths, _, _ = train_test_split(
    train_val_paths, train_val_labels, test_size=val_frac / (train_frac + val_frac), random_state=seed,
    stratify=train_val_labels
)


def create_dataset(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


train_ds = create_dataset(train_paths)
val_ds = create_dataset(val_paths)
test_ds = create_dataset(test_paths)


# ───────────────────────────────── 7. Augmentation for variable-length sequences ──────────────────────────────────────
# Slightly tweak the spectrum: shift, scale, and add some Poisson noise
def augment(spec, lbl):
    seq_len = tf.shape(spec)[0]
    shift = tf.random.uniform([], 0, seq_len, dtype=tf.int32)
    spec = tf.roll(spec, shift, axis=0)

    scale = tf.random.uniform([], 0.8, 1.2)
    spec = spec * scale

    noise = tf.random.normal(shape=tf.shape(spec), mean=0.0, stddev=0.02)
    spec = tf.clip_by_value(spec + noise, 0.0, 1.0)

    return spec, lbl


train_ds = (
    train_ds.shuffle(3000, seed=seed)
    .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size, padded_shapes=([None, 1], []))
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = val_ds.padded_batch(batch_size, padded_shapes=([None, 1], [])).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.padded_batch(batch_size, padded_shapes=([None, 1], [])).prefetch(tf.data.AUTOTUNE)


# ─────────── 8.The 1D CNN Model ───────────
# A powerful but regularized ConvNet for 1D signal classification:
def build_cnn_Pu_U(num_classes):
    inputs = layers.Input(shape=(None, 1))
    # Conv. Block 1
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout1D(0.1)(x)
    x = layers.MaxPooling1D(2)(x)
    #Conv. Block 2
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout1D(0.1)(x)
    x = layers.MaxPooling1D(2)(x)
    #Conv. Block 3
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    #Dense layers
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)


# Instantiate and compile model:
model = build_cnn_Pu_U(num_classes)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ─────────── 9. Capture and print model summary ───────────
# Log model architecture to a text file and display
summary_lines = []
model.summary(print_fn=lambda line: summary_lines.append(line))
with open("model_summary.txt", "w", encoding="utf-8") as f:
    for line in summary_lines:
        f.write(line + "\n")
for line in summary_lines:
    sys.__stdout__.write(line + "\n")

# ─────────── 10. Train with callbacks ───────────
# Training with early stopping, learning rate adjustment, and best-weight saving
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, min_delta=0.01, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
    keras.callbacks.ModelCheckpoint(filepath='best_weights.weights.h5', save_best_only=True, save_weights_only=True),
]

history = model.fit(train_ds, validation_data=val_ds, epochs=num_epochs, callbacks=callbacks)

# ─────────── 11. Evaluate & Save Model ───────────
# Load best weights, evaluate on test set, and save model
model.load_weights('best_weights.weights.h5')
loss, acc = model.evaluate(test_ds)
print(f"\nTest accuracy: {acc:.4f}, Test loss: {loss:.4f}")
model.save("CNN_1d_Pu&.h5")

# ─────────── 12. Plot Training Curves ───────────
# Visualize accuracy and loss trends over epochs
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.savefig('training_curves_Pu&U.png', dpi=120)
plt.close()
print("Saved training curves to 'training_curves_Pu&U.png'")

# ─────────── 13. Compute and Plot Confusion Matrix ───────────
# Evaluate per-class performance using confusion matrix and classification report
y_true, y_pred = [], []
for x_batch, y_batch in test_ds:
    preds = model.predict(x_batch).argmax(axis=1)
    y_true.append(y_batch.numpy())
    y_pred.append(preds)

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)
cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
print("\nConfusion Matrix (rows=true, cols=pred):")
print(cm)
report = classification_report(y_true, y_pred, target_names=classes, digits=4)
print("\nClassification Report:")
print(report)

plt.figure(figsize=(6, 5))
if _HAS_SEABORN:
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
else:
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    ticks = np.arange(num_classes)
    plt.xticks(ticks, classes, rotation=45, ha='right')
    plt.yticks(ticks, classes)
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix_rebalanced.png', dpi=120)
plt.close()
print("Saved confusion matrix to 'confusion_matrix_pu&U.png'")
log_file.close()
