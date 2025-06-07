import cv2
import numpy as np
import os

# === Settings ===
input_folder = "Raw-Spectra"         # Input folder
output_folder = "Cleaned-Spectra_last"     # Output folder
target_size = (512, 512)            # High resolution for CNN
preview = True                        # Preview mode

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# === Helper Functions ===

def hard_crop_margins(img, margin_percent=0.1):
    """Crop fixed margins to remove labels/titles."""
    h, w = img.shape[:2]
    top = int(margin_percent * h)
    bottom = int((1 - margin_percent) * h)
    left = int(margin_percent * w)
    right = int((1 - margin_percent) * w)
    cropped = img[top:bottom, left:right]
    return cropped

def normalize_image(img):
    """Normalize pixel intensities to 0-1."""
    img = img.astype(np.float32)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    return img

def resize_with_padding(img, target_size):
    """Resize while keeping aspect ratio, add padding."""
    h, w = img.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad with black
    canvas = np.zeros((target_h, target_w), dtype=np.float32)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def sharpen_image(img):
    """Apply light sharpening to enhance plot lines."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened

def preprocess_image(img_path):
    """Full cleaning pipeline."""
    # Load and grayscale
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Hard crop margins
    cropped = hard_crop_margins(gray, margin_percent=0.1)

    # Threshold to isolate plot
    _, thresh = cv2.threshold(cropped, 245, 255, cv2.THRESH_BINARY_INV)

    # Find bounding box
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        fine_cropped = cropped[y:y+h, x:x+w]
    else:
        fine_cropped = cropped

    # Light denoising
    blurred = cv2.GaussianBlur(fine_cropped, (3, 3), 0)

    # Sharpen
    sharpened = sharpen_image(blurred)

    # Normalize
    normalized = normalize_image(sharpened)

    # Resize with padding
    final_img = resize_with_padding(normalized, target_size)

    return gray, final_img

# === Process All Images ===

for filename in os.listdir(input_folder):
    if filename.lower().endswith('.png'):
        img_path = os.path.join(input_folder, filename)
        original_gray, cleaned_img = preprocess_image(img_path)

        # Build label from filename
        label_name = filename.split('_')[0]  # Before underscore if exists
        if label_name == filename:  # If no underscore, use full name without extension
            label_name = os.path.splitext(filename)[0]

        # Create a subfolder for each label (optional, nice for organization)
        label_folder = os.path.join(output_folder, label_name)
        os.makedirs(label_folder, exist_ok=True)

        # Count how many files already exist for this label
        existing_files = [f for f in os.listdir(label_folder) if f.endswith('.png')]
        file_number = len(existing_files) + 1

        # Build new filename
        new_filename = f"{label_name}_{file_number:03d}.png"
        out_path = os.path.join(new_filename)

        # Save image
        cv2.imwrite(out_path, (cleaned_img * 255).astype(np.uint8))

        print(f"Saved: {out_path}")

        # === PREVIEW Mode ===
        # if preview:
        #     # Resize original for comparison
        #     orig_cropped = hard_crop_margins(normalize_image(original_gray), margin_percent=0.1)
        #     orig_resized = resize_with_padding(orig_cropped, target_size)
        #
        #     # Stack original and cleaned images side by side
        #     combined = np.hstack((orig_resized, cleaned_img))
        #     combined = (combined * 255).astype(np.uint8)
        #
        #     cv2.imshow('Original (left) vs Cleaned (right)', combined)
        #     key = cv2.waitKey(0)
        #     if key == 27:  # ESC to exit preview early
        #         preview = False
        #     cv2.destroyAllWindows()
