import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image


VAL_DIR       = 'night2day/val'   
MODEL_PATH    = 'model/GAN_Generator.keras'
IMG_SIZE      = 256
NUM_SAMPLES   = 5
USE_HUGGINGFACE = True   


def load_from_huggingface(num_samples=NUM_SAMPLES):
    """
    Load a few validation samples directly from the HuggingFace dataset
    (works when internet is available).
    imageA = night, imageB = day
    """
    from datasets import load_dataset
    # Only download a slice — no need to pull all 20K for prediction
    ds = load_dataset("huggan/night2day", split=f"train[:{num_samples * 10}]")

    night_images = []
    day_images   = []

    for sample in ds:
        night = _preprocess_pil(sample["imageA"])
        day   = _preprocess_pil(sample["imageB"])
        night_images.append(night)
        day_images.append(day)

    return np.array(night_images), np.array(day_images)


def load_from_local(val_dir, num_samples=NUM_SAMPLES):
    """
    Load paired images from a local directory.
    Supports two formats:
      1. Side-by-side pairs (single image, night on left, day on right)
      2. Separate files in val_dir/night/ and val_dir/day/ subfolders
    """
    # Format 2: separate subfolders
    night_dir = os.path.join(val_dir, 'night')
    day_dir   = os.path.join(val_dir, 'day')
    if os.path.isdir(night_dir) and os.path.isdir(day_dir):
        night_paths = sorted([
            os.path.join(night_dir, f)
            for f in os.listdir(night_dir) if f.endswith(('.jpg', '.png'))
        ])
        day_paths = sorted([
            os.path.join(day_dir, f)
            for f in os.listdir(day_dir) if f.endswith(('.jpg', '.png'))
        ])
        night_images = [_preprocess_pil(Image.open(p).convert("RGB")) for p in night_paths]
        day_images   = [_preprocess_pil(Image.open(p).convert("RGB")) for p in day_paths]
        return np.array(night_images), np.array(day_images)

    # Format 1: side-by-side pairs (same format as original maps dataset)
    image_paths = sorted([
        os.path.join(val_dir, f)
        for f in os.listdir(val_dir) if f.endswith(('.jpg', '.png'))
    ])
    night_images, day_images = [], []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        half = w // 2
        night = _preprocess_pil(img.crop((0, 0, half, h)))   # left = night
        day   = _preprocess_pil(img.crop((half, 0, w, h)))   # right = day
        night_images.append(night)
        day_images.append(day)

    return np.array(night_images), np.array(day_images)


def _preprocess_pil(img):
    """PIL Image → numpy float32 in [-1, 1] at 256×256. Matches training pipeline."""
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32)
    return (arr / 127.5) - 1.0   # [-1, 1]


def _to_display(arr):
    """Convert [-1, 1] float array back to uint8 [0, 255] for matplotlib."""
    return np.clip(((arr + 1.0) * 127.5), 0, 255).astype(np.uint8)


def show_predictions(generator, night_images, day_images, num_samples=NUM_SAMPLES):
    """
    For num_samples random images, show:
      [Night Input] | [Ground Truth Day] | [Generated Day]
    """
    indices = np.random.choice(len(night_images), min(num_samples, len(night_images)), replace=False)

    for idx in indices:
        night_input  = night_images[idx]
        ground_truth = day_images[idx]

        # Run generator — input must be [-1, 1], output is also [-1, 1]
        batch_input = np.expand_dims(night_input, axis=0)
        predicted   = generator.predict(batch_input, verbose=0)[0]

        # Convert all to display-ready uint8
        night_vis  = _to_display(night_input)
        gt_vis     = _to_display(ground_truth)
        pred_vis   = _to_display(predicted)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Sample #{idx}", fontsize=13)

        axes[0].imshow(night_vis);  axes[0].set_title("Night Input");       axes[0].axis('off')
        axes[1].imshow(gt_vis);     axes[1].set_title("Ground Truth (Day)"); axes[1].axis('off')
        axes[2].imshow(pred_vis);   axes[2].set_title("Generated Day");      axes[2].axis('off')

        plt.tight_layout()
        plt.show()

def main():
    # 1. Load model
    try:
        generator = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded: {MODEL_PATH}")
        print(f"  Input shape:  {generator.input_shape}")
        print(f"  Output shape: {generator.output_shape}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Load validation data
    try:
        if USE_HUGGINGFACE:
            print("Loading samples from HuggingFace (night2day)...")
            night_images, day_images = load_from_huggingface(num_samples=NUM_SAMPLES)
        else:
            print(f"Loading samples from local directory: {VAL_DIR}")
            night_images, day_images = load_from_local(VAL_DIR, num_samples=NUM_SAMPLES)

        print(f"Loaded {len(night_images)} pairs — shape: {night_images[0].shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 3. Show predictions
    show_predictions(generator, night_images, day_images, num_samples=NUM_SAMPLES)


if __name__ == "__main__":
    main()