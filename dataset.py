#dataset
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from PIL import Image
import io

datasize = '40%'

class DatasetLoader:
    def __init__(self, dataset_path=None, image_size=128):
        self.image_size = image_size
        # dataset_path is unused — data is loaded from HuggingFace

    def _to_pil(self, img_field):
        """
        HuggingFace image columns can come back in two different forms
        depending on the dataset version and how it was converted to Parquet:

          1. Already a PIL.Image object  → use directly
          2. A dict like {'bytes': b'...', 'path': '...'}  → decode bytes manually

        This handles both so the code doesn't break regardless of HF version.
        """
        if isinstance(img_field, Image.Image):
            return img_field

        if isinstance(img_field, dict):
            raw_bytes = img_field.get("bytes")
            if raw_bytes:
                return Image.open(io.BytesIO(raw_bytes))
            # Fallback: if only a path is present (rare), open from disk
            path = img_field.get("path")
            if path:
                return Image.open(path)

        raise ValueError(
            f"Unrecognised image field type: {type(img_field)}. "
            f"Value: {str(img_field)[:120]}"
        )

    def _preprocess(self, img_field):
        """Convert a raw HF image field → float32 numpy array in [-1, 1]."""
        img = self._to_pil(img_field)
        img = img.convert("RGB").resize(
            (self.image_size, self.image_size), Image.BICUBIC
        )
        arr = np.array(img, dtype=np.float32)
        return (arr / 127.5) - 1.0   # [-1, 1] to match tanh output

    def load_data(self):
        print("Loading huggan/night2day from HuggingFace...")
        ds = load_dataset("huggan/night2day", split=f"train[:{datasize}]")

        night_images = []
        day_images   = []

        for i, sample in enumerate(ds):
            night_images.append(self._preprocess(sample["imageA"]))  # night → input
            day_images.append(self._preprocess(sample["imageB"]))    # day   → target
            if (i + 1) % 2000 == 0:
                print(f"  Processed {i + 1}/{len(ds)} pairs...")

        print(f"Done. Loaded {len(night_images)} pairs.")
        return np.array(night_images), np.array(day_images)

    def get_dataset(self, images, masks, batch_size=1):
        dataset = tf.data.Dataset.from_tensor_slices((images, masks))
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset