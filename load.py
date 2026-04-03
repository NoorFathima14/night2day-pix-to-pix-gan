from datasets import load_dataset
import numpy as np
from PIL import Image

# Downloads ~3.3GB, cached after first run
datasize = '40%'
ds = load_dataset("huggan/night2day", split=f"train[:{datasize}]")

print(f"Total pairs: {len(ds)}")
print(f"Columns: {ds.column_names}")  # ['imageA', 'imageB']