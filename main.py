from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
import cv2
import os

# ---------------------------------------------------------------------------
# GPU config — disable if running on CPU-only / M1 Mac
# Comment this out if you have an NVIDIA GPU available at inference time
# ---------------------------------------------------------------------------
tf.config.set_visible_devices([], 'GPU')

app = FastAPI(title="Night → Day Translator")

# Serve static files (HTML, CSS, any saved images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------------------------
# Model loading
# No custom objects needed — generator.py is now clean
# ---------------------------------------------------------------------------
print("Loading generator model...")
try:
    model = tf.keras.models.load_model("model/GAN_Generator.keras")
    # Warm-up pass so first real request isn't slow
    _ = model.predict(np.zeros((1, 256, 256, 3), dtype=np.float32), verbose=0)
    print("Model loaded and warmed up successfully.")
except Exception as e:
    print(f"ERROR: Could not load model — {e}")
    raise RuntimeError(f"Model loading failed: {e}")


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def preprocess(image_path: str):
    """
    Load image from disk, resize to 256×256, normalize to [-1, 1].
    Returns (model_input, original_h, original_w).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at: {image_path}")

    original_h, original_w = img.shape[:2]

    # Convert BGR → RGB (model was trained on RGB data via PIL/HuggingFace)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    # Normalize to [-1, 1] — matches training pipeline
    img = (img.astype(np.float32) / 127.5) - 1.0
    return np.expand_dims(img, axis=0), original_h, original_w


def postprocess(output: np.ndarray, original_h: int, original_w: int) -> np.ndarray:
    """
    Convert model output ([-1,1] float) back to uint8 RGB,
    then resize to original dimensions and convert to BGR for cv2.imwrite.
    """
    # Denormalize: [-1, 1] → [0, 255]
    img = ((output[0] + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    # Resize back to original resolution
    if (original_h, original_w) != (256, 256):
        img = cv2.resize(img, (original_w, original_h), interpolation=cv2.INTER_AREA)

    # Convert RGB → BGR for OpenCV saving
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def run_inference(input_path: str, output_path: str):
    """Full pipeline: load → preprocess → predict → postprocess → save."""
    input_tensor, orig_h, orig_w = preprocess(input_path)
    raw_output = model.predict(input_tensor, verbose=0)
    result = postprocess(raw_output, orig_h, orig_w)
    cv2.imwrite(output_path, result)


# ---------------------------------------------------------------------------
# Routes — static pages
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_main():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/upload", response_class=HTMLResponse)
async def serve_upload():
    with open("static/upload.html", "r") as f:
        return HTMLResponse(content=f.read())


# ---------------------------------------------------------------------------
# Routes — inference
# ---------------------------------------------------------------------------

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    """
    Upload a night image → receive a day image.
    Returns URLs for both input and output images.
    """
    input_path  = "static/input.png"
    output_path = "static/output.png"

    # Save uploaded file
    try:
        contents = await file.read()
        with open(input_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    # Run model
    try:
        run_inference(input_path, output_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return {
        "input_image_url":  "/static/input.png",
        "output_image_url": "/static/output.png",
    }


# ---------------------------------------------------------------------------
# Routes — utility
# ---------------------------------------------------------------------------

@app.post("/reset")
async def reset():
    """Clear any saved images from a previous session."""
    paths = ["static/input.png", "static/output.png"]
    removed = []
    try:
        for p in paths:
            if os.path.exists(p):
                os.remove(p)
                removed.append(p)
        return {"message": "Reset successful", "removed": removed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")


@app.get("/health")
async def health():
    """Quick liveness check — useful for debugging on Kaggle/local."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape),
    }