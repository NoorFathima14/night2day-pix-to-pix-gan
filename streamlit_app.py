from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


DEFAULT_MODEL_CANDIDATES = [
    Path("model/GAN_Generator.keras"),
    Path("model/generator_latest.keras"),
    Path("model/best_generator.keras"),
]
DEFAULT_IMAGE_SIZE = 256


def configure_tensorflow() -> None:
    """Prefer CPU inference to avoid startup issues on machines without GPU support."""
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass


def find_available_models() -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    for path in DEFAULT_MODEL_CANDIDATES:
        if path.exists():
            resolved = str(path.resolve())
            if resolved not in seen:
                candidates.append(path)
                seen.add(resolved)

    model_dir = Path("model")
    if model_dir.exists():
        for path in sorted(model_dir.glob("*.keras")):
            name = path.name.lower()
            if "generator" not in name:
                continue
            resolved = str(path.resolve())
            if resolved not in seen:
                candidates.append(path)
                seen.add(resolved)

    return candidates


@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> tf.keras.Model:
    model = tf.keras.models.load_model(model_path)
    height, width = get_model_image_size(model)
    _ = model.predict(np.zeros((1, height, width, 3), dtype=np.float32), verbose=0)
    return model


def get_model_image_size(model: tf.keras.Model) -> tuple[int, int]:
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, list):
        shape = shape[0]

    if not shape or len(shape) < 4:
        return DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE

    height = shape[1] or DEFAULT_IMAGE_SIZE
    width = shape[2] or DEFAULT_IMAGE_SIZE
    return int(height), int(width)


def preprocess_image(image: Image.Image, target_size: tuple[int, int]) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((target_size[1], target_size[0]), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims((array / 127.5) - 1.0, axis=0)


def postprocess_image(output: np.ndarray, original_size: tuple[int, int]) -> Image.Image:
    image = ((output[0] + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    pil_image = Image.fromarray(image)
    if pil_image.size != original_size:
        pil_image = pil_image.resize(original_size, Image.BICUBIC)
    return pil_image


def render_sidebar(default_model_path: str) -> str:
    st.sidebar.header("Settings")
    st.sidebar.caption("Choose a generator checkpoint or type a custom path.")

    discovered_models = find_available_models()
    select_options = ["Custom path"] + [str(path) for path in discovered_models]

    if default_model_path in select_options:
        default_index = select_options.index(default_model_path)
    elif discovered_models:
        default_index = 1
    else:
        default_index = 0

    selected_option = st.sidebar.selectbox(
        "Model checkpoint",
        options=select_options,
        index=default_index,
    )

    custom_path = st.sidebar.text_input(
        "Model path",
        value=default_model_path if selected_option == "Custom path" else selected_option,
        help="Path to a saved TensorFlow/Keras generator model.",
    )

    st.sidebar.markdown(
        "Expected model behavior: night image in, translated day image out."
    )
    return custom_path.strip()


def image_to_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_image(image: Image.Image) -> None:
    try:
        st.image(image, use_container_width=True)
    except TypeError:
        st.image(image, use_column_width=True)


def main() -> None:
    configure_tensorflow()
    st.set_page_config(
        page_title="Night to Day Pix2Pix",
        page_icon=":sunrise:",
        layout="wide",
    )

    st.title("Night to Day Image Translation")
    st.write(
        "Upload a night-time scene and run the trained Pix2Pix generator to produce a day-time version."
    )

    available_models = find_available_models()
    default_model_path = (
        str(available_models[0]) if available_models else str(DEFAULT_MODEL_CANDIDATES[0])
    )
    model_path = render_sidebar(default_model_path)

    uploaded_file = st.file_uploader(
        "Upload an input image",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
    )

    with st.expander("Project notes", expanded=False):
        st.markdown(
            "- The app normalizes images to `[-1, 1]`, matching the project inference code.\n"
            "- Output is resized back to the uploaded image dimensions for easier comparison.\n"
            "- If the model file is missing, place a generator checkpoint under `model/` or provide a custom path."
        )

    if not model_path:
        st.warning("Provide a valid model path to enable inference.")
        return

    model_file = Path(model_path)
    if not model_file.exists():
        st.error(f"Model file not found: `{model_file}`")
        return

    try:
        model = load_model(str(model_file))
    except Exception as exc:
        st.error(f"Failed to load model from `{model_file}`.")
        st.exception(exc)
        return

    model_height, model_width = get_model_image_size(model)
    st.success(
        f"Model loaded from `{model_file}` with input size `{model_height}x{model_width}`."
    )

    if uploaded_file is None:
        st.info("Upload an image to begin.")
        return

    try:
        input_image = Image.open(uploaded_file)
    except Exception as exc:
        st.error("The uploaded file could not be opened as an image.")
        st.exception(exc)
        return

    input_image = ImageOps.exif_transpose(input_image).convert("RGB")

    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Original")
        render_image(input_image)
        st.caption(f"Uploaded size: {input_image.width} x {input_image.height}")

    run_inference = st.button("Generate day image", type="primary", use_container_width=True)

    if not run_inference:
        with right_col:
            st.subheader("Generated")
            st.info("Run inference to preview the translated result.")
        return

    with st.spinner("Running generator..."):
        model_input = preprocess_image(input_image, (model_height, model_width))
        output = model.predict(model_input, verbose=0)
        output_image = postprocess_image(output, input_image.size)

    with right_col:
        st.subheader("Generated")
        render_image(output_image)
        st.caption(f"Returned size: {output_image.width} x {output_image.height}")

    st.download_button(
        "Download generated image",
        data=image_to_bytes(output_image),
        file_name=f"{Path(uploaded_file.name).stem}_day.png",
        mime="image/png",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
