import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import quote

import gradio as gr
import requests
from ultralytics import YOLO


SPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_FILE = "best.pt"
DEFAULT_MODEL_PATH = SPACE_ROOT / DEFAULT_MODEL_FILE
MODEL_CACHE_DIR = SPACE_ROOT / ".model-cache"


def _download_model_from_hugging_face(repo_id: str, model_file: str, repo_type: str, revision: str) -> Path:
    """Download a YOLO weight file from Hugging Face into a local cache."""
    repo_id = repo_id.strip()
    model_file = model_file.strip().lstrip("/")
    repo_type = (repo_type or "space").strip().lower()
    revision = revision.strip() or "main"

    if not repo_id:
        raise ValueError("HUGGINGFACE_MODEL_REPO_ID is required")
    if not model_file:
        raise ValueError("HUGGINGFACE_MODEL_FILE is required")
    if repo_type not in {"space", "model", "dataset"}:
        raise ValueError(f"Unsupported HUGGINGFACE_MODEL_REPO_TYPE: {repo_type}")

    cache_dir = MODEL_CACHE_DIR / repo_type / repo_id.replace("/", "__") / revision
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_path = cache_dir / Path(model_file).name

    if cached_path.exists():
        return cached_path

    repo_prefix = "spaces/" if repo_type == "space" else ""
    encoded_repo_id = "/".join(quote(part, safe="") for part in repo_id.split("/"))
    encoded_model_file = "/".join(quote(part, safe="") for part in model_file.split("/"))
    download_url = f"https://huggingface.co/{repo_prefix}{encoded_repo_id}/resolve/{quote(revision, safe='')}/{encoded_model_file}"

    token = os.getenv("HUGGINGFACE_TOKEN", "").strip()
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    response = requests.get(download_url, headers=headers, stream=True, timeout=120)
    response.raise_for_status()

    with open(cached_path, "wb") as file_handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file_handle.write(chunk)

    return cached_path


def _resolve_model_path() -> Path:
    """Resolve the best available model path for the Space."""
    model_path = os.getenv("MODEL_PATH", "").strip()
    if model_path:
        candidate = Path(model_path).expanduser()
        if candidate.exists():
            return candidate

    repo_id = os.getenv("HUGGINGFACE_MODEL_REPO_ID", "").strip()
    if repo_id:
        downloaded_path = _download_model_from_hugging_face(
            repo_id=repo_id,
            model_file=os.getenv("HUGGINGFACE_MODEL_FILE", DEFAULT_MODEL_FILE),
            repo_type=os.getenv("HUGGINGFACE_MODEL_REPO_TYPE", "space"),
            revision=os.getenv("HUGGINGFACE_MODEL_REVISION", "main"),
        )
        return downloaded_path

    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH

    raise FileNotFoundError(
        "No YOLO model found. Set MODEL_PATH or configure HUGGINGFACE_MODEL_REPO_ID/HUGGINGFACE_MODEL_FILE."
    )


@lru_cache(maxsize=1)
def get_model() -> YOLO:
    """Load the YOLO model once and reuse it for all requests."""
    model_path = _resolve_model_path()
    return YOLO(str(model_path))


def predict(image, confidence: float = 0.25) -> Tuple[Any, Dict[str, Any]]:
    """Run inference and return the plotted image plus structured predictions."""
    if image is None:
        raise gr.Error("Please upload an image before running detection.")

    model = get_model()
    results = model.predict(source=image, conf=confidence, save=False, verbose=False)

    if not results:
        return image, {"error": "No predictions were produced."}

    result = results[0]
    plotted = result.plot()
    detections_raw = result.tojson()

    try:
        detections = json.loads(detections_raw) if isinstance(detections_raw, str) else detections_raw
    except json.JSONDecodeError:
        detections = {"raw": detections_raw}

    return plotted, detections


CSS = """
:root {
    --bg-start: #f2f8f1;
    --bg-end: #dbe8db;
    --panel: rgba(255, 255, 255, 0.82);
    --panel-border: rgba(39, 89, 59, 0.12);
    --text: #12311f;
    --muted: #4f6658;
    --accent: #1f7a4d;
}

body, .gradio-container {
    background: linear-gradient(160deg, var(--bg-start), var(--bg-end));
    color: var(--text);
}

.wrap {
    max-width: 1120px !important;
}

.hero {
    border: 1px solid var(--panel-border);
    border-radius: 28px;
    background: linear-gradient(135deg, rgba(31, 122, 77, 0.94), rgba(18, 49, 31, 0.94));
    color: white;
    padding: 1.35rem 1.5rem;
    box-shadow: 0 18px 40px rgba(18, 49, 31, 0.16);
}

.hero h1 {
    margin: 0;
    font-size: clamp(1.7rem, 3.8vw, 2.8rem);
    line-height: 1.05;
}

.hero p {
    margin: 0.5rem 0 0;
    opacity: 0.92;
}

.panel {
    border-radius: 24px !important;
    border: 1px solid var(--panel-border) !important;
    background: var(--panel) !important;
    backdrop-filter: blur(10px);
    box-shadow: 0 12px 30px rgba(18, 49, 31, 0.08);
}

.panel h3, .panel h4, .panel label {
    color: var(--text) !important;
}

.panel .wrap {
    max-width: none !important;
}

.gr-button-primary {
    background: linear-gradient(135deg, #1f7a4d, #2ea66b) !important;
    border: 0 !important;
}

.gr-button-secondary {
    border: 1px solid rgba(31, 122, 77, 0.26) !important;
}
"""


with gr.Blocks(title="Tea Leaf Detection API", theme=gr.themes.Soft(), css=CSS) as demo:
    with gr.Column(elem_classes=["wrap"]):
        gr.Markdown(
            """
            <div class="hero">
              <h1>Tea Leaf Detection API</h1>
              <p>Upload a tea leaf image, run YOLO inference, and call the same endpoint from code via the Gradio API.</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["panel"]):
                input_image = gr.Image(type="pil", label="Tea leaf image")
                confidence = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=float(os.getenv("CONFIDENCE_THRESHOLD", "0.25")),
                    step=0.05,
                    label="Confidence threshold",
                )
                run_button = gr.Button("Run detection", variant="primary")

            with gr.Column(scale=1, elem_classes=["panel"]):
                output_image = gr.Image(type="numpy", label="Detection result")
                output_json = gr.JSON(label="Predictions")

        gr.Markdown(
            "The prediction function is exposed as an API endpoint, so you can call it with `api_name=\"/predict\"`."
        )

        run_button.click(
            fn=predict,
            inputs=[input_image, confidence],
            outputs=[output_image, output_json],
            api_name="predict",
        )

demo.queue()


if __name__ == "__main__":
    demo.launch()