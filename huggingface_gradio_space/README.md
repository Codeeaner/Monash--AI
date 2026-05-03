---
title: Tea Leaf Detection API
emoji: 🍃
colorFrom: green
colorTo: gray
sdk: gradio
app_file: app.py
---

# Tea Leaf Detection API

Upload a tea leaf image, run YOLO inference, and call the same prediction function as an API endpoint.

## Files to upload

- `app.py`
- `requirements.txt`
- `README.md`
- `.env.example`
- `best.pt` or a Hugging Face model repo reference

## Environment variables

Set these in Hugging Face Secrets or Space variables:

```env
MODEL_PATH=
CONFIDENCE_THRESHOLD=0.25
HUGGINGFACE_MODEL_REPO_ID=
HUGGINGFACE_MODEL_FILE=best.pt
HUGGINGFACE_MODEL_REPO_TYPE=space
HUGGINGFACE_MODEL_REVISION=main
HUGGINGFACE_TOKEN=
```

## API usage

The prediction function is exposed at `/api/predict` through Gradio.

Example client call:

```python
from gradio_client import Client

client = Client("your-username/your-space")
result = client.predict(image_path, 0.25, api_name="/predict")
```