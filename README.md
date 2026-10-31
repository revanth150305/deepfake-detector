# Hybrid Deepfake Detection System — Demo

Upload a face image and the app scores it with three branches (visual texture,
facial geometry, illumination) and combines them into a Reality Score (0–100).
A score of 90 or more is classified as REAL.

## Run with Docker (recommended)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/).
Open a terminal in this folder and run:

```bash
docker build -t deepfake-demo .
docker run --rm -p 7860:7860 deepfake-demo
```

Then open **http://localhost:7860**. Press `Ctrl+C` in the terminal to stop it.

The first build downloads about 1–2 GB of packages and takes a few minutes.
Later runs start in seconds.

If port 7860 is in use, change the left-hand number, for example
`-p 8080:7860`, and open http://localhost:8080.

## Run without Docker (Python 3.11)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python app.py
```

## Folder contents

| Path | What it is |
|---|---|
| `app.py` | The Gradio demo app |
| `models/` | Trained models: EfficientNet-B0, XGBoost (geometry), Gradient Boosting (illumination), Logistic Regression (fusion) |
| `samples/Fake`, `samples/Real` | Example images with known labels |
| `Dockerfile`, `requirements.txt` | Environment setup |

