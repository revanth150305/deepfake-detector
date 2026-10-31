"""
Hybrid Deepfake Detection System - demo app.

Three branches score an image for authenticity, and a Logistic Regression
fusion layer combines them into a Reality Score (0-100):
  1. Visual        - fine-tuned EfficientNet-B0
  2. Geometry      - MediaPipe Face Mesh landmarks -> 26 features -> XGBoost
  3. Illumination  - 42 lighting features -> Gradient Boosting

Run:
    python app.py                 # http://127.0.0.1:7860
    python app.py --share         # public gradio.live link
"""

import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import cv2
import gradio as gr
import joblib
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = ROOT / "models"
SAMPLES_DIR = ROOT / "samples"
DEFAULT_THRESHOLD = 90.0  # paper: R >= 90 -> authentic


# =====================================================
# GEOMETRY FEATURES (26) - matches the training code
# (distances normalised by the image diagonal)
# =====================================================

def get_geometry_features(points, h, w):
    if points is None:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b)

    img_diag = np.sqrt(h**2 + w**2) + 1e-6

    p = points
    face_width = dist(p[234], p[454])
    face_height = dist(p[10], p[152])
    eye_distance = dist(p[33], p[263])
    left_eye_width = dist(p[33], p[133])
    right_eye_width = dist(p[263], p[362])
    left_eye_height = dist(p[159], p[145])
    right_eye_height = dist(p[386], p[374])
    mouth_width = dist(p[61], p[291])
    mouth_height = dist(p[13], p[14])
    nose_width = dist(p[98], p[327])
    nose_height = dist(p[168], p[1])
    nose_to_mouth = dist(p[1], (p[61] + p[291]) / 2)
    left_face_height = dist(p[33], p[152])
    right_face_height = dist(p[263], p[152])
    symmetry = abs(dist(p[33], p[1]) - dist(p[263], p[1]))

    fw = face_width + 1e-6
    fh = face_height + 1e-6

    return np.array([
        # normalised by image diagonal
        face_width / img_diag,
        face_height / img_diag,
        eye_distance / img_diag,
        left_eye_width / img_diag,
        right_eye_width / img_diag,
        left_eye_height / img_diag,
        right_eye_height / img_diag,
        mouth_width / img_diag,
        mouth_height / img_diag,
        nose_width / img_diag,
        nose_height / img_diag,
        nose_to_mouth / img_diag,
        symmetry / img_diag,
        # shape ratios (face-relative)
        face_width / fh,
        mouth_width / fw,
        mouth_height / fh,
        nose_width / fw,
        nose_height / fh,
        eye_distance / fw,
        left_eye_width / fw,
        right_eye_width / fw,
        left_eye_height / fh,
        right_eye_height / fh,
        left_face_height / fh,
        right_face_height / fh,
        symmetry / fw,
    ], dtype=np.float32)


# =====================================================
# ILLUMINATION FEATURES (12 + 3 + 8 + 6 + 8 + 5 = 42)
# =====================================================

def compute_sh_coefficients(face_region):
    if face_region.size == 0:
        return np.zeros(12)
    h, w = face_region.shape[:2]
    ys = np.linspace(-1, 1, h)
    xs = np.linspace(-1, 1, w)
    xv, yv = np.meshgrid(xs, ys)
    r = np.sqrt(xv**2 + yv**2)
    zv = np.sqrt(np.maximum(0, 1 - r**2))
    basis = np.stack([np.ones_like(xv), xv, yv, zv], axis=-1)
    coeffs = []
    face_f = face_region.astype(np.float32) / 255.0
    for c in range(3):
        B = basis.reshape(-1, 4)
        I = face_f[:, :, c].reshape(-1)
        c_coeffs, _, _, _ = np.linalg.lstsq(B, I, rcond=None)
        coeffs.extend(c_coeffs.tolist())
    return np.array(coeffs, dtype=np.float32)


def estimate_light_direction(face_region):
    if face_region.size == 0:
        return np.zeros(3)
    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mean_i = np.mean(gray)
    w = np.abs(gray - mean_i) / (np.sum(np.abs(gray - mean_i)) + 1e-6)
    Lx = np.sum(gx * w)
    Ly = np.sum(gy * w)
    Lz = np.sqrt(max(0, 1 - Lx**2 - Ly**2))
    norm = np.sqrt(Lx**2 + Ly**2 + Lz**2) + 1e-6
    return np.array([Lx / norm, Ly / norm, Lz / norm], dtype=np.float32)


def analyze_specular_highlights(face_region):
    if face_region.size == 0:
        return np.zeros(8)
    hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV).astype(np.float32)
    V, S = hsv[:, :, 2], hsv[:, :, 1]
    spec_mask = ((V > 200) & (S < 50)).astype(np.float32)
    total_px = spec_mask.size + 1e-6
    spec_ratio = spec_mask.sum() / total_px
    if spec_mask.sum() > 0:
        spec_V = V[spec_mask > 0]
        mean_v = np.mean(spec_V)
        std_v = np.std(spec_V)
        ys, xs = np.where(spec_mask > 0)
        cy_s = np.mean(ys) / (face_region.shape[0] + 1e-6)
        cx_s = np.mean(xs) / (face_region.shape[1] + 1e-6)
        spread_y = np.std(ys) / (face_region.shape[0] + 1e-6)
        spread_x = np.std(xs) / (face_region.shape[1] + 1e-6)
    else:
        mean_v = std_v = cy_s = cx_s = spread_y = spread_x = 0.0
    return np.array([
        spec_ratio, mean_v / 255, std_v / 255,
        cy_s, cx_s, spread_y, spread_x,
        float(spec_mask.sum() > 0),
    ], dtype=np.float32)


def face_background_consistency(image, lm_arr, w, h):
    if lm_arr is None or image.size == 0:
        return np.zeros(6)
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        xs = (lm_arr[:, 0] * w).astype(int)
        ys = (lm_arr[:, 1] * h).astype(int)
        x1, x2 = max(0, xs.min()), min(w - 1, xs.max())
        y1, y2 = max(0, ys.min()), min(h - 1, ys.max())
        face_lab = lab[y1:y2, x1:x2]
        mask = np.ones(lab.shape[:2], dtype=bool)
        mask[y1:y2, x1:x2] = False
        bg_lab = lab[mask]
        if face_lab.size == 0 or bg_lab.size == 0:
            return np.zeros(6)
        f_lab = face_lab.reshape(-1, 3)
        b_lab = bg_lab.reshape(-1, 3)
        dL = abs(f_lab[:, 0].mean() - b_lab[:, 0].mean()) / 255
        da = abs(f_lab[:, 1].mean() - b_lab[:, 1].mean()) / 255
        db = abs(f_lab[:, 2].mean() - b_lab[:, 2].mean()) / 255
        dC = np.sqrt(da**2 + db**2)
        return np.array([dL, da, db, dC,
                         f_lab[:, 0].std() / 255,
                         b_lab[:, 0].std() / 255], dtype=np.float32)
    except Exception:
        return np.zeros(6)


def gradient_shading_features(face_region):
    if face_region.size == 0:
        return np.zeros(8)
    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx)
    mag_hist, _ = np.histogram(mag, bins=32, range=(0, 255), density=True)
    ang_hist, _ = np.histogram(ang, bins=18, range=(-np.pi, np.pi), density=True)
    eps = 1e-10
    H_mag = -np.sum(mag_hist * np.log(mag_hist + eps))
    H_ang = -np.sum(ang_hist * np.log(ang_hist + eps))
    return np.array([
        H_mag, H_ang,
        mag.mean() / 255, mag.std() / 255,
        gx.mean() / 255, gx.std() / 255,
        gy.mean() / 255, gy.std() / 255,
    ], dtype=np.float32)


def shadow_consistency_features(face_region):
    if face_region.size == 0:
        return np.zeros(5)
    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    shadow_mask = (gray < 0.6).astype(np.float32)
    ratio = shadow_mask.mean()
    if shadow_mask.sum() > 0:
        ys, xs = np.where(shadow_mask > 0)
        cy = np.mean(ys) / (face_region.shape[0] + 1e-6)
        cx = np.mean(xs) / (face_region.shape[1] + 1e-6)
        std_y = np.std(ys) / (face_region.shape[0] + 1e-6)
        std_x = np.std(xs) / (face_region.shape[1] + 1e-6)
    else:
        cy = cx = std_y = std_x = 0.0
    return np.array([ratio, cy, cx, std_y, std_x], dtype=np.float32)


def get_light_features(face_crop, image, lm_arr, w, h):
    return np.concatenate([
        compute_sh_coefficients(face_crop),
        estimate_light_direction(face_crop),
        analyze_specular_highlights(face_crop),
        face_background_consistency(image, lm_arr, w, h),
        gradient_shading_features(face_crop),
        shadow_consistency_features(face_crop),
    ])


# =====================================================
# DETECTOR
# =====================================================

class DeepfakeDetector:

    def __init__(self, models_dir=DEFAULT_MODELS_DIR):
        self.device = torch.device("cpu")
        d = Path(models_dir)

        self.effnet = models.efficientnet_b0(weights=None)
        self.effnet.classifier[1] = nn.Linear(self.effnet.classifier[1].in_features, 2)
        self.effnet.load_state_dict(torch.load(d / "best_effb0.pth", map_location=self.device))
        self.effnet.eval()

        self.geo_model = joblib.load(d / "xgboost_geometry_model.pkl")
        self.geo_scaler = joblib.load(d / "geometry_scaler.pkl")
        self.light_model = joblib.load(d / "light_gbm_model.pkl")
        self.light_scaler = joblib.load(d / "light_gbm_scaler.pkl")
        self.fusion = joblib.load(d / "fusion_lr_model.pkl")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
        )

    @property
    def fusion_weights(self):
        w = self.fusion.coef_[0]
        return {"visual": float(w[0]), "geometry": float(w[1]),
                "illumination": float(w[2]), "bias": float(self.fusion.intercept_[0])}

    def predict(self, image_bgr):
        """Run all three branches + fusion on a BGR image."""
        h, w = image_bgr.shape[:2]

        # One MediaPipe pass, shared by the geometry and illumination branches
        results = self.face_mesh.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            points = np.array([[p.x * w, p.y * h] for p in lm.landmark], dtype=np.float32)
            lm_arr = np.array([(p.x, p.y, p.z) for p in lm.landmark], dtype=np.float16)
            xs, ys = points[:, 0], points[:, 1]
            bbox = (max(0, int(xs.min())), min(w, int(xs.max())),
                    max(0, int(ys.min())), min(h, int(ys.max())))
        else:
            points = lm_arr = None
            cx, cy, s = w // 2, h // 2, min(w, h) // 3
            bbox = (cx - s, cx + s, cy - s, cy + s)

        x1, x2, y1, y2 = bbox
        face_crop = image_bgr[y1:y2, x1:x2]
        if face_crop.size == 0:
            face_crop = image_bgr

        s1 = self._visual_score(image_bgr)
        s2 = self._geometry_score(points, h, w)
        s3 = self._light_score(face_crop, image_bgr, lm_arr, w, h)

        scores = np.array([[s1, s2, s3]], dtype=np.float32)
        reality = float(self.fusion.predict_proba(scores)[0][1]) * 100

        return {
            "reality_score": reality,
            "visual": s1, "geometry": s2, "illumination": s3,
            "face_found": points is not None,
            "points": points, "bbox": bbox,
        }

    def _visual_score(self, image_bgr):
        t = self.transform(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        with torch.no_grad():
            return torch.softmax(self.effnet(t), dim=1)[0][1].item()

    def _geometry_score(self, points, h, w):
        feats = get_geometry_features(points, h, w)
        if feats is None:
            return 0.5
        return float(self.geo_model.predict_proba(
            self.geo_scaler.transform(feats.reshape(1, -1)))[0][1])

    def _light_score(self, face_crop, image, lm_arr, w, h):
        try:
            feats = get_light_features(face_crop, image, lm_arr, w, h).reshape(1, -1)
            return float(self.light_model.predict_proba(
                self.light_scaler.transform(feats))[0][1])
        except Exception:
            return 0.5


# =====================================================
# GRADIO UI
# =====================================================

def draw_landmarks(image_rgb, result):
    vis = image_rgb.copy()
    h, w = vis.shape[:2]
    r = max(1, round(min(h, w) / 300))
    if result["points"] is not None:
        for x, y in result["points"].astype(int):
            cv2.circle(vis, (int(x), int(y)), r, (0, 220, 120), -1, cv2.LINE_AA)
    x1, x2, y1, y2 = result["bbox"]
    color = (0, 220, 120) if result["face_found"] else (255, 170, 0)
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, max(1, r), cv2.LINE_AA)
    return vis


def verdict_html(score, threshold, face_found):
    real = score >= threshold
    color = "#16a34a" if real else "#dc2626"
    label = "REAL" if real else "FAKE / AI-GENERATED"
    note = "" if face_found else (
        "<div style='margin-top:8px;font-size:13px;color:#b45309'>"
        "No face detected. The geometry branch fell back to a neutral 0.5, "
        "and illumination used a centre crop.</div>")
    return f"""
<div style="border:2px solid {color};border-radius:12px;padding:18px 20px;">
  <div style="font-size:13px;letter-spacing:.08em;opacity:.7">MODEL VERDICT</div>
  <div style="font-size:30px;font-weight:700;color:{color};line-height:1.2">{label}</div>
  <div style="margin-top:10px;font-size:15px">Reality Score
    <b style="font-size:22px">{score:.2f}</b> / 100
    <span style="opacity:.6">&nbsp;(threshold {threshold:.0f})</span></div>
  <div style="margin-top:8px;height:10px;border-radius:5px;background:rgba(127,127,127,.25);overflow:hidden">
    <div style="width:{score:.1f}%;height:100%;background:{color}"></div></div>
  {note}
</div>"""


def breakdown_md(result, weights):
    rows = [("Visual (EfficientNet-B0)", "visual"),
            ("Geometry (MediaPipe + XGBoost)", "geometry"),
            ("Illumination (Gradient Boosting)", "illumination")]
    logit = weights["bias"]
    lines = ["| Branch | Authenticity score | Fusion weight | Contribution (w x s) |",
             "|---|---:|---:|---:|"]
    for name, key in rows:
        s, wt = result[key], weights[key]
        logit += wt * s
        lines.append(f"| {name} | {s * 100:.2f}% | {wt:+.2f} | {wt * s:+.2f} |")
    lines.append(f"| Bias | | | {weights['bias']:+.2f} |")
    lines.append(f"| **Logit -> sigmoid x 100** | | | **{logit:+.2f} -> {result['reality_score']:.2f}** |")
    return "\n".join(lines)


def build_app(detector):
    weights = detector.fusion_weights
    # samples/<Label>/<Label>_imageN.jpg -> example labelled "<Label> / imageN"
    sample_paths = sorted(SAMPLES_DIR.glob("*/*.jpg")) if SAMPLES_DIR.exists() else []
    examples = [str(p) for p in sample_paths]
    example_labels = [p.stem.replace("_", " / ", 1) for p in sample_paths]

    def analyze(image_path, threshold):
        if not image_path:
            raise gr.Error("Upload an image first.")
        image_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise gr.Error("Could not read that image.")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = detector.predict(image_bgr)
        branch_scores = {
            "Visual": result["visual"],
            "Geometry": result["geometry"],
            "Illumination": result["illumination"],
        }
        return (verdict_html(result["reality_score"], threshold, result["face_found"]),
                branch_scores,
                draw_landmarks(image, result),
                breakdown_md(result, weights))

    with gr.Blocks(title="Hybrid Deepfake Detector") as app:
        gr.Markdown(
            "# Hybrid Deepfake Detection System\n"
            "Three independent branches - **visual texture**, **facial geometry** and "
            "**illumination consistency** - each score the image, then a Logistic Regression "
            "fusion layer combines them into a **Reality Score** (0-100).")

        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(label="Face image", type="filepath", height=340)
                threshold = gr.Slider(50, 99, value=DEFAULT_THRESHOLD, step=1,
                                      label="Decision threshold (Reality Score >= threshold -> REAL)")
                btn = gr.Button("Analyze", variant="primary", size="lg")
                if examples:
                    gr.Examples(examples=examples, inputs=img_in,
                                example_labels=example_labels,
                                label="Sample images (true label shown)",
                                examples_per_page=12)

            with gr.Column(scale=1):
                verdict = gr.HTML("<div style='opacity:.6;padding:18px'>Upload an image and click "
                                  "<b>Analyze</b>.</div>")
                branches = gr.Label(label="Branch authenticity scores (1 = real)", num_top_classes=3)
                with gr.Tabs():
                    with gr.Tab("Face landmarks"):
                        landmarks = gr.Image(label="MediaPipe Face Mesh (468 points)",
                                             interactive=False, height=300)
                    with gr.Tab("Fusion breakdown"):
                        breakdown = gr.Markdown()

        outputs = [verdict, branches, landmarks, breakdown]
        btn.click(analyze, inputs=[img_in, threshold], outputs=outputs)
        img_in.upload(analyze, inputs=[img_in, threshold], outputs=outputs)

        gr.Markdown(
            "---\n*Trained on the Kaggle 'Deepfake and Real Images' dataset (StyleGAN faces). "
            "Results on other generators, face-swaps or heavily compressed images may differ.*")
    return app


def main():
    parser = argparse.ArgumentParser(description="Hybrid deepfake detection demo")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1",
                        help="use 0.0.0.0 inside Docker so the port can be reached")
    args = parser.parse_args()

    print("Loading models...")
    detector = DeepfakeDetector(args.models_dir)
    print("Models loaded.")
    build_app(detector).launch(server_name=args.host, server_port=args.port, share=args.share,
                               theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
