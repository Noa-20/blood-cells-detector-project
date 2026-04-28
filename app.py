"""
Blood Cell Detector — Streamlit Web App
========================================
Interactive interface for a YOLO26-based blood-cell detection model.
Upload a peripheral blood smear image and get annotated predictions
with cell counts for 7 classes: RBC, Platelets, Neutrophil,
Lymphocyte, Monocyte, Eosinophil, and Basophil.
"""

from __future__ import annotations

import io
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ── Paths ───────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "blood_detector_model.pt"
TEST_DIR = ROOT / "test_images"

# ── Cell-type colours (RGB for PIL / Streamlit) ────────────────────
CLASS_COLORS = {
    "RBC":        (220, 60, 60),
    "Platelets":  (50, 205, 80),
    "Neutrophil": (60, 130, 246),
    "Lymphocyte": (139, 92, 246),
    "Monocyte":   (236, 152, 42),
    "Eosinophil": (244, 63, 174),
    "Basophil":   (20, 184, 166),
}

WBC_SUBTYPES = {"Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"}

# Annotation colours in BGR for OpenCV drawing
COLOR_MAP_BGR = {k: (b, g, r) for k, (r, g, b) in CLASS_COLORS.items()}


# ── Model loading (cached) ─────────────────────────────────────────
@st.cache_resource(show_spinner="Loading detection model …")
def load_model():
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))
    return model


# ── Annotation helpers ──────────────────────────────────────────────
def color_for_bgr(name: str) -> tuple[int, int, int]:
    return COLOR_MAP_BGR.get(name, (200, 200, 200))


def annotate_image(img_bgr, results):
    """Draw bounding boxes and labels on the image (in-place)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness, box_thickness = 0.45, 1, 2

    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    confs = results.boxes.conf.cpu().numpy()
    names = results.names

    for (x1, y1, x2, y2), cls_id, conf in zip(boxes, classes, confs):
        name = names[int(cls_id)]
        col = color_for_bgr(name)
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        # Box
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), col, box_thickness)

        # Label background
        label = f"{name} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        label_y = y1 - 4
        if label_y - th - 4 < 0:
            label_y = y2 + th + 6
        cv2.rectangle(
            img_bgr,
            (x1, label_y - th - 4),
            (x1 + tw + 6, label_y + 2),
            col, -1,
        )
        cv2.putText(
            img_bgr, label, (x1 + 3, label_y - 2),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )

    return img_bgr, boxes, classes, confs, names


def build_counts_table(classes, names):
    """Return a list of dicts for the summary table."""
    counts = Counter(names[int(c)] for c in classes)
    # Fixed order
    order = ["RBC", "Platelets", "Neutrophil", "Lymphocyte",
             "Monocyte", "Eosinophil", "Basophil"]
    rows = []
    for cls_name in order:
        count = counts.get(cls_name, 0)
        if count > 0:
            category = "WBC" if cls_name in WBC_SUBTYPES else cls_name
            rows.append({"Class": cls_name, "Category": category, "Count": count})
    return rows


# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Blood Cell Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hero header */
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #dc3c3c 0%, #8b5cf6 50%, #14b8a6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
    letter-spacing: -0.02em;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    margin-bottom: 1.5rem;
    font-weight: 300;
}

/* Cell badge */
.cell-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px 3px;
    color: white;
    letter-spacing: 0.01em;
}

/* Stats card */
.stat-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(139,92,246,0.08) 100%);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    margin-bottom: 8px;
}
.stat-number {
    font-size: 2rem;
    font-weight: 700;
    color: #a78bfa;
}
.stat-label {
    font-size: 0.82rem;
    color: #94a3b8;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] label {
    color: #cbd5e1 !important;
}

/* Footer */
.footer-text {
    text-align: center;
    color: #64748b;
    font-size: 0.78rem;
    padding: 2rem 0 1rem 0;
    border-top: 1px solid rgba(100,116,139,0.2);
    margin-top: 3rem;
}

/* Divider */
.gradient-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #8b5cf6, #14b8a6, transparent);
    border: none;
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Detection Settings")
    st.markdown("---")

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05, max_value=0.95, value=0.25, step=0.05,
        help="Minimum confidence score to keep a detection. "
             "Lower → more detections (but more false positives).",
    )
    iou_thresh = st.slider(
        "IoU threshold (NMS)",
        min_value=0.1, max_value=0.95, value=0.70, step=0.05,
        help="Non-Maximum Suppression IoU. Higher → allows more "
             "overlapping boxes.",
    )
    img_size = st.selectbox(
        "Inference resolution",
        options=[416, 640, 800, 1024],
        index=1,
        help="Image size fed to the model. 640 is the training "
             "resolution and recommended default.",
    )

    st.markdown("---")
    st.markdown("### 🎨 Class Legend")
    for cls_name, (r, g, b) in CLASS_COLORS.items():
        st.markdown(
            f'<span class="cell-badge" style="background:rgb({r},{g},{b})">'
            f"{cls_name}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "<p style='color:#64748b;font-size:0.75rem;text-align:center'>"
        "Model: YOLO26m · 7 classes · 640px<br>"
        "CPU inference</p>",
        unsafe_allow_html=True,
    )


# ── Header ──────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">🔬 Blood Cell Detector</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">'
    "Upload a peripheral blood smear image and detect <strong>7 cell types</strong> "
    "instantly — powered by YOLO26 deep learning."
    "</p>",
    unsafe_allow_html=True,
)
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ── Image input ─────────────────────────────────────────────────────
tab_upload, tab_examples = st.tabs(["📤 Upload Image", "🖼️ Example Images"])

input_image = None
image_source_name = None

with tab_upload:
    uploaded = st.file_uploader(
        "Drop a blood smear image here",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG",
    )
    if uploaded is not None:
        input_image = Image.open(uploaded).convert("RGB")
        image_source_name = uploaded.name

with tab_examples:
    example_files = sorted(TEST_DIR.glob("*")) if TEST_DIR.exists() else []
    example_files = [f for f in example_files if f.suffix.lower() in {".png", ".jpg", ".jpeg"}]

    if example_files:
        cols = st.columns(min(len(example_files), 3))
        for idx, ef in enumerate(example_files):
            col = cols[idx % 3]
            with col:
                ex_img = Image.open(ef)
                st.image(ex_img, caption=ef.stem.replace("_", " ").title(), use_container_width=True)
                if st.button(f"Use this image", key=f"ex_{idx}", use_container_width=True):
                    input_image = ex_img.convert("RGB")
                    image_source_name = ef.name
    else:
        st.info("No example images found in `test_images/` folder.")


# ── Run inference ───────────────────────────────────────────────────
if input_image is not None:
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # Convert PIL → numpy BGR for OpenCV
    img_np = np.array(input_image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Load model and predict
    with st.spinner("🔍 Running cell detection …"):
        model = load_model()
        results = model.predict(
            source=img_np,
            conf=confidence,
            iou=iou_thresh,
            imgsz=img_size,
            device="cpu",
            save=False,
            verbose=False,
        )
        result = results[0]

    # Annotate
    annotated_bgr, boxes, classes, confs, names = annotate_image(
        img_bgr.copy(), result
    )
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # ── Results header ──────────────────────────────────────────
    st.markdown(f"### 📊 Detection Results — *{image_source_name}*")

    # Stats row
    total_cells = len(boxes)
    wbc_count = sum(1 for c in classes if names[int(c)] in WBC_SUBTYPES)
    rbc_count = sum(1 for c in classes if names[int(c)] == "RBC")
    plt_count = sum(1 for c in classes if names[int(c)] == "Platelets")

    c1, c2, c3, c4 = st.columns(4)
    for col, (label, value) in zip(
        [c1, c2, c3, c4],
        [("Total Cells", total_cells), ("RBC", rbc_count),
         ("WBC", wbc_count), ("Platelets", plt_count)],
    ):
        col.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-number">{value}</div>'
            f'<div class="stat-label">{label}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Side-by-side images ─────────────────────────────────────
    col_orig, col_pred = st.columns(2)
    with col_orig:
        st.markdown("**Original**")
        st.image(input_image, use_container_width=True)
    with col_pred:
        st.markdown("**Detected**")
        st.image(annotated_rgb, use_container_width=True)

    # ── Detailed counts table ───────────────────────────────────
    rows = build_counts_table(classes, names)
    if rows:
        st.markdown("#### 📋 Cell Count Breakdown")

        # Build HTML table with colored badges
        table_html = (
            '<table style="width:100%;border-collapse:collapse;margin-top:8px">'
            "<thead><tr>"
            '<th style="text-align:left;padding:8px 12px;border-bottom:2px solid #334155">Class</th>'
            '<th style="text-align:left;padding:8px 12px;border-bottom:2px solid #334155">Category</th>'
            '<th style="text-align:right;padding:8px 12px;border-bottom:2px solid #334155">Count</th>'
            "</tr></thead><tbody>"
        )
        for row in rows:
            r, g, b = CLASS_COLORS.get(row["Class"], (150, 150, 150))
            table_html += (
                f"<tr>"
                f'<td style="padding:8px 12px;border-bottom:1px solid #1e293b">'
                f'<span class="cell-badge" style="background:rgb({r},{g},{b})">{row["Class"]}</span></td>'
                f'<td style="padding:8px 12px;border-bottom:1px solid #1e293b;color:#94a3b8">{row["Category"]}</td>'
                f'<td style="padding:8px 12px;border-bottom:1px solid #1e293b;text-align:right;'
                f'font-weight:600;font-size:1.1rem">{row["Count"]}</td>'
                f"</tr>"
            )
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)

    # ── Download annotated image ────────────────────────────────
    st.markdown("")
    annotated_pil = Image.fromarray(annotated_rgb)
    buf = io.BytesIO()
    annotated_pil.save(buf, format="JPEG", quality=92)
    st.download_button(
        label="⬇️ Download annotated image",
        data=buf.getvalue(),
        file_name=f"detected_{image_source_name}",
        mime="image/jpeg",
        use_container_width=True,
    )

else:
    # Placeholder when no image is selected
    st.markdown("")
    st.markdown(
        '<div style="text-align:center;padding:4rem 2rem;'
        "background:linear-gradient(135deg,rgba(99,102,241,0.06),rgba(20,184,166,0.06));"
        'border-radius:16px;border:1px dashed rgba(139,92,246,0.3)">'
        '<p style="font-size:3rem;margin-bottom:0.5rem">🔬</p>'
        '<p style="color:#94a3b8;font-size:1.1rem;font-weight:500">'
        "Upload a blood smear image or select an example to get started"
        "</p>"
        '<p style="color:#64748b;font-size:0.85rem">'
        "Supports JPG, JPEG, PNG • Recommended resolution: 640×640 or higher"
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Footer ──────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer-text">'
    "🔬 <strong>Blood Cell Detector</strong> · YOLO26m fine-tuned on peripheral blood smears<br>"
    "7 classes: RBC · Platelets · Neutrophil · Lymphocyte · Monocyte · Eosinophil · Basophil<br><br>"
    "⚠️ <em>Research use only — not validated for clinical diagnosis.</em>"
    "</div>",
    unsafe_allow_html=True,
)
