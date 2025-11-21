import os
import uuid

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision

from PIL import Image
import numpy as np

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["RESULT_FOLDER"] = os.path.join("static", "results")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SEG_CLASSES = 4  # background, plaque, gum, tooth

def load_segmentation_model(weights_path: str):
    model = torchvision.models.segmentation.deeplabv3_resnet50(
        weights=None,
        aux_loss=True  # ⬅️ IMPORTANT
    )
    model.classifier[4] = nn.Conv2d(256, NUM_SEG_CLASSES, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, NUM_SEG_CLASSES, kernel_size=1)

    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)  # maintenant ça doit passer
    model.to(DEVICE)
    model.eval()
    return model


model_seg = load_segmentation_model("deeplab_teeth_plaque_best.pth")

transform_seg = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def run_segmentation_and_overlay(img_path: str):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    img_resized = img.resize((512, 512))
    x = transform_seg(img_resized).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model_seg(x)["out"]
        pred = out.argmax(dim=1)[0].cpu().numpy()  # [512, 512]

    mask_plaque_small = (pred == 1).astype(np.uint8)  # 1 = plaque
    mask_img = Image.fromarray(mask_plaque_small * 255).resize((w, h), resample=Image.NEAREST)
    mask_plaque = np.array(mask_img) > 0  # bool [H, W]

    plaque_pixels = mask_plaque.sum()
    total_pixels = mask_plaque.size
    plaque_ratio = float(plaque_pixels) / float(total_pixels) if total_pixels > 0 else 0.0

    img_np = np.array(img).astype(np.float32)
    overlay = img_np.copy()
    overlay[mask_plaque] = np.array([255, 0, 0], dtype=np.float32)

    alpha = 0.5
    blended = (img_np * (1 - alpha) + overlay * alpha).clip(0, 255).astype(np.uint8)
    blended_img = Image.fromarray(blended)

    result_filename = f"{uuid.uuid4().hex}.png"
    result_path = os.path.join(app.config["RESULT_FOLDER"], result_filename)
    blended_img.save(result_path)

    return result_filename, plaque_ratio * 100.0

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", error="No file part")

        file = request.files["image"]
        if file.filename == "":
            return render_template("index.html", error="No selected file")

        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(upload_path)

        result_filename, plaque_percent = run_segmentation_and_overlay(upload_path)

        return render_template(
            "index.html",
            original_image=os.path.join(app.config["UPLOAD_FOLDER"], unique_name),
            result_image=os.path.join(app.config["RESULT_FOLDER"], result_filename),
            plaque_percent=f"{plaque_percent:.1f}",
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
