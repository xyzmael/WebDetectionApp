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

# Configuration des dossiers
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["RESULT_FOLDER"] = os.path.join("static", "results")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SEG_CLASSES = 4


# --- Chargement du Modèle ---
def load_segmentation_model(weights_path: str):
    model = torchvision.models.segmentation.deeplabv3_resnet50(
        weights=None,
        aux_loss=True
    )
    model.classifier[4] = nn.Conv2d(256, NUM_SEG_CLASSES, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, NUM_SEG_CLASSES, kernel_size=1)

    # Assurez-vous que le fichier .pth est bien à la racine ou indiquez le bon chemin
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    else:
        print(f"ATTENTION: Le fichier de poids '{weights_path}' est introuvable.")

    model.to(DEVICE)
    model.eval()
    return model


# Charger le modèle (vérifiez que le nom du fichier est correct)
model_seg = load_segmentation_model("deeplab_teeth_plaque_best.pth")

transform_seg = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


# --- Fonction de Traitement (Version Overlay Bleu) ---
def run_segmentation_and_overlay(img_path: str):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    img_resized = img.resize((512, 512))
    x = transform_seg(img_resized).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model_seg(x)["out"]
        pred = out.argmax(dim=1)[0].cpu().numpy()  # [512, 512]

    # --- 1. Récupération des masques ---
    mask_plaque_small = (pred == 1).astype(np.uint8)
    mask_tooth_small = (pred == 3).astype(np.uint8)

    # Redimensionnement 'Nearest' pour garder 0 ou 1
    mask_plaque_img = Image.fromarray(mask_plaque_small * 255).resize((w, h), resample=Image.NEAREST)
    mask_tooth_img = Image.fromarray(mask_tooth_small * 255).resize((w, h), resample=Image.NEAREST)

    mask_plaque = np.array(mask_plaque_img) > 0
    mask_tooth = np.array(mask_tooth_img) > 0

    # --- 2. Calcul du pourcentage ---
    plaque_pixels = mask_plaque.sum()
    tooth_pixels = mask_tooth.sum()
    total_dental_surface = plaque_pixels + tooth_pixels

    if total_dental_surface > 0:
        plaque_ratio = float(plaque_pixels) / float(total_dental_surface)
    else:
        plaque_ratio = 0.0

    # --- 3. Création de l'overlay Bleu ---
    img_np = np.array(img).astype(np.float32)
    overlay = img_np.copy()

    # Appliquer le bleu [0, 0, 255] sur la plaque
    overlay[mask_plaque] = np.array([0, 0, 255], dtype=np.float32)

    alpha = 0.5
    blended = (img_np * (1 - alpha) + overlay * alpha).clip(0, 255).astype(np.uint8)
    blended_img = Image.fromarray(blended)

    result_filename = f"{uuid.uuid4().hex}.png"
    result_path = os.path.join(app.config["RESULT_FOLDER"], result_filename)
    blended_img.save(result_path)

    return result_filename, plaque_ratio * 100.0


# --- La Route Flask Modifiée ---
@app.route("/", methods=["GET", "POST"])
def index():
    results = []  # Liste pour stocker les résultats de chaque image

    if request.method == "POST":
        # Notez le "images" au pluriel qui correspond au name="images" du HTML
        if "images" not in request.files:
            return render_template("index.html", error="No file part")

        # Utilisation de getlist pour récupérer TOUS les fichiers
        files = request.files.getlist("images")

        if not files or files[0].filename == '':
            return render_template("index.html", error="No selected file")

        # Boucle sur chaque fichier uploadé
        for file in files:
            if file:
                filename = secure_filename(file.filename)
                unique_name = f"{uuid.uuid4().hex}_{filename}"
                upload_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)

                # Sauvegarder l'image originale
                file.save(upload_path)

                # Lancer l'analyse
                result_filename, plaque_percent = run_segmentation_and_overlay(upload_path)

                # Ajouter au tableau de résultats pour le template
                # On utilise .replace('\\', '/') pour assurer la compatibilité des chemins URL sous Windows
                results.append({
                    "name": filename,
                    "percent": f"{plaque_percent:.1f}",
                    "original": os.path.join("static", "uploads", unique_name).replace('\\', '/'),
                    "result": os.path.join("static", "results", result_filename).replace('\\', '/')
                })

        # On renvoie la liste complète 'results' au template
        return render_template("index.html", results=results)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)