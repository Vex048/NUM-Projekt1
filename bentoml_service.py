import glob
import os
import re
import urllib.request
from typing import List, Optional, Tuple

import bentoml
from PIL import Image as PilImage
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torchvision import transforms

from src.models.resnet_classifier import ResNetClassifier
from src.models.efficientnet_classifier import EfficientNetClassifier
from src.models.densenet_classifier import DenseNetClassifier
from src.models.baseline_classifier import BaselineClassifier

CHECKPOINTS_DIR = os.getenv("CHECKPOINTS_DIR", "./checkpoints")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "./artifacts/best.ckpt")
MODEL_URL = os.getenv("MODEL_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
DATA_DIR = os.getenv("DATA_DIR", "./dataset/archive")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CLASS_BY_PREFIX = {
    "resnet": ResNetClassifier,
    "efficientnet": EfficientNetClassifier,
    "densenet": DenseNetClassifier,
    "baseline": BaselineClassifier,
}


def find_best_checkpoint(checkpoints_dir: str) -> Tuple[str, float]:
    pattern = os.path.join(checkpoints_dir, "*.ckpt")
    best_path = None
    best_loss = float("inf")

    for path in glob.glob(pattern):
        match = re.search(r"val_loss=([0-9]+(?:\.[0-9]+)?)", os.path.basename(path))
        if not match:
            continue
        loss = float(match.group(1))
        if loss < best_loss:
            best_loss = loss
            best_path = path

    if not best_path:
        raise FileNotFoundError(
            f"No checkpoint with val_loss found in {checkpoints_dir}"
        )

    return best_path, best_loss


def extract_val_loss(checkpoint_path: str) -> Optional[float]:
    match = re.search(
        r"val_loss=([0-9]+(?:\.[0-9]+)?)", os.path.basename(checkpoint_path)
    )
    return float(match.group(1)) if match else None


def infer_model_name(checkpoint_path: str) -> str:
    base = os.path.basename(checkpoint_path).lower()
    for prefix in MODEL_CLASS_BY_PREFIX:
        if base.startswith(prefix):
            return prefix
    if MODEL_NAME:
        return MODEL_NAME.lower()
    raise ValueError(f"Cannot infer model name from checkpoint: {checkpoint_path}")


def load_class_names_from_metadata(data_dir: str) -> Optional[List[str]]:
    metadata_path = os.path.join(data_dir, "HAM10000_metadata.csv")
    if not os.path.exists(metadata_path):
        return None

    df = pd.read_csv(metadata_path)
    label_encoder = LabelEncoder()
    label_encoder.fit(df["dx"])
    return label_encoder.classes_.tolist()


def build_preprocess() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def resolve_checkpoint() -> Tuple[str, Optional[float]]:
    if MODEL_URL:
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        if not os.path.exists(CHECKPOINT_PATH):
            urllib.request.urlretrieve(MODEL_URL, CHECKPOINT_PATH)
        return CHECKPOINT_PATH, extract_val_loss(CHECKPOINT_PATH)

    if os.path.exists(CHECKPOINT_PATH):
        return CHECKPOINT_PATH, extract_val_loss(CHECKPOINT_PATH)

    artifacts_dir = os.path.dirname(CHECKPOINT_PATH) or "./artifacts"
    if os.path.isdir(artifacts_dir):
        candidates = glob.glob(os.path.join(artifacts_dir, "*.ckpt"))
        if candidates:
            candidates_with_loss = [
                (path, extract_val_loss(path)) for path in candidates
            ]
            candidates_with_loss = [
                (path, loss) for path, loss in candidates_with_loss if loss is not None
            ]
            if candidates_with_loss:
                best_path, best_loss = min(candidates_with_loss, key=lambda item: item[1])
                return best_path, best_loss
            newest_path = max(candidates, key=os.path.getmtime)
            return newest_path, None

    if os.path.isdir(CHECKPOINTS_DIR):
        best_path, best_loss = find_best_checkpoint(CHECKPOINTS_DIR)
        return best_path, best_loss

    raise FileNotFoundError(
        "No checkpoint found. Set CHECKPOINT_PATH or MODEL_URL, or include a .ckpt file "
        "under ./artifacts or ./checkpoints in the Bento."
    )


def load_model() -> Tuple[torch.nn.Module, List[str], str, Optional[float]]:
    checkpoint_path, best_loss = resolve_checkpoint()
    model_name = infer_model_name(checkpoint_path)
    model_class = MODEL_CLASS_BY_PREFIX[model_name]

    class_names = load_class_names_from_metadata(DATA_DIR)
    model = model_class.load_from_checkpoint(
        checkpoint_path,
        class_names=class_names,
        map_location=DEVICE,
    )
    model.eval()
    model.to(DEVICE)

    if not class_names:
        class_names = getattr(model, "class_names", None)

    if not class_names:
        class_names = [str(i) for i in range(getattr(model, "num_classes", 0) or 0)]

    return model, class_names, checkpoint_path, best_loss


@bentoml.service(name="ham10000_inference")
class Ham10000Service:
    def __init__(self) -> None:
        self.model, self.class_names, self.best_ckpt, self.best_loss = load_model()
        self.preprocess = build_preprocess()

    @bentoml.api
    def predict(self, image: PilImage.Image, top_k: int = 1) -> dict:
        image = image.convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = self.model(tensor)
            probs_tensor = torch.softmax(logits, dim=1)[0].cpu()
            probs = probs_tensor.tolist()
            pred_idx = int(torch.argmax(logits, dim=1).item())

        top_k = max(1, min(int(top_k), len(probs)))
        top_values, top_indices = torch.topk(probs_tensor, k=top_k)
        top_k_results = []
        for score, idx in zip(top_values.tolist(), top_indices.tolist()):
            top_k_results.append(
                {
                    "label": self.class_names[idx] if self.class_names else str(idx),
                    "probability": score,
                    "index": idx,
                }
            )

        return {
            "label": self.class_names[pred_idx] if self.class_names else str(pred_idx),
            "probabilities": probs,
            "top_k": top_k_results,
            "class_names": self.class_names,
            "checkpoint": os.path.basename(self.best_ckpt),
            "val_loss": self.best_loss,
        }
