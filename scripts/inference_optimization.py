import argparse
import copy
import io
import os
import sys
import time
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchmetrics import F1Score

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data.ham10000_datamodule import HAM10000DataModule
from src.models.resnet_classifier import ResNetClassifier
from src.models.efficientnet_classifier import EfficientNetClassifier
from src.models.densenet_classifier import DenseNetClassifier
from src.models.baseline_classifier import BaselineClassifier

MODEL_CLASS_BY_PREFIX = {
    "resnet": ResNetClassifier,
    "efficientnet": EfficientNetClassifier,
    "densenet": DenseNetClassifier,
    "baseline": BaselineClassifier,
}


def infer_model_name(checkpoint_path: str, override_name: Optional[str] = None) -> str:
    if override_name:
        return override_name.lower()
    base = os.path.basename(checkpoint_path).lower()
    for prefix in MODEL_CLASS_BY_PREFIX:
        if base.startswith(prefix):
            return prefix
    raise ValueError(
        "Cannot infer model name from checkpoint. "
        "Provide --model-name (resnet|efficientnet|densenet|baseline)."
    )


def load_model(
    checkpoint_path: str,
    class_names: List[str],
    device: torch.device,
    model_name_override: Optional[str] = None,
) -> nn.Module:
    model_name = infer_model_name(checkpoint_path, model_name_override)
    model_class = MODEL_CLASS_BY_PREFIX[model_name]
    model = model_class.load_from_checkpoint(
        checkpoint_path, class_names=class_names, map_location=device
    )
    model.eval()
    model.to(device)
    return model


def get_state_dict_size_bytes(state_dict: Dict[str, torch.Tensor]) -> int:
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getbuffer().nbytes


def get_model_size_bytes(model: nn.Module) -> int:
    return get_state_dict_size_bytes(model.state_dict())


def make_int16_quantized(model: nn.Module) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
    state = model.state_dict()
    q_state: Dict[str, torch.Tensor] = {}
    scales: Dict[str, float] = {}

    for name, tensor in state.items():
        if not tensor.is_floating_point():
            q_state[name] = tensor
            continue
        max_abs = tensor.abs().max().item()
        scale = max_abs / 32767.0 if max_abs > 0 else 1.0
        q_tensor = torch.round(tensor / scale).clamp(-32768, 32767).to(torch.int16)
        q_state[name] = q_tensor
        scales[name] = scale

    dequant_state: Dict[str, torch.Tensor] = {}
    for name, tensor in q_state.items():
        if tensor.dtype == torch.int16:
            dequant_state[name] = tensor.float() * scales[name]
        else:
            dequant_state[name] = tensor

    dequant_model = copy.deepcopy(model).cpu()
    dequant_model.load_state_dict(dequant_state, strict=False)
    dequant_model.eval()

    # Store scales in the size estimate to account for per-tensor scale values.
    scale_tensors = {
        f"{name}__scale": torch.tensor(scale, dtype=torch.float32)
        for name, scale in scales.items()
    }
    q_state_with_scales = {**q_state, **scale_tensors}

    return dequant_model, q_state_with_scales


def make_int8_dynamic_quantized(model: nn.Module) -> nn.Module:
    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


class FeatureSelector(nn.Module):
    def __init__(self, indices: torch.Tensor):
        super().__init__()
        self.register_buffer("indices", indices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.index_select(1, self.indices)


def structured_prune_last_linear(model: nn.Module, keep_ratio: float) -> nn.Module:
    last_linear_name = None
    last_linear = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_linear_name = name
            last_linear = module

    if last_linear is None:
        return model

    weight = last_linear.weight.detach()
    importance = weight.abs().sum(dim=0)
    keep_count = max(1, int(importance.numel() * keep_ratio))
    keep_indices = torch.topk(importance, k=keep_count).indices.sort().values

    new_linear = nn.Linear(
        keep_count, last_linear.out_features, bias=last_linear.bias is not None
    )
    new_linear.weight.data = weight[:, keep_indices].clone()
    if last_linear.bias is not None:
        new_linear.bias.data = last_linear.bias.detach().clone()

    selector = FeatureSelector(keep_indices)

    def replace_module(root: nn.Module, name: str, module: nn.Module) -> None:
        parts = name.split(".")
        target = root
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], module)

    replace_module(model, last_linear_name, nn.Sequential(selector, new_linear))
    return model


def apply_unstructured_pruning(model: nn.Module, amount: float) -> nn.Module:
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    return model


def count_nonzero_params(model: nn.Module) -> int:
    total = 0
    for param in model.parameters():
        if param.is_floating_point():
            total += int((param != 0).sum().item())
        else:
            total += int(param.numel())
    return total


def run_f1(
    model: nn.Module, loader: Iterable, device: torch.device, num_classes: int
) -> float:
    metric = F1Score(task="multiclass", num_classes=num_classes).to(device)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            dtype = next(model.parameters()).dtype
            if dtype == torch.float16:
                x = x.half()
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            metric.update(preds, y)
    return float(metric.compute().item())


def time_inference(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    warmup: int,
    iters: int,
) -> float:
    model.eval()
    batches = iter(loader)

    def next_batch():
        nonlocal batches
        try:
            return next(batches)
        except StopIteration:
            batches = iter(loader)
            return next(batches)

    with torch.inference_mode():
        for _ in range(warmup):
            x, _ = next_batch()
            x = x.to(device)
            dtype = next(model.parameters()).dtype
            if dtype == torch.float16:
                x = x.half()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

    times: List[float] = []
    with torch.inference_mode():
        for _ in range(iters):
            x, _ = next_batch()
            x = x.to(device)
            dtype = next(model.parameters()).dtype
            if dtype == torch.float16:
                x = x.half()
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    return sum(times) / max(1, len(times))


def prepare_datamodule(
    data_dir: str, batch_size: int, num_workers: int
) -> HAM10000DataModule:
    datamodule = HAM10000DataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        use_sampler=False,
    )
    datamodule.setup(stage="test")
    return datamodule


def main() -> None:
    parser = argparse.ArgumentParser(description="HW2 inference optimization")
    parser.add_argument(
        "--checkpoint",
        default="./artifacts/best.ckpt",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--model-name",
        choices=sorted(MODEL_CLASS_BY_PREFIX.keys()),
        help="Model architecture when checkpoint filename has no prefix",
    )
    parser.add_argument(
        "--data-dir",
        default="./dataset/archive",
        help="Dataset root directory",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--unstructured-amount", type=float, default=0.3)
    parser.add_argument("--structured-keep", type=float, default=0.7)
    args = parser.parse_args()

    device = torch.device(args.device)
    datamodule = prepare_datamodule(args.data_dir, args.batch_size, args.num_workers)
    test_loader = datamodule.test_dataloader()
    class_names = getattr(datamodule, "classes", None)
    if not class_names:
        class_names = [str(i) for i in range(7)]

    base_model = load_model(args.checkpoint, class_names, device, args.model_name)
    num_classes = len(class_names)

    results = []

    float_f1 = run_f1(base_model, test_loader, device, num_classes)
    float_time = time_inference(
        base_model, test_loader, device, args.warmup, args.iters
    )
    float_size = get_model_size_bytes(base_model)
    results.append(("float32", float_f1, float_time, float_size))

    fp16_model = copy.deepcopy(base_model).to(device)
    fp16_model = fp16_model.half()
    fp16_f1 = run_f1(fp16_model, test_loader, device, num_classes)
    fp16_time = time_inference(fp16_model, test_loader, device, args.warmup, args.iters)
    fp16_size = get_model_size_bytes(fp16_model)
    results.append(("float16", fp16_f1, fp16_time, fp16_size))

    int16_model, int16_state = make_int16_quantized(base_model)
    int16_model = int16_model.to(device)
    int16_f1 = run_f1(int16_model, test_loader, device, num_classes)
    int16_time = time_inference(
        int16_model, test_loader, device, args.warmup, args.iters
    )
    int16_size = get_state_dict_size_bytes(int16_state)
    results.append(("int16 (sim)", int16_f1, int16_time, int16_size))

    if device.type != "cpu":
        print("Warning: int8 dynamic quantization runs on CPU. Using CPU for int8.")
    int8_device = torch.device("cpu")
    int8_base = load_model(args.checkpoint, class_names, int8_device, args.model_name)
    int8_model = make_int8_dynamic_quantized(int8_base)
    int8_f1 = run_f1(int8_model, test_loader, int8_device, num_classes)
    int8_time = time_inference(
        int8_model, test_loader, int8_device, args.warmup, args.iters
    )
    int8_size = get_model_size_bytes(int8_model)
    results.append(("int8 (dynamic)", int8_f1, int8_time, int8_size))

    # Pruning section
    pruned_base = load_model(args.checkpoint, class_names, device, args.model_name)
    unstructured = apply_unstructured_pruning(
        copy.deepcopy(pruned_base), args.unstructured_amount
    )
    un_f1 = run_f1(unstructured, test_loader, device, num_classes)
    un_time = time_inference(unstructured, test_loader, device, args.warmup, args.iters)
    un_nonzero = count_nonzero_params(unstructured)

    structured = structured_prune_last_linear(
        copy.deepcopy(pruned_base), args.structured_keep
    )
    structured.to(device)
    st_f1 = run_f1(structured, test_loader, device, num_classes)
    st_time = time_inference(structured, test_loader, device, args.warmup, args.iters)
    st_nonzero = count_nonzero_params(structured)

    baseline_nonzero = count_nonzero_params(pruned_base)

    print("\nQuantization results:")
    print("Precision\tF1\tTime(s)/batch\tModel size (MB)")
    for name, f1, t, size in results:
        print(f"{name}\t{f1:.4f}\t{t:.6f}\t{size / (1024 ** 2):.2f}")

    print("\nPruning results:")
    print("Variant\tF1\tTime(s)/batch\tNon-zero params")
    print(f"baseline\t{float_f1:.4f}\t{float_time:.6f}\t{baseline_nonzero}")
    print(f"unstructured\t{un_f1:.4f}\t{un_time:.6f}\t{un_nonzero}")
    print(f"structured\t{st_f1:.4f}\t{st_time:.6f}\t{st_nonzero}")


if __name__ == "__main__":
    main()
