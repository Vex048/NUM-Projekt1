import argparse
import yaml
import torch
import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
import os
import sys

# Dodajemy folder główny projektu (NUM) do PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ham10000_datamodule import HAM10000DataModule
from src.models.resnet_classifier import ResNetClassifier
from src.models.baseline_classifier import BaselineClassifier
from src.models.efficientnet_classifier import EfficientNetClassifier
from src.models.densenet_classifier import DenseNetClassifier

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def objective(trial, config_path):
    config = load_config(config_path)
    
    lr = trial.suggest_categorical("lr", [1e-3,5e-4, 1e-4, 5e-5, 1e-5])
    optimizer = "adamw" 
    weight_decay = trial.suggest_categorical("weight_decay", [0.1,0.05,0.01])
    scheduler = trial.suggest_categorical("scheduler", ["plateau"])
    batch_size = trial.suggest_categorical("batch_size", [32, 64,128])
    strategy = trial.suggest_categorical("class_imbalance_strategy", ["sampler"])
    
    config["model"]["lr"] = lr
    config["model"]["optimizer"] = optimizer
    config["model"]["weight_decay"] = weight_decay
    config["model"]["scheduler"] = scheduler
    
    if "data" not in config:
        config["data"] = {}
    config["data"]["batch_size"] = batch_size
    config["data"]["class_imbalance_strategy"] = strategy
    
    config["project_name"] = config.get("project_name", "skin-lesion-classification") + "-optuna-effnet-v3"
    run_name = f"trial_{trial.number}_adamw_lr{lr}_bs{batch_size}-v2"
    
    wandb_logger = WandbLogger(
        project=config["project_name"],
        name=run_name,
        log_model=False, # to save space
        config=config,
    )

    data_cfg = config.get("data", {})
    use_sampler = strategy in ["sampler", "both"]
    
    datamodule = HAM10000DataModule(
        data_dir=data_cfg.get("data_dir", "./dataset/archive"),
        batch_size=data_cfg.get("batch_size", 64),
        num_workers=data_cfg.get("num_workers", 4),
        use_sampler=use_sampler,
    )
    datamodule.setup(stage="fit")

    class_names = getattr(datamodule, "classes", [str(i) for i in range(10)])
    if strategy in ["weights", "both"]:
        class_weights = getattr(datamodule, "class_weights", None)
    else:
        class_weights = None
        
    num_classes = len(class_names)
    
    model_name = config.get("model", {}).get("name", "resnet")
    
    if model_name == "resnet":
        model = ResNetClassifier(
            num_classes=num_classes,
            lr=lr,
            weight_decay=weight_decay,
            optimizer_name=optimizer,
            pretrained=config.get("model", {}).get("pretrained", True),
            class_names=class_names,
            class_weights=class_weights,
        )
    elif model_name == "baseline":
        model = BaselineClassifier(
            num_classes=num_classes,
            lr=lr,
            weight_decay=weight_decay,
            optimizer_name=optimizer,
            scheduler_name=scheduler,
            class_names=class_names,
            class_weights=class_weights,
        )
    elif model_name == "efficientnet":
        model = EfficientNetClassifier(
            num_classes=num_classes,
            lr=lr,
            weight_decay=weight_decay,
            optimizer_name=optimizer,
            scheduler_name=scheduler,
            pretrained=config.get("model", {}).get("pretrained", True),
            class_names=class_names,
            class_weights=class_weights,
            freeze_backbone=config.get("model", {}).get("freeze_backbone", False),
        )
    elif model_name == "densenet":
        model = DenseNetClassifier(
            num_classes=num_classes,
            lr=lr,
            weight_decay=weight_decay,
            optimizer_name=optimizer,
            scheduler_name=scheduler,
            pretrained=config.get("model", {}).get("pretrained", True),
            class_names=class_names,
            class_weights=class_weights,
        )
    else:
        raise ValueError(f"Nieznany model: {model_name}")

    trainer_cfg = config.get("trainer", {})
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5, # smaller patience for tuning so trials finish faster
        verbose=False,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=trainer_cfg.get("max_epochs", 10),
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", 1),
        precision=trainer_cfg.get("precision", "32-true"),
        logger=wandb_logger,
        callbacks=[early_stop_callback],
        enable_checkpointing=False, # Don't save checkpoints during tuning
    )

    trainer.fit(model, datamodule=datamodule)
    
    val_loss = trainer.callback_metrics.get("val_loss")
    wandb.finish()
    
    if val_loss is not None:
        return val_loss.item()
    return float("inf")

def main():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/optuna.yaml",
        help="Ścieżka do pliku konfiguracyjnego YAML",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Liczba prób Optuny",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Błąd: Plik konfiguracyjny {args.config} nie istnieje!")
        return

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, args.config), n_trials=args.n_trials)

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == "__main__":
    main()
