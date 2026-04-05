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

def objective(trial, config_path, current_model_name):

    config = load_config(config_path)
    
    optimizer = "adamw"
    
    if current_model_name == "resnet":
        lr = trial.suggest_categorical("lr", [1e-3, 5e-4, 1e-4,5e-5])
        batch_size = trial.suggest_categorical("batch_size", [64, 128])
        weight_decay = trial.suggest_categorical("weight_decay", [1e-3, 1e-4,1e-5])
    elif current_model_name == "efficientnet":

        lr = trial.suggest_categorical("lr", [5e-4, 1e-4, 5e-5])
        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        weight_decay = trial.suggest_categorical("weight_decay", [1e-3, 1e-4,1e-5])
    elif current_model_name == "densenet":

        lr = trial.suggest_categorical("lr", [5e-4, 1e-4, 5e-5])
        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        weight_decay = trial.suggest_categorical("weight_decay", [1e-3, 1e-4,1e-5])
    else:
        lr = trial.suggest_categorical("lr", [1e-4])
        batch_size = 32
        weight_decay = 1e-4
        
    scheduler = trial.suggest_categorical("scheduler", ["cosine", "plateau"])
    strategy = trial.suggest_categorical("class_imbalance_strategy", ["weights", "sampler"])
    
    # Podmiana w konfiguracji bazowo-zapisanej
    if "model" not in config:
        config["model"] = {}
    config["model"]["name"] = current_model_name
    config["model"]["lr"] = lr
    config["model"]["optimizer"] = optimizer
    config["model"]["weight_decay"] = weight_decay
    config["model"]["scheduler"] = scheduler
    
    if "data" not in config:
        config["data"] = {}
    config["data"]["batch_size"] = batch_size
    config["data"]["class_imbalance_strategy"] = strategy
    

    base_project_name = config.get("project_name", "skin-lesion-classification - 3models-TEST")
    config["project_name"] = f"{base_project_name}-optuna-3ModelsTest-{current_model_name}"
    run_name = f"trial_{trial.number}_{optimizer}_lr{lr}_bs{batch_size}"
    
    wandb_logger = WandbLogger(
        project=config["project_name"],
        name=run_name,
        log_model=False,
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
    

    pretrained = config.get("model", {}).get("pretrained", True)
    
    if current_model_name == "resnet":
        model = ResNetClassifier(
            num_classes=num_classes, lr=lr, weight_decay=weight_decay,
            optimizer_name=optimizer, scheduler_name=scheduler,
            pretrained=pretrained, class_names=class_names, class_weights=class_weights
        )
    elif current_model_name == "efficientnet":
        model = EfficientNetClassifier(
            num_classes=num_classes, lr=lr, weight_decay=weight_decay,
            optimizer_name=optimizer, scheduler_name=scheduler,
            pretrained=pretrained, class_names=class_names, class_weights=class_weights
        )
    elif current_model_name == "densenet":
        model = DenseNetClassifier(
            num_classes=num_classes, lr=lr, weight_decay=weight_decay,
            optimizer_name=optimizer, scheduler_name=scheduler,
            pretrained=pretrained, class_names=class_names, class_weights=class_weights
        )
    else:
        raise ValueError(f"Nieznany model: {current_model_name}")


    trainer_cfg = config.get("trainer", {})
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3, 
        verbose=False,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=trainer_cfg.get("max_epochs", 10),
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", 1),
        precision=trainer_cfg.get("precision", "16-mixed"), 
        logger=wandb_logger,
        callbacks=[early_stop_callback],
        enable_checkpointing=False,
    )

    trainer.fit(model, datamodule=datamodule)
    
    val_loss = trainer.callback_metrics.get("val_loss")
    wandb.finish()
    
    if val_loss is not None:
        return val_loss.item()
    return float("inf")

def main():
    parser = argparse.ArgumentParser(description="Multimodel Optuna Tuning")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Ścieżka do pliku konfiguracyjnego YAML",
    )
    parser.add_argument(
        "--n_trials", type=int, default=20,
        help="Liczba prób Optuny NA KAŻDY Z MODELI",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Błąd: Plik konfiguracyjny {args.config} nie istnieje!")
        return

    models_to_tune = ["efficientnet", "densenet"]

    for model_name in models_to_tune:
        print(f"\n{'='*70}")
        print(f"   ROZPOCZYNAM STROJENIE DLA MODELU: {model_name.upper()}")
        print(f"{'='*70}\n")
        
        study = optuna.create_study(
            study_name=f"ham10000_{model_name}",
            direction="minimize"
        )
        study.optimize(
            lambda trial: objective(trial, args.config, model_name), 
            n_trials=args.n_trials
        )

        print(f"\n[PODSUMOWANIE {model_name.upper()}]")
        print(f"Zakończonych prób: {len(study.trials)}")
        print("Najlepszy trial:")
        best_trial = study.best_trial
        print(f"  Wartość (val_loss): {best_trial.value}")
        print("  Parametry: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
        print("\n")

if __name__ == "__main__":
    main()
