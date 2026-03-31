import argparse
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
import os
from src.data.ham10000_datamodule import HAM10000DataModule
from src.models.resnet_classifier import ResNetClassifier
from src.models.baseline_classifier import BaselineClassifier
from src.models.efficientnet_classifier import EfficientNetClassifier
from src.models.densenet_classifier import DenseNetClassifier


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_datamodule(config: dict):
    data_cfg = config.get("data", {})
    strategy = data_cfg.get("class_imbalance_strategy", "weights")
    use_sampler = strategy in ["sampler", "both"]
    
    if data_cfg.get("name") == "ham10000":
        return HAM10000DataModule(
            data_dir=data_cfg.get("data_dir", "./dataset/archive"),
            batch_size=data_cfg.get("batch_size", 64),
            num_workers=data_cfg.get("num_workers", 4),
            use_sampler=use_sampler,
        )
    else:
        raise ValueError(f"Unknown dataset: {data_cfg.get('name')}")


def get_model(config: dict, class_names: list, class_weights: torch.Tensor = None):
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name")
    num_classes = model_cfg.get("num_classes", 10)

    lr = model_cfg.get("lr", 1e-3)
    weight_decay = model_cfg.get("weight_decay", 1e-4)
    optimizer = model_cfg.get("optimizer", "adam")
    scheduler = model_cfg.get("scheduler", "cosine")

    if model_name == "resnet":
        pretrained = model_cfg.get("pretrained", True)
        return ResNetClassifier(
            num_classes=num_classes,
            lr=lr,
            weight_decay=weight_decay,
            optimizer_name=optimizer,
            scheduler_name=scheduler,
            pretrained=pretrained,
            class_names=class_names,
            class_weights=class_weights,
        )
    elif model_name == "baseline":
        return BaselineClassifier(
            num_classes=num_classes,
            lr=lr,
            weight_decay=weight_decay,
            optimizer_name=optimizer,
            scheduler_name=scheduler,
            class_names=class_names,
            class_weights=class_weights,
        )
    elif model_name == "efficientnet":
        pretrained = model_cfg.get("pretrained", True)
        freeze_backbone = model_cfg.get("freeze_backbone", False)
        return EfficientNetClassifier(
            num_classes=num_classes,
            lr=lr,
            weight_decay=weight_decay,
            optimizer_name=optimizer,
            scheduler_name=scheduler,
            pretrained=pretrained,
            class_names=class_names,
            class_weights=class_weights,
            freeze_backbone=freeze_backbone,
        )
    elif model_name == "densenet":
        pretrained = model_cfg.get("pretrained", True)
        return DenseNetClassifier(
            num_classes=num_classes,
            lr=lr,
            weight_decay=weight_decay,
            optimizer_name=optimizer,
            scheduler_name=scheduler,
            pretrained=pretrained,
            class_names=class_names,
            class_weights=class_weights,
        )
    else:
        raise ValueError(f"Nieznany model: {model_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Trenowanie klasyfikacji obrazów (PyTorch Lightning)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/official_train.yaml",
        help="Ścieżka do pliku konfiguracyjnego YAML",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Błąd: Plik konfiguracyjny {args.config} nie istnieje!")
        return

    config = load_config(args.config)

    # Init wandb logger
    wandb_logger = WandbLogger(
        project=config.get("project_name", "skin-lesion-classification"),
        name=config.get("run_name", "efficientnet-ham10000-adamw"),
        log_model="all",
        config=config,
    )
    # Init datamodule
    datamodule = get_datamodule(config)
    datamodule.setup(stage="fit")

    class_names = getattr(datamodule, "classes", [str(i) for i in range(10)])
    
    strategy = config.get("data", {}).get("class_imbalance_strategy", "weights")
    if strategy in ["weights", "both"]:
        class_weights = getattr(datamodule, "class_weights", None)
    else:
        class_weights = None
        
    config["model"]["num_classes"] = len(class_names)

    # Init Torch lightning
    model = get_model(config, class_names, class_weights)

    # Callbacki
    trainer_cfg = config.get("trainer", {})
    model_name = config.get("model", {}).get("name", "model")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",   
        dirpath="./checkpoints",
        filename=model_name + "-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=trainer_cfg.get("patience", 5),
        verbose=True,
        mode="min",
    )

    # Init obiektu Trenera
    trainer = pl.Trainer(
        max_epochs=trainer_cfg.get("max_epochs", 10),
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", 1),
        precision=trainer_cfg.get("precision", "32-true"),
        gradient_clip_val=trainer_cfg.get("gradient_clip_val", 0.0),
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    # Trening oraz test
    trainer.fit(model, datamodule=datamodule)

    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    wandb.finish()


if __name__ == "__main__":
    main()
