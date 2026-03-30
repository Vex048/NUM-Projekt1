import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import wandb


class BaseClassifier(pl.LightningModule):
    """
    Abstrakcyjna klasa bazowa dla klasyfikatorów.
    Zajmuje się liczeniem metryk (Loss, Accuracy, F1) oraz logowaniem do W&B
    """

    def __init__(
        self,
        num_classes: int = 7,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer_name: str = "adam",
        scheduler_name: str = "cosine",
        class_names: list = None,
    ):
        super().__init__()
        # Zapisujemy wszystko do logowania w pliku hparams.yaml przez Wandb
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name.lower()
        self.scheduler_name = scheduler_name.lower()
        self.class_names = class_names or [str(i) for i in range(num_classes)]

        # Na razie klasyczna ufnkcji straty- tu pewnie trzeba będzie coś zmienić, bo są niezbalansowane dane
        # self.criterion = nn.CrossEntropyLoss()
        # Use lossFunction for inbalcaned data classes
        # WEights need to be tuned - using the amount of the samples in each class as a starting point
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0] * num_classes))

        # Inicjalizacja metryk
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        self.train_f1(preds, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)
        self.val_f1(preds, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=False)

        # Logujemy obrazy tylko w pierwszym batchu by nie wysycić przepustosowści W&B
        if batch_idx == 0:
            n = min(x.size(0), 8)
            images = x[:n]
            labels = y[:n]
            predictions = preds[:n]

            # Denormalizacja (odwrócenie ImageNetowych statystyk)
            # do estetycznego rysowania w W&B
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
            images = images * std + mean
            images = torch.clamp(
                images, 0, 1
            )  # Ucinamy żeby W&B się nie buntowało na wartości rzędu 1.0001

            wandb_images = []
            for img, pred, label in zip(images, predictions, labels):
                pred_name = self.class_names[pred.item()]
                label_name = self.class_names[label.item()]
                caption = f"Pred: {pred_name} | Truth: {label_name}"
                wandb_images.append(wandb.Image(img, caption=caption))

            if isinstance(self.logger, pl.loggers.WandbLogger):
                self.logger.experiment.log(
                    {"val/predictions": wandb_images, "global_step": self.global_step}
                )

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)
        self.test_f1(preds, y)

        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc)
        self.log("test_f1", self.test_f1)

    def configure_optimizers(self):

        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if self.scheduler_name == "cosine":
            max_epochs = self.trainer.max_epochs if self.trainer.max_epochs else 100
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        elif self.scheduler_name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2, verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
            }

        else:
            return optimizer
