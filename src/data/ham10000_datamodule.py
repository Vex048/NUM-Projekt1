import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

class HAM10000Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dirs: list, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dirs = img_dirs
        self.transform = transform

    def _get_image_path(self, image_id):
        img_name = f"{image_id}.jpg"
        for d in self.img_dirs:
            path = os.path.join(d, img_name)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Nie znaleziono zdjęcia dla: {img_name}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['image_id']
        label = row['label'] 
        
        img_path = self._get_image_path(img_id)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

class HAM10000DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './dataset/archive', batch_size: int = 64, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metadata_path = os.path.join(self.data_dir, "HAM10000_metadata.csv")
        self.img_dirs = [
            os.path.join(self.data_dir, "ham10000_images_part_1"),
            os.path.join(self.data_dir, "ham10000_images_part_2")
        ]
        
        # Klasyczna Transformacja obrazów oraz augmentacja
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Będzie potrzebne do mapowania klasy stringów w inty 
        self.label_encoder = LabelEncoder()
        # Te klasy będą odtworzone dynamicznie w setup()
        self.classes = []


    def setup(self, stage: str = None):
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Plik metadanych nie istnieje: {self.metadata_path}")
            
        df = pd.read_csv(self.metadata_path)
        
        # Mapowanie stringów (akiec, bcc itp.) do labeli numerycznych
        df['label'] = self.label_encoder.fit_transform(df['dx'])
        self.classes = self.label_encoder.classes_.tolist()
        
        # Bezpieczny Podział zestawu - tak żeby nie było żadnego data leak'u
        gss = GroupShuffleSplit(test_size=0.20, n_splits=1, random_state=42)
        train_idx, temp_idx = next(gss.split(df, groups=df["lesion_id"]))

        train_df = df.iloc[train_idx]
        temp_df = df.iloc[temp_idx]

        gss_val = GroupShuffleSplit(test_size=0.50, n_splits=1, random_state=42)
        val_idx, test_idx = next(gss_val.split(temp_df, groups=temp_df["lesion_id"]))

        val_df = temp_df.iloc[val_idx]
        test_df = temp_df.iloc[test_idx]

        if stage == 'fit' or stage is None:
            self.train_dataset = HAM10000Dataset(train_df, self.img_dirs, transform=self.train_transform)
            self.val_dataset = HAM10000Dataset(val_df, self.img_dirs, transform=self.val_test_transform)
            
        if stage == 'test' or stage is None:
            self.test_dataset = HAM10000Dataset(test_df, self.img_dirs, transform=self.val_test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
