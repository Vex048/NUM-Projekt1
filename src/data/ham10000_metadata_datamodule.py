import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder


class Ham10000MetaDataDataset(Dataset):
    def __init__(self, df_metadata: pd.DataFrame):
        self.df = df_metadata

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        # Previously create label class as num of dx class
        label = row["label"]

        no_label_row = row.drop(["label", "lesion_id", "image_id"])

        return no_label_row, label


class Ham10000MetaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_csv: str = "./datasets/archive/HAM10000_metadata.csv",
        batch_size: int = 32,
        num_workers: int = 2,
    ):
        super().__init__()
        self.data_csv = data_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.df: pd.DataFrame = pd.read_csv(
            data_csv
        )  # To można przerzucić w prszyłosci gdzies indziej
        # Setup

    def setup(self):
        # Preprocessing metadanych znalezienie jakiś nullów, feature enginnering
        # Carefull here, to not leak any data to test and val set, beocuse data can be weird here

        raise NotImplementedError()

    def find_replace_nulls(self):
        nulls = self.df.isnull().sum()
        raise NotImplementedError()

    def one_hot_encode_sex(self):
        raise NotImplementedError()

    def normalize_df(self):
        raise NotImplementedError()

    def one_hot_localization(self):
        # Not sure if its worth one hot encode - it will be huge propably
        raise NotImplementedError()

    # Also need to check if dx_type is important here or it can be cut
    #
