import pandas as pd

DATASET_PATH = "dataset/archive/HAM10000_metadata.csv"
dataset_8_8_l = "dataset/archive/hmnist_8_8_L.csv"
dataset_8_8_rgb = "dataset/archive/hmnist_8_8_RGB.csv"
dataset_28_28_l = "dataset/archive/hmnist_28_28_L.csv"
dataset_28_28_rgb = "dataset/archive/hmnist_28_28_RGB.csv"

images_1 = "dataset/archive/ham10000_images_part_1"
images_2 = "dataset/archive/ham10000_images_part_2"


if __name__ == "__main__":
    df = pd.read_csv(DATASET_PATH)
    print(df.head())
    print(df.info())
    print(df.describe())

    df2 = pd.read_csv(dataset_8_8_l)
    print(df2.head())
    print(df2.info())
    print(df2.describe())

    df3 = pd.read_csv(dataset_8_8_rgb)
    print(df3.head())
    print(df3.info())
    print(df3.describe())

    df4 = pd.read_csv(dataset_28_28_l)
    print(df4.head())
    print(df4.info())
    print(df4.describe())

    df5 = pd.read_csv(dataset_28_28_rgb)
    print(df5.head())
    print(df5.info())
    print(df5.describe())
