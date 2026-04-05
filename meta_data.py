import pandas as pd
import matplotlib.pyplot as plt


PATH_TO_CSV = "archive/HAM10000_metadata.csv"


def read_csv(path_to_csv: str) -> pd.DataFrame:
    return pd.read_csv(path_to_csv)


if __name__ == "__main__":
    df = read_csv(PATH_TO_CSV)
    print(df.head())
    print(df.info())
    print(df["dx"].value_counts())
    plt.hist(df["dx"], histtype="bar")
    plt.savefig("hist - class")
