import pandas as pd
import matplotlib.pyplot as plt
import os

METADATA_CSV_PATH = "datasets/archive/HAM10000_metadata.csv"
IMAGES_DIRS = [
    "datasets/archive/ham10000_images_part_1",
    "datasets/archive/ham10000_images_part_2",
]


def check_for_corelation_between_sex_and_class(df: pd.DataFrame):

    plt.figure(figsize=(10, 6))
    for cls in df["dx"].unique():
        subset = df[df["dx"] == cls]
        plt.hist(subset["sex"], alpha=0.5, label=cls, edgecolor="black")
    plt.title("Zależność między płcią a klasą")
    plt.xlabel("Płeć")
    plt.ylabel("Liczba próbek")
    plt.legend()
    plt.savefig("data_analysis/sex_vs_class.png")
    plt.close()


def check_for_corelation_between_age_and_class(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    for cls in df["dx"].unique():
        subset = df[df["dx"] == cls]
        plt.scatter(subset["age"], [cls] * len(subset), alpha=0.5, label=cls)
    plt.title("Zależność między wiekiem a klasą")
    plt.xlabel("Wiek")
    plt.ylabel("Klasa")
    plt.legend()
    plt.savefig("data_analysis/age_vs_class.png")
    plt.close()


import matplotlib.pyplot as plt
import pandas as pd


def check_age_across_classes(df: pd.DataFrame):
    classes = df["dx"].unique()
    num_classes = len(classes)

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 8))
    axes = axes.flatten()

    for i, cls in enumerate(classes):
        subset = df[df["dx"] == cls]
        axes[i].hist(subset["age"].dropna(), bins=20, edgecolor="black", color="coral")

        axes[i].set_title(f"Klasa: {cls}")
        axes[i].set_xlabel("Wiek")
        axes[i].set_ylabel("Liczba próbek")
        axes[i].set_xlim(0, 90)
    if num_classes < len(axes):
        for j in range(num_classes, len(axes)):
            fig.delaxes(axes[j])

    plt.tight_layout()

    plt.savefig("data_analysis/age_distribution_by_class.png")
    plt.close()


def check_most_common_locations_for_each_class(df: pd.DataFrame):
    for cls in df["dx"].unique():
        subset = df[df["dx"] == cls]
        location_counts = subset["localization"].value_counts()
        print(f"Najczęstsze lokalizacje dla klasy {cls}:")
        print(location_counts)
        print()


def check_overall_age_distribution(df: pd.DataFrame):
    # Here i want to have onyl histogram of age distribution, without splitting it by class
    print("Podstawowe statystyki wieku:")
    df_age = df["age"].dropna()
    print(df_age.describe())
    plt.hist(df_age, edgecolor="black")
    plt.xlabel("Wiek")
    plt.ylabel("Ilośc próbek")
    plt.savefig("data_analysis/age_distribution.png")
    plt.close()


def check_class_imbalance(df: pd.DataFrame):
    class_counts = df["dx"].value_counts()
    print("Rozkład klas w zbiorze danych:")
    print(class_counts)
    print("\nProcentowy rozkład klas:")
    print((class_counts / len(df)) * 100)
    plt.hist(df["dx"], edgecolor="black")
    plt.title("Rozkład klas w zbiorze danych")
    plt.xlabel("Klasa")
    plt.ylabel("Liczba próbek")
    plt.savefig("data_analysis/class_distribution.png")
    plt.close()


def get_each_class_image(df: pd.DataFrame):
    class_samples = {}
    for cls in df["dx"].unique():
        sample = df[df["dx"] == cls].iloc[0]
        class_samples[cls] = sample["image_id"]
    return class_samples


def get_image_path(image_id):
    img_name = f"{image_id}.jpg"
    for d in IMAGES_DIRS:
        path = os.path.join(d, img_name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Nie znaleziono zdjęcia dla: {img_name}")


def save_class_samples_images(class_samples):
    for cls, image_id in class_samples.items():
        img_path = get_image_path(image_id)
        img = plt.imread(img_path)
        plt.imshow(img)
        plt.title(f"Klasa: {cls}")
        plt.axis("off")
        plt.savefig(f"data_analysis/sample_{cls}.png")
        plt.close()


if __name__ == "__main__":
    df = pd.read_csv(METADATA_CSV_PATH)
    check_class_imbalance(df)
    check_overall_age_distribution(df)
    check_age_across_classes(df)
    check_for_corelation_between_sex_and_class(df)
    check_for_corelation_between_age_and_class(df)
    check_most_common_locations_for_each_class(df)
    class_samples = get_each_class_image(df)
    save_class_samples_images(class_samples)
