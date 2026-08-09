import os
import pandas as pd

def check_distribution(csv_path, split_name):
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    dist = df['diagnosis'].value_counts(normalize=True) * 100
    print(f"\nPhân bố lớp tập {split_name} (Tổng: {len(df)} mẫu):")
    print(dist.round(2).astype(str) + "%")

if __name__ == "__main__":
    base_dir = "data/processed/"
    check_distribution(os.path.join(base_dir, "train_split.csv"), "TRAIN")
    check_distribution(os.path.join(base_dir, "val_split.csv"), "VALIDATION")
    check_distribution(os.path.join(base_dir, "test_split.csv"), "TEST")