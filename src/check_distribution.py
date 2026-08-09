import pandas as pd

def print_dist(csv_path, name):
    try:
        df = pd.read_csv(csv_path)
        counts = df['diagnosis'].value_counts()
        total = len(df)
        print(f"\n--- Phân bố tập {name} (Tổng: {total} ca) ---")
        for cls, count in counts.items():
            print(f"{cls}: {count} ({count/total*100:.1f}%)")
    except Exception as e:
        print(f"Không tìm thấy {csv_path}")

print_dist("data/processed/train_split.csv", "TRAIN")
print_dist("data/processed/val_split.csv", "VALIDATION")
print_dist("data/processed/test_split.csv", "TEST")