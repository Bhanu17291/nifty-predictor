"""
Fixes the train/test split in features_v2.csv
Moves cutoff from 2022 to 2023
Run: python fix_split.py
"""
import pandas as pd
import os

DATA_DIR = "data"

print("Loading features_v2.csv...")
df = pd.read_csv(os.path.join(DATA_DIR, "features_v2.csv"),
                 index_col=0, parse_dates=True)
df = df.sort_index()

# Show current split
old_train = df[df.index.year <= 2022]
old_test  = df[df.index.year >= 2023]
print(f"OLD split — Train: {len(old_train)} rows | Test: {len(old_test)} rows")

# Apply new split
train = df[df.index.year <= 2023]
test  = df[df.index.year >= 2024]
print(f"NEW split — Train: {len(train)} rows | Test: {len(test)} rows")

if len(test) < 50:
    print("ERROR: Not enough 2024 data in features_v2.csv")
    print("You need to re-run data_fetch.py and features_v2.py first")
    print("Make sure END date in data_fetch.py includes 2024 data")
else:
    train.to_csv(os.path.join(DATA_DIR, "train_v2.csv"))
    test.to_csv(os.path.join(DATA_DIR,  "test_v2.csv"))
    print("Saved new train_v2.csv and test_v2.csv")
    print("Now run: python ensemble_model.py")