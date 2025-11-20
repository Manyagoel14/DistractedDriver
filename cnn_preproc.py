import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.utils import load_img, img_to_array
import pickle

# ======================================
# PATHS
# ======================================
BASE_DIR = "/content/drive/MyDrive/Distracted-Driver-Detection_1"
TRAIN_DIR = f"{BASE_DIR}/dataset/imgs/train"
TEST_DIR  = f"{BASE_DIR}/dataset/imgs/test"
CSV_DIR   = f"{BASE_DIR}/csv_files"
PICKLE_DIR = f"{BASE_DIR}/pickle_files"
NPY_DIR = f"{BASE_DIR}/npy_files"

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PICKLE_DIR, exist_ok=True)
os.makedirs(NPY_DIR, exist_ok=True)

# ======================================
# CREATE CSV
# ======================================
def create_csv(DATA_DIR, filename):
    data = []
    items = os.listdir(DATA_DIR)

    # Check folder-type dataset
    if os.path.isdir(os.path.join(DATA_DIR, items[0])):
        for cls in items:
            cdir = os.path.join(DATA_DIR, cls)
            if not os.path.isdir(cdir): continue
            for file in os.listdir(cdir):
                data.append({"Filename": os.path.join(cdir, file), "ClassName": cls})
    else:
        for file in items:
            data.append({"Filename": os.path.join(DATA_DIR, file), "ClassName": "test"})

    pd.DataFrame(data).to_csv(f"{CSV_DIR}/{filename}", index=False)

# Make CSVs
create_csv(TRAIN_DIR, "train.csv")
create_csv(TEST_DIR, "test.csv")

# ======================================
# LOAD & ENCODE LABELS
# ======================================
train_df = pd.read_csv(f"{CSV_DIR}/train.csv")
labels_list = sorted(list(train_df["ClassName"].unique()))
labels_map = {label: idx for idx, label in enumerate(labels_list)}

train_df["ClassName"] = train_df["ClassName"].map(labels_map)

# Save labels map
with open(f"{PICKLE_DIR}/labels_map.pkl", "wb") as f:
    pickle.dump(labels_map, f)

# ======================================
# CONVERT IMAGES → TENSORS
# ======================================
def paths_to_tensor(paths):
    tensors = []
    for p in tqdm(paths):
        try:
            img = load_img(p, target_size=(128, 128))
            arr = img_to_array(img)
            tensors.append(arr)
        except:
            pass
    return np.array(tensors)

# Convert
X = paths_to_tensor(train_df["Filename"].values) / 255.0
y = train_df["ClassName"].values

# Save numpy arrays
np.save(f"{NPY_DIR}/X.npy", X)
np.save(f"{NPY_DIR}/y.npy", y)

print("✅ Preprocessing Complete!")
print("Saved:")
print(f"- {NPY_DIR}/X.npy")
print(f"- {NPY_DIR}/y.npy")
print(f"- {PICKLE_DIR}/labels_map.pkl")
