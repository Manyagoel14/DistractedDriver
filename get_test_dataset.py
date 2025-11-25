import os
import pandas as pd

test_path = "./test"
test_act_path = "./test_actual"
train_path = "./csv_files/train.csv"
image_folder = "./dataset/imgs/train"

os.makedirs(test_path, exist_ok=True)
os.makedirs(test_act_path, exist_ok=True)

df = pd.read_csv(train_path)

class_name = {
    "c0": "SAFE_DRIVING",
    "c1": "TEXTING_RIGHT",
    "c2": "TALKING_PHONE_RIGHT",
    "c3": "TEXTING_LEFT",
    "c4": "TALKING_PHONE_LEFT",
    "c5": "OPERATING_RADIO",
    "c6": "DRINKING",
    "c7": "REACHING_BEHIND",
    "c8": "HAIR_AND_MAKEUP",
    "c9": "TALKING_TO_PASSENGER"
}

grouped = df.groupby("ClassName")

test_rows = []
test_actual_rows = []

for cls, group in grouped:
    subset = group.sample(n=1000, random_state=42)

    for idx, row in subset.iterrows():
        img_name = row["Filename"]
        label = row["ClassName"]

        # ---------- test.csv ----------
        test_rows.append({
            "path": os.path.join(img_name).replace("\\", "/"),
            "class_name": "test"
        })

        # ---------- test_actual.csv ----------
        new_name = img_name.replace(".jpg", "") + "_" + class_name[label] + ".jpg"
        test_actual_rows.append({
            "path": os.path.join(img_name).replace("\\", "/"),
            "actual_name": label
        })

# Save CSV files
pd.DataFrame(test_rows).to_csv("test.csv", index=False)
pd.DataFrame(test_actual_rows).to_csv("test_actual.csv", index=False)

print("Done! CSV files created.")