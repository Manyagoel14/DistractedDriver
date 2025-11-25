import os
import shutil
import pandas as pd

test_act_path = "./test_actual"
test_path = "./test"
train_path = "./csv_files/train.csv"
image_folder = "./dataset/imgs/train"
os.makedirs(test_act_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

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

for cls, group in grouped:
    subset = group.sample(n=100, random_state=42)

    for idx, row in subset.iterrows():
        img_name = row["Filename"]
        label = row["ClassName"]

        # Normalize slashes
        clean = img_name.replace("\\", "/")

        # Extract path after '/dataset/imgs/train/'
        marker = "/dataset/imgs/train/"
        clean = clean.split(marker)[-1]     # "c7/img_55214.jpg"

        # Build local source path
        src = os.path.join(image_folder, clean).replace("\\", "/")


        # Copy original image to test/
        base = clean.split("/")[-1]         # img_55214.jpg
        shutil.copy(src, os.path.join(test_path, base))

        # Create renamed file for test_actual/
        new_name = base.replace(".jpg", "") + "_" + class_name[label] + ".jpg"
        dst = os.path.join(test_act_path, new_name)

        shutil.copy(src, dst)

print("Done! 100 files per class copied.")
