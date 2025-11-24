import os, glob, pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# CONFIG (folders from your repo)
TEST_CSV = "csv_files/test.csv"
TEST_IMG_DIR = "dataset/imgs/test"
MODEL_GLOBS = ["model/**/*.h5", "model/**/*.keras", "model/**/*.tf", "model/**/*.pb"]
MODEL_GLOBS = [p for g in MODEL_GLOBS for p in glob.glob(g, recursive=True)]
PICKLE_CANDIDATES = glob.glob("pickle_files/*.pkl")

# load test file list
df = pd.read_csv(TEST_CSV)
img_col = None
for c in ["image","img","Image","image_path","path","filename"]:
    if c in df.columns:
        img_col = c; break
if img_col is None:
    img_col = df.columns[0]
df["img_name"] = df[img_col].apply(lambda x: os.path.basename(str(x)))
df["full_path"] = df["img_name"].apply(lambda x: os.path.join(TEST_IMG_DIR, x))
df = df.head(10000)
exists = df["full_path"].apply(os.path.exists)
print(f"Test samples total: {len(df)}, found on disk: {exists.sum()}")

# (optional) if you have ground-truth labels in csv, set y_true:
y_true = None
if "label" in df.columns:
    y_true = df["label"].values
else:
    print("No ground-truth labels in test CSV -> script will compute only predicted distributions.")

# helper to build class_names from pickle if available
def load_class_names():
    for p in PICKLE_CANDIDATES:
        try:
            with open(p,'rb') as f:
                labels = pickle.load(f)
            if isinstance(labels, dict):
                inv = {v:k for k,v in labels.items()}
                return [inv[i] for i in sorted(inv.keys())]
            elif isinstance(labels, (list,tuple)):
                return list(labels)
        except:
            continue
    return None

class_names = load_class_names()

# evaluate models
results = []
for model_path in MODEL_GLOBS:
    try:
        print("Loading", model_path)
        model = load_model(model_path)
    except Exception as e:
        print("Failed to load", model_path, ":", e)
        continue

    H, W = model.input_shape[1], model.input_shape[2]
    preds = []
    img_names = []
    for idx, row in df.iterrows():
        p = row["full_path"]
        if not os.path.exists(p):
            preds.append(None); img_names.append(os.path.basename(p)); continue
        img = load_img(p, target_size=(H,W))
        arr = img_to_array(img).astype("float32")/255.0
        arr = np.expand_dims(arr,0)
        y = model.predict(arr, verbose=0)
        pi = int(np.argmax(y,axis=1)[0])
        preds.append(pi)
        img_names.append(os.path.basename(p))

    # if class_names loaded, map indices to labels
    if class_names and len(class_names) > 0:
        pred_labels = [class_names[i] if (i is not None and i < len(class_names)) else None for i in preds]
    else:
        pred_labels = preds

    # compute metrics if y_true exists and lengths match (may need mapping)
    metrics = {"model": model_path, "num_preds": sum([1 for p in preds if p is not None])}
    if y_true is not None and len(y_true) == len(preds):
        mask = [p is not None for p in preds]
        y_pred = [preds[i] for i, m in enumerate(mask) if m]
        y_t = [y_true[i] for i, m in enumerate(mask) if m]
        metrics["accuracy"] = accuracy_score(y_t, y_pred)
        metrics["macro_f1"] = f1_score(y_t, y_pred, average="macro")
        metrics["precision"] = precision_score(y_t, y_pred, average="macro", zero_division=0)
        metrics["recall"] = recall_score(y_t, y_pred, average="macro", zero_division=0)
    results.append(metrics)

res_df = pd.DataFrame(results).sort_values("macro_f1" if "macro_f1" in results[0] else "num_preds", ascending=False)
print(res_df)
# best model by macro_f1 (if available)
if "macro_f1" in res_df.columns:
    print("Best model by macro F1:", res_df.iloc[0]["model"], "score:", res_df.iloc[0]["macro_f1"])
else:
    print("No labels available to rank models. Inspect predicted class distribution manually.")
