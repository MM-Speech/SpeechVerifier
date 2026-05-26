import json
import os
from datasets import Dataset, DatasetInfo, Features, Value, Image
from PIL import Image as PILImage
from huggingface_hub import login
from tqdm import tqdm

with open("/mnt/private_hk/data/Video-R1-data/Video-R1-260k.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

filtered_data = [entry for entry in raw_data if entry.get("data_type") == "image"]

for entry in tqdm(filtered_data):
    image_path = os.path.join('/mnt/private_hk/data/Video-R1-data',entry["path"])
    if os.path.exists(image_path):
        entry["image"] = image_path
    else:
        entry["image"] = None

filtered_data = [e for e in filtered_data if e["image"] is not None]

dataset = Dataset.from_list(filtered_data).cast_column("image", Image())


# login(token="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")

# dataset.push_to_hub("conctsai/video-r1-image")
print("Dataset is pushed to the hub.")
dataset.save_to_disk("/mnt/private_hk/data/Image-R1")
