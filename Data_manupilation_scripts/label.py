import os
import csv

root = "colormap_truncate"   # change if needed
output_csv = "labels_new.csv"

# Accepted image extensions
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# List class folders sorted for consistent IDs
classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

# Map: class_name → integer label
class_to_id = {cls_name: idx for idx, cls_name in enumerate(classes)}

rows = []

for cls_name in classes:
    cls_dir = os.path.join(root, cls_name)
    class_id = class_to_id[cls_name]

    for f in os.listdir(cls_dir):
        if os.path.splitext(f)[1].lower() in IMG_EXTS:
            rows.append([f, cls_name, class_id])

# Write CSV
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "class_name", "class_id"])
    writer.writerows(rows)

print(f"New label CSV created: {output_csv}")
print(f"Classes found: {class_to_id}")
print(f"Total labeled images: {len(rows)}")
