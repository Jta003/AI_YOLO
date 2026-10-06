import json
import math
import random
import shutil
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse


SCRIPT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = Path(r"D:\Codeจารรุจิ\AI_YOLO\Frame\images")
OUTPUT_DIR = Path(r"D:\Codeจารรุจิ\AI_YOLO\dataset")
TRAIN_SPLIT = 0.8
SEED = 42

# True  = YOLO-OBB (เก็บกล่องหมุน 4 มุม) -> เทรนด้วย: yolo obb train ... model=yolov8n-obb.pt
# False = YOLO detect (กล่องตรงแกนที่ครอบกล่องหมุนพอดี) -> เทรนด้วย: yolo detect train ... model=yolov8n.pt
USE_OBB = False

# True = ลบโฟลเดอร์ dataset เดิมก่อนสร้างใหม่ (กันข้อมูลเก่าปนกับข้อมูลใหม่)
CLEAN_OUTPUT = True


def find_json_file(folder: Path) -> Path:
    """Find the first .json file in the folder where this script is located."""
    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        sys.exit(
            f"ERROR: No .json file found in folder: {folder}\n"
            f"Place the exported JSON file from Label Studio in the same folder as this script first."
        )
    if len(json_files) > 1:
        print(f"WARNING: Found more than 1 .json file in this folder, using the first one: {json_files[0].name}")
        for f in json_files:
            print(f"    - {f.name}")
    return json_files[0]


def get_image_filename(task):
    """
    Extract the actual image filename (basename) from task['data'] in Label Studio.
    Supports multiple formats: a direct 'image' key, local-files URL (?d=...), or a generic URL.
    """
    data = task.get("data", {})

    image_value = None
    for key in ("image", "img", "picture", "photo"):
        if key in data:
            image_value = data[key]
            break

    if image_value is None:
        for key, value in data.items():
            if isinstance(value, str) and (
                "image" in key.lower() or "/data/local-files" in value
            ):
                image_value = value
                break

    if image_value is None:
        return None

    parsed = urlparse(image_value)
    if "d=" in parsed.query:
        rel_path = unquote(parsed.query.split("d=", 1)[1])
        # normalize both Windows (\) and POSIX (/) separators before taking the basename
        rel_path = rel_path.replace("\\", "/")
        return Path(rel_path).name

    return unquote(Path(parsed.path).name)


def collect_classes(tasks):
    """Scan all tasks for labels actually used, to build the class list."""
    classes = set()
    for task in tasks:
        for annotation in task.get("annotations", []):
            if annotation.get("was_cancelled"):
                continue
            for result in annotation.get("result", []):
                if result.get("type") != "rectanglelabels":
                    continue
                labels = result.get("value", {}).get("rectanglelabels", [])
                classes.update(labels)
    return sorted(classes)


def get_image_size(result, image_path):
    """Get (width, height) in pixels. Prefer values stored in the annotation, fall back to reading the image."""
    w = result.get("original_width")
    h = result.get("original_height")
    if w and h:
        return w, h
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            return im.size
    except Exception:
        sys.exit(
            "ERROR: Cannot determine image size (no original_width/original_height in JSON "
            "and Pillow could not read the image). Run: pip install pillow"
        )


def get_corners(value, img_w, img_h):
    """
    Return the 4 corners (in pixels) of a Label Studio rectangle after rotation.
    Label Studio rotates the box around its top-left corner (x, y), so we must work in pixels
    (not percent) so that the rotation is not distorted by the image aspect ratio.
    Corner order: top-left, top-right, bottom-right, bottom-left.
    """
    x = value["x"] / 100.0 * img_w
    y = value["y"] / 100.0 * img_h
    w = value["width"] / 100.0 * img_w
    h = value["height"] / 100.0 * img_h
    angle = math.radians(value.get("rotation", 0) or 0)
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    corners = []
    for dx, dy in ((0, 0), (w, 0), (w, h), (0, h)):
        corners.append((x + dx * cos_a - dy * sin_a,
                        y + dx * sin_a + dy * cos_a))
    return corners


def clamp01(v):
    return max(0.0, min(1.0, v))


def convert_task_to_yolo_lines(task, class_to_id, image_path):
    """
    Convert the annotations of one task into YOLO lines.
      USE_OBB = True : <class_id> x1 y1 x2 y2 x3 y3 x4 y4   (normalized 0-1)
      USE_OBB = False: <class_id> <cx> <cy> <w> <h>          (normalized 0-1)
    """
    lines = []
    for annotation in task.get("annotations", []):
        if annotation.get("was_cancelled"):
            continue
        for result in annotation.get("result", []):
            if result.get("type") != "rectanglelabels":
                continue
            value = result.get("value", {})
            labels = value.get("rectanglelabels", [])
            if not labels:
                continue

            img_w, img_h = get_image_size(result, image_path)
            corners = get_corners(value, img_w, img_h)

            for label in labels:
                class_id = class_to_id[label]

                if USE_OBB:
                    coords = " ".join(
                        f"{clamp01(px / img_w):.6f} {clamp01(py / img_h):.6f}"
                        for px, py in corners
                    )
                    lines.append(f"{class_id} {coords}")
                else:
                    xs = [c[0] for c in corners]
                    ys = [c[1] for c in corners]
                    x_min, x_max = clamp01(min(xs) / img_w), clamp01(max(xs) / img_w)
                    y_min, y_max = clamp01(min(ys) / img_h), clamp01(max(ys) / img_h)
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    lines.append(
                        f"{class_id} {x_center:.6f} {y_center:.6f} "
                        f"{x_max - x_min:.6f} {y_max - y_min:.6f}"
                    )
    return lines


def main():
    random.seed(SEED)

    json_path = find_json_file(SCRIPT_DIR)
    print(f"Using JSON file: {json_path.name}")
    print(f"Mode: {'YOLO-OBB (rotated boxes)' if USE_OBB else 'YOLO detect (axis-aligned boxes)'}")

    if not IMAGES_DIR.exists():
        sys.exit(f"ERROR: Source images folder not found at path: {IMAGES_DIR}\nUpdate IMAGES_DIR at the top of this script.")

    with open(json_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    if not isinstance(tasks, list):
        sys.exit(
            "ERROR: The JSON file is not a list of tasks as expected — "
            "check that the export format selected was 'JSON' (not JSON-MIN or another format)."
        )

    classes = collect_classes(tasks)
    if not classes:
        sys.exit(
            "ERROR: No bounding box labels (rectanglelabels) found in this file — "
            "check that the labeling setup is Object Detection with Bounding Boxes "
            "and that at least one task has been labeled."
        )
    class_to_id = {name: idx for idx, name in enumerate(classes)}
    print(f"Found {len(classes)} classes: {classes}")

    if CLEAN_OUTPUT and OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        print(f"Cleaned old output folder: {OUTPUT_DIR}")

    for split in ("train", "val"):
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    valid_tasks = []
    skipped_no_file = 0
    skipped_no_annotation = 0

    for task in tasks:
        filename = get_image_filename(task)
        if filename is None:
            skipped_no_file += 1
            continue
        src_image_path = IMAGES_DIR / filename
        if not src_image_path.exists():
            skipped_no_file += 1
            continue
        if not task.get("annotations"):
            skipped_no_annotation += 1
            continue
        valid_tasks.append((task, filename, src_image_path))

    if skipped_no_file:
        print(f"WARNING: Skipped {skipped_no_file} tasks: source image file not found in IMAGES_DIR")
    if skipped_no_annotation:
        print(f"WARNING: Skipped {skipped_no_annotation} tasks: no annotation present")

    if not valid_tasks:
        sys.exit(
            "ERROR: No usable tasks found — check that IMAGES_DIR points to the correct folder "
            "and that filenames match what is referenced in Label Studio."
        )

    random.shuffle(valid_tasks)
    split_idx = int(len(valid_tasks) * TRAIN_SPLIT)
    train_tasks = valid_tasks[:split_idx]
    val_tasks = valid_tasks[split_idx:]

    def process_split(split_tasks, split_name):
        count = 0
        for task, filename, src_image_path in split_tasks:
            lines = convert_task_to_yolo_lines(task, class_to_id, src_image_path)
            if not lines:
                continue

            dst_image_path = OUTPUT_DIR / "images" / split_name / filename
            shutil.copy2(src_image_path, dst_image_path)

            label_path = (OUTPUT_DIR / "labels" / split_name / filename).with_suffix(".txt")
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            count += 1
        return count

    n_train = process_split(train_tasks, "train")
    n_val = process_split(val_tasks, "val")

    if n_train == 0:
        sys.exit("ERROR: No images were successfully converted — check paths and annotations again.")

    print(f"Train: {n_train} images")
    print(f"Val:   {n_val} images")

    (OUTPUT_DIR / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    data_yaml_content = (
        f"path: {OUTPUT_DIR.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"nc: {len(classes)}\n"
        f"names: {classes}\n"
    )
    (OUTPUT_DIR / "data.yaml").write_text(data_yaml_content, encoding="utf-8")

    print(f"\ndata.yaml and classes.txt written to: {OUTPUT_DIR}")
    print("Ready to train with Ultralytics, e.g.:")
    if USE_OBB:
        print(f'  yolo obb train data="{OUTPUT_DIR / "data.yaml"}" model=yolov8n-obb.pt epochs=100')
    else:
        print(f'  yolo detect train data="{OUTPUT_DIR / "data.yaml"}" model=yolov8n.pt epochs=100')


if __name__ == "__main__":
    main()