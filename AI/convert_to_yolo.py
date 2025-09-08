import json, os
from pathlib import Path
from PIL import Image
import argparse

# ROI 클래스 정의 (8개)
roi_classes = [
    "facepart::forehead",
    "facepart::glabella",
    "facepart::left_crowsfeet",
    "facepart::right_crowsfeet",
    "facepart::left_cheek",
    "facepart::right_cheek",
    "facepart::lips",
    "facepart::chin"
]
roi_to_id = {cls: i for i, cls in enumerate(roi_classes)}

def convert(json_path, img_root, out_dir):
    coco = json.load(open(json_path, "r"))
    id2img = {im["id"]: im for im in coco["images"]}
    id2cat = {c["id"]: c["name"] for c in coco["categories"]}
    os.makedirs(out_dir, exist_ok=True)

    for ann in coco["annotations"]:
        cat_name = id2cat[ann["category_id"]]
        if cat_name not in roi_to_id:
            continue
        if "bbox" not in ann or not ann["bbox"]:
            continue

        img_meta = id2img[ann["image_id"]]
        W, H = img_meta["width"], img_meta["height"]
        x, y, w, h = ann["bbox"]

        # YOLO normalized format
        xc, yc = (x + w/2)/W, (y + h/2)/H
        ww, hh = w/W, h/H
        cls = roi_to_id[cat_name]

        # 라벨 txt 저장
        txt_path = Path(out_dir) / (Path(img_meta["file_name"]).stem + ".txt")
        with open(txt_path, "a") as f:
            f.write(f"{cls} {xc} {yc} {ww} {hh}\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, required=True)
    ap.add_argument("--img-root", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()

    convert(args.json, args.img_root, args.out_dir)
