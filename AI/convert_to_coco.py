# convert_to_yolo.py
import json, os, argparse
from pathlib import Path

ROI_CLASSES = [
    "facepart::forehead",
    "facepart::glabella",
    "facepart::left_crowsfeet",
    "facepart::right_crowsfeet",
    "facepart::left_cheek",
    "facepart::right_cheek",
    "facepart::lips",
    "facepart::chin",
]
ROI_TO_ID = {name: i for i, name in enumerate(ROI_CLASSES)}

def convert(json_path: str, img_root: str, out_dir: str):
    coco = json.load(open(json_path, "r"))
    id2img = {im["id"]: im for im in coco["images"]}
    id2cat = {c["id"]: c["name"] for c in coco["categories"]}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cnt_total = 0
    cnt_by_cls = {k:0 for k in ROI_CLASSES}

    for ann in coco["annotations"]:
        cat_name = id2cat.get(ann.get("category_id"))
        if cat_name not in ROI_TO_ID:
            continue
        bbox = ann.get("bbox")
        if not bbox:
            continue

        im = id2img[ann["image_id"]]
        W, H = im["width"], im["height"]
        x, y, w, h = bbox
        xc, yc = (x + w/2) / W, (y + h/2) / H
        ww, hh = w / W, h / H
        cls_id = ROI_TO_ID[cat_name]

        # 이미지의 상대경로(file_name)를 그대로 따라 라벨 경로 생성
        rel_img = Path(im["file_name"])            # e.g. 1_digital_camera/0002/xxx.jpg
        out_txt = (out_dir / rel_img).with_suffix(".txt")
        out_txt.parent.mkdir(parents=True, exist_ok=True)

        with open(out_txt, "a") as f:
            f.write(f"{cls_id} {xc} {yc} {ww} {hh}\n")

        cnt_total += 1
        cnt_by_cls[cat_name] += 1

    print(f"✅ YOLO 라벨 생성: {cnt_total}개")
    for k,v in cnt_by_cls.items():
        print(f" - {k}: {v}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--img-root", required=True)  # 참조만, 경로 검증용으로 남겨둠
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    convert(args.json, args.img_root, args.out_dir)
