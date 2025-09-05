'''
python3 infer_and_score_skin_roi.py \
  --weights ./runs/skin_roi_resnet50/best.pth \
  --roi-label-space ./runs/skin_roi_resnet50/roi_label_space.json \
  --train-json /home/bbang/Workspace/Intel07_Intelproject_Team3/dataset/merged_training.json \
  --save-stats-to ./runs/skin_roi_resnet50/scoring_stats.json \
  --image /path/to/face.jpg \
  --out-json ./out_pred_scored.json \
  --roi-mode auto
'''





import os, json, argparse, sys
from pathlib import Path
from collections import defaultdict, OrderedDict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms





class RoiMultiHead(nn.Module):
    def __init__(self, roi_label_space, backbone="resnet50"):
        super().__init__()
        self.roi_reg_keys = roi_label_space["regression_keys"]  
        self.roi_cls_keys = roi_label_space["class_keys"]        
        self.roi_cls_maps = roi_label_space["class_key_to_index"]

        if backbone == "resnet50":
            m = torchvision.models.resnet50(weights=None)  
            in_dim = m.fc.in_features
            m.fc = nn.Identity()
            self.backbone = m
        else:
            raise ValueError("Only resnet50 supported.")

        self.reg_heads = nn.ModuleDict()
        self.cls_heads = nn.ModuleDict()
        for roi, R_keys in self.roi_reg_keys.items():
            if len(R_keys) > 0:
                self.reg_heads[roi] = nn.Linear(in_dim, len(R_keys))
        for roi, C_keys in self.roi_cls_keys.items():
            if len(C_keys) > 0:
                heads = nn.ModuleList()
                for ck in C_keys:
                    nclass = len(self.roi_cls_maps.get(roi, {}).get(ck, {}).get("values", []))
                    heads.append(nn.Linear(in_dim, nclass))
                self.cls_heads[roi] = heads

    def forward(self, x, roi_names):
        feats = self.backbone(x)  
        B = feats.size(0)
        out = []
        for i in range(B):
            roi = roi_names[i]
            f = feats[i:i+1]
            reg_pred = self.reg_heads[roi](f) if roi in self.reg_heads else None
            cls_preds = None
            if roi in self.cls_heads:
                cls_preds = [head(f) for head in self.cls_heads[roi]]
            out.append((reg_pred, cls_preds))
        return out





def clamp_box(x, y, w, h, W, H):
    x = max(0, min(int(x), W-1)); y = max(0, min(int(y), H-1))
    w = max(1, min(int(w), W-x)); h = max(1, min(int(h), H-y))
    return [x,y,w,h]

def load_manual_rois(path, W, H):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    rois = {}
    for k, v in d.items():
        if isinstance(v, (list, tuple)) and len(v) == 4:
            rois[k] = clamp_box(v[0], v[1], v[2], v[3], W, H)
    return rois

def auto_rois_by_mediapipe(pil):
    """mediapipe로 얼굴 박스→ 비율 분할 ROI 근사. 실패/미설치 시 None."""
    try:
        import mediapipe as mp
    except Exception:
        return None

    img = np.array(pil)
    mp_face = mp.solutions.face_detection
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
        res = fd.process(img[:,:,::-1])  
        if not res.detections:
            return None
        det = res.detections[0]
        bb = det.location_data.relative_bounding_box
        H, W = img.shape[0], img.shape[1]
        x = int(bb.xmin * W); y = int(bb.ymin * H)
        w = int(bb.width * W); h = int(bb.height * H)
        x,y,w,h = clamp_box(x,y,w,h,W,H)

    rois = {}
    top    = y
    mid_y  = y + int(h*0.45)
    lip_y  = y + int(h*0.65)
    bot    = y + h

    left   = x
    mid_x  = x + int(w*0.5)
    right  = x + w

    rois["forehead"] = clamp_box(x, top, w, int(h*0.25), W, H)
    rois["glabella"] = clamp_box(x+int(w*0.35), top+int(h*0.22), int(w*0.3), int(h*0.12), W, H)
    rois["left_crowsfeet"]  = clamp_box(x, y+int(h*0.30), int(w*0.25), int(h*0.25), W, H)
    rois["right_crowsfeet"] = clamp_box(x+int(w*0.75), y+int(h*0.30), int(w*0.25), int(h*0.25), W, H)
    rois["left_cheek"]  = clamp_box(x+int(w*0.08),  mid_y, int(w*0.30), int(h*0.28), W, H)
    rois["right_cheek"] = clamp_box(x+int(w*0.62),  mid_y, int(w*0.30), int(h*0.28), W, H)
    rois["lips"]        = clamp_box(x+int(w*0.30), lip_y, int(w*0.40), int(h*0.15), W, H)
    rois["chin"]        = clamp_box(x+int(w*0.25), y+int(h*0.82), int(w*0.50), int(h*0.18), W, H)
    rois["full"]        = [x,y,w,h]
    return rois

def full_roi(pil):
    W, H = pil.size
    return {"full": [0,0,W,H]}

def preprocess_crop(pil_crop, img_size):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return tf(pil_crop)





HIGHER_BETTER_KEYWORDS = ["moisture","elasticity","water","hydration"]
LOWER_BETTER_KEYWORDS  = ["pore","wrinkle","pigment","pigmentation","spot","acne","blemish","sebum","redness"]

def guess_direction(key: str):
    k = key.lower()
    if any(s in k for s in HIGHER_BETTER_KEYWORDS): return "higher_is_better"
    if any(s in k for s in LOWER_BETTER_KEYWORDS):  return "lower_is_better"
    return "higher_is_better"

def build_scoring_stats_from_train(train_json_path):
    J = json.loads(Path(train_json_path).read_text(encoding="utf-8"))
    id2name = {c["id"]: c["name"] for c in J["categories"]}

    values = defaultdict(list)     
    class_vals = defaultdict(set)  

    for a in J["annotations"]:
        if "value" not in a: 
            continue
        name = id2name[a["category_id"]]
        v = a["value"]
        try:
            fv = float(v)
        except:
            continue
        if name.startswith("eq_"):
            values[name].append(fv)
        elif name.startswith("ann_"):
            try: class_vals[name].add(int(round(fv)))
            except: pass

    import numpy as np
    stats = {"regression": {}, "classification": {}}
    for k, arr in values.items():
        if len(arr) < 10:
            lo, hi = (float(np.min(arr)), float(np.max(arr)))
        else:
            lo, hi = (float(np.percentile(arr, 5)), float(np.percentile(arr, 95)))
            if lo == hi:
                lo, hi = (float(np.min(arr)), float(np.max(arr)))
        stats["regression"][k] = {"p5": lo, "p95": hi, "direction": guess_direction(k)}

    for k, s in class_vals.items():
        if not s: 
            continue
        mn, mx = int(min(s)), int(max(s))
        
        stats["classification"][k] = {"min": mn, "max": mx, "direction": "lower_is_better"}

    return stats

def clamp01(x): 
    return max(0.0, min(1.0, x))

def score_regression(val, st):
    lo, hi = st["p5"], st["p95"]
    if hi <= lo:
        return 50.0
    t = (val - lo) / (hi - lo)
    t = clamp01(t)
    return t*100.0 if st.get("direction","higher_is_better")=="higher_is_better" else (1.0-t)*100.0

def score_classification(grade, st):
    mn, mx = st["min"], st["max"]
    if mx <= mn:
        return 50.0
    t = (grade - mn) / float(mx - mn)
    t = clamp01(t)
    return (1.0-t)*100.0 if st.get("direction","lower_is_better")=="lower_is_better" else t*100.0

def score_report(pred_report, stats):
    out = {}
    roi_scores = []
    for roi, dic in pred_report.items():
        roi_out = {}
        
        if "regression" in dic and isinstance(dic["regression"], dict):
            reg_scored = {}
            for k, v in dic["regression"].items():
                st = stats["regression"]?.get(k) if False else stats["regression"].get(k)  
                if st is None: 
                    continue
                reg_scored[k] = round(score_regression(float(v), st), 2)
            if reg_scored:
                roi_out["regression_score"] = reg_scored

        
        if "classification" in dic and isinstance(dic["classification"], dict):
            cls_scored = {}
            for k, g in dic["classification"].items():
                st = stats["classification"].get(k)
                if st is None: 
                    continue
                cls_scored[k] = round(score_classification(int(g), st), 2)
            if cls_scored:
                roi_out["classification_score"] = cls_scored

        
        flat = []
        for blk in ["regression_score","classification_score"]:
            if blk in roi_out:
                flat.extend(list(roi_out[blk].values()))
        if flat:
            roi_out["roi_score"] = round(sum(flat)/len(flat), 2)
            roi_scores.append(roi_out["roi_score"])

        if roi_out:
            out[roi] = roi_out

    if roi_scores:
        out["_overall_score"] = round(sum(roi_scores)/len(roi_scores), 2)
    return out





def run_infer_and_score(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    
    roi_label_space = json.loads(Path(args.roi_label_space).read_text(encoding="utf-8"))
    model = RoiMultiHead(roi_label_space, backbone="resnet50")
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    
    with Image.open(args.image).convert("RGB") as pil:
        W, H = pil.size
        if args.roi_mode == "manual":
            assert args.roi_json, "--roi-json 필요(manual 모드)"
            rois = load_manual_rois(args.roi_json, W, H)
        elif args.roi_mode == "auto":
            rois = auto_rois_by_mediapipe(pil)
            if rois is None:
                print("[warn] mediapipe 실패/미설치 → full 모드로 폴백")
                rois = full_roi(pil)
        else:
            rois = full_roi(pil)

        
        valid = []
        for name, box in rois.items():
            has_reg = len(roi_label_space["regression_keys"].get(name, [])) > 0
            has_cls = len(roi_label_space["class_keys"].get(name, [])) > 0
            if has_reg or has_cls:
                valid.append((name, box))

        if len(valid) == 0:
            print("[error] 사용할 수 있는 ROI 없음. --roi-mode/--roi-json 확인.")
            sys.exit(1)

        xs, roi_names = [], []
        for name, (x,y,w,h) in valid:
            crop = pil.crop((x, y, x+w, y+h))
            xs.append(preprocess_crop(crop, args.img_size))
            roi_names.append(name)
        X = torch.stack(xs, dim=0).to(device)

        
        with torch.no_grad():
            outputs = model(X, roi_names)

    
    pred_report = {}
    for i, (roi_name, _) in enumerate(valid):
        reg_pred, cls_preds = outputs[i]
        roi_rep = {}

        
        R_keys = roi_label_space["regression_keys"].get(roi_name, [])
        if reg_pred is not None and len(R_keys) > 0:
            vals = reg_pred.squeeze(0).cpu().tolist()
            roi_rep["regression"] = {k: float(v) for k, v in zip(R_keys, vals)}

        
        C_keys = roi_label_space["class_keys"].get(roi_name, [])
        cls_maps = roi_label_space["class_key_to_index"].get(roi_name, {})
        if cls_preds is not None and len(C_keys) > 0:
            pred_classes = {}
            for k, lg in enumerate(cls_preds):
                key = C_keys[k]
                idx = int(lg.squeeze(0).argmax().item())
                idx_to_val = cls_maps.get(key, {}).get("idx_to_val", {})
                if isinstance(idx_to_val, dict):
                    pred_val = int(idx_to_val.get(str(idx), idx_to_val.get(idx, idx)))
                else:
                    pred_val = int(idx_to_val[idx])
                pred_classes[key] = pred_val
            roi_rep["classification"] = pred_classes

        pred_report[roi_name] = roi_rep

    
    if args.stats_json and Path(args.stats_json).exists():
        stats = json.loads(Path(args.stats_json).read_text(encoding="utf-8"))
    else:
        if not args.train_json:
            print("[warn] --stats-json 없고 --train-json 도 없어서 점수화 통계를 만들 수 없습니다. 원시 예측만 출력합니다.")
            stats = None
        else:
            print("[info] scoring stats 없음 → train JSON에서 자동 생성")
            stats = build_scoring_stats_from_train(args.train_json)
            if args.save_stats_to:
                Path(args.save_stats_to).parent.mkdir(parents=True, exist_ok=True)
                Path(args.save_stats_to).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[ok] saved stats: {args.save_stats_to}")

    
    out_obj = {"prediction": pred_report}
    if stats is not None:
        out_obj["scored"] = score_report(pred_report, stats)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] saved: {args.out_json}")
    else:
        print(json.dumps(out_obj, ensure_ascii=False, indent=2))





def parse_args():
    ap = argparse.ArgumentParser(description="Inference + 0~100 Scoring (ROI-based)")
    
    ap.add_argument("--weights", required=True, help="학습된 체크포인트(best.pth)")
    ap.add_argument("--roi-label-space", required=True, help="roi_label_space.json 경로")
    ap.add_argument("--image", required=True, help="입력 얼굴 이미지 경로")

    
    ap.add_argument("--out-json", default="", help="최종 결과 저장(JSON). 생략 시 stdout")

    
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--roi-mode", choices=["auto","manual","full"], default="auto",
                    help="ROI 생성 방식: auto(mediapipe), manual(roi-json), full(전체)")
    ap.add_argument("--roi-json", default="", help="--roi-mode manual일 때 ROI 정의 JSON")
    ap.add_argument("--device", default="", help="'cuda' 또는 'cpu' (기본 자동)")

    
    ap.add_argument("--stats-json", default="", help="기존 scoring_stats.json 경로(있으면 사용)")
    ap.add_argument("--train-json", default="", help="없으면 여기서 통계를 자동 산출(merged_training.json)")
    ap.add_argument("--save-stats-to", default="", help="자동 산출 시, 통계를 이 경로로 저장")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_infer_and_score(args)
