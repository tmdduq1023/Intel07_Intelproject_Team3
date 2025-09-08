


import os, json, argparse, random, time, copy
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import torch.backends.cudnn as cudnn





def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def clamp_bbox(x, y, w, h, W, H):
    x = max(0, min(int(x), W-1)); y = max(0, min(int(y), H-1))
    w = max(1, min(int(w), W - x)); h = max(1, min(int(h), H - y))
    return x, y, w, h





FACEPART_TO_LABEL_SUBSTR = {
    "forehead":      ["forehead"],
    "glabella":      ["glabellus","glabella"],
    "left_crowsfeet":["l_perocular","left_crowsfeet"],
    "right_crowsfeet":["r_perocular","right_crowsfeet"],
    "left_cheek":    ["l_cheek","left_cheek"],
    "right_cheek":   ["r_cheek","right_cheek"],
    "lips":          ["lip","lips","perioral"],
    "chin":          ["chin"],
    "full":          ["full","all","face"],
}





def build_roi_label_space(train_json_path):
    coco = load_json(train_json_path)
    id2name = {c["id"]: c["name"] for c in coco["categories"]}

    roi_regression = defaultdict(set)
    roi_class_vals = defaultdict(lambda: defaultdict(set))

    for a in coco["annotations"]:
        if "value" not in a:
            continue
        name = id2name[a["category_id"]]
        val = a["value"]

        roi_hits = []
        for roi, subs in FACEPART_TO_LABEL_SUBSTR.items():
            if any(s in name for s in subs):
                roi_hits.append(roi)
        if not roi_hits:
            continue

        if name.startswith("eq_"):
            for r in roi_hits:
                roi_regression[r].add(name)
        elif name.startswith("ann_"):
            try:
                iv = int(round(float(val)))
                for r in roi_hits:
                    roi_class_vals[r][name].add(iv)
            except:
                pass

    roi_label_space = {"regression_keys": {}, "class_keys": {}, "class_key_to_index": {}}
    for roi in FACEPART_TO_LABEL_SUBSTR.keys():
        R = sorted(list(roi_regression[roi])) if roi in roi_regression else []
        C = sorted(list(roi_class_vals[roi].keys())) if roi in roi_class_vals else []
        roi_label_space["regression_keys"][roi] = R
        roi_label_space["class_keys"][roi] = C
        for ck in C:
            vals = sorted(list(roi_class_vals[roi][ck]))
            v2i = {v:i for i,v in enumerate(vals)}
            i2v = {i:v for v,i in v2i.items()}
            roi_label_space["class_key_to_index"].setdefault(roi, {})[ck] = {
                "values": vals, "val_to_idx": v2i, "idx_to_val": i2v
            }
    return roi_label_space


def compute_reg_std_from_train(train_json_path):
    """eq_* 키별 mean/std 산출 (z-score 학습용)"""
    coco = load_json(train_json_path)
    id2name = {c["id"]: c["name"] for c in coco["categories"]}
    vals = defaultdict(list)
    for a in coco["annotations"]:
        if "value" not in a: 
            continue
        name = id2name[a["category_id"]]
        if not name.startswith("eq_"): 
            continue
        try:
            v = float(a["value"])
            vals[name].append(v)
        except:
            pass
    stats = {}
    for k, arr in vals.items():
        if len(arr) == 0: 
            continue
        m = float(np.mean(arr))
        s = float(np.std(arr) + 1e-6)
        stats[k] = {"mean": m, "std": s}
    return stats


def compute_class_weights_from_train(train_json_path, roi_label_space):
    """ann_* 키별 등급 분포로 weight(1/log(1+freq)) 산출"""
    coco = load_json(train_json_path)
    id2name = {c["id"]: c["name"] for c in coco["categories"]}

    
    counts = {}
    for roi, C_keys in roi_label_space["class_keys"].items():
        for ck in C_keys:
            nclass = len(roi_label_space["class_key_to_index"][roi][ck]["values"])
            counts[ck] = np.zeros(nclass, dtype=np.int64)

    
    for a in coco["annotations"]:
        if "value" not in a:
            continue
        name = id2name[a["category_id"]]
        if not name.startswith("ann_"): 
            continue
        try:
            val = int(round(float(a["value"])))
        except:
            continue
        
        
        
        map_found = None
        for roi in roi_label_space["class_key_to_index"]:
            if name in roi_label_space["class_key_to_index"][roi]:
                map_found = roi_label_space["class_key_to_index"][roi][name]
                break
        if map_found is None: 
            continue
        if val not in map_found["val_to_idx"]:
            continue
        idx = map_found["val_to_idx"][val]
        counts[name][idx] += 1

    weights = {}
    for ck, arr in counts.items():
        freq = arr.astype(np.float32) + 1e-6
        w = 1.0 / np.log1p(freq)
        w = w / (w.mean() + 1e-8)
        weights[ck] = w.tolist()
    return weights





class RoiDataset(Dataset):
    def __init__(self, json_path, data_root, roi_label_space, img_size=384, is_train=True,
                 pad_ratio=0.10, reg_std=None):
        super().__init__()
        self.coco = load_json(json_path)
        self.data_root = Path(data_root)
        self.id2img = {im["id"]: im for im in self.coco["images"]}
        self.id2name = {c["id"]: c["name"] for c in self.coco["categories"]}
        self.anns_by_img = defaultdict(list)
        for a in self.coco["annotations"]:
            self.anns_by_img[a["image_id"]].append(a)

        self.roi_reg_keys = roi_label_space["regression_keys"]
        self.roi_cls_keys = roi_label_space["class_keys"]
        self.roi_cls_maps = roi_label_space["class_key_to_index"]

        self.is_train = is_train
        self.pad_ratio_base = pad_ratio
        self.reg_std = reg_std or {}

        
        self.samples = []
        for img_id, im in self.id2img.items():
            W, H = im.get("width"), im.get("height")
            if W is None or H is None: 
                continue
            for a in self.anns_by_img.get(img_id, []):
                if "value" in a:
                    continue
                cname = self.id2name[a["category_id"]]
                if not cname.startswith("facepart::"): 
                    continue
                roi_name = cname.split("facepart::",1)[1]
                bbox = a.get("bbox", [])
                if not bbox or len(bbox)!=4: 
                    continue
                x,y,w,h = bbox
                
                pad = np.random.uniform(0.08, 0.15) if self.is_train else self.pad_ratio_base
                px, py = int(w*pad), int(h*pad)
                x,y,w,h = clamp_bbox(x-px, y-py, w+2*px, h+2*py, W, H)
                
                if len(self.roi_reg_keys.get(roi_name, []))==0 and len(self.roi_cls_keys.get(roi_name, []))==0:
                    continue
                self.samples.append((img_id, roi_name, [x,y,w,h]))

        
        if is_train:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.08,0.08,0.05,0.02)], p=0.3),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, roi, bbox = self.samples[idx]
        im = self.id2img[img_id]
        img_path = self.data_root / im["file_name"]
        with Image.open(img_path).convert("RGB") as pil:
            x0,y0,w,h = bbox
            crop = pil.crop((x0, y0, x0+w, y0+h))
            x = self.tf(crop)

        R_keys = self.roi_reg_keys.get(roi, [])
        C_keys = self.roi_cls_keys.get(roi, [])
        reg_t = torch.full((len(R_keys),), float('nan'), dtype=torch.float32)
        cls_t = torch.full((len(C_keys),), -1, dtype=torch.long)

        for a in self.anns_by_img.get(img_id, []):
            if "value" not in a:
                continue
            name = self.id2name[a["category_id"]]
            v = a["value"]
            if name in R_keys:
                j = R_keys.index(name)
                try:
                    fv = float(v)
                    if name in self.reg_std:
                        m, s = self.reg_std[name]["mean"], self.reg_std[name]["std"]
                        reg_t[j] = (fv - m) / max(1e-6, s)
                    else:
                        reg_t[j] = fv
                except:
                    pass
            if name in C_keys:
                try:
                    iv = int(round(float(v)))
                    maps = self.roi_cls_maps.get(roi, {}).get(name, None)
                    if maps and iv in maps["val_to_idx"]:
                        cls_t[C_keys.index(name)] = maps["val_to_idx"][iv]
                except:
                    pass

        meta = {"img_id": img_id, "roi": roi}
        return x, reg_t, cls_t, meta


def roi_collate(batch):
    xs, regs, clss, metas = zip(*batch)
    xs = torch.stack(xs, dim=0)

    reg_lens = [r.numel() for r in regs]
    cls_lens = [c.numel() for c in clss]
    max_R = max(reg_lens) if reg_lens else 0
    max_C = max(cls_lens) if cls_lens else 0

    if max_R > 0:
        reg_pad = torch.full((len(regs), max_R), float('nan'), dtype=torch.float32)
        for i, r in enumerate(regs):
            if r.numel() > 0:
                reg_pad[i, :r.numel()] = r
    else:
        reg_pad = torch.empty((len(regs), 0), dtype=torch.float32)

    if max_C > 0:
        cls_pad = torch.full((len(clss), max_C), -1, dtype=torch.long)
        for i, c in enumerate(clss):
            if c.numel() > 0:
                cls_pad[i, :c.numel()] = c
    else:
        cls_pad = torch.empty((len(clss), 0), dtype=torch.long)

    return xs, reg_pad, cls_pad, metas





class RoiMultiHead(nn.Module):
    def __init__(self, backbone="resnet50", roi_label_space=None, pretrained=True, head_dropout=0.2):
        super().__init__()
        assert roi_label_space is not None
        self.roi_reg_keys = roi_label_space["regression_keys"]
        self.roi_cls_keys = roi_label_space["class_keys"]
        self.roi_cls_maps = roi_label_space["class_key_to_index"]

        if backbone == "resnet50":
            m = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            )
            in_dim = m.fc.in_features
            m.fc = nn.Identity()
            self.backbone = m
        else:
            raise ValueError("resnet50만 구현")

        self.dropout = nn.Dropout(p=head_dropout)

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
        feats = self.dropout(feats)
        B = feats.size(0)
        out_reg, out_cls = [None]*B, [None]*B

        for i in range(B):
            roi = roi_names[i]
            f = feats[i:i+1]
            reg_pred = None
            if roi in self.reg_heads:
                reg_pred = self.reg_heads[roi](f)
            cls_preds = None
            if roi in self.cls_heads:
                cls_preds = [head(f) for head in self.cls_heads[roi]]
            out_reg[i], out_cls[i] = reg_pred, cls_preds

        return out_reg, out_cls





def masked_huber(pred, target):
    """target의 NaN 마스크로 Huber(smooth_l1) 계산"""
    if pred is None or pred.numel()==0:
        return None, 0
    if pred.dim()==2 and target.dim()==1:
        target = target.unsqueeze(0)
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return None, 0
    loss = F.smooth_l1_loss(pred[mask], target[mask], beta=1.0)
    return loss, int(mask.sum().item())

def cls_loss_and_acc(logits, target, weight=None, ignore_index=-1):
    """logits: (1,C), target: (1,)"""
    ce = F.cross_entropy(logits, target, weight=weight, ignore_index=ignore_index)
    mask = (target != ignore_index)
    if mask.sum() > 0:
        pred = logits.argmax(dim=1)
        acc = (pred[mask] == target[mask]).float().mean()
        return ce, float(acc.item()), 1
    return ce, None, 0





def train_one_epoch(model, loader, optim, device, lambda_reg=1.0, lambda_cls=1.0, scaler=None,
                    reg_std=None, class_weight_map=None, ema_model=None, ema_decay=0.999):
    model.train()
    pbar = tqdm(loader, desc="train", dynamic_ncols=True, leave=False)

    steps = 0
    avg = {"loss":0.0, "mae":0.0, "acc":0.0}
    for x, reg_t, cls_t, metas in pbar:
        x = x.to(device, non_blocking=True)
        reg_t = reg_t.to(device)
        cls_t = cls_t.to(device)
        roi_names = [m["roi"] for m in metas]

        optim.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=(scaler is not None)):
            out_reg, out_cls = model(x, roi_names)
            total_loss = x.new_tensor(0.0)
            
            total_mae, n_reg = 0.0, 0
            total_acc, n_cls = 0.0, 0

            for i in range(x.size(0)):
                roi = roi_names[i]
                
                reg_pred = out_reg[i]
                if reg_pred is not None and reg_pred.size(1) > 0:
                    target_slice = reg_t[i, :reg_pred.size(1)]
                    
                    huber, n_used = masked_huber(reg_pred, target_slice)
                    if huber is not None and n_used > 0:
                        total_loss = total_loss + lambda_reg * huber
                    
                    mask = ~torch.isnan(target_slice)
                    if mask.any():
                        p = reg_pred.squeeze(0)[mask]
                        t = target_slice[mask]
                        if reg_std:
                            
                            R_keys = model.roi_reg_keys.get(roi, [])
                            idxs = mask.nonzero(as_tuple=False).squeeze(1).tolist()
                            for k_idx, j in enumerate(idxs):
                                key = R_keys[j]
                                m = reg_std.get(key, {}).get("mean", 0.0)
                                s = reg_std.get(key, {}).get("std", 1.0)
                                pv = float(p[k_idx].item())*s + m
                                tv = float(t[k_idx].item())*s + m
                                total_mae += abs(pv - tv); n_reg += 1
                        else:
                            total_mae += float((p - t).abs().mean().item()); n_reg += 1

                
                cls_preds = out_cls[i]
                if cls_preds is not None and len(cls_preds) > 0:
                    C_keys = model.roi_cls_keys.get(roi, [])
                    for j, lg in enumerate(cls_preds):
                        key = C_keys[j]
                        tgt = cls_t[i, j].unsqueeze(0)  
                        if int(tgt.item()) < 0:
                            continue
                        w = None
                        if class_weight_map and (key in class_weight_map):
                            w = torch.tensor(class_weight_map[key], dtype=torch.float32, device=device)
                        ce, acc, valid = cls_loss_and_acc(lg, tgt, weight=w)
                        total_loss = total_loss + lambda_cls * ce
                        if valid and acc is not None:
                            total_acc += acc; n_cls += 1

        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            total_loss.backward()
            optim.step()

        
        if ema_model is not None:
            with torch.no_grad():
                for p, p_ema in zip(model.parameters(), ema_model.parameters()):
                    p_ema.data.mul_(ema_decay).add_(p.data, alpha=1.0-ema_decay)

        steps += 1
        avg["loss"] += float(total_loss.item())
        if n_reg > 0: avg["mae"] += (total_mae / max(1,n_reg))
        if n_cls > 0: avg["acc"] += (total_acc / max(1,n_cls))

        pbar.set_postfix({
            "loss": f"{avg['loss']/steps:.4f}",
            "MAE":  f"{(avg['mae']/steps):.4f}" if steps>0 else "0.0000",
            "Acc":  f"{(avg['acc']/steps):.4f}" if steps>0 else "0.0000",
        })

    return avg["loss"]/max(1,steps), avg["mae"]/max(1,steps), avg["acc"]/max(1,steps)


@torch.no_grad()
def evaluate(model, loader, device, reg_std=None):
    model.eval()
    pbar = tqdm(loader, desc="valid", dynamic_ncols=True, leave=False)
    steps = 0
    avg = {"loss":0.0, "mae":0.0, "acc":0.0}

    for x, reg_t, cls_t, metas in pbar:
        x = x.to(device, non_blocking=True)
        reg_t = reg_t.to(device)
        cls_t = cls_t.to(device)
        roi_names = [m["roi"] for m in metas]

        out_reg, out_cls = model(x, roi_names)
        total_loss = x.new_tensor(0.0)
        total_mae, n_reg = 0.0, 0
        total_acc, n_cls = 0.0, 0

        for i in range(x.size(0)):
            roi = roi_names[i]
            
            reg_pred = out_reg[i]
            if reg_pred is not None and reg_pred.size(1) > 0:
                target_slice = reg_t[i, :reg_pred.size(1)]
                huber, n_used = masked_huber(reg_pred, target_slice)
                if huber is not None and n_used > 0:
                    total_loss = total_loss + huber
                mask = ~torch.isnan(target_slice)
                if mask.any():
                    p = reg_pred.squeeze(0)[mask]
                    t = target_slice[mask]
                    if reg_std:
                        R_keys = model.roi_reg_keys.get(roi, [])
                        idxs = mask.nonzero(as_tuple=False).squeeze(1).tolist()
                        for k_idx, j in enumerate(idxs):
                            key = R_keys[j]
                            m = reg_std.get(key, {}).get("mean", 0.0)
                            s = reg_std.get(key, {}).get("std", 1.0)
                            pv = float(p[k_idx].item())*s + m
                            tv = float(t[k_idx].item())*s + m
                            total_mae += abs(pv - tv); n_reg += 1
                    else:
                        total_mae += float((p - t).abs().mean().item()); n_reg += 1

            
            cls_preds = out_cls[i]
            if cls_preds is not None and len(cls_preds) > 0:
                C_keys = model.roi_cls_keys.get(roi, [])
                for j, lg in enumerate(cls_preds):
                    tgt = cls_t[i, j].unsqueeze(0)
                    if int(tgt.item()) < 0:
                        continue
                    ce = F.cross_entropy(lg, tgt)
                    total_loss = total_loss + ce
                    pred = lg.argmax(dim=1)
                    total_acc += float((pred == tgt).float().mean().item()); n_cls += 1

        steps += 1
        avg["loss"] += float(total_loss.item())
        if n_reg > 0: avg["mae"] += (total_mae / max(1,n_reg))
        if n_cls > 0: avg["acc"] += (total_acc / max(1,n_cls))

        pbar.set_postfix({
            "loss": f"{avg['loss']/steps:.4f}",
            "MAE":  f"{(avg['mae']/steps):.4f}" if steps>0 else "0.0000",
            "Acc":  f"{(avg['acc']/steps):.4f}" if steps>0 else "0.0000",
        })

    return avg["loss"]/max(1,steps), avg["mae"]/max(1,steps), avg["acc"]/max(1,steps)





class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.num_bad = 0

    def step(self, score):
        if self.best is None or score < self.best - self.min_delta:
            self.best = score
            self.num_bad = 0
            return True
        else:
            self.num_bad += 1
            return False

    def should_stop(self):
        return self.num_bad >= self.patience





def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="/home/bbang/Workspace/Intel07_Intelproject_Team3/dataset")
    ap.add_argument("--train-json", type=str, default="/home/bbang/Workspace/Intel07_Intelproject_Team3/dataset/merged_training.json")
    ap.add_argument("--val-json",   type=str, default="/home/bbang/Workspace/Intel07_Intelproject_Team3/dataset/merged_validation.json")
    ap.add_argument("--test-json",  type=str, default="")
    ap.add_argument("--out-dir",    type=str, default="./runs/skin_roi_v3")
    ap.add_argument("--img-size",   type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs",     type=int, default=50)
    ap.add_argument("--workers",    type=int, default=8)
    ap.add_argument("--lr-backbone",type=float, default=5e-5)
    ap.add_argument("--lr-head",    type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--lambda-reg", type=float, default=1.0)
    ap.add_argument("--lambda-cls", type=float, default=1.0)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--early-patience", type=int, default=10)
    ap.add_argument("--early-min-delta", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--ema-decay",  type=float, default=0.999)
    ap.add_argument("--head-dropout", type=float, default=0.2)
    
    ap.add_argument("--reg-std-json", type=str, default="", help="미리 계산된 회귀 z-score 통계(JSON)")
    ap.add_argument("--auto-reg-std", action="store_true", help="train json에서 회귀 통계를 자동 산출")
    ap.add_argument("--save-reg-std-to", type=str, default="", help="자동 산출 시 저장 경로(미지정 시 out-dir/reg_std.json)")
    ap.add_argument("--use-class-weights", action="store_true", help="ann_* 분포 기반 클래스 가중치 사용")
    args = ap.parse_args()

    set_seed(args.seed)
    cudnn.benchmark = True
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    roi_label_space = build_roi_label_space(args.train_json)
    save_json(roi_label_space, os.path.join(args.out_dir, "roi_label_space.json"))

    
    reg_std = None
    if args.reg-std-json if False else False:  
        pass
    if args.reg_std_json and Path(args.reg_std_json).exists():
        reg_std = load_json(args.reg_std_json)
        print(f"[info] loaded reg std from {args.reg_std_json}")
    elif args.auto_reg_std:
        reg_std = compute_reg_std_from_train(args.train_json)
        save_path = args.save_reg_std_to or os.path.join(args.out_dir, "reg_std.json")
        save_json(reg_std, save_path)
        print(f"[ok] computed & saved reg std to {save_path}")
    else:
        reg_std = None
        print("[info] reg std not used")

    
    class_weight_map = None
    if args.use_class_weights:
        class_weight_map = compute_class_weights_from_train(args.train_json, roi_label_space)
        save_json(class_weight_map, os.path.join(args.out_dir, "class_weights.json"))
        print("[ok] computed class weights")

    
    train_set = RoiDataset(args.train_json, args.data_root, roi_label_space,
                           img_size=args.img_size, is_train=True, pad_ratio=0.10, reg_std=reg_std)
    val_set   = RoiDataset(args.val_json,   args.data_root, roi_label_space,
                           img_size=args.img_size, is_train=False, pad_ratio=0.10, reg_std=reg_std)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              prefetch_factor=4, persistent_workers=True,
                              collate_fn=roi_collate)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=max(1, args.workers//2), pin_memory=True,
                              prefetch_factor=2, persistent_workers=True,
                              collate_fn=roi_collate)

    
    model = RoiMultiHead(backbone="resnet50", roi_label_space=roi_label_space,
                         pretrained=True, head_dropout=args.head_dropout).to(device)
    backbone_params, head_params = [], []
    for n,p in model.named_parameters():
        (backbone_params if "backbone" in n else head_params).append(p)
    optim = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params,     "lr": args.lr_head},
    ], weight_decay=args.weight_decay)

    
    warmup_e = max(0, min(args.warmup_epochs, args.epochs-1))
    scheds = []
    miles = []
    if warmup_e > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.1, total_iters=warmup_e)
        scheds.append(warmup); miles.append(warmup_e)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, args.epochs - warmup_e))
    scheds.append(cosine)
    sched = torch.optim.lr_scheduler.SequentialLR(optim, schedulers=scheds, milestones=miles or [0])

    scaler = GradScaler(device="cuda") if torch.cuda.is_available() else None

    
    ema_model = copy.deepcopy(model).to(device).eval()
    for p in ema_model.parameters(): p.requires_grad = False

    
    stopper = EarlyStopping(patience=args.early_patience, min_delta=args.early_min_delta)
    best_path  = os.path.join(args.out_dir, "best_ema.pth")
    last_path  = os.path.join(args.out_dir, "last.pth")
    best_epoch = 0

    for ep in range(1, args.epochs+1):
        tr_loss, tr_mae, tr_acc = train_one_epoch(
            model, train_loader, optim, device,
            lambda_reg=args.lambda_reg, lambda_cls=args.lambda_cls,
            scaler=scaler, reg_std=reg_std, class_weight_map=class_weight_map,
            ema_model=ema_model, ema_decay=args.ema_decay
        )
        
        vl_loss, vl_mae, vl_acc = evaluate(ema_model, val_loader, device, reg_std=reg_std)
        sched.step()

        score = vl_mae + (1.0 - (vl_acc if not np.isnan(vl_acc) else 0.0))
        print(f"[{ep:03d}/{args.epochs}] train: loss={tr_loss:.4f} MAE={tr_mae:.4f} Acc={tr_acc:.4f} | "
              f"valid(EMA): loss={vl_loss:.4f} MAE={vl_mae:.4f} Acc={vl_acc:.4f} | score={score:.4f}")

        improved = stopper.step(score)
        
        torch.save({"model": model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "roi_label_space": roi_label_space,
                    "reg_std": reg_std,
                    "epoch": ep,
                    "val_mae": vl_mae,
                    "val_acc": vl_acc,
                    "score": score}, last_path)

        if improved:
            torch.save({"model": model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "roi_label_space": roi_label_space,
                        "reg_std": reg_std,
                        "epoch": ep,
                        "val_mae": vl_mae,
                        "val_acc": vl_acc,
                        "score": score}, best_path)
            best_epoch = ep
            print(f"  → Saved best EMA to {best_path}")
        elif stopper.should_stop():
            print(f"Early stopping triggered at epoch {ep} (best @ {best_epoch})")
            break

    print(f"Done. Best epoch={best_epoch}, checkpoint={best_path}")

    
    if args.test_json and Path(args.test_json).exists():
        test_set = RoiDataset(args.test_json, args.data_root, roi_label_space,
                              img_size=args.img_size, is_train=False, pad_ratio=0.10, reg_std=reg_std)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True,
                                 prefetch_factor=2, persistent_workers=True,
                                 collate_fn=roi_collate)
        ts_loss, ts_mae, ts_acc = evaluate(ema_model, test_loader, device, reg_std=reg_std)
        print(f"[Test/EMA] loss={ts_loss:.4f} | MAE={ts_mae:.4f} | Acc={ts_acc:.4f}")
        save_json({"test_loss": ts_loss, "test_mae": ts_mae, "test_acc": ts_acc},
                  os.path.join(args.out_dir, "test_metrics.json"))


if __name__ == "__main__":
    main()
