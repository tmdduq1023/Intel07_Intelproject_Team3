

"""
부위(ROI) 기반 멀티태스크 학습 + tqdm 상태바(라인 누적 방지) + Early Stopping

입력:
  - merged_training.json / merged_validation.json / merged_test.json
  - origin 이미지 (images.file_name은 DATASET_ROOT 기준 상대경로)

특징:
  - ROI bbox( facepart::... )로 crop → 각 부위별 회귀(eq_*) / 분류(ann_*) 타깃만 학습
  - 배치 내 서로 다른 ROI의 타깃 차이 → collate에서 패딩(NaN/-1) + 학습 시 슬라이싱 처리
  - tqdm 상태바: leave=False 로 라인 누적 없이 실시간 갱신
  - EarlyStopping(patience, min_delta) 지원
"""

import os, json, argparse, random, time
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
    "glabella":      ["glabellus", "glabella"],
    "left_crowsfeet":["l_perocular", "left_crowsfeet"],
    "right_crowsfeet":["r_perocular", "right_crowsfeet"],
    "left_cheek":    ["l_cheek", "left_cheek"],
    "right_cheek":   ["r_cheek", "right_cheek"],
    "lips":          ["lip", "lips", "perioral"],
    "chin":          ["chin"],
    "full":          ["full", "all", "face"],
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





class RoiDataset(Dataset):
    def __init__(self, json_path, data_root, roi_label_space, img_size=384, is_train=True, pad_ratio=0.10):
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
        self.pad_ratio = pad_ratio

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
                if not bbox or len(bbox) != 4:
                    continue
                x,y,w,h = bbox
                px, py = int(w*self.pad_ratio), int(h*self.pad_ratio)
                x,y,w,h = clamp_bbox(x-px, y-py, w+2*px, h+2*py, W, H)
                
                if len(self.roi_reg_keys.get(roi_name, [])) == 0 and len(self.roi_cls_keys.get(roi_name, [])) == 0:
                    continue
                self.samples.append((img_id, roi_name, [x,y,w,h]))

        if is_train:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
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
                try: reg_t[j] = float(v)
                except: pass
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
    def __init__(self, backbone="resnet50", roi_label_space=None, pretrained=True):
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





def masked_mae(pred, target):
    if pred is None:
        return None, 0
    if pred.dim() == 2 and target.dim() == 1:
        target = target.unsqueeze(0)
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return pred.new_tensor(0.0), 0
    mae = (pred[mask] - target[mask]).abs().mean()
    return mae, int(mask.sum().item())

def cls_loss_and_acc(logits_list, target, ignore_index=-1):
    if logits_list is None or len(logits_list) == 0 or target.numel() == 0:
        return None, 0, None, 0
    total_ce, total_acc, n_valid = 0.0, 0.0, 0
    for k, lg in enumerate(logits_list):
        t = target[k].unsqueeze(0)  
        ce = F.cross_entropy(lg, t, ignore_index=ignore_index)
        mask = (t != ignore_index)
        if mask.sum() > 0:
            pred = lg.argmax(dim=1)
            acc = (pred[mask] == t[mask]).float().mean()
            total_ce += ce.item()
            total_acc += acc.item()
            n_valid += 1
        else:
            total_ce += 0.0
    if n_valid == 0:
        return None, 0, None, 0
    
    return ce, n_valid, total_acc / n_valid, n_valid





def train_one_epoch(model, loader, optim, device, lambda_reg=1.0, lambda_cls=1.0, scaler=None):
    model.train()
    pbar = tqdm(loader, desc="train", dynamic_ncols=True, leave=False)
    avg = {"loss":0.0, "mae":0.0, "acc":0.0}
    steps = 0

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
                
                reg_pred = out_reg[i]
                if reg_pred is not None and reg_pred.size(1) > 0:
                    target_slice = reg_t[i, :reg_pred.size(1)]
                    mae, n_used = masked_mae(reg_pred, target_slice)
                    if mae is not None and n_used > 0:
                        total_mae += mae.item(); n_reg += 1
                        target_slice = torch.nan_to_num(target_slice.unsqueeze(0),
                                                        nan=reg_pred.detach().mean().item())
                        total_loss = total_loss + lambda_reg * F.mse_loss(reg_pred, target_slice)

                
                cls_preds = out_cls[i]
                if cls_preds is not None and len(cls_preds) > 0:
                    Clen = len(cls_preds)
                    ce, nvalid_ce, acc, nvalid_acc = cls_loss_and_acc(cls_preds, cls_t[i, :Clen])
                    if ce is not None and nvalid_ce > 0:
                        total_loss = total_loss + lambda_cls * ce
                    if acc is not None and nvalid_acc > 0:
                        total_acc += acc; n_cls += 1

        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            total_loss.backward()
            optim.step()

        steps += 1
        avg["loss"] += float(total_loss.item())
        if n_reg > 0: avg["mae"] += (total_mae / n_reg)
        if n_cls > 0: avg["acc"] += (total_acc / n_cls)

        pbar.set_postfix({
            "loss": f"{avg['loss']/steps:.4f}",
            "MAE":  f"{(avg['mae']/steps):.4f}" if steps>0 else "0.0000",
            "Acc":  f"{(avg['acc']/steps):.4f}" if steps>0 else "0.0000",
        })

    return avg["loss"]/max(1,steps), avg["mae"]/max(1,steps), avg["acc"]/max(1,steps)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    pbar = tqdm(loader, desc="valid", dynamic_ncols=True, leave=False)
    avg = {"loss":0.0, "mae":0.0, "acc":0.0}
    steps = 0

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
            reg_pred = out_reg[i]
            if reg_pred is not None and reg_pred.size(1) > 0:
                target_slice = reg_t[i, :reg_pred.size(1)]
                mae, n_used = masked_mae(reg_pred, target_slice)
                if mae is not None and n_used > 0:
                    total_mae += mae.item(); n_reg += 1
                    target_slice = torch.nan_to_num(target_slice.unsqueeze(0),
                                                    nan=reg_pred.detach().mean().item())
                    total_loss = total_loss + F.mse_loss(reg_pred, target_slice)

            cls_preds = out_cls[i]
            if cls_preds is not None and len(cls_preds) > 0:
                Clen = len(cls_preds)
                ce, nvalid_ce, acc, nvalid_acc = cls_loss_and_acc(cls_preds, cls_t[i, :Clen])
                if ce is not None and nvalid_ce > 0:
                    total_loss = total_loss + ce
                if acc is not None and nvalid_acc > 0:
                    total_acc += acc; n_cls += 1

        steps += 1
        avg["loss"] += float(total_loss.item())
        if n_reg > 0: avg["mae"] += (total_mae / n_reg)
        if n_cls > 0: avg["acc"] += (total_acc / n_cls)

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
    ap.add_argument("--out-dir",    type=str, default="./runs/skin_roi")
    ap.add_argument("--img-size",   type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs",     type=int, default=50)
    ap.add_argument("--workers",    type=int, default=16)
    ap.add_argument("--lr-backbone",type=float, default=5e-5)
    ap.add_argument("--lr-head",    type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lambda-reg", type=float, default=1.0)
    ap.add_argument("--lambda-cls", type=float, default=1.0)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--early-patience", type=int, default=10)
    ap.add_argument("--early-min-delta", type=float, default=1e-4)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    
    roi_label_space = build_roi_label_space(args.train_json)
    save_json(roi_label_space, os.path.join(args.out_dir, "roi_label_space.json"))

    
    train_set = RoiDataset(args.train_json, args.data_root, roi_label_space, img_size=args.img_size, is_train=True)
    val_set   = RoiDataset(args.val_json,   args.data_root, roi_label_space, img_size=args.img_size, is_train=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, collate_fn=roi_collate)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True, collate_fn=roi_collate)

    
    model = RoiMultiHead(backbone="resnet50", roi_label_space=roi_label_space, pretrained=True).to(device)
    backbone_params, head_params = [], []
    for n,p in model.named_parameters():
        (backbone_params if "backbone" in n else head_params).append(p)
    optim = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params,     "lr": args.lr_head},
    ], weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    scaler = GradScaler(device="cuda") if torch.cuda.is_available() else None

    
    stopper = EarlyStopping(patience=args.early_patience, min_delta=args.early_min_delta)
    best_path  = os.path.join(args.out_dir, "best.pth")
    best_epoch = 0

    for ep in range(1, args.epochs+1):
        tr_loss, tr_mae, tr_acc = train_one_epoch(model, train_loader, optim, device,
                                                  lambda_reg=args.lambda_reg, lambda_cls=args.lambda_cls, scaler=scaler)
        vl_loss, vl_mae, vl_acc = evaluate(model, val_loader, device)
        sched.step()

        
        score = vl_mae + (1.0 - vl_acc if not np.isnan(vl_acc) else 1.0)

        
        print(f"[{ep:03d}/{args.epochs}] "
              f"train: loss={tr_loss:.4f} MAE={tr_mae:.4f} Acc={tr_acc:.4f} | "
              f"valid: loss={vl_loss:.4f} MAE={vl_mae:.4f} Acc={vl_acc:.4f} | "
              f"score={score:.4f}")

        improved = stopper.step(score)
        if improved:
            torch.save({"model": model.state_dict(),
                        "roi_label_space": roi_label_space,
                        "epoch": ep,
                        "val_mae": vl_mae,
                        "val_acc": vl_acc,
                        "score": score}, best_path)
            best_epoch = ep
            print(f"  → Saved best to {best_path}")
        elif stopper.should_stop():
            print(f"Early stopping triggered at epoch {ep} (best @ {best_epoch})")
            break

    print(f"Done. Best epoch={best_epoch}, checkpoint={best_path}")

    
    if args.test_json and Path(args.test_json).exists():
        test_set = RoiDataset(args.test_json, args.data_root, roi_label_space, img_size=args.img_size, is_train=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True, collate_fn=roi_collate)
        ts_loss, ts_mae, ts_acc = evaluate(model, test_loader, device)
        print(f"[Test] loss={ts_loss:.4f} | MAE={ts_mae:.4f} | Acc={ts_acc:.4f}")
        save_json({"test_loss": ts_loss, "test_mae": ts_mae, "test_acc": ts_acc},
                  os.path.join(args.out_dir, "test_metrics.json"))


if __name__ == "__main__":
    main()
