import argparse, json, sys
from pathlib import Path

# 8개 부위(이 순서가 YOLO class id)
REGION_CLASSES = [
    "forehead",
    "glabella",
    "left_periocular",
    "right_periocular",
    "left_cheek",
    "right_cheek",
    "lips",
    "chin",
]
REGION_TO_ID = {n: i for i, n in enumerate(REGION_CLASSES)}

def extract_region(key: str) -> str | None:
    """annotations의 key에서 부위만 추출 (지표 단어 제외)"""
    k = key.lower()
    if k.startswith("ann_"):
        k = k[4:]  # ann_ 접두사 제거

    # 눈가(좌/우) - perocular/periocular 모두 허용
    if k.startswith("l_perocular") or k.startswith("l_periocular"):
        return "left_periocular"
    if k.startswith("r_perocular") or k.startswith("r_periocular"):
        return "right_periocular"

    # 볼(좌/우)
    if k.startswith("l_cheek"):
        return "left_cheek"
    if k.startswith("r_cheek"):
        return "right_cheek"

    # 이마
    if k.startswith("forehead"):
        return "forehead"

    # 미간(glabella/glabellus 철자 변형 허용)
    if k.startswith("glabella") or k.startswith("glabellus"):
        return "glabella"

    # 입술
    if k.startswith("lip") or k.startswith("lips") or k == "lip_dryness":
        return "lips"

    # 턱
    if k.startswith("chin"):
        return "chin"

    return None

def is_xyxy(x1, y1, x2, y2, W, H):
    return (x2 > x1) and (y2 > y1) and (0 <= x1 < W) and (0 <= y1 < H) and (0 < x2 <= W) and (0 < y2 <= H)

def convert_one(json_path: Path) -> int:
    try:
        data = json.load(open(json_path, "r", encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] JSON 로드 실패: {json_path} ({e})", file=sys.stderr)
        return 0

    images = data.get("images")
    if not isinstance(images, dict):
        return 0  # per-image 구조가 아니면 스킵

    W = images.get("width")
    H = images.get("height")
    bb = images.get("bbox")
    if not (isinstance(W, (int, float)) and isinstance(H, (int, float)) and W > 0 and H > 0):
        print(f"[WARN] width/height 비정상: {json_path}", file=sys.stderr)
        return 0
    if not (isinstance(bb, (list, tuple)) and len(bb) >= 4):
        print(f"[WARN] bbox 없음: {json_path}", file=sys.stderr)
        return 0

    x1, y1, x2, y2 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
    if is_xyxy(x1, y1, x2, y2, W, H):
        x, y, w, h = x1, y1, (x2 - x1), (y2 - y1)
    else:
        # [x, y, w, h]로 간주
        x, y, w, h = x1, y1, x2, y2
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > W or y + h > H:
            print(f"[WARN] bbox 형식/범위 이상: {json_path} {bb}", file=sys.stderr)
            return 0

    # annotations에서 부위 키 찾기 (여러 개면 첫 매칭 사용)
    ann = data.get("annotations") or {}
    region_name = None
    for k in ann.keys():
        r = extract_region(str(k))
        if r:
            region_name = r
            break
    if not region_name:
        print(f"[WARN] 부위 키 미검출: {json_path}", file=sys.stderr)
        return 0

    cls = REGION_TO_ID.get(region_name)
    if cls is None:
        print(f"[WARN] 정의되지 않은 부위: {region_name} ({json_path})", file=sys.stderr)
        return 0

    # YOLO 정규화
    xc = (x + w / 2.0) / float(W)
    yc = (y + h / 2.0) / float(H)
    ww = w / float(W)
    hh = h / float(H)

    out_txt = json_path.with_suffix(".txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"{cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")
    return 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled-root", required=True, help="예: ../dataset/Training/labeled_data")
    args = ap.parse_args()

    root = Path(args.labeled_root).resolve()
    files, lines = 0, 0
    for jp in root.rglob("*.json"):
        n = convert_one(jp)
        if n > 0:
            files += 1
            lines += n
    print(f"✅ 완료: txt 생성 {files}개, 총 라벨 {lines}개", file=sys.stderr)

if __name__ == "__main__":
    main()
