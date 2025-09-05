# convert_perimage_inplace.py
import argparse, json, sys
from pathlib import Path

# YOLO 클래스 이름(원하시는 순서로 고정)
ROI_CLASSES = [
    "forehead", "glabella",
    "left_crowsfeet", "right_crowsfeet",
    "left_cheek", "right_cheek",
    "lips", "chin",
]
ROI_TO_ID = {n:i for i,n in enumerate(ROI_CLASSES)}

# facepart(숫자) → ROI 이름 매핑
# 주신 예시로부터 3=left_crowsfeet, 4=right_crowsfeet가 확실합니다.
# 나머지는 데이터셋 규칙에 맞게 필요시 바꿔주세요.
FACEPART_TO_ROI = {
    1: "forehead",
    2: "glabella",
    3: "left_crowsfeet",   # 예시 파일들: 1011_01_F/Fb/Ft/L15/L30 등에서 facepart=3 (왼눈가) 
    4: "right_crowsfeet",  # 예시 파일들: 1011_01_R15/R30의 facepart=4 (오른눈가) 
    5: "left_cheek",
    6: "right_cheek",
    7: "lips",
    8: "chin",
}

def convert_one(json_path: Path) -> int:
    """
    per-image JSON 하나를 읽어 같은 폴더에 YOLO txt 생성.
    반환값: 쓴 라벨 라인 수(0이면 스킵)
    """
    try:
        data = json.load(open(json_path, "r", encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] JSON 로드 실패: {json_path} ({e})", file=sys.stderr)
        return 0

    # 필수 키: images 안의 width/height/facepart/bbox
    images = data.get("images")
    if not isinstance(images, dict):
        # COCO 통합본 같은 형식이 들어오면 이 스크립트 대상이 아님
        return 0

    W = images.get("width"); H = images.get("height")
    facepart = images.get("facepart")
    bbox = images.get("bbox")

    if not (isinstance(W, int) and isinstance(H, int) and W>0 and H>0):
        print(f"[WARN] width/height 없음: {json_path}", file=sys.stderr); return 0
    if not (isinstance(bbox, (list,tuple)) and len(bbox) >= 4):
        print(f"[WARN] bbox 없음: {json_path}", file=sys.stderr); return 0
    if facepart is None:
        print(f"[WARN] facepart 없음: {json_path}", file=sys.stderr); return 0

    # facepart → ROI 이름 → YOLO class id
    roi_name = FACEPART_TO_ROI.get(int(facepart))
    if roi_name not in ROI_TO_ID:
        print(f"[WARN] facepart={facepart} 매핑 불명: {json_path}", file=sys.stderr); return 0
    cls_id = ROI_TO_ID[roi_name]

    x, y, w, h = map(float, bbox[:4])
    if w <= 0 or h <= 0:
        print(f"[WARN] 비정상 bbox: {json_path}", file=sys.stderr); return 0

    # YOLO 정규화
    xc = (x + w/2.0) / float(W)
    yc = (y + h/2.0) / float(H)
    ww = w / float(W)
    hh = h / float(H)

    # 같은 폴더, 같은 이름으로 .txt 생성
    out_txt = json_path.with_suffix(".txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"{cls_id} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")

    return 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled-root", required=True,
                    help="예: ../dataset/Training/labeled_data (이 하위의 모든 *.json을 변환)")
    args = ap.parse_args()

    root = Path(args.labeled_root).resolve()
    jsons = list(root.rglob("*.json"))
    total = 0; files = 0
    for jp in jsons:
        n = convert_one(jp)
        total += n
        if n > 0: files += 1
    print(f"✅ 완료: txt 생성 {files}개 파일, 총 라벨 라인 {total}개")

if __name__ == "__main__":
    main()
