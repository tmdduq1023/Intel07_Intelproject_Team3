import os
import json
from PIL import Image
from tqdm import tqdm
import shutil
import multiprocessing # <<< 멀티프로세싱 모듈 추가

def process_image(task):
    """
    단일 이미지를 리사이즈하고 저장하는 작업자(worker) 함수.
    이 함수가 여러 CPU 코어에서 동시에 실행됩니다.
    """
    original_img_path, new_img_path, new_size = task

    try:
        # 이미지 파일이 저장될 하위 폴더 생성 (예: 'images/')
        os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
        
        with Image.open(original_img_path) as img:
            resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
            resized_img = resized_img.convert('RGB')
            resized_img.save(new_img_path)
        return None # 성공
    except FileNotFoundError:
        return f"Warning: Source image not found at {original_img_path}. Skipping." # 실패 정보 반환
    except Exception as e:
        return f"Error processing {original_img_path}: {e}" # 기타 에러 정보 반환


def main():
    """
    메인 함수: 작업 목록을 만들고 프로세스 풀에 작업을 분배합니다.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    new_size=(224, 224)

    # --- 경로 설정 및 폴더 생성 (이전과 동일) ---
    json_path = os.path.join(base_path, 'coco_skin_dataset.json')
    output_base_dir = os.path.join(base_path, 'dataset_preprocessed')
    
    if os.path.exists(output_base_dir):
        print(f"'{output_base_dir}' exists. Removing it to start fresh.")
        shutil.rmtree(output_base_dir)
    os.makedirs(output_base_dir)
    print(f"Created directory: {output_base_dir}")

    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # <<< 1. 모든 작업 목록을 미리 생성 ---
    tasks = []
    for img_info in coco_data['images']:
        original_img_path = os.path.join(base_path, img_info['file_name'])
        new_img_path = os.path.join(output_base_dir, img_info['file_name'])
        tasks.append((original_img_path, new_img_path, new_size))

    print(f"Preparing to process {len(tasks)} images using multiple cores...")

    # <<< 2. 멀티프로세싱 풀(Pool) 생성 ---
    # 사용 가능한 모든 CPU 코어를 사용하도록 설정
    num_workers = multiprocessing.cpu_count()
    print(f"Creating a pool of {num_workers} worker processes.")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        # <<< 3. 작업을 풀에 분배하고 tqdm으로 진행 상황 표시 ---
        # imap_unordered는 작업이 완료되는 순서대로 결과를 반환하여 효율적입니다.
        results = list(tqdm(pool.imap_unordered(process_image, tasks), total=len(tasks), desc="Processing Images"))

    # 에러가 있었다면 출력
    errors = [res for res in results if res is not None]
    if errors:
        print("\n--- Encountered some errors during processing ---")
        for error in errors[:10]: # 최대 10개만 출력
            print(error)

    # JSON 파일 복사 (이전과 동일)
    new_json_path = os.path.join(output_base_dir, 'coco_skin_dataset.json')
    shutil.copy2(json_path, new_json_path)
    print(f"\nCopied JSON file to {new_json_path}")
    print("Preprocessing finished successfully!")


if __name__ == '__main__':
    # 멀티프로세싱 사용 시 Windows/macOS에서 필요할 수 있는 설정
    multiprocessing.freeze_support()
    main()