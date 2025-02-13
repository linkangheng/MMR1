
import datasets
import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

SPLIT = 'test'
IMG_DIR = f'/mnt/jfs-test/data/CountBenchQA/images/{SPLIT}'

os.makedirs(IMG_DIR, exist_ok=True)
def save_image(image_item, target_path):
    try:
        image_item.save(target_path)
        return True, target_path
    except Exception as e:
        print(f"保存失败已跳过 {target_path}: {str(e)}")
        return False, target_path

dataset = datasets.load_dataset("/mnt/jfs-test/data/CountBenchQA")

with open(f'/data/ICCV2025/PaR/MMR1/eval/prompts/countbenchqa_{SPLIT}491_counting_problems.jsonl', 'w') as f:
    tasks = []
    for idx, data in tqdm(enumerate(dataset[SPLIT]), total=491):
        image_item = data['image']
        filename = f"{str(idx).zfill(5)}.jpg"
        target_path = os.path.join(IMG_DIR, filename)
        tasks.append((image_item, target_path))

    with ThreadPoolExecutor(max_workers=32) as executor:
        results = []
        for result in tqdm(executor.map(lambda x: save_image(*x), tasks), total=len(tasks), desc="下载进度"):
            results.append(result)
    
    success_paths = {path for success, path in results if success}
    for idx, data in enumerate(dataset[SPLIT]):
        filename = f"{str(idx).zfill(5)}.jpg"
        target_path = os.path.join(IMG_DIR, filename)
        if target_path not in success_paths:
            continue
        info = {
            "image_path": target_path,
            "question": data['question'],
            "ground_truth": data['number'],
        }
        f.write(json.dumps(info) + '\n')