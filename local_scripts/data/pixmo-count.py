
import datasets
import os
import requests
from hashlib import md5
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

SPLIT = 'validation'
IMG_DIR = f'/mnt/jfs-test/data/pixmo-count/images/{SPLIT}'

os.makedirs(IMG_DIR, exist_ok=True)

def download_image(url, target_path):
    """下载并保存图片到指定路径（返回下载结果和路径）"""
    if os.path.exists(target_path):
        return True, target_path
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(target_path, 'wb') as f:
            f.write(response.content)
        return True, target_path
    except Exception as e:
        print(f"下载失败已跳过 {url}: {str(e)}")
        if os.path.exists(target_path):
            os.remove(target_path)
        return False, target_path

dataset = datasets.load_dataset("/mnt/jfs-test/data/pixmo-count")

with open(f'/data/ICCV2025/PaR/MMR1/eval/prompts/pixmo_count_{SPLIT}540_counting_problems.jsonl', 'w') as f:
    tasks = []
    for idx, data in tqdm(enumerate(dataset[SPLIT]), total=540):
        url = data['image_url']
        filename = f"{str(idx).zfill(5)}.jpg"
        target_path = os.path.join(IMG_DIR, filename)
        tasks.append((url, target_path))

    with ThreadPoolExecutor(max_workers=32) as executor:
        results = []
        for result in tqdm(executor.map(lambda x: download_image(*x), tasks), total=len(tasks), desc="下载进度"):
            results.append(result)
    
    success_paths = {path for success, path in results if success}
    for idx, data in enumerate(dataset[SPLIT]):
        filename = f"{str(idx).zfill(5)}.jpg"
        target_path = os.path.join(IMG_DIR, filename)
        if target_path not in success_paths:
            continue
        info = {
            "image_path": target_path,
            "question": f"How many {data['label']} are there in the image?",
            "ground_truth": data['count'],
        }
        f.write(json.dumps(info) + '\n')