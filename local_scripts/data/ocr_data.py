import json
import os
import random
import re

import pandas as pd
from PIL import Image as PILImage
from datasets import Dataset, Features, Value, Image, DatasetDict
from tqdm import tqdm

random.seed(42)

def format_grounding_explanation(answer: str):
    formatted_text = f"<answer>{answer}</answer>"
    return formatted_text

def has_valid_image_size_from_path(image_path):
    try:
        image = PILImage.open(image_path)
        width, height = image.size
        return height >= 28 and width >= 28
    except Exception as e:
        # If there's an error opening the image (e.g., invalid file or path), return False
        print(f"Error opening image {image_path}: {e}")
        return False

input_file = '/mnt/jfs-test/data/books_arxiv_pdf_png_page_50k/jsons/books_arxiv_pdf_png_en_page_50k.json'
# weights_path = '/data/LLaVA/llava/ppo_vl/ppo_datasets/weights_0.json'
data = {
    'image': [],
    'image_path': [],
    'problem': [],
    'solution': [],
}

with open(input_file, 'r') as f:
    data_all = json.load(f)

# with open(weights_path, 'r') as f:
#     weights = json.load(f)

data_all_filtered = []
for data_tmp in tqdm(data_all, desc="Filtering ...", unit="image"):
    image_path = os.path.join('/mnt/jfs-test/data/books_arxiv_pdf_png_page_50k/', data_tmp['image'])
    data_tmp['image'] = image_path
    is_valid = has_valid_image_size_from_path(image_path)
    if is_valid == True:
        data_all_filtered.append(data_tmp)
    else:
        print(image_path)

print('len(data_all): ', len(data_all))
print('len(data_all_filtered): ', len(data_all_filtered))


def sort_func(data):
    return data['reward']

def make_data_order(data_all_filtered, weights):
    for data in data_all_filtered:
        for weight in weights:
            if data['id'] == weight['id']:
                data['reward'] = weight['weight']
        if 'reward' not in data.keys() or not isinstance(data['reward'], float):
            print(data)


    data_all_filtered.sort(reverse=True, key=sort_func)
    return data_all_filtered

# data_all_filtered = make_data_order(data_all_filtered, weights)


random.shuffle(data_all_filtered)
for item in data_all_filtered:
    image = item.get('image')
    problem = item['conversations'][0]['value']
    # solution = format_grounding_explanation(item['answer'])
    solution = item['conversations'][1]['value']
    data['image'].append(image)
    data['image_path'].append(image)
    data['problem'].append(problem)
    data['solution'].append(solution)

features = Features({
    'image': Image(),
    'image_path': Value('string'),
    'problem': Value('string'),
    'solution': Value('string')
})

# train_dataset = Dataset.from_pandas(df, features=features)


train_dataset = Dataset.from_dict(
    data,
    features=features
)

train_dataset = DatasetDict({
    'train': train_dataset,
})

train_save_path = "/mnt/jfs-test/data/ocr"

# train_dataset.to_parquet(train_save_path)
train_dataset.save_to_disk(train_save_path)
print(f"Saved to {train_save_path}")
