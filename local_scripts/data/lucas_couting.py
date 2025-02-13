# question: How many items are there in the image?
# answer: <answer> 10 </answer>

import json
import os
import random
import re

import pandas as pd
from PIL import Image as PILImage
from datasets import Dataset, Features, Value, Image, DatasetDict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

random.seed(42)

def load_image(image_path):
    import megfile
    from io import BytesIO
    import os
    os.environ['OSS_ENDPOINT'] = 'http://oss.i.basemind.com'
    if 's3://' in image_path:
        with megfile.smart_open(image_path, "rb") as f:
            bytes_data = f.read()
        image = PILImage.open(BytesIO(bytes_data), "r").convert('RGB')
    else:
        image = PILImage.open(image_path).convert('RGB')
    return image

def format_counting_explanation(answer: str):
    formatted_text = f"<answer> {answer} </answer>"
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

def get_qa_pairs(conversation: list):
    qa_pairs = []
    for i in range(0, len(conversation), 2):
        question = conversation[i]['value']
        answer = conversation[i+1]['value']
        qa_pairs.append((question, answer))
    return qa_pairs

def count_items(solution: str):
    pattern = r'\(([0-9]+), ([0-9]+)\)'
    matches = re.findall(pattern, solution)
    return len(matches)

# define the formats
input_file = '/mnt/shared-storage/groups/hypertext/sft-data/jihe/count-min20box-237147.json'
question_format = 'How many {item} are there in the image?'
data = {
    'image': [],
    'image_path': [],
    'problem': [],
    'solution': [],
}

# load and process the source data
with open(input_file, 'r') as f:
    data_all = json.load(f)
processed_data = []
for item in data_all:
    image_path = item['image']
    conversation = item['conversations']
    for q,a in get_qa_pairs(conversation):
        processed_data.append({
            'image_path': image_path,
            'problem': question_format.format(item=q),
            'solution': format_counting_explanation(count_items(a))
        })

print('len(data_all): ', len(data_all))
print('len(processed_data): ', len(processed_data))
# organize the data into the format
random.shuffle(processed_data)

processed_data = processed_data[:50_000]

with ThreadPoolExecutor(max_workers=64) as executor:
    image_paths = [item['image_path'] for item in processed_data]
    images = list(tqdm(executor.map(load_image, image_paths), total=len(image_paths), desc='Loading images'))

# 修改原有循环，直接使用预加载的图片
for idx, item in tqdm(enumerate(processed_data), total=len(processed_data), desc='organizing data'):
    image = item.get('image_path')
    problem = item['problem']
    solution = item['solution']
    data['image'].append(images[idx])
    data['image_path'].append(image)
    data['problem'].append(problem)
    data['solution'].append(solution)

features = Features({
    'image': Image(),
    'image_path': Value('string'),
    'problem': Value('string'),
    'solution': Value('string')
})

train_dataset = Dataset.from_dict(
    data,
    features=features
)

train_dataset = DatasetDict({
    'train': train_dataset,
})

train_save_path = "/mnt/jfs-test/data/lucas_counting/count-min20box-763043"
os.makedirs(train_save_path, exist_ok=True)
train_dataset.save_to_disk(train_save_path)
print(f"Saved to {train_save_path}")