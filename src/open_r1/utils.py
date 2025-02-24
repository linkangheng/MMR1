import json

def get_qa_pairs(conversation: list):
    qa_pairs = []
    for i in range(0, len(conversation), 2):
        question = conversation[i]['value']
        answer = conversation[i+1]['value']
        qa_pairs.append((question, answer))
    return qa_pairs

def load_image(image_path):
    from PIL import Image
    import megfile
    from io import BytesIO
    import os
    os.environ['OSS_ENDPOINT'] = 'http://oss.i.basemind.com'
    if 's3://' in image_path:
        with megfile.smart_open(image_path, "rb") as f:
            bytes_data = f.read()
        image = Image.open(BytesIO(bytes_data), "r").convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')
    return image

# make conversation for text hf-dataset
def make_conversation(
        example,
        system_prompt,
        question_template=None,
        answer_template=None
    ):
    question = example["problem"] if question_template is None else question_template.format(question=example["problem"])
    answer = example["solution"] if answer_template is None else answer_template.format(answer=example["solution"])
    return {
        "prompt": json.dumps([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]),
    }

# make conversation for multi-modal hf-dataset
def make_conversation_image(
        example,
        system_prompt,
        question_template=None,
        answer_template=None
    ):
    question = example["problem"] if question_template is None else question_template.format(question=example["problem"])
    question = question + "." if not question.endswith(".") else question
    return {
        "prompt": json.dumps([
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]),
    }

# make conversation for json dataset
def json_map(
        example,
        system_prompt,
        question_template=None,
        answer_template=None
    ):
    '''
    1. example:
        {
            "problem": <str>,
            "image": <image_path>,
            "solution": <int/str>
        }
    2. system_prompt: <str>
    3. question_template: <str>, e.g. "xx {question}?"
    4. answer_template: <str>, e.g. "<answer> {answer} </answer>."
    '''
    image_path = example['image']
    question = example['problem'] if question_template is None else question_template.format(question=example['problem'])
    solution = example['solution'] if answer_template is None else answer_template.format(answer=example['solution'])
    

    
    return {
        "image": load_image(image_path),
        "prompt": json.dumps([
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]),
        "solution": solution,
        "problem": question,
    }
def save_dict_to_json(dict_data, filename):
    """
    Save a dictionary to a JSON file.
    
    Args:
        dict_data (dict): Dictionary to be saved
        filename (str): Path to the output JSON file
    """
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(json.dumps(dict_data, indent=4))
    except Exception as e:
        print(f"error saving dictionary to {filename}: {e}")

def save_args_to_txt(args, filename):
    """
    将 argparse 解析的参数保存到 txt 文件中
    :param args: argparse.Namespace 对象，包含解析后的参数
    :param filename: 要保存的文件名
    """
    with open(filename, 'w') as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
