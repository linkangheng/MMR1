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

