from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os
import time
import argparse
import pandas as pd

def parse_float_sequence_within(input_str):
    """Extract the first sequence of four floating-point numbers from the string"""
    patterns = [
        r"\[\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\s*\]",  # [x1,y1,x2,y2]
        r"\(\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\s*\)",  # (x1,y1,x2,y2)
        r"\(\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)\s*,\s*\(\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\s*\)"  # (x1,y1),(x2,y2)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, input_str)
        if match:
            return [float(match.group(i)) for i in range(1, 5)]
    return [0.0, 0.0, 0.0, 0.0]

def extract_bbox_answer(content):
    """Extract bounding box from model output"""
    is_qwen2vl = "<|box_start|>" in content
    bbox = parse_float_sequence_within(content)
    return (bbox if is_qwen2vl else [int(x*1000) for x in bbox]), is_qwen2vl

def compute_iou(box1, box2):
    """Compute IoU"""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    intersection = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    
    return intersection / (area1 + area2 - intersection + 1e-10)

def compute_accuracy(box1, box2, threshold=0.5):
    """Compute accuracy"""
    return compute_iou(box1, box2) >= threshold

def compute_center_accuracy(box1, box2):
    """Compute center accuracy"""
    cx = (box2[0] + box2[2]) / 2
    cy = (box2[1] + box2[3]) / 2
    return (box1[0] <= cx <= box1[2]) and (box1[1] <= cy <= box1[3])

class RefCOCOEvaluator:
    """Evaluator class encapsulating the evaluation logic"""
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize model
        if "qwen2" in args.model_path.lower():
            generator = Qwen2VLForConditionalGeneration
        elif "qwen2.5" in args.model_path.lower():
            generator = Qwen2_5_VLForConditionalGeneration
        else:
            raise ValueError(f"Unsupported model: {args.model_path}")
        
        self.model = generator.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=self.device,
        )
        self.processor = AutoProcessor.from_pretrained(args.model_path)
        
        # Evaluation metrics
        self.scorers = {
            "IoU": compute_iou,
            "ACC@0.1": lambda x,y: compute_accuracy(x,y,0.1),
            "ACC@0.3": lambda x,y: compute_accuracy(x,y,0.3),
            "ACC@0.5": lambda x,y: compute_accuracy(x,y,0.5),
            "ACC@0.75": lambda x,y: compute_accuracy(x,y,0.75),
            "ACC@0.95": lambda x,y: compute_accuracy(x,y,0.95),
            "Center_ACC": compute_center_accuracy,
        }
    
    def evaluate_dataset(self, dataset):
        """Evaluate a single dataset"""
        print(f"Processing {dataset}...")
        ds_path = os.path.join(self.args.data_root, f"{dataset}.json")
        data = json.load(open(ds_path, "r"))
        if self.args.sample_num > 0:
            data = data[:self.args.sample_num]
        
        # Prepare input data
        messages = []
        for x in data:
            img_path = os.path.join(self.args.image_root, x['image'])
            messages.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{img_path}"},
                    {"type": "text", "text": f"Output the bounding box of the {x['problem']} in the image."}
                ]
            }])
        
        # Batch inference
        results = []
        for i in tqdm(range(0, len(messages), self.args.batch_size)):
            batch = messages[i:i+self.args.batch_size]
            text = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
            
            # Process visual input
            image_inputs, video_inputs = process_vision_info(batch)
            inputs = self.processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                padding_side="left",
                return_tensors="pt"
            ).to(self.device)
            
            # Generate output
            outputs = self.model.generate(**inputs, max_new_tokens=128, do_sample=False, use_cache=True)
            decoded = self.processor.batch_decode(
                [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False
            )
            results.extend(decoded)
        
        # Calculate result metrics
        return self._calculate_metrics(data, results, dataset)

    def _calculate_metrics(self, data, outputs, dataset):
        """Calculate evaluation metrics"""
        scores = {k: 0.0 for k in self.scorers}
        final_results = []
        
        for example, output in zip(data, outputs):
            pred_box, is_normalized = extract_bbox_answer(output)
            gt_box = example['normalized_solution'] if is_normalized else example['solution']
            
            result = {
                'question': example['problem'],
                'ground_truth': gt_box,
                'model_output': output,
                'extracted_answer': pred_box,
                'scores': {}
            }
            
            for name, scorer in self.scorers.items():
                score = scorer(gt_box, pred_box)
                result['scores'][name] = score
                scores[name] += score
            
            final_results.append(result)
        
        # Calculate average score
        avg_scores = {k: round(v/len(data)*100, 2) for k,v in scores.items()}
        return {
            'dataset': dataset,
            'average_scores': avg_scores,
            'details': final_results
        }

def main(args):
    """Main execution flow"""
    evaluator = RefCOCOEvaluator(args)
    
    # Determine the datasets to evaluate
    ALL_DATASETS = [
        'refcoco_val', 'refcoco_testA', 'refcoco_testB',
        'refcocop_val', 'refcocop_testA', 'refcocop_testB',
        'refcocog_val', 'refcocog_test'
    ]
    target_datasets = [ALL_DATASETS[args.task_id]]
    
    # Execute evaluation
    for ds in target_datasets:
        result = evaluator.evaluate_dataset(ds)
        
        # Save results
        output_path = f"./logs/rec_results_{ds}_{os.path.basename(args.model_path)}_ts{args.timestamp}.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'model': args.model_path,
                'config': vars(args),
                **result
            }, f, indent=2)
        
        print(f"\nResults for {ds}:")
        for k,v in result['average_scores'].items():
            print(f"{k}: {v}%")
        print(f"Saved to {output_path}")

def organize_results(args):
    """Summarize results"""
    ALL_DATASETS = [
        'refcoco_val', 'refcoco_testA', 'refcoco_testB',
        'refcocop_val', 'refcocop_testA', 'refcocop_testB',
        'refcocog_val', 'refcocog_test'
    ]
    summary = []
    for ds in ALL_DATASETS:
        result_file = f"./logs/rec_results_{ds}_{os.path.basename(args.model_path)}_ts{args.timestamp}.json"
        if os.path.exists(result_file):
            with open(result_file) as f:
                data = json.load(f)
                summary.append({
                    'dataset': ds,
                    **data['average_scores']
                })
    
    # Generate summary report
    df = pd.DataFrame(summary)
    df['dataset'] = pd.Categorical(df['dataset'], categories=ALL_DATASETS, ordered=True)
    df = df.sort_values('dataset')
    
    report_path = f"./logs/summary_{os.path.basename(args.model_path)}_ts{args.timestamp}.csv"
    df.to_csv(report_path, index=False)
    print(f"Summary report saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Localization Evaluation Script")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--data_root", type=str, default="/mnt/jfs-test/data/refcoco", help="Data root directory")
    parser.add_argument("--image_root", type=str, default="/mnt/shared-storage/groups/hypertext/danielyu/data/Cambrian-10M/coco/", help="Image root directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--task_id", type=int, default=0, help="ID of the dataset to evaluate")
    parser.add_argument("--sample_num", type=int, default=-1, help="Number of samples (for debugging)")
    parser.add_argument("--timestamp", type=str, default=time.strftime("%Y%m%d_%H%M%S"), help="Timestamp")
    parser.add_argument("--mode", choices=['eval', 'orgnize'], default='eval', help="Operation mode")
    args = parser.parse_args()

    if args.mode == 'eval':
        main(args)
    else:
        organize_results(args)