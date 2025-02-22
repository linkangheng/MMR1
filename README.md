# MMR1

## ⚙️ **Environment**

- `git clone https://github.com/linkangheng/MMR1.git`
- Manual installation of `lighteval`
  ```bash
  git clone https://github.com/huggingface/lighteval.git && cd lighteval && git checkout 4f381b352c0e467b5870a97d41cb66b487a2c503 && pip install ".[math]" && rm -rf lighteval
  ```
- `cd MMR1 && pip install -e ".[dev]"`
- Update the transformers to the 4.49.0.dev0 to support the Qwen2.5_VL
- `pip install vllm == 0.7.2 trl == 0.15.0.dev0` to support vLLM

## 🚨 **Notes**

- You are supported to get the permission of `B:kanelin-jfs` and rlaunch your machine by `--mount=juicefs+s3://oss.i.shaipower.com/kanelin-jfs:/mnt/jfs-test` to access the dataset and the model.
- Whenever `MMR1` updates, you may reinstall the `MMR1` by `cd MMR1 && pip install -e ".[dev]"`
- If your dataset contains S3 paths, you may run `unset http_proxy https_proxy all_proxy no_proxy` before training.

## 📋 **ToDos**

- [x] Accelerate and support `vllm`
- [x] Fix the bug of gradient checkpointing
- [x] Support scale up rollout size
- [x] Support Multi-machine parallel
- [x] Support `Kimi-KL`
- [x] Support `OCR` Tasks
- [ ] Support `Detection` Tasks
- [ ] Remove all the absolute path

## 📅 **Update Logs**
### 🤯2025.02.16
- Move all constants to constants.py
- Add the `train_sample_size` to config the number of training samples
- Add system_prompt_template, question_template, answer_template to set the system prompt, question template, answer template, if you want to use a custom template, you can design your own template in the `constants.py` and set the `system_prompt_template`, `question_template`, `answer_template` to your custom template name.
- **Support json dataset as input**, we recommend you to use the `json` format to store your dataset, all you need to create a `json` file which contains dicts with keys `problem`, `solution`, `image`. e.g.
```json
{
    "problem": <str>,
    "solution": <int/str>,
    "image": <image_path>
}
```

### 🤯2025.02.17
#### 🖼️ OCR Task Support
The model now supports training on OCR (Optical Character Recognition) tasks.

**Usage:**
```bash
bash local_scripts/train/train_qwen2_2b_vl_ocr_demo.sh
```

---
#### 🔥Temperature Control
Control the sampling temperature during training using `--temperature_func`.

Available Functions:

- **Linear Scheduling:**
  
  - Set `--temperature_func linear` with:

    - `--temperature_begin`: Initial temperature (must be ≤ temperature_end).

    - `--temperature_end`: Final temperature.

  - Temperature linearly increases from `--temperature_begin` to `--temperature_end` over training steps.

- **Constant Scheduling:**
  - Set `--temperature_func constant` with `--temperature` to apply a fixed temperature value.

---
#### 🎛️ KL Divergence Control

- **Note:** You can view [kl approximator introduction](https://zhuanlan.zhihu.com/p/139084847) to know what is k1, k2(kimikl), k3.Our 'fullkimi' follow the kimi1.5 paper loss.
- **Note:** No matter whether to use kl or which kl to use, we've provide module to compute kl while training. So, ref model is loaded even if use_kl is set to False

#### K1: Context-Distribution KL

- Set `--kl_approximator k1` with `--use_kl True` to use k1 kl for training

- **Definition:**
 - `k1 = logprobs-ref_logprobs`
- **Parameters:**
  - `--beta`: Weight for loss (default: 0.04).

#### K3: Adaptive Response KL
- Set `--kl_approximator k1` with `--use_kl True` to use k1 kl for training
- **Definition:** 
 - `k3 = exp(ref_logprob-logprob) - (ref_logprob - loggprob) - 1`
- **Parameters:**
  - `--beta`: Weight for loss (default: 0.04).

#### KimiKL: Task-Specific KL
- **Definition:** 
  - `kimikl(k2) = 0.5*(logprob-ref_logprob)**2`
- **Parameters:**
  - `--beta`: Weight for KimiKL loss (default: 0.04).

#### KimiFull: Full-Distribution KL
- **Definition:** 
<td><img src="./assets/fullkimi_loss.png"></td>

- **Parameters:**
  - `--beta`: Weight for KimiFull loss (default: 0.04).

---
### 📉 Entropy Regularization
- **Definition:**
  - Entropy loss is computed as L_entropy = -entropy_weight * H(p), where H(p) is the entropy of the model’s output distribution. This term incentivizes the model to sharpen (low entropy) or diversify (high entropy) predictions based on the task.

- **Parameters:**
  - `--entropy_reg`: Enable entropy regularization (default: False).
  - `--entropy_weight`:
    - Use positive values to encourage higher entropy (e.g., for creative generation).
    - Use negative values to reduce entropy (e.g., for discriminative tasks like OCR or grounding).

---
#### 📊 Enhanced Training Logs
Additional metrics are now logged to wandb and local:

##### Reward Logs:
- `completion`: Model response when rollouting.
- `solution`: Intermediate reasoning steps (if applicable).
- `reward`: Task-specific reward signals.

##### Model State Metrics:
- `KLs`: K1, K3, KimiKL, and KimiFull divergence values.
- `entropy`: Output distribution entropy.
- `temperature`: Current temperature value.
To enable logging, ensure wandb is configured in your environment.

## 🚀 **Quick Start**
We now support *counting*, *grounding*, *ocr* tasks. You can easily run the demo scripts in `local_scripts/train/`.

## 🥩 **Mini-Batch**
Optimize GRPO memory usage by redefining per_device_batch_size as generations per device, introduces a more flexible approach:

- Instead of defining per_device_batch_size as the number of prompts per device, it now represents the number of generations per device.
- This allows for much greater flexibility in choosing the number of generations (G) and the batch size per device.
- The only constraint is that the global batch size (num_processes * per_device_batch_size) must be divisible by G.

Note that these settings should be equivalent:

```python
num_generations = ...  # eg, 8
num_prompts_per_device = ...  # eg, 1
# main
GRPOConfig(num_generations=num_generations, per_device_batch_size=num_prompts_per_device, ...)
# this PR
GRPOConfig(num_generations=num_generations, per_device_batch_size=num_generations*num_prompts_per_device, ...)
```

<table align="center" cellpadding="0" cellspacing="0">
  <tr>
    <td align="center"><h3>Original Training</h3></td>
    <td align="center"><h3>Mini-Batch Training</h3></td>
  </tr>
  <tr>
    <td><img src="./assets/original_training.png"></td>
    <td><img src="./assets/mini_batch_training.png"></td>
  </tr>
</table>
