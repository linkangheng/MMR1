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
- [ ] Support `Kimi-KL`
- [ ] Support `OCR` Tasks
- [ ] Support `Detection` Tasks
- [ ] Remove all the absolute path

## 📅 **Update Logs**
### 2025.02.16
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

## 🚀 **Quick Start**

You can run the following command to quickly start the training of `LLaVA-GRPO-Perpo`.

```bash
bash local_scripts/train/train_llava_perpo.sh
```

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
