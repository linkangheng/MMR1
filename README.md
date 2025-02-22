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
### 2025.02.17
#### add some hyperparameters aboout Temperature controling and KL controling
##### Temperature controling

you can view the ```src/open_r1/arguments.py``` for detail info of every hyperparameters and use the ```train_qwen22b_perpo.sh``` to train qwen baseline on perpo grounding task. 

`-use_kl`: whether to use kl in loss. If false, no kl will be included into loss. But you can also view kl change trends in pandb.

`-kl_approximator`: which type kl to use for computing loss.you can use k1(not good), k3(official in grpo, unbias, lowest variance), 
kimikl(only the kl used in kimi1.5), kimifull(the same setting as the core idea of kimi1.5, 
your value of sync_ref_model, ref_model_mixup_alpha and ref_model_sync_steps will be invalid, they are all set the same as kimi1.5)

`-entropy_reg`: whether to use entropy regularization while training. For discriminative tasks like grounding, ocr and counting, we expect entropy to decrease.
For literary creation task, we expect entropy to increase. this can be controlled by entropy_weight.

`-entropy_weight`: the weight for entropy loss. It's only valid when entropy_reg is true. If it's positive, the entropy is to increase. If it's negetive, the entropy is to decrease.

`-temperature_func`: which temperature function to use while training. Unlike reward_funcs, you can only use one temperature function. The available function is "linear" and "constant"

`-learning_rate`: the laerning_rate for begining training. The learning rate will end to 0.

`-sync_ref_model`: whether to update ref modeel while training.

`-ref_model_mixup_alpha`: the alpha to mix policy model and ref moodel: `π_ref = α * π_θ + (1 - α) * π_ref_prev`. In kimi1.5, they set the value 1.0

`-ref_model_sync_steps`: the steps for updating ref model. In kimi1.5, they set the value 1

##### KL controling

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
