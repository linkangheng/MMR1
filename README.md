# MMR1

⚙️ **Environment**

- Follow the instruction in https://github.com/FanqingM/R1-Multimodal-Journey
- Update the transformers to the 4.49.0.dev0 to support the Qwen2.5_VL
- `pip install vllm == 0.7.2 trl == 0.15.0.dev0` to support vLLM

🚨 **Notes**

- You are supported to get the permission of `B:kanelin-jfs` and rlaunch your machine by `--mount=juicefs+s3://oss.i.shaipower.com/kanelin-jfs:/mnt/jfs-test` to access the dataset and the model.

📋 **ToDos**

- [x] Accelerate and support `vllm`
- [ ] Support scale up rollout size
- [ ] Support `Kimi-KL`
- [ ] Support `OCR` Tasks
- [ ] Support `Detection` Tasks
- [ ] Remove all the absolute path

🚀 **Quick Start**

You can run the following command to quickly start the training of `LLaVA-GRPO-Perpo`.

```bash
bash local_scripts/train/train_llava_perpo.sh
```
