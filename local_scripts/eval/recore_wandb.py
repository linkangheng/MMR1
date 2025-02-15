# record score to wandb
import wandb

api = wandb.Api()

_runs = api.runs(f"linkangheng/MMR1")
id_map = {}
for run in _runs:
    id_map[run.name] = run.id

runs = [
    ("qwen2-vl-2b_vllm_perpo_roll2", 78.8, 78.23),
    ("qwen2-vl-2b_vllm_perpo_roll4", 80.15, 79.23),
    ("qwen2-vl-2b_vllm_perpo_roll8", 80.04, 79.58),
    ("qwen2-vl-2b_vllm_perpo_roll_8_minibatch", 80.62, 79.75),
    ("qwen2-vl-2b_vllm_perpo_roll16", 80.07, 79.56),
    ("qwen2-vl-2b_vllm_perpo_roll32", 80.39, 79.97),
    ("qwen2-vl-2b_vllm_perpo_roll64", 80.43, 80.06),
    ("qwen2-vl-2b_vllm_perpo_roll126", 80.62, 80.24),
    ("qwen2-vl-2b_vllm_perpo_roll252", 80.56, 79.98),
    ("qwen2-vl-2b_vllm_perpo_roll329", 80.76, 80.14),
]

for run_name, refcocog_val, refcocog_test in runs:
    run = api.run(f"linkangheng/MMR1/{id_map[run_name]}")
    run.summary["refcocog_val"] = refcocog_val
    run.summary["refcocog_test"] = refcocog_test
    run.update()
wandb.finish()
