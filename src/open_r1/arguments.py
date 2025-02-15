from trl import ScriptArguments
from dataclasses import dataclass, field
from typing import Optional
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    prompt_template: Optional[str] = field(
        default="reasoning",
        metadata={"help": "Prompt template. Possible values: 'llava', 'qwen2', 'reasoning', 'grounding', 'ocr"},
    )
    use_kl: bool = field(
        default=True, 
        metadata={"help":"whether to use kl with ref model"}
    )
    kl_approximator: str = field(
        default="k3", 
        metadata={"help": "which kl compute to use, k1, k3, or kimikl"}
    )
    reward_rule: int = field(
        default=1, 
        metadata={"help": "reward rule for training"}
    )
    reward_scale: float = field(
        default=1, 
        metadata={"help": "reward scale of all rewards"}
    )
    reward_baseline: float = field(
        default=0, 
        metadata={"help": "reward baseline for training"}
    )
    train_vision: bool = field(
        default=False, 
        metadata={"help":"whether to train vision encoder"}
    )
    entropy_reg : bool = field(
        default=False, 
        metadata={"help": "whether to use entropy reg"}
    )
    entropy_weight: float = field(
        default=0.01, 
        metadata={"help": "entropy_reg_loss weight"}
    )
    temperature_func: str = field(
        default="linear", 
        metadata={"help":"which temperature func to use"}
    )
    temperature_begin: float = field(
        default=0.1
    )
    temperature_end: float = field(
        default=1.0
    )
    order_dataset: str = field(
        default='random'
    )
