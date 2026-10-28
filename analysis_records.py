from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TopBatchRecord:
    rank: int
    batch_index: int
    mean_loss: float
    max_token_loss: float


@dataclass
class TopLogitRecord:
    token_id: int
    logit: float
    token: Optional[str]


@dataclass
class AppliedSwapRecord:
    relative_position: int
    absolute_position: int
    old_token_id: int
    old_token: Optional[str]
    new_token_id: int
    new_token: Optional[str]


@dataclass
class SwapAnalysisRecord:
    applied_swaps: list[AppliedSwapRecord] = field(default_factory=list)
    loss_after_swap: float = 0.0
    loss_delta: float = 0.0
    target_logit_before: float = 0.0
    target_logit_after: float = 0.0
    target_logit_delta: float = 0.0
    top_logits_after_swap: list[TopLogitRecord] = field(default_factory=list)


@dataclass
class TokenEventRecord:
    rank: int
    loss: float
    recomputed_loss: float
    batch_index: int
    example_index: int
    position: int
    input_token_id: int
    input_token: Optional[str]
    target_token_id: int
    target_token: Optional[str]
    context_start: int
    context_ids: list[int] = field(default_factory=list)
    context_text: Optional[str] = None
    top_logits: list[TopLogitRecord] = field(default_factory=list)
    swap_analysis: Optional[SwapAnalysisRecord] = None


@dataclass
class IterativeStepRecord:
    step: int
    token_position: int
    old_token_id: int
    old_token: Optional[str]
    new_token_id: int
    new_token: Optional[str]
    loss_before_step: float
    loss_after_step: float
    loss_delta_vs_prev: float
    loss_delta_vs_base: float
    went_down_vs_prev: bool
    went_down_vs_base: bool


@dataclass
class IterativeModeResult:
    mode: str
    final_loss: float
    monotonic_nonincreasing_loss: bool
    always_down_vs_base_loss: bool
    steps: list[IterativeStepRecord] = field(default_factory=list)


@dataclass
class TopEventRecord:
    rank: int
    loss: float
    batch_index: int
    example_index: int
    position: int
    target_token_id: int
    target_token: Optional[str]


@dataclass
class IterativeContextSwapResult:
    enabled: bool
    scope: str
    mode: str
    replacement_token_id: int
    alternate_replacement_token_id: int
    top_event: TopEventRecord
    context_positions: list[int] = field(default_factory=list)
    base_loss: float = 0.0
    final_loss: float = 0.0
    monotonic_nonincreasing_loss: bool = False
    always_down_vs_base_loss: bool = False
    steps: list[IterativeStepRecord] = field(default_factory=list)
    modes: dict[str, IterativeModeResult] = field(default_factory=dict)


@dataclass
class AnalysisSettingsRecord:
    top_batches: int
    top_token_events: int
    context_window: int
    top_logits_k: int
    tokenizer_name: str
    save_full_logits: bool
    full_logits_dtype: str
    iterative_context_swap_enabled: bool
    iterative_context_swap_token_id: int
    iterative_context_swap_alternate_token_id: int
    iterative_context_swap_scope: str
    iterative_context_swap_mode: str
    swap_relative_positions: list[int] = field(default_factory=list)
    swap_replacement_token_ids: list[int] = field(default_factory=list)


@dataclass
class AnalysisReport:
    run_name: str
    checkpoint_step: int
    checkpoint_dir: str
    split: str
    num_batches: int
    analysis: AnalysisSettingsRecord
    top_batches_by_mean_loss: list[TopBatchRecord] = field(default_factory=list)
    top_token_events: list[TokenEventRecord] = field(default_factory=list)
    iterative_context_swap_top_event: Optional[IterativeContextSwapResult] = None
