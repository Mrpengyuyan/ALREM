from typing import Dict, Iterable, Optional, Tuple

import torch

from .lora_utils import (
    collect_target_modules,
    estimate_lora_params_rank_pattern,
    estimate_lora_params_uniform,
    infer_layer_index,
)


def infer_num_layers(model: torch.nn.Module) -> int:
    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "num_hidden_layers", None) is not None:
        return int(cfg.num_hidden_layers)
    max_idx = -1
    for name, _ in model.named_modules():
        idx = infer_layer_index(name)
        if idx is not None:
            max_idx = max(max_idx, idx)
    if max_idx >= 0:
        return max_idx + 1
    raise ValueError("Unable to infer number of layers from model.")


def compute_cut_indices(
    num_layers: int,
    cut_ratio_early: Optional[float],
    cut_ratio_mid: Optional[float],
    early_end: Optional[int],
    mid_end: Optional[int],
) -> Tuple[int, int]:
    if early_end is None:
        if cut_ratio_early is None:
            cut_ratio_early = 0.25
        early_end = int(num_layers * cut_ratio_early)
    if mid_end is None:
        if cut_ratio_mid is None:
            cut_ratio_mid = 0.75
        mid_end = int(num_layers * cut_ratio_mid)
    if early_end < 0 or mid_end < 0:
        raise ValueError("early_end and mid_end must be non-negative.")
    if early_end >= mid_end:
        raise ValueError("early_end must be smaller than mid_end.")
    if mid_end > num_layers:
        raise ValueError("mid_end must be <= num_layers.")
    return early_end, mid_end


def build_rank_pattern(
    model: torch.nn.Module,
    target_modules: Iterable[str],
    r_high: int,
    r_low: int,
    cut_ratio_early: Optional[float] = None,
    cut_ratio_mid: Optional[float] = None,
    early_end: Optional[int] = None,
    mid_end: Optional[int] = None,
) -> Tuple[Dict[str, int], int, int, int]:
    num_layers = infer_num_layers(model)
    early_end, mid_end = compute_cut_indices(
        num_layers, cut_ratio_early, cut_ratio_mid, early_end, mid_end
    )
    rank_pattern: Dict[str, int] = {}
    infos = collect_target_modules(model, target_modules)
    for info in infos:
        if info.layer_idx is None:
            rank = r_high
        elif info.layer_idx < early_end or info.layer_idx >= mid_end:
            rank = r_high
        else:
            rank = r_low
        rank_pattern[info.name] = int(rank)
    return rank_pattern, num_layers, early_end, mid_end


def build_alpha_pattern(
    rank_pattern: Dict[str, int], mode: str, fixed: Optional[int]
) -> Dict[str, int]:
    if mode == "fixed":
        alpha = fixed if fixed is not None else 16
        return {k: int(alpha) for k in rank_pattern}
    return {k: int(2 * v) for k, v in rank_pattern.items()}


def estimate_alrem_params(
    model: torch.nn.Module,
    target_modules: Iterable[str],
    r_high: int,
    r_low: int,
    cut_ratio_early: Optional[float] = None,
    cut_ratio_mid: Optional[float] = None,
    early_end: Optional[int] = None,
    mid_end: Optional[int] = None,
) -> Tuple[int, Dict[str, int], int, int, int]:
    rank_pattern, num_layers, early_end, mid_end = build_rank_pattern(
        model=model,
        target_modules=target_modules,
        r_high=r_high,
        r_low=r_low,
        cut_ratio_early=cut_ratio_early,
        cut_ratio_mid=cut_ratio_mid,
        early_end=early_end,
        mid_end=mid_end,
    )
    target_params = estimate_lora_params_rank_pattern(model, rank_pattern)
    return target_params, rank_pattern, num_layers, early_end, mid_end


def solve_r_match(
    target_params: int,
    model: torch.nn.Module,
    target_modules: Iterable[str],
    r_min: int = 1,
    r_max: int = 128,
) -> Tuple[int, int, float]:
    base_per_rank = estimate_lora_params_uniform(model, target_modules, r=1)
    if base_per_rank == 0:
        return r_min, 0, 1.0
    best_r = r_min
    best_params = base_per_rank * r_min
    best_err = abs(best_params - target_params) / max(target_params, 1)
    for r in range(r_min, r_max + 1):
        params = base_per_rank * r
        err = abs(params - target_params) / max(target_params, 1)
        if err < best_err:
            best_r = r
            best_params = params
            best_err = err
    return best_r, best_params, best_err


def build_module_aware_rank_pattern(
    model: torch.nn.Module,
    attn_modules: Iterable[str],
    mlp_modules: Iterable[str],
    attn_config: Dict[str, Any],
    mlp_config: Dict[str, Any],
) -> Tuple[Dict[str, int], int, int, int]:
    """
    Build rank pattern for ALREM v2 with separate strategies for Attention and MLP.
    Attention: Sandwich (Bottom/Top=High, Middle=Low)
    MLP: Capacity Preserving (Uniform or Sandwich)
    """
    num_layers = infer_num_layers(model)
    
    # 1. Attention Rank Pattern (Sandwich)
    attn_r_high = attn_config.get("r_high", 32)
    attn_r_low = attn_config.get("r_low", 4)
    attn_cut_early = attn_config.get("cut_ratio_early", 0.2)
    attn_cut_mid = attn_config.get("cut_ratio_mid", 0.8)
    
    early_end, mid_end = compute_cut_indices(
        num_layers, attn_cut_early, attn_cut_mid, None, None
    )
    
    rank_pattern: Dict[str, int] = {}
    
    # Process Attention Modules
    attn_infos = collect_target_modules(model, attn_modules)
    for info in attn_infos:
        if info.layer_idx is None:
            rank = attn_r_high
        elif info.layer_idx < early_end or info.layer_idx >= mid_end:
            rank = attn_r_high
        else:
            rank = attn_r_low
        rank_pattern[info.name] = int(rank)
        
    # 2. MLP Rank Pattern (Default: Uniform)
    mlp_r = mlp_config.get("r_uniform", 16)
    mlp_infos = collect_target_modules(model, mlp_modules)
    for info in mlp_infos:
        # If we want a sandwich strategy for MLP too (optional A2), we can add logic here
        # For ALREM v2 Main, it's uniform
        rank_pattern[info.name] = int(mlp_r)
        
    return rank_pattern, num_layers, early_end, mid_end


def estimate_alrem_v2_params(
    model: torch.nn.Module,
    attn_modules: Iterable[str],
    mlp_modules: Iterable[str],
    attn_config: Dict[str, Any],
    mlp_config: Dict[str, Any],
) -> Tuple[int, Dict[str, int], int, int, int]:
    rank_pattern, num_layers, early_end, mid_end = build_module_aware_rank_pattern(
        model=model,
        attn_modules=attn_modules,
        mlp_modules=mlp_modules,
        attn_config=attn_config,
        mlp_config=mlp_config,
    )
    target_params = estimate_lora_params_rank_pattern(model, rank_pattern)
    return target_params, rank_pattern, num_layers, early_end, mid_end
