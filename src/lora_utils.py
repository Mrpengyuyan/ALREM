import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch


_LAYER_PATTERNS = [
    re.compile(r"\.layers\.(\d+)\."),
    re.compile(r"^layers\.(\d+)\."),
    re.compile(r"\.h\.(\d+)\."),
    re.compile(r"^h\.(\d+)\."),
]


@dataclass
class ModuleInfo:
    name: str
    module: torch.nn.Module
    layer_idx: Optional[int]


def infer_layer_index(name: str) -> Optional[int]:
    for pat in _LAYER_PATTERNS:
        m = pat.search(name)
        if m:
            return int(m.group(1))
    return None


def collect_target_modules(
    model: torch.nn.Module, target_modules: Iterable[str]
) -> List[ModuleInfo]:
    target_set = set(target_modules)
    infos: List[ModuleInfo] = []
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf not in target_set:
            continue
        if not hasattr(module, "weight"):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2:
            continue
        infos.append(ModuleInfo(name=name, module=module, layer_idx=infer_layer_index(name)))
    return infos


def lora_params_for_module(module: torch.nn.Module, r: int) -> int:
    if r <= 0:
        return 0
    weight = module.weight
    out_dim, in_dim = weight.shape[0], weight.shape[1]
    return int(r * (in_dim + out_dim))


def estimate_lora_params_uniform(
    model: torch.nn.Module, target_modules: Iterable[str], r: int
) -> int:
    infos = collect_target_modules(model, target_modules)
    return sum(lora_params_for_module(info.module, r) for info in infos)


def estimate_lora_params_rank_pattern(
    model: torch.nn.Module, rank_pattern: Dict[str, int]
) -> int:
    module_map = {name: module for name, module in model.named_modules()}
    total = 0
    for name, r in rank_pattern.items():
        module = module_map.get(name)
        if module is None or not hasattr(module, "weight"):
            continue
        total += lora_params_for_module(module, r)
    return total


def compute_total_lora_params(model: torch.nn.Module, peft_model: torch.nn.Module) -> int:
    total = 0
    for name, param in peft_model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            total += param.numel()
    return total


def compute_lora_params_by_layer(peft_model: torch.nn.Module) -> Dict[str, int]:
    by_layer: Dict[str, int] = {}
    for name, param in peft_model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" not in name:
            continue
        layer_idx = infer_layer_index(name)
        key = "layer_%s" % (layer_idx if layer_idx is not None else "na")
        by_layer[key] = by_layer.get(key, 0) + param.numel()
    return by_layer


def summarize_ranks(rank_pattern: Dict[str, int]) -> Tuple[int, Dict[str, List[int]]]:
    per_layer: Dict[str, List[int]] = {}
    for name, r in rank_pattern.items():
        layer_idx = infer_layer_index(name)
        key = "layer_%s" % (layer_idx if layer_idx is not None else "na")
        per_layer.setdefault(key, []).append(r)
    return len(per_layer), per_layer
