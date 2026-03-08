# Archive Scope (Non-mainline Paths)

This file records compatibility paths kept in code but outside the current
SPARQL core paper protocol.

## Keep but Non-core

1. ALREM v2 module-aware rank allocation
- Files: `src/alrem_rank_pattern.py`, `src/train_sft.py`
- Status: compatibility only
- Guardrail: requires explicit `enable_legacy_alrem_v2=true` in config
- Reason: current protocol focuses on Uniform / Matched / ALREM-main /
  ALREM-strong / Reverse-sandwich.

2. High-stakes subset construction
- Files: `scripts/prepare_data.py`, `src/entity_filter.py`
- Status: optional side analysis artifact
- Guardrail: disabled by default, enable with `--build-high-stakes-subset`
- Reason: not part of the core primary benchmark matrix.

## Principle

These paths should not be removed blindly, but should be excluded from:
- main result table aggregation
- core protocol validation
- default run scripts unless explicitly requested.
