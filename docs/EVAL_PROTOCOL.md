# SPARQL Unified Evaluation Protocol

This document defines the enforced shared protocol for fair comparison across:
- adapter-based runs
- ICL zero-shot runs
- ICL few-shot runs

The canonical protocol file is:
- `configs/sparql_eval_shared.yaml`

## Required Alignment

All runs must align on:
- same `test_data_path`
- same `test_languages` (currently `en,de,es,ru`)
- same decode settings:
  - `max_seq_len`
  - `max_new_tokens`
  - `do_sample`
  - `temperature`
  - `top_p`
- same execution/cache behavior:
  - `cache_dir`
  - `offline_only`

For main-table protocol runs:
- `main_table_protocol=true`
- `strict_schema=true` (hard requirement)
- `result_partition=unified_codechain`

## Predictions Schema

`predictions.jsonl` should include:
- `idx`
- `qid`
- `language`
- `question`
- `gold_sparql`
- `pred_sparql`
- `generation_time_sec`
- `mode` (`adapter` / `icl_zero` / `icl_fewshot`)
- `run_id`
- `protocol_id`

Naming rules:
- `protocol_id`: `<name>:v<version>` (example: `sparql_eval_shared:v1`)
- `run_id`: `<run_name>__<mode>__<protocol_id_tag>__s<seed>`
  - example: `sparql_icl_zero__icl_zero__sparql_eval_shared_v1__s42`

## Metadata Schema

Each run should write `run_metadata.json` with:
- run identity (`run_id`, `run_name`, `mode`)
- model/config metadata
- decode metadata
- protocol metadata (`protocol_name`, `protocol_version`, `protocol_id`)
- result metadata (`result_partition`)
- seed/timestamp

Under `strict_schema=true`, evaluator enforces:
- `predictions.jsonl` + `run_metadata.json` must both be present
- `mode` / `run_id` / `protocol_id` schema checks
- protocol-aligned decode/cache/test-language fields
- `run_id` naming-rule consistency (`run_name` + `mode` + `protocol_id` + `seed`)
- `result_partition` consistency with protocol file

Allowed `result_partition` values:
- `unified_codechain`
- `external_reproduced`
- `external_reported`

Table reporting rule:
- `external_reported` rows must be presented in a separate table section.
- Do not perform same-class significance tests between `external_reported` and `unified_codechain` rows.

## Error Taxonomy

The primary error classes are fixed to:
- `generation_empty`
- `syntax_or_parse_error`
- `execution_error`
- `wrong_answer`

Fine-grained diagnostics should be stored in `error_detail`.
