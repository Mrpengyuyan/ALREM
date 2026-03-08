"""Shared naming/partition rules for SPARQL experiment runs."""

from __future__ import annotations

import re
from typing import Any, Set

ALLOWED_RESULT_PARTITIONS: Set[str] = {
    "unified_codechain",
    "external_reproduced",
    "external_reported",
}

PROTOCOL_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]*:v[0-9]+[a-z0-9._-]*$")
RUN_ID_PATTERN = re.compile(
    r"^[a-z0-9][a-z0-9._-]*__(adapter|icl_zero|icl_fewshot)__[a-z0-9][a-z0-9._-]*__s[0-9]+$"
)


def _normalize_token(value: Any, *, field_name: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-._")
    if not text:
        raise ValueError(f"{field_name} is empty after normalization.")
    return text


def validate_protocol_id(protocol_id: str) -> str:
    value = str(protocol_id or "").strip()
    if not value:
        raise ValueError("protocol_id must not be empty.")
    if not PROTOCOL_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "protocol_id must match '<name>:v<version>' naming rule, "
            f"got: {value}"
        )
    return value


def build_run_id(
    *,
    run_name: str,
    mode: str,
    protocol_id: str,
    seed: int,
) -> str:
    run_name_token = _normalize_token(run_name, field_name="run_name")
    mode_token = _normalize_token(mode, field_name="mode")
    protocol_tag = _normalize_token(protocol_id.replace(":", "_"), field_name="protocol_id")
    return f"{run_name_token}__{mode_token}__{protocol_tag}__s{int(seed)}"


def validate_run_id(run_id: str) -> str:
    value = str(run_id or "").strip()
    if not value:
        raise ValueError("run_id must not be empty.")
    if not RUN_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "run_id must follow '<run_name>__<mode>__<protocol_id_tag>__s<seed>' naming rule, "
            f"got: {value}"
        )
    return value


def validate_result_partition(
    value: Any,
    *,
    strict_schema: bool,
    default_value: str = "unified_codechain",
) -> str:
    text = str(value or "").strip().lower()
    if not text:
        if strict_schema:
            raise ValueError("result_partition is required when strict_schema=true.")
        return default_value
    if text not in ALLOWED_RESULT_PARTITIONS:
        raise ValueError(
            f"Unsupported result_partition: {text}. "
            f"Allowed={sorted(ALLOWED_RESULT_PARTITIONS)}"
        )
    return text
