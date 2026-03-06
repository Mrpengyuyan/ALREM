import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger("alrem.sparql_executor")


class SPARQLCache:
    def __init__(
        self,
        cache_dir: str,
        endpoint: str = "https://query.wikidata.org/sparql",
        min_interval_sec: float = 1.5,
        timeout_sec: int = 60,
        force_offline: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.endpoint = endpoint
        self.min_interval_sec = max(0.0, float(min_interval_sec))
        self.timeout_sec = int(timeout_sec)
        self.force_offline = force_offline
        self._last_remote_ts = 0.0

    def _cache_path(self, sparql_query: str) -> Path:
        cache_key = hashlib.md5(sparql_query.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{cache_key}.json"

    def _save_cache(self, path: Path, payload: Dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def _load_cache(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _normalize_binding_cell(self, cell: Dict[str, Any]) -> str:
        value = str(cell.get("value", ""))
        datatype = str(cell.get("datatype", ""))
        lang = str(cell.get("xml:lang", ""))
        parts = [value]
        if datatype:
            parts.append(f"^^{datatype}")
        if lang:
            parts.append(f"@{lang}")
        return "".join(parts)

    def _normalize_answers(self, raw_result: Dict[str, Any]) -> List[str]:
        if "boolean" in raw_result:
            return [str(raw_result["boolean"]).lower()]

        bindings = raw_result.get("results", {}).get("bindings", [])
        if not isinstance(bindings, list):
            return []

        rows = set()
        for binding in bindings:
            if not isinstance(binding, dict):
                continue
            row_parts = []
            for var_name in sorted(binding.keys()):
                cell = binding[var_name]
                if isinstance(cell, dict):
                    cell_text = self._normalize_binding_cell(cell)
                else:
                    cell_text = str(cell)
                row_parts.append(f"{var_name}={cell_text}")
            rows.add("|".join(row_parts))
        return sorted(rows)

    def _respect_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_remote_ts
        remaining = self.min_interval_sec - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _remote_query(self, sparql_query: str) -> Dict[str, Any]:
        try:
            from SPARQLWrapper import JSON, SPARQLWrapper
        except Exception as exc:
            raise ImportError(
                "SPARQLWrapper is required for remote SPARQL execution. "
                "Install it via requirements.txt or run in offline cache mode."
            ) from exc
        self._respect_rate_limit()
        wrapper = SPARQLWrapper(self.endpoint, agent="ALREM-SPARQLCache/1.0")
        wrapper.setQuery(sparql_query)
        wrapper.setReturnFormat(JSON)
        if hasattr(wrapper, "setTimeout"):
            wrapper.setTimeout(self.timeout_sec)
        try:
            raw_result = wrapper.query().convert()
        finally:
            self._last_remote_ts = time.monotonic()
        if not isinstance(raw_result, dict):
            raise RuntimeError("SPARQL endpoint returned non-JSON or invalid JSON result.")
        return raw_result

    def execute(self, sparql_query: str, offline_only: bool = False) -> Dict[str, Any]:
        query_text = sparql_query.strip()
        if not query_text:
            raise ValueError("sparql_query must be non-empty.")

        cache_path = self._cache_path(query_text)
        cached = self._load_cache(cache_path)
        if cached is not None:
            if cached.get("ok", True):
                cached["from_cache"] = True
                return cached
            LOGGER.warning(
                "Ignoring cached failed SPARQL result for key=%s; will retry.",
                cache_path.name,
            )

        effective_offline = offline_only or self.force_offline
        if effective_offline:
            raise FileNotFoundError(
                "SPARQL cache miss in offline mode. "
                f"Missing cache file: {cache_path}. "
                "Run once with offline_only=False (and network access) to pre-cache this query."
            )

        payload: Dict[str, Any]
        try:
            raw_result = self._remote_query(query_text)
            payload = {
                "query": query_text,
                "ok": True,
                "status": "success",
                "error": "",
                "endpoint": self.endpoint,
                "from_cache": False,
                "normalized_answers": self._normalize_answers(raw_result),
                "raw": raw_result,
                "timestamp": int(time.time()),
            }
        except Exception as exc:
            payload = {
                "query": query_text,
                "ok": False,
                "status": "error",
                "error": str(exc),
                "endpoint": self.endpoint,
                "from_cache": False,
                "normalized_answers": [],
                "raw": {},
                "timestamp": int(time.time()),
            }
        if payload.get("ok", False):
            self._save_cache(cache_path, payload)
        else:
            LOGGER.warning("SPARQL remote query failed and will not be cached: %s", payload.get("error", ""))
        return payload

    def pre_cache_gold(self, qald_data: List[Dict[str, Any]]) -> None:
        queries = []
        seen = set()
        for item in qald_data:
            query = str(item.get("sparql", "")).strip()
            if not query or query in seen:
                continue
            seen.add(query)
            queries.append(query)

        LOGGER.info("Pre-caching %d unique SPARQL queries.", len(queries))
        success = 0
        failed = 0
        for idx, query in enumerate(queries, start=1):
            result = self.execute(query, offline_only=False)
            if result.get("ok", False):
                success += 1
            else:
                failed += 1
                LOGGER.warning("SPARQL pre-cache failed (%d/%d): %s", idx, len(queries), result.get("error", ""))
        LOGGER.info("Pre-cache completed: success=%d failed=%d", success, failed)
