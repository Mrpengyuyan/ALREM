import logging
import re
from typing import Any, Dict, List, Set

from .sparql_executor import SPARQLCache

LOGGER = logging.getLogger("alrem.entity_filter")

_ENTITY_RE = re.compile(r"\bwd:(Q\d+)\b", flags=re.IGNORECASE)
_ENTITY_URI_RE = re.compile(r"https?://www\.wikidata\.org/entity/(Q\d+)", flags=re.IGNORECASE)
_QID_RE = re.compile(r"Q\d+$", flags=re.IGNORECASE)

# Curated high-risk type anchors (instance of / P31).
_LEGAL_TYPE_IDS = {
    "Q7748",     # law
    "Q49371",    # legislation
    "Q2334719",  # legal case
    "Q41487",    # court
    "Q428148",   # legal norm / regulation-like entity
    "Q131569",   # treaty
}

_MEDICAL_TYPE_IDS = {
    "Q12136",     # disease
    "Q11190",     # medicine
    "Q169872",    # symptom
    "Q796194",    # medical procedure
    "Q30612",     # clinical trial
    "Q28885102",  # pharmaceutical drug
}

_LEGAL_KEYWORDS = (
    "law",
    "legal",
    "regulation",
    "court",
    "treaty",
    "statute",
    "法规",
    "法律",
    "司法",
)

_MEDICAL_KEYWORDS = (
    "medical",
    "medicine",
    "drug",
    "disease",
    "symptom",
    "hospital",
    "clinical",
    "医疗",
    "药",
    "疾病",
)


def _normalize_qid(entity_id: str) -> str:
    value = entity_id.strip()
    if value.lower().startswith("wd:"):
        value = value.split(":", 1)[1]
    value = value.upper()
    if not _QID_RE.match(value):
        return ""
    return value


def _extract_qid_from_value(value: str) -> str:
    text = value.strip()
    uri_match = _ENTITY_URI_RE.search(text)
    if uri_match:
        return uri_match.group(1).upper()
    if _QID_RE.match(text.upper()):
        return text.upper()
    if text.lower().startswith("wd:"):
        return _normalize_qid(text)
    return ""


def extract_entities(sparql: str) -> List[str]:
    if not sparql:
        return []
    found: List[str] = []
    seen: Set[str] = set()
    for match in _ENTITY_RE.findall(sparql):
        qid = match.upper()
        if qid in seen:
            continue
        seen.add(qid)
        found.append(qid)
    for match in _ENTITY_URI_RE.findall(sparql):
        qid = match.upper()
        if qid in seen:
            continue
        seen.add(qid)
        found.append(qid)
    return found


def get_entity_types(ids: List[str], cache: SPARQLCache) -> Dict[str, List[str]]:
    unique_ids: List[str] = []
    seen = set()
    for item in ids:
        qid = _normalize_qid(item)
        if not qid or qid in seen:
            continue
        seen.add(qid)
        unique_ids.append(qid)

    type_map: Dict[str, List[str]] = {}
    for qid in unique_ids:
        query = (
            "SELECT DISTINCT ?type WHERE {\n"
            f"  wd:{qid} wdt:P31 ?type .\n"
            "}"
        )
        result = cache.execute(query)
        if not result.get("ok", False):
            LOGGER.warning("Failed to query types for %s: %s", qid, result.get("error", ""))
            type_map[qid] = []
            continue

        types = set()
        raw = result.get("raw", {})
        bindings = raw.get("results", {}).get("bindings", [])
        if isinstance(bindings, list):
            for binding in bindings:
                if not isinstance(binding, dict):
                    continue
                type_cell = binding.get("type")
                if isinstance(type_cell, dict):
                    type_value = str(type_cell.get("value", "")).strip()
                    type_qid = _extract_qid_from_value(type_value)
                    if type_qid:
                        types.add(type_qid)

        if not types:
            # Fallback to normalized rows if raw bindings are unavailable.
            for row in result.get("normalized_answers", []):
                for token in row.split("|"):
                    if "=" not in token:
                        continue
                    _, value = token.split("=", 1)
                    type_qid = _extract_qid_from_value(value)
                    if type_qid:
                        types.add(type_qid)

        type_map[qid] = sorted(types)
    return type_map


def _contains_keywords(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def filter_high_stakes_subset(data: List[Dict[str, Any]], cache: SPARQLCache) -> List[Dict[str, Any]]:
    if not data:
        return []

    entity_map: Dict[int, List[str]] = {}
    all_entities: List[str] = []
    for idx, sample in enumerate(data):
        entities = extract_entities(str(sample.get("sparql", "")))
        entity_map[idx] = entities
        all_entities.extend(entities)

    type_map = get_entity_types(all_entities, cache)
    subset: List[Dict[str, Any]] = []

    for idx, sample in enumerate(data):
        question = str(sample.get("question", ""))
        entities = entity_map.get(idx, [])
        tags = set()

        if _contains_keywords(question, _LEGAL_KEYWORDS):
            tags.add("legal")
        if _contains_keywords(question, _MEDICAL_KEYWORDS):
            tags.add("medical")

        for entity in entities:
            types = set(type_map.get(entity, []))
            if types & _LEGAL_TYPE_IDS:
                tags.add("legal")
            if types & _MEDICAL_TYPE_IDS:
                tags.add("medical")

        if not tags:
            continue

        enriched = dict(sample)
        enriched["entities"] = entities
        enriched["risk_tags"] = sorted(tags)
        subset.append(enriched)

    LOGGER.info("High-stakes subset size: %d / %d", len(subset), len(data))
    return subset
