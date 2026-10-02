from __future__ import annotations

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "string"},
                },
                "required": ["name"],
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                    "quote": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["source", "target"],
            },
        },
    },
    "required": ["entities", "relationships"],
}

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "relationship_key": {"type": "string"},
                    "accepted": {"type": "boolean"},
                    "reason": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["relationship_key", "accepted"],
            },
        }
    },
    "required": ["verdicts"],
}

RESOLVE_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "canonical_name": {"type": "string"},
                    "type": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "string"},
                },
                "required": ["canonical_name", "aliases"],
            },
        },
        "rejected_merges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "names": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
                "required": ["names"],
            },
        },
    },
    "required": ["entities"],
}

REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "reports": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "community": {"type": "integer"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "findings": {"type": "array", "items": {"type": "string"}},
                    "evidence_chunk_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["community", "title", "summary"],
            },
        }
    },
    "required": ["reports"],
}

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "used_chunk_ids": {"type": "array", "items": {"type": "string"}},
        "enough_evidence": {"type": "boolean"},
    },
    "required": ["answer", "enough_evidence"],
}
