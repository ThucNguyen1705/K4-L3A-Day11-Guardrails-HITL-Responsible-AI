"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from core.config import DEMO_SECRETS

# Longest first so "db.vinbank.internal:5432" is masked before "db.vinbank.internal".
_SECRET_RX = re.compile(
    "|".join(re.escape(s) for s in sorted(DEMO_SECRETS, key=len, reverse=True) if s)
    or r"(?!x)x",
    re.IGNORECASE,
)


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


def mask_secrets(text: str | None) -> str:
    """Mask protected lab secrets so the audit log itself cannot leak them."""
    return _SECRET_RX.sub("[SECRET]", text or "")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        # request_id -> pending entry (input side, waiting for its output)
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None) -> str:
        """Store input + start timestamp keyed by request_id; return the request_id."""
        request_id = request_id or f"req-{uuid.uuid4().hex[:12]}"
        self._open[request_id] = {
            "request_id": request_id,
            "user_id": user_id,
            "input": mask_secrets(text),
            "input_chars": len(text or ""),
            "started_at": utc_now_iso(),
            "_t0": time.perf_counter(),
        }
        return request_id

    def _find_open(self, user_id: str, request_id: str | None) -> tuple[str | None, dict | None]:
        if request_id and request_id in self._open:
            return request_id, self._open.pop(request_id)
        # No id given: close the most recent open request of this user.
        for rid in reversed(list(self._open)):
            if self._open[rid]["user_id"] == user_id:
                return rid, self._open.pop(rid)
        return request_id, None

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
        **extra,
    ) -> dict:
        """Store output, layer decision, latency; append to self.logs."""
        rid, entry = self._find_open(user_id, request_id)
        now_iso = utc_now_iso()
        if entry is None:
            # Output without a matching input is still worth keeping.
            entry = {
                "request_id": rid or f"req-{uuid.uuid4().hex[:12]}",
                "user_id": user_id,
                "input": None,
                "input_chars": 0,
                "started_at": now_iso,
                "_t0": None,
            }

        t0 = entry.pop("_t0", None)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2) if t0 is not None else None
        record = {
            **entry,
            "output": mask_secrets(text),
            "blocked": bool(blocked),
            "layer": layer,
            "latency_ms": latency_ms,
            "finished_at": now_iso,
        }
        for key, value in extra.items():
            record[key] = mask_secrets(value) if isinstance(value, str) else value
        self.logs.append(record)
        return record

    def export_json(self, filepath: str | None = None) -> str:
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            raise OSError(f"Cannot write audit log to {path}: {exc}") from exc
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
