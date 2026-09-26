"""
Assignment 11 — Monitoring & Alerts.

Tracks block rate, rate-limit hits, judge fail rate.
Fires alerts when thresholds are exceeded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def default_metrics_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "metrics.json")


@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class MonitoringAlert:
    """Aggregate counters from pipeline plugins and emit alerts."""

    block_rate_threshold: float = 0.5
    rate_limit_hit_threshold: int = 5
    judge_fail_rate_threshold: float = 0.3
    alerts: list[Alert] = field(default_factory=list)

    # Counters — update these from your pipeline after each request
    total_requests: int = 0
    blocked_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0
    blocked_by_layer: dict[str, int] = field(default_factory=dict)
    redacted_responses: int = 0

    def record_request(
        self,
        *,
        blocked: bool,
        layer: str | None = None,
        redacted: bool = False,
        judge_checked: bool = False,
        judge_failed: bool = False,
    ) -> None:
        """Update counters for one finished request."""
        self.total_requests += 1
        if blocked:
            self.blocked_requests += 1
            key = layer or "unknown"
            self.blocked_by_layer[key] = self.blocked_by_layer.get(key, 0) + 1
        if layer == "rate_limiter":
            self.rate_limit_hits += 1
        if redacted:
            self.redacted_responses += 1
        if judge_checked:
            self.judge_checks += 1
            if judge_failed:
                self.judge_fails += 1

    def check_metrics(self) -> list[Alert]:
        """Compute rates; keep one Alert per metric that exceeds its threshold."""
        snap = self.snapshot()
        candidates = [
            (
                "block_rate",
                snap["block_rate"],
                self.block_rate_threshold,
                snap["block_rate"] > self.block_rate_threshold,
                "Block rate {v:.0%} > {t:.0%}: possible attack wave or over-blocking guardrail.",
            ),
            (
                "rate_limit_hits",
                float(self.rate_limit_hits),
                float(self.rate_limit_hit_threshold),
                self.rate_limit_hits >= self.rate_limit_hit_threshold,
                "{v:.0f} rate-limit hits (threshold {t:.0f}): a client may be flooding the API.",
            ),
            (
                "judge_fail_rate",
                snap["judge_fail_rate"],
                self.judge_fail_rate_threshold,
                self.judge_checks > 0 and snap["judge_fail_rate"] > self.judge_fail_rate_threshold,
                "Judge fail rate {v:.0%} > {t:.0%}: model output quality/safety degraded.",
            ),
        ]

        # Re-evaluating replaces the previous alert for that metric instead of
        # stacking duplicates every time check_metrics() runs.
        current = {a.metric: a for a in self.alerts}
        for metric, value, threshold, fired, template in candidates:
            if fired:
                current[metric] = Alert(
                    metric=metric,
                    value=round(value, 4),
                    threshold=threshold,
                    message=template.format(v=value, t=threshold),
                )
            else:
                current.pop(metric, None)
        self.alerts = list(current.values())
        return self.alerts

    def export_json(self, filepath: str | None = None) -> str:
        """Write metrics + alerts to JSON under repo-root ``outputs/`` by default.
        Use ``filepath or default_metrics_path()`` so running from ``src/`` does not
        create ``src/outputs/``.
        """
        path = Path(filepath or default_metrics_path())
        self.check_metrics()
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "thresholds": {
                "block_rate": self.block_rate_threshold,
                "rate_limit_hits": self.rate_limit_hit_threshold,
                "judge_fail_rate": self.judge_fail_rate_threshold,
            },
            **self.snapshot(),
            "blocked_by_layer": dict(self.blocked_by_layer),
            "redacted_responses": self.redacted_responses,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            raise OSError(f"Cannot write metrics to {path}: {exc}") from exc
        return str(path)

    def snapshot(self) -> dict:
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": block_rate,
            "rate_limit_hits": self.rate_limit_hits,
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": judge_fail_rate,
            "alerts": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "threshold": a.threshold,
                    "message": a.message,
                }
                for a in self.alerts
            ],
        }
