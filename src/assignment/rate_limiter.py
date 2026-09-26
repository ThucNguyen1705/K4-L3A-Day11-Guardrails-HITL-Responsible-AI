"""
Assignment 11 — Rate Limiter.

Sliding-window, per-user rate limiting. Blocks abuse that other
guardrail layers do not address (flooding / cost attacks).
"""
from __future__ import annotations

from collections import defaultdict, deque
import time

from google.adk.plugins import base_plugin
from google.genai import types


class RateLimitPlugin(base_plugin.BasePlugin):
    """Block users who exceed max_requests within window_seconds."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        super().__init__(name="rate_limiter")
        if max_requests < 1 or window_seconds < 1:
            raise ValueError("max_requests and window_seconds must be >= 1")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.user_windows: dict[str, deque] = defaultdict(deque)
        self.blocked_count = 0
        self.total_count = 0

    def _block_response(self, message: str) -> types.Content:
        return types.Content(
            role="model",
            parts=[types.Part.from_text(text=message)],
        )

    def remaining(self, user_id: str, now: float | None = None) -> int:
        """Requests the user may still send in the current window."""
        now = time.time() if now is None else now
        window = self.user_windows[user_id]
        self._evict(window, now)
        return max(0, self.max_requests - len(window))

    def _evict(self, window: deque, now: float) -> None:
        cutoff = now - self.window_seconds
        while window and window[0] <= cutoff:
            window.popleft()

    async def on_user_message_callback(self, *, invocation_context, user_message):
        """Return Content to block, or None to allow."""
        self.total_count += 1
        user_id = getattr(invocation_context, "user_id", None) or "anonymous"
        now = time.time()
        window = self.user_windows[user_id]

        # 1. Drop timestamps that have slid out of the window
        self._evict(window, now)

        # 2. Over the limit → block without calling the LLM.
        #    Blocked attempts are not recorded, so a flooding client cannot
        #    push its own unblock time further into the future forever.
        if len(window) >= self.max_requests:
            wait = max(1.0, self.window_seconds - (now - window[0]))
            self.blocked_count += 1
            return self._block_response(
                f"Rate limit exceeded: max {self.max_requests} requests per "
                f"{self.window_seconds}s. Try again in {wait:.0f}s."
            )

        # 3. Under the limit → record and let it through
        window.append(now)
        return None
