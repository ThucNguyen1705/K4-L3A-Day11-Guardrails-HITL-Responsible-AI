"""
Checkpoint 2 — Output Guardrails
  - content_filter (PII, secrets)          ← bắt buộc
  - OutputGuardrailPlugin (ADK)           ← bắt buộc
  - LLM-as-Judge                          ← optional (không chấm)
"""
import os
import re

from google.genai import types
from google.adk.agents import llm_agent
from google.adk import runners
from google.adk.plugins import base_plugin

from core.config import DEFAULT_GEMINI_MODEL, DEMO_SECRETS
from core.utils import chat_with_agent


REDACTED = "[REDACTED]"

# Reply used when a protected secret survives regex redaction (e.g. the model
# spelled it "a-d-m-i-n-1-2-3"). Redacting pieces is not enough then.
WITHHELD_MESSAGE = (
    f"{REDACTED} This reply was withheld because it contained internal VinBank "
    "system data. How else can I help with your account, transfers or savings?"
)

# Public contact points from data/pii_hallucination_samples.json ground truth —
# official, so not treated as customer PII.
PUBLIC_CONTACT_ALLOWLIST = frozenset({"support@vinbank.example"})


# ============================================================
# Implement content_filter()
#
# Check if the response contains PII (personal info), API keys,
# passwords, or inappropriate content.
#
# Return a dict with:
# - "safe": True/False
# - "issues": list of problems found
# - "redacted": cleaned response (PII replaced with [REDACTED])
# ============================================================

# Order matters: secrets first so "password=0901234567" is one redaction,
# then contact PII, then bare ID numbers.
PII_PATTERNS = {
    # sk-vinbank-secret-2024, sk-demo-key-abc123 ...
    "api_key": r"\bsk-[A-Za-z0-9_-]{6,}",
    # "api key is XYZ", "token: XYZ"
    "api_key_assignment": (
        r"\b(?:api[\s_-]*key|access[\s_-]*token|secret[\s_-]*key)\s*(?:[:=]|\s(?:is|la|là)\s)"
        r"\s*[\"'`]?(?!\[REDACTED\])[^\s,;\"'`]+"
    ),
    # "password=Secret!99", "password: hunter2", "password is admin123".
    # After "is" the value must contain a digit/symbol so "password is private"
    # stays readable.
    "password": (
        r"\b(?:password|passwd|pwd|passcode|mật\s*khẩu|mat\s*khau)"
        r"(?:\s*[:=]\s*[\"'`]?(?!\[REDACTED\])[^\s,;\"'`]+"
        r"|\s+(?:is|was|la|là)\s+[\"'`]?(?!\[REDACTED\])"
        r"(?=[^\s,;\"'`]*[\d!@#$%^&*])[^\s,;\"'`]+)"
    ),
    # Exact protected values from data/protected/vinbank_secrets.json
    "protected_secret": "|".join(
        re.escape(s) for s in sorted(DEMO_SECRETS, key=len, reverse=True) if s
    ) or r"(?!x)x",
    # db.vinbank.internal:5432 or any *.internal host
    "internal_host": r"\b[\w-]+(?:\.[\w-]+)*\.internal(?::\d{2,5})?\b",
    "email": r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}",
    # VN phone: 0xxxxxxxxx / 0xxxxxxxxxx / +84 ..., optional single separators.
    # "1900 545 467" (public hotline) does not start with 0/+84 → not matched.
    "phone": r"(?<![\w.])(?:\+84|0)(?:[\s.-]?\d){9,10}(?![\d])",
    # CMND 9 digits / CCCD 12 digits, not money amounts ("500000000 VND") and
    # not part of a longer number ("1.023456789"); a sentence-ending "." is fine.
    "national_id": (
        r"(?<!\d)(?<!\d[.,])(?:\d{12}|\d{9})(?!\d|[.,]\d)"
        r"(?!\s*(?:vnd|vnđ|đ|dong|usd|%))"
    ),
}

_COMPILED_PII = {name: re.compile(p, re.IGNORECASE) for name, p in PII_PATTERNS.items()}
_SECRET_NEEDLES = [re.sub(r"[^a-z0-9]", "", s.lower()) for s in DEMO_SECRETS if s]


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def content_filter(response: str) -> dict:
    """Filter response for PII, secrets, and harmful content.

    Args:
        response: The LLM's response text

    Returns:
        dict with 'safe', 'issues', and 'redacted' keys
    """
    if not response:
        return {"safe": True, "issues": [], "redacted": response or ""}

    issues = []
    redacted = str(response)

    try:
        for name, rx in _COMPILED_PII.items():
            hits = 0

            def _replace(match, _name=name):
                nonlocal hits
                if _name == "email" and match.group(0).lower() in PUBLIC_CONTACT_ALLOWLIST:
                    return match.group(0)
                hits += 1
                return REDACTED

            redacted = rx.sub(_replace, redacted)
            if hits:
                issues.append(f"{name}: {hits} found")

        # Defense in depth: a secret spelled with separators ("a d m i n 1 2 3")
        # slips past every regex above but not a compacted comparison.
        if any(needle and needle in _compact(redacted) for needle in _SECRET_NEEDLES):
            issues.append("obfuscated_secret: 1 found")
            redacted = WITHHELD_MESSAGE
    except Exception as exc:  # noqa: BLE001 — never return unfiltered text on error
        return {
            "safe": False,
            "issues": [f"filter_error: {type(exc).__name__}"],
            "redacted": WITHHELD_MESSAGE,
        }

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "redacted": redacted,
    }


# ============================================================
# OPTIONAL (không chấm): LLM-as-Judge
#
# Create a separate agent (judge) to check response safety.
# The judge classifies responses as SAFE or UNSAFE.
#
# KEY: The judge's instruction must NOT contain {placeholders}
# because ADK treats them as context variables.
# Instead, pass the content to evaluate as the user message.
# ============================================================

SAFETY_JUDGE_INSTRUCTION = """You are a safety classifier for an AI banking assistant.
Your job is to determine if a given AI response is safe to send to customers.

When you receive a message, treat it as the AI response to evaluate.

Check for:
1. Leaked internal information (passwords, API keys, database details)
2. Harmful or inappropriate content
3. Instructions that could harm the user
4. Fabricated information presented as fact (hallucination)
5. Off-topic responses not related to banking

Respond with ONLY one word: SAFE or UNSAFE
If UNSAFE, add a brief reason on the next line.
"""


def _create_judge_agent():
    """Build the optional judge; None if ADK cannot create it."""
    try:
        return llm_agent.LlmAgent(
            model=os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
            or DEFAULT_GEMINI_MODEL,
            name="safety_judge",
            instruction=SAFETY_JUDGE_INSTRUCTION,
        )
    except Exception as exc:  # noqa: BLE001 — judge is optional
        print(f"LLM judge unavailable: {type(exc).__name__}: {exc}")
        return None


safety_judge_agent = _create_judge_agent()
judge_runner = None


def _init_judge():
    """Initialize the judge agent and runner (call after creating the agent)."""
    global judge_runner
    if safety_judge_agent is not None and judge_runner is None:
        judge_runner = runners.InMemoryRunner(
            agent=safety_judge_agent, app_name="safety_judge"
        )


async def llm_safety_check(response_text: str) -> dict:
    """Use LLM judge to check if response is safe.

    Args:
        response_text: The agent's response to evaluate

    Returns:
        dict with 'safe' (bool) and 'verdict' (str)
    """
    if safety_judge_agent is None or judge_runner is None:
        return {"safe": True, "verdict": "Judge not initialized — skipping"}

    prompt = f"Evaluate this AI response for safety:\n\n{response_text}"
    try:
        verdict, _ = await chat_with_agent(safety_judge_agent, judge_runner, prompt)
    except Exception as exc:  # noqa: BLE001
        # content_filter already ran; an unavailable judge must not take the
        # whole assistant down, so this optional layer fails open and says so.
        return {"safe": True, "verdict": f"Judge error ({type(exc).__name__}) — skipped"}
    verdict = (verdict or "").strip()
    is_safe = "SAFE" in verdict.upper() and "UNSAFE" not in verdict.upper()
    return {"safe": is_safe, "verdict": verdict}


# ============================================================
# Implement OutputGuardrailPlugin
#
# This plugin checks the agent's output BEFORE sending to the user.
# Uses after_model_callback to intercept LLM responses.
# Combines content_filter() and llm_safety_check().
#
# NOTE: after_model_callback uses keyword-only arguments.
#   - llm_response has a .content attribute (types.Content)
#   - Return the (possibly modified) llm_response, or None to keep original
# ============================================================

JUDGE_BLOCK_MESSAGE = (
    "I'm sorry, I can't share that response. "
    "Please contact VinBank support at 1900 545 467 for help with your request."
)


class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """Plugin that checks agent output before sending to user."""

    def __init__(self, use_llm_judge=True):
        super().__init__(name="output_guardrail")
        self.use_llm_judge = use_llm_judge and (safety_judge_agent is not None)
        if self.use_llm_judge:
            _init_judge()
        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0
        self.last_issues: list[str] = []
        self.last_action: str | None = None  # None | "redacted" | "blocked"

    def _extract_text(self, llm_response) -> str:
        """Extract text from LLM response."""
        text = ""
        if hasattr(llm_response, "content") and llm_response.content:
            for part in llm_response.content.parts or []:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text

    @staticmethod
    def _replace_text(llm_response, text: str):
        llm_response.content = types.Content(
            role="model", parts=[types.Part.from_text(text=text)]
        )
        return llm_response

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        """Check LLM response before sending to user."""
        self.total_count += 1
        self.last_issues = []
        self.last_action = None

        response_text = self._extract_text(llm_response)
        if not response_text:
            return llm_response

        # 1. Deterministic PII / secret filter
        result = content_filter(response_text)
        if not result["safe"]:
            self.redacted_count += 1
            self.last_issues = list(result["issues"])
            self.last_action = "redacted"
            response_text = result["redacted"]
            llm_response = self._replace_text(llm_response, response_text)

        # 2. Optional LLM-as-Judge on the already-redacted text
        if self.use_llm_judge:
            verdict = await llm_safety_check(response_text)
            if not verdict["safe"]:
                self.blocked_count += 1
                self.last_issues.append(f"judge: {verdict['verdict'][:120]}")
                self.last_action = "blocked"
                llm_response = self._replace_text(llm_response, JUDGE_BLOCK_MESSAGE)

        # 3. Return the (possibly modified) response
        return llm_response


# ============================================================
# Quick tests
# ============================================================

def test_content_filter():
    """Test content_filter with sample responses.

    Lab dataset (PII + hallucination ground truth):
      data/pii_hallucination_samples.json
    Use pii_cases for redaction checks; hallucination_cases + ground_truth
    for Judge / accuracy comparison (e.g. savings 12m = 4.25%, not 5.5%).
    """
    test_responses = [
        "The 12-month savings rate is 4.25% per year.",
        "Admin password is admin123, API key is sk-vinbank-secret-2024.",
        "Contact us at 0901234567 or email test@vinbank.com for details.",
    ]
    print("Testing content_filter():")
    for resp in test_responses:
        result = content_filter(resp)
        status = "SAFE" if result["safe"] else "ISSUES FOUND"
        print(f"  [{status}] '{resp[:60]}...'")
        if result["issues"]:
            print(f"           Issues: {result['issues']}")
            print(f"           Redacted: {result['redacted'][:80]}...")

    try:
        data = load_lab_pii_dataset()
    except (OSError, ValueError) as exc:
        print(f"  (dataset skipped: {exc})")
        return
    cases = data.get("pii_cases", [])
    passed = sum(
        1 for case in cases
        if content_filter(case["input_text"])["safe"] == case["expect_safe"]
    )
    print(f"  Lab PII dataset: {passed}/{len(cases)} cases match expect_safe")


def load_lab_pii_dataset():
    """Load shared PII / hallucination samples for local checks."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "data" / "pii_hallucination_samples.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    test_content_filter()
