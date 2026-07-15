"""Stage C — LLM-as-judge on the borderline band only.

Reaches ~10-20% of ad-eligible traffic (p in [tau_b, tau_high)), which
bounds cost.
"""

from __future__ import annotations

from ..config import AdGateConfig
from ..prompts import JUDGE_PROMPT
from ..schemas import Ad, Message


class LLMJudge:
    def __init__(self, config: AdGateConfig) -> None:
        self.config = config

    def judge(self, messages: list[Message], ad: Ad) -> tuple[bool, str]:
        """(verdict, reason). Precision-first: any failure (API error,
        unparseable reply) counts as a rejection — never show on doubt.

        IMPROVEMENT (jury research): a 3-vote panel of diverse model
        families beats a single judge; single call keeps cost minimal.
        """
        import json

        import litellm  # deferred: only needed when use_judge is on

        convo = "\n".join(f"{m.role}: {m.content}" for m in messages[-6:])
        ad_text = f"[{ad.brand}] {ad.title} — {ad.body} (CTA: {ad.cta})"
        try:
            resp = litellm.completion(
                model=self.config.judge_model,
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": f"CONVERSATION:\n{convo}\n\nAD:\n{ad_text}"},
                ],
                temperature=0,
                max_tokens=120,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            return bool(data["show"]), str(data.get("reason", ""))
        except Exception as e:  # noqa: BLE001 — fail closed by design
            return False, f"judge error ({type(e).__name__}): fail toward abstention"
