"""Phase 2: turn a sequence of recognized ASL glosses into fluent English.

Uses Claude (Anthropic SDK) when ANTHROPIC_API_KEY is set; otherwise falls back
to a plain join of the glosses so the live demo still runs fully offline.

The recognizer emits ASL *glosses* (uppercase word labels) in signing order. ASL
grammar isn't English word order — it drops articles and the copula, fronts the
topic, and marks tense separately — so "ME WANT COFFEE" should come out as
"I'd like a coffee," not "Me want coffee." That reshaping is what the model does.
"""
from __future__ import annotations

import os

from . import config as C

SYSTEM_PROMPT = (
    "You translate sequences of American Sign Language (ASL) glosses into natural "
    "English. ASL grammar differs from English: it commonly drops articles and the "
    "copula, fronts the topic, and marks tense and negation separately. You are "
    "given the glosses a signer produced, in order, as uppercase labels. Return the "
    "single most natural English sentence they most likely meant. Reply with ONLY "
    "that sentence — no quotes, no explanation, no alternatives."
)


class Translator:
    """Glosses -> one English sentence. Safe to construct with no API key."""

    def __init__(self, model: str = C.LLM_MODEL):
        self.model = model
        self._client = None
        self.available = False
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import anthropic

                self._client = anthropic.Anthropic()
                self.available = True
            except Exception:
                # Missing package or bad credentials — degrade to the fallback.
                self._client = None
                self.available = False

    def to_sentence(self, glosses: list[str]) -> str:
        if not glosses:
            return ""
        if not self.available:
            return self._fallback(glosses)
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=128,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": " ".join(g.upper() for g in glosses)}
                ],
            )
            text = next((b.text for b in resp.content if b.type == "text"), "").strip()
            return text or self._fallback(glosses)
        except Exception:
            # Network/rate-limit/etc. — never let the demo crash over a sentence.
            return self._fallback(glosses)

    @staticmethod
    def _fallback(glosses: list[str]) -> str:
        words = " ".join(g.replace("_", " ").lower() for g in glosses).strip()
        return words[:1].upper() + words[1:] if words else ""
