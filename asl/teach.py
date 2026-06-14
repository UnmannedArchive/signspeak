"""English sentence -> "how to sign it": ASL gloss order + reference clips.

Given a spoken/typed sentence, produce an ordered list of signs to perform. With
an ANTHROPIC_API_KEY we ask Claude to reorder into ASL grammar and describe each
sign; without one we fall back to a simple content-word extraction. Either way,
for each gloss we locate a real reference video — from the WLASL clips already on
disk, or downloaded on demand from the same live sources — so the learner can
watch the actual sign.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.parse

from . import config as C

_DIRECT = {
    "signstock.blob.core.windows.net",
    "media.asldeafined.com",
    "media.spreadthesign.com",
    "s3-us-west-1.amazonaws.com",
}
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# Function words that have no distinct ASL sign in a simple word-for-word pass.
_STOP = {"a", "an", "the", "to", "of", "is", "are", "am", "be", "been",
         "do", "does", "did", "will", "would", "and", "for"}

_index = None


def _load_index() -> dict:
    global _index
    if _index is None:
        if C.WLASL_JSON.exists():
            data = json.load(open(C.WLASL_JSON))
            _index = {e["gloss"]: e["instances"] for e in data}
        else:
            _index = {}
    return _index


def _valid(path) -> bool:
    import cv2

    cap = cv2.VideoCapture(str(path))
    ok, _ = cap.read()
    cap.release()
    return ok


def ensure_clip(gloss: str) -> str | None:
    """Return a video_id with a playable reference clip for `gloss`, or None.

    Checks clips already on disk first, then downloads one from a direct-mp4
    source. Pure-network failures degrade to None (caller fingerspells instead).
    """
    insts = _load_index().get(gloss.lower(), [])
    for i in insts:  # already downloaded?
        p = C.WLASL_VIDEO_DIR / f"{i['video_id']}.mp4"
        if p.exists() and _valid(p):
            return i["video_id"]
    for i in insts:  # fetch from a fast direct source
        if urllib.parse.urlparse(i.get("url", "")).netloc not in _DIRECT:
            continue
        p = C.WLASL_VIDEO_DIR / f"{i['video_id']}.mp4"
        p.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sS", "-m", "25", "-L", "-A", _UA, "-o", str(p), i["url"]],
                       capture_output=True)
        if p.exists() and _valid(p):
            return i["video_id"]
        p.unlink(missing_ok=True)
    return None


def heuristic_glosses(sentence: str) -> list[tuple[str, str]]:
    """No-API fallback: keep content words in order. Returns (gloss, how_to)."""
    words = re.findall(r"[a-zA-Z']+", sentence.lower())
    return [(w, "") for w in words if w not in _STOP]


def _claude_glosses(sentence: str):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic

        client = anthropic.Anthropic()
        prompt = (
            "Translate this English sentence into American Sign Language. "
            "Output ASL glosses (uppercase single English words naming each sign) "
            "in ASL grammatical order, and a one-sentence how-to for each sign "
            "covering handshape, location, and movement. Prefer common single-word "
            "glosses. Return ONLY JSON of the form "
            '{"steps":[{"gloss":"WORD","how_to":"..."}]}.\n\nSentence: ' + sentence
        )
        resp = client.messages.create(
            model=C.LLM_MODEL, max_tokens=700,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
        text = text[text.find("{"): text.rfind("}") + 1]
        steps = json.loads(text)["steps"]
        return [(s["gloss"].strip(), s.get("how_to", "").strip()) for s in steps if s.get("gloss")]
    except Exception:
        return None


def plan(sentence: str) -> dict:
    """Full plan: ASL gloss order + a reference clip (or fingerspell flag) per sign."""
    sentence = (sentence or "").strip()
    pairs = _claude_glosses(sentence) or heuristic_glosses(sentence)
    steps = []
    for gloss, how_to in pairs:
        video_id = ensure_clip(gloss)
        steps.append({
            "gloss": gloss.upper(),
            "video_id": video_id,
            "how_to": how_to,
            "fingerspell": video_id is None,
        })
    return {
        "sentence": sentence,
        "gloss": [s["gloss"] for s in steps],
        "steps": steps,
        "source": "claude" if os.environ.get("ANTHROPIC_API_KEY") else "offline",
    }
