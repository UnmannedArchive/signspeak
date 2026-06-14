"""Tests for the phase-2 sentence layer (no API calls)."""
from asl.infer_live import SentenceBuilder
from asl.translator import Translator


def test_sentence_builder_dedups_and_appends_on_change():
    b = SentenceBuilder(pause_s=2.0)
    b.note_commit("WANT", now=0.0)
    b.note_commit("WANT", now=0.1)   # same word held -> not re-appended
    b.note_commit("COFFEE", now=0.2)
    b.note_commit("WANT", now=0.3)   # changed back -> appended again
    assert b.glosses == ["WANT", "COFFEE", "WANT"]


def test_sentence_builder_finalize_timing():
    b = SentenceBuilder(pause_s=2.0)
    b.note_commit("HELLO", now=10.0)
    assert not b.should_finalize(now=11.0)        # only 1s since last sign
    assert b.should_finalize(now=12.5)            # pause elapsed
    b.finalized = True
    assert not b.should_finalize(now=20.0)        # already finalized
    b.clear()
    assert b.glosses == [] and not b.should_finalize(now=99.0)


def test_translator_offline_fallback_formats_glosses():
    # _fallback is what runs with no API key / on any API error.
    assert Translator._fallback(["want", "coffee"]) == "Want coffee"
    assert Translator._fallback(["I_LOVE_YOU"]) == "I love you"
    assert Translator._fallback([]) == ""
