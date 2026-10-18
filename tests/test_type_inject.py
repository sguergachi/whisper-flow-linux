"""Typing has to work without the host's injector daemon.

ydotool can fail silently - on this machine KWin disables its device and
discards every keystroke while ydotool still exits 0 - so the app carries
its own injection device. The keymap is what decides whether the words
survive the trip, so it is pinned here: letters, digits, punctuation, and
the shifted forms, on the US layout the injector assumes.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

type_inject = pytest.importorskip("whisper_flow.type_inject")


def test_letters_and_digits_are_plain_keys():
    from evdev import ecodes

    assert type_inject._codes_for("a") == (ecodes.KEY_A, False)
    assert type_inject._codes_for("Z") == (ecodes.KEY_Z, True)
    assert type_inject._codes_for("7") == (ecodes.KEY_7, False)


def test_punctuation_maps_to_its_key():
    from evdev import ecodes

    assert type_inject._codes_for(" ") == (ecodes.KEY_SPACE, False)
    assert type_inject._codes_for(",") == (ecodes.KEY_COMMA, False)
    assert type_inject._codes_for("/") == (ecodes.KEY_SLASH, False)
    assert type_inject._codes_for("\n") == (ecodes.KEY_ENTER, False)


def test_shifted_characters_use_their_base_key():
    from evdev import ecodes

    # "!" is shift+1, ":" is shift+;, "?" is shift+/
    assert type_inject._codes_for("!") == (ecodes.KEY_1, True)
    assert type_inject._codes_for(":") == (ecodes.KEY_SEMICOLON, True)
    assert type_inject._codes_for("?") == (ecodes.KEY_SLASH, True)
    assert type_inject._codes_for('"') == (ecodes.KEY_APOSTROPHE, True)
    assert type_inject._codes_for("_") == (ecodes.KEY_MINUS, True)


def test_characters_with_no_us_key_are_skipped():
    """Smart quotes and em dashes arrive in real transcripts and have no key."""
    assert type_inject._codes_for("\u2014") is None
    assert type_inject._codes_for("\u2019") is None
    assert type_inject._codes_for("\u00e9") is None
