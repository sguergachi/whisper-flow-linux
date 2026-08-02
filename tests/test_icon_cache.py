"""The tray icon must not be drawn on the path that starts a recording.

_render_mic_icon supersamples to 512px and runs a 41-pixel MaxFilter over
it: ~615ms measured. It was called on every recording start and every stop,
ahead of opening the microphone, so a fixed picture of a microphone was
delaying the microphone by over a second per dictation.
"""

import time
from unittest.mock import patch

from whisper_flow import daemon as daemon_module


def setup_function():
    daemon_module._icon_cache.clear()


def teardown_function():
    daemon_module._icon_cache.clear()


def test_each_icon_is_drawn_once_however_often_it_is_asked_for():
    with patch.object(daemon_module, "_render_mic_icon",
                      wraps=daemon_module._render_mic_icon) as render:
        for _ in range(5):
            daemon_module._cached_icon(daemon_module.ICON_IDLE)
            daemon_module._cached_icon(daemon_module.ICON_RECORDING)
    assert render.call_count == 2


def test_the_two_states_are_different_pictures():
    idle = daemon_module._cached_icon(daemon_module.ICON_IDLE)
    recording = daemon_module._cached_icon(daemon_module.ICON_RECORDING)
    assert idle is not recording
    assert idle.tobytes() != recording.tobytes()


def test_asking_for_an_icon_again_is_effectively_free():
    daemon_module.prerender_icons()
    started = time.perf_counter()
    for _ in range(100):
        daemon_module._cached_icon(daemon_module.ICON_RECORDING)
    per_call_ms = (time.perf_counter() - started) * 1000 / 100
    # The real render is ~615ms; anything near that means the cache is gone.
    assert per_call_ms < 1.0, f"{per_call_ms:.3f}ms per lookup"


def test_prerendering_fills_the_cache_for_both_states():
    daemon_module.prerender_icons()
    assert daemon_module.ICON_IDLE in daemon_module._icon_cache
    assert daemon_module.ICON_RECORDING in daemon_module._icon_cache


def test_starting_a_recording_does_not_draw_anything(monkeypatch):
    """The specific regression: no render on the hotkey path."""
    daemon_module.prerender_icons()
    with patch.object(daemon_module, "_render_mic_icon") as render:
        daemon_module._cached_icon(daemon_module.ICON_RECORDING)
        daemon_module._cached_icon(daemon_module.ICON_IDLE)
    render.assert_not_called()
