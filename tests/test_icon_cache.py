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


def test_rendering_an_icon_is_not_expensive_any_more():
    """The halo used to be grown at the supersampled size.

    A 41-pixel kernel over 512x512 cost 609ms of a 615ms render, for a halo
    that is five pixels wide once reduced to 64. Downscaling first makes it
    0.3ms. The budget is generous - this is guarding against a return to
    over a second, not policing milliseconds.
    """
    daemon_module._icon_cache.clear()
    started = time.perf_counter()
    daemon_module.prerender_icons()
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert elapsed_ms < 250, f"rendering both icons took {elapsed_ms:.0f}ms"


def test_the_icon_still_has_its_halo():
    """The halo is what makes a light glyph read on a light panel."""
    icon = daemon_module._cached_icon(daemon_module.ICON_IDLE)
    alpha = icon.getchannel("A")
    glyph = icon.convert("RGBA")

    # Somewhere there must be pixels that are dark but not transparent:
    # that is the halo, sitting just outside the bright glyph.
    haloed = [
        (x, y) for x in range(icon.width) for y in range(icon.height)
        if alpha.getpixel((x, y)) > 20 and sum(glyph.getpixel((x, y))[:3]) < 120
    ]
    assert haloed, "no dark semi-opaque pixels; the halo is gone"


def test_the_icon_is_the_size_the_tray_expects():
    icon = daemon_module._cached_icon(daemon_module.ICON_IDLE)
    assert icon.size == (daemon_module.ICON_SIZE, daemon_module.ICON_SIZE)
    assert icon.mode == "RGBA"
