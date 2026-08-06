"""The tray icon must not be drawn on the path that starts a recording.

icon.tray_icon supersamples to 512px and runs a 41-pixel MaxFilter over
it: ~615ms measured. It was called on every recording start and every stop,
ahead of opening the microphone, so a fixed picture of a microphone was
delaying the microphone by over a second per dictation.
"""

import time
from unittest.mock import patch

from whisper_flow import daemon as daemon_module
from whisper_flow import icon as icon_module


def setup_function():
    daemon_module._icon_cache.clear()


def teardown_function():
    daemon_module._icon_cache.clear()


def test_each_icon_is_drawn_once_however_often_it_is_asked_for():
    with patch.object(icon_module, "tray_icon",
                      wraps=icon_module.tray_icon) as render:
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
    with patch.object(icon_module, "tray_icon") as render:
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
    assert icon.size == (icon_module.ICON_SIZE, icon_module.ICON_SIZE)
    assert icon.mode == "RGBA"


# ------------------------------------------------------- the app icon
def test_the_app_icon_is_the_tray_mark(tmp_path):
    """One drawing, so the exe, the shortcuts and the tray cannot disagree.

    The .ico is generated at package time from this same module rather than
    checked in, which is the only way they stay the same picture.
    """
    from PIL import Image

    from whisper_flow import icon

    path = icon.write_ico(str(tmp_path / "app.ico"))
    with Image.open(path) as image:
        assert image.format == "ICO"
        sizes = {size for size in image.info["sizes"]}
    assert sizes == {(s, s) for s in icon.ICO_SIZES}


def test_every_size_is_drawn_rather_than_resampled(tmp_path):
    """A 16px frame resampled from 256 loses the stem and the base.

    The glyph is mostly thin strokes, so the smallest size is the one that
    has to be drawn at its own resolution to survive at all.
    """
    from whisper_flow import icon

    small = icon.draw_mic(16, icon.APP_COLOR)
    assert small.size == (16, 16)
    # Ink in the bottom third is the stem and the base; a resampled frame
    # washes them out to nothing.
    bottom = small.crop((0, 11, 16, 16)).getchannel("A")
    assert max(bottom.getdata()) > 40, "the stand vanished at 16px"


def test_the_tray_icon_still_carries_its_halo(tmp_path):
    """It is what makes a light glyph readable on a light panel.

    The app icon deliberately drops it - there is no panel behind an .ico -
    so the two must not have been collapsed into one drawing.
    """
    from whisper_flow import icon

    bare = icon.draw_mic(icon.ICON_SIZE, icon.ICON_IDLE)
    tray = icon.tray_icon(icon.ICON_IDLE)
    assert sum(tray.getchannel("A").getdata()) > \
        sum(bare.getchannel("A").getdata()), "no halo around the tray glyph"


def test_a_window_asks_for_the_small_sizes_only(tmp_path):
    """The taskbar and title bar never show anything above 64px.

    This one is drawn while a window is opening rather than at package time,
    and the 256px frame alone is most of what that costs - for a size only
    Explorer ever asks for.
    """
    from PIL import Image

    from whisper_flow import icon

    path = icon.write_ico(str(tmp_path / "window.ico"),
                          sizes=icon.WINDOW_ICO_SIZES)
    with Image.open(path) as image:
        sizes = {size for size in image.info["sizes"]}
    assert sizes == {(s, s) for s in icon.WINDOW_ICO_SIZES}
    assert max(icon.WINDOW_ICO_SIZES) <= 64, (
        "a window icon this large is drawn for nobody")
    # And the small sizes a window does use are all there.
    assert {(16, 16), (32, 32), (48, 48)} <= sizes


def test_the_window_icon_is_cheaper_than_the_full_set(tmp_path):
    """It is drawn on the path between the click and the window appearing.

    A budget rather than a stopwatch: what matters is that a window is not
    paying for the 256px frame, which is most of the cost of the whole set
    and is only ever shown by Explorer. Timed against the full set rather
    than against a fixed number of milliseconds, which would only measure
    how busy the machine running the tests is.
    """
    from whisper_flow import icon

    # PIL loads its PNG codec on the first save, which costs more than either
    # of these and would land entirely on whichever runs first.
    icon.write_ico(str(tmp_path / "warm.ico"), sizes=(16,))

    started = time.perf_counter()
    icon.write_ico(str(tmp_path / "window.ico"), sizes=icon.WINDOW_ICO_SIZES)
    window = time.perf_counter() - started

    started = time.perf_counter()
    icon.write_ico(str(tmp_path / "app.ico"))
    everything = time.perf_counter() - started

    assert window < everything / 2, (
        f"the window icon costs {window*1000:.0f}ms against "
        f"{everything*1000:.0f}ms for the full set; it is not saving anything")


def test_the_build_still_gets_every_size(tmp_path):
    """The default is the executable's, and the executable needs 256."""
    from PIL import Image

    from whisper_flow import icon

    with Image.open(icon.write_ico(str(tmp_path / "app.ico"))) as image:
        assert (256, 256) in image.info["sizes"]
