"""The overlay stays up while the transcript is being worked out.

Letting go of the hotkey used to take the pill down at once, and the words
arrived whenever the transcription finished - a gap with nothing on screen to
say whether anything was happening or whether the dictation had failed. On a
machine transcribing slowly that gap was twenty seconds.

The overlay itself needs GTK 4 and, off Windows, gtk4-layer-shell, so what can
be checked here is the arithmetic behind the animation, the supervisor that
puts the overlay into the state, and the daemon that decides when.
"""

import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from whisper_flow import hud_anim


# ------------------------------------------------------------ the animation
def test_the_sweep_stays_inside_the_bars():
    """A level above 1 draws a bar out through the top of the pill."""
    for tick in range(0, 400):
        levels = hud_anim.sweep(tick * 0.05, 30)
        assert len(levels) == 30
        assert all(0.0 <= level <= 1.0 for level in levels), (
            f"level out of range at t={tick * 0.05}: "
            f"{min(levels)}..{max(levels)}")


def test_the_bump_actually_travels():
    """Otherwise it is a static pattern, which says nothing is happening.

    Sampled inside one pass. Across the wrap the position goes from the last
    bar back to the first, which is the bump continuing rather than stopping,
    and is covered by the repetition and never-empties tests instead.
    """
    period = 1.0 / hud_anim.SWEEP_HZ
    where = []
    for fraction in (0.15, 0.30, 0.45, 0.60, 0.75, 0.90):
        levels = hud_anim.sweep(period * fraction, 30)
        where.append(levels.index(max(levels)))
    assert where == sorted(where), f"the bump went backwards: {where}"
    assert where[0] < where[-1], f"the bump did not move: {where}"
    assert len(set(where)) >= 5, f"the bump barely moved: {where}"


def test_the_sweep_comes_round_again():
    """It runs for as long as the transcription does, so it has to loop.

    Compared by where the bump is rather than by the exact heights: the
    swell underneath has its own, longer period and does not repeat with it.
    """
    period = 1.0 / hud_anim.SWEEP_HZ
    for fraction in (0.0, 0.25, 0.5, 0.75):
        first = hud_anim.sweep(period * fraction, 30)
        later = hud_anim.sweep(period * (fraction + 4), 30)
        assert first.index(max(first)) == later.index(max(later)), (
            f"the sweep did not repeat at {fraction} of a pass")
    # And mid-pass it is unmistakably lit. The bump starts just off the left
    # edge so that it enters rather than appearing, so the ends are dimmer.
    assert max(hud_anim.sweep(period * 0.5, 30)) > 0.7


def test_the_sweep_never_empties_out():
    """A pill of flat bars reads as a hung overlay, not a working one.

    The bump is measured around the ends rather than straight along them, so
    it re-enters at the left as it leaves at the right. Without that there
    is a moment at the end of every pass with nothing lit at all - which is
    the one thing this animation exists not to say. It dims at the crossover,
    with both ends carrying half a bump, and that is as low as it goes.
    """
    peaks = [max(hud_anim.sweep(tick * 0.005, 30)) for tick in range(800)]
    assert min(peaks) > 0.35, (
        f"the sweep faded to {min(peaks):.3f}; the bump does not wrap")
    assert max(peaks) > 0.9


def test_the_spinner_turns_one_way():
    angles = [hud_anim.spinner_angle(t * 0.1) for t in range(10)]
    assert angles == sorted(angles)
    assert angles[0] == 0.0


def test_a_degenerate_bar_count_does_not_explode():
    assert hud_anim.sweep(1.0, 0) == []
    assert len(hud_anim.sweep(1.0, 1)) == 1


# ------------------------------------------------------- the supervisor
@pytest.fixture
def hud(monkeypatch):
    from whisper_flow import hud as hud_module

    overlay = hud_module.HUD()
    monkeypatch.setattr(hud_module, "RESIDENT", True)
    sent = []
    monkeypatch.setattr(overlay, "_command",
                        lambda line: (sent.append(line), True)[1])
    return overlay, sent


def test_processing_tells_a_resident_overlay(hud, tmp_path):
    overlay, sent = hud
    level_file = tmp_path / "levels"
    level_file.write_bytes(b"")
    overlay.processing(str(level_file))
    assert sent == ["processing"]


def test_processing_also_leaves_a_marker_for_an_overlay_with_no_pipe(
        hud, tmp_path):
    """Off Windows the overlay is spawned per recording and has no stdin.

    It polls the level file already, so the marker beside it is the only
    channel that reaches every platform.
    """
    from whisper_flow import hud as hud_module

    overlay, _ = hud
    level_file = tmp_path / "levels"
    level_file.write_bytes(b"")
    overlay.processing(str(level_file))
    assert (tmp_path / ("levels" + hud_module.PROCESSING_SUFFIX)).exists()


def test_the_marker_is_removed_so_the_next_recording_is_clean(hud, tmp_path):
    from whisper_flow import hud as hud_module

    overlay, _ = hud
    level_file = tmp_path / "levels"
    level_file.write_bytes(b"")
    overlay.processing(str(level_file))
    overlay.clear_processing(str(level_file))
    assert not (tmp_path / ("levels" + hud_module.PROCESSING_SUFFIX)).exists()


def test_clearing_a_marker_that_is_not_there_is_not_an_error(hud, tmp_path):
    overlay, _ = hud
    overlay.clear_processing(str(tmp_path / "never-existed"))
    overlay.clear_processing("")


def test_an_overlay_with_no_pipe_is_not_commanded(monkeypatch, tmp_path):
    """_command would start a *resident* overlay where there is none."""
    from whisper_flow import hud as hud_module

    monkeypatch.setattr(hud_module, "RESIDENT", False)
    overlay = hud_module.HUD()
    monkeypatch.setattr(
        overlay, "_command",
        lambda line: pytest.fail("commanded a non-resident overlay"))
    level_file = tmp_path / "levels"
    level_file.write_bytes(b"")
    overlay.processing(str(level_file))
    assert (tmp_path / ("levels" + hud_module.PROCESSING_SUFFIX)).exists()


# ------------------------------------------------------------- the daemon
@pytest.fixture
def daemon(tmp_path, monkeypatch):
    from whisper_flow.daemon import WhisperFlowDaemon

    instance = WhisperFlowDaemon.__new__(WhisperFlowDaemon)
    instance.hud = Mock()
    instance.tray_icon = None
    instance.is_recording = True
    instance.current_mode = "transcribe"
    instance.recording_start_time = 0.0
    instance.stop_recording_event = None
    instance.hotkey_manager = Mock()
    instance.transcribe_app = Mock()
    level_file = tmp_path / "levels"
    level_file.write_bytes(b"")
    instance._level_file = str(level_file)
    return instance, level_file


def test_letting_go_leaves_the_overlay_up_and_processing(daemon):
    """Hiding it here is the gap this whole change exists to close."""
    instance, level_file = daemon
    instance._stop_recording_locked()

    instance.hud.processing.assert_called_once_with(str(level_file))
    instance.hud.hide.assert_not_called()
    assert level_file.exists(), (
        "the level file is the overlay's own orphan guard - deleting it here "
        "closes the very overlay we just asked to keep waiting")


def test_the_overlay_comes_down_when_the_processing_does(daemon):
    instance, level_file = daemon
    instance._stop_recording_locked()
    instance._take_down_overlay()

    instance.hud.hide.assert_called_once()
    instance.hud.clear_processing.assert_called_once_with(str(level_file))
    assert not level_file.exists()
    assert instance._level_file is None


def test_taking_the_overlay_down_twice_is_harmless(daemon):
    instance, _ = daemon
    instance._take_down_overlay()
    instance._take_down_overlay()


def test_stopping_releases_the_modifiers_we_are_holding(daemon):
    """The stuck Super key: this is where the dictation ends and we let go."""
    instance, _ = daemon
    instance._stop_recording_locked()
    instance.transcribe_app.system_manager.release_stuck_modifiers \
        .assert_called_once()


# --------------------------------------------------------- the overlay source
def _hud_app_source() -> str:
    return (Path(__file__).resolve().parents[1]
            / "src/whisper_flow/hud_app.py").read_text(encoding="utf-8")


def test_the_overlay_stops_reading_levels_once_it_is_processing():
    """There are none coming, and the file is about to be deleted."""
    body = _hud_app_source().split("def _levels_loop(", 1)[1].split(
        "\n    def ", 1)[0]
    assert "PROCESSING_SUFFIX" in body, (
        "an overlay with no command pipe learns about processing from the "
        "marker file, and this is the only place it looks")
    assert "self.processing" in body


def test_a_new_recording_leaves_the_processing_state():
    """Or the second dictation of a session shows a spinner for its waveform."""
    body = _hud_app_source().split("def begin_show(", 1)[1].split(
        "\n    def ", 1)[0]
    assert "self.processing = False" in body


def test_the_overlay_understands_the_processing_command():
    source = _hud_app_source()
    body = source.split("def _command_loop(", 1)[1]
    assert '"processing"' in body and "begin_processing" in body


def test_the_marker_suffix_matches_on_both_sides():
    """Two modules that cannot import each other have to agree by hand."""
    from whisper_flow import hud as hud_module

    source = _hud_app_source()
    assert f'PROCESSING_SUFFIX = "{hud_module.PROCESSING_SUFFIX}"' in source


def test_the_animation_is_loaded_as_a_sibling_and_shipped():
    """hud_app cannot import through the package, so hud_anim is a file.

    A frozen build only carries files the spec names, and the failure mode
    is the overlay dying at the first import - in front of the user, with
    the pill never appearing at all.
    """
    assert '_load_sibling("hud_anim")' in _hud_app_source()
    spec = (Path(__file__).resolve().parents[1]
            / "packaging/windows/whisper-flow.spec").read_text(encoding="utf-8")
    assert "hud_anim.py" in spec, "the overlay's animation is not bundled"
    assert os.path.exists(Path(__file__).resolve().parents[1]
                          / "src/whisper_flow/hud_anim.py")
