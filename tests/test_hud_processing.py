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


def test_quiet_room_is_low_risk():
    """Clean speech over a soft floor should stay near white."""
    # Quiet floor ~80, speech peaks ~2000 → high SNR.
    history = [80.0] * 40 + [1800.0, 2200.0, 1500.0, 900.0] * 5
    risk = hud_anim.noise_risk(history)
    assert risk < hud_anim.RISK_YELLOW_AT, f"clean room risk too high: {risk}"


def test_cafe_bed_is_high_risk():
    """Loud continuous bed with little dynamic range → red zone."""
    history = [900.0 + (i % 5) * 30 for i in range(60)]
    risk = hud_anim.noise_risk(history)
    assert risk >= hud_anim.RISK_YELLOW_AT, f"café bed not risky enough: {risk}"


def test_low_snr_whisper_is_risky():
    """Whisper barely above room tone maps toward yellow/red."""
    # Floor ~500, peaks ~700 → ~3 dB SNR.
    history = [500.0] * 30 + [650.0, 700.0, 680.0, 620.0] * 8
    risk = hud_anim.noise_risk(history)
    assert risk >= hud_anim.RISK_YELLOW_AT * 0.8, f"low SNR risk low: {risk}"


def test_risk_rgb_moves_white_to_yellow_to_red():
    r0, g0, b0 = hud_anim.risk_rgb(0.0)
    ry, gy, by = hud_anim.risk_rgb(hud_anim.RISK_YELLOW_AT)
    rr, gr, br = hud_anim.risk_rgb(1.0)
    assert g0 > gy >= gr  # green falls white → yellow → red
    assert rr > r0 * 0.9
    assert br < b0


def test_short_history_does_not_flash_red():
    assert hud_anim.noise_risk([]) == 0.0
    assert hud_anim.noise_risk([100.0, 200.0]) == 0.0


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


def test_the_stop_suffix_matches_on_both_sides():
    """The overlay drops the stop-request marker; the daemon polls it."""
    from whisper_flow import hud as hud_module

    source = _hud_app_source()
    assert f'STOP_SUFFIX = "{hud_module.STOP_SUFFIX}"' in source
    assert "level_file + STOP_SUFFIX" in source, (
        "the overlay must ask for the stop beside the level file, which is "
        "the only path the daemon can watch")


def test_the_stop_button_is_the_pill_grown_not_a_floating_tab():
    """The button is the capsule's own lower half - one outline, no gap.

    A tab drawn below the pill reads as a separate control floating beside
    the HUD; the stop button must read as part of it. The outline is one
    continuous path that grows the pill's straight sides down past its
    bottom cap.
    """
    source = _hud_app_source()
    assert "_extended_pill(cr, x, y, w, HEIGHT, STOP_BTN_H)" in source, (
        "the stop button must be drawn as one extended capsule with the "
        "pill - not as a separate shape hanging below it")
    assert "STOP_GAP" not in source, (
        "any air between the pill and the button makes it a floating tab")


# ------------------------------------------------------- the stop button
def test_auto_modes_show_the_stop_button(daemon):
    """Only the modes that end on silence need a mouse-visible way out.

    Push-to-talk already has one: the user is holding the key.
    """
    instance, level_file = daemon
    instance.current_mode = "auto_transcribe"
    instance._hud_point = None
    instance._show_hud_now()
    assert instance.hud.show.call_args.kwargs["stop_button"] is True

    instance.hud.reset_mock()
    instance.current_mode = "transcribe"
    instance._show_hud_now()
    assert instance.hud.show.call_args.kwargs["stop_button"] is False


def test_pressing_the_stop_button_ends_the_recording(daemon):
    """The overlay's marker is the daemon's stop signal.

    The overlay is a separate process and cannot call back, so it drops a
    file beside the level file and the daemon polls for it.
    """
    import threading
    import time

    instance, level_file = daemon
    instance.current_mode = "auto_transcribe"
    marker = level_file.with_suffix(level_file.suffix + ".stop")
    stopped = []

    def stop():
        stopped.append(1)
        instance.is_recording = False

    instance._stop_recording = stop
    # The watcher sweeps stale markers as it starts, so the press has to land
    # after that, the way a real one does.
    thread = threading.Thread(
        target=instance._watch_hud_stop_request, args=("auto_transcribe",),
        daemon=True)
    thread.start()
    time.sleep(0.1)
    marker.write_bytes(b"1")
    thread.join(timeout=2)

    assert stopped == [1]
    assert not marker.exists(), "the marker must be consumed, not re-seen"


def test_a_stale_stop_marker_cannot_stop_the_next_recording(daemon):
    """An overlay that died after writing its marker must not end the next
    recording the instant it starts."""
    instance, level_file = daemon
    instance.current_mode = "auto_transcribe"
    marker = level_file.with_suffix(level_file.suffix + ".stop")
    marker.write_bytes(b"1")
    instance.is_recording = False

    instance._watch_hud_stop_request("auto_transcribe")

    assert not marker.exists(), "a leftover marker must be swept at the start"


def test_taking_the_overlay_down_clears_the_stop_marker(daemon):
    instance, level_file = daemon
    instance._stop_recording_locked()
    instance._take_down_overlay()

    instance.hud.clear_stop_marker.assert_called_once_with(str(level_file))


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


# ------------------------------------------- the overlay must not wedge things
def test_a_failing_overlay_does_not_leave_the_daemon_recording(
        temp_config_dir, monkeypatch):
    """start_recording's rollback runs straight through the overlay.

    RESIDENT is Windows-only, so the Linux job never reached the branch where
    putting the overlay into processing tries to start a process - and the
    rollback path asks it to do that in the middle of whatever just failed.
    An exception there skipped the state reset below it, leaving is_recording
    set with no thread to clear it: the watchdog ignores a recording with no
    thread, so every later press was dropped as busy for the rest of the
    daemon's life. That is the failure this whole state exists downstream of,
    and it is worth more than the overlay.
    """
    import sys
    from unittest.mock import patch

    from whisper_flow import hud as hud_module

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_daemon import _stub_daemon

    monkeypatch.setattr(hud_module, "RESIDENT", True)
    daemon = _stub_daemon(temp_config_dir)

    # Fails the overlay's own temp file as well as the level file, which is
    # exactly what the rollback test does and how this surfaced.
    with patch("whisper_flow.daemon.tempfile.mkstemp",
               side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            daemon.start_recording("transcribe")

    assert daemon.is_recording is False, (
        "the overlay threw during rollback and the daemon is still recording")
    assert daemon.current_mode is None
    assert daemon.recording_thread is None


def test_the_resident_overlay_survives_a_disk_that_will_not_take_its_log(
        monkeypatch):
    """It runs on the recording path; an exception here is not a missing
    overlay but a wedged daemon."""
    from unittest.mock import patch

    from whisper_flow import hud as hud_module

    monkeypatch.setattr(hud_module, "RESIDENT", True)
    overlay = hud_module.HUD()
    with patch("whisper_flow.hud.tempfile.mkstemp",
               side_effect=OSError("disk full")):
        assert overlay._resident_process() is None
        overlay.processing("")           # must not raise
