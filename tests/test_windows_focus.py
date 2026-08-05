"""The transcript has to land in the window it was dictated into.

save_active_window() had one implementation, kdotool's, and kdotool does not
exist on Windows - so `_saved_window` was None on every Windows recording and
the text went to whatever had focus at the moment it was typed. That is fine
while the words keep up with the speaking and quite wrong for the tail, which
is typed whenever the closing transcription finishes: on a machine where that
took twenty seconds, the utterance arrived in whichever field had been clicked
into since.

These run anywhere: IS_WINDOWS and the system_win module are both patched, so
nothing here touches Win32.
"""

import pytest

from whisper_flow import system as system_module


class _FakeWin:
    """Stands in for system_win, recording what the focus path did."""

    def __init__(self, foreground=100):
        self.foreground = foreground
        self.focused = []
        self.typed = []

    def foreground_window(self):
        return self.foreground

    def focus_window(self, hwnd):
        self.focused.append(hwnd)
        self.foreground = hwnd
        return True

    def type_text(self, text):
        self.typed.append((text, self.foreground))
        return True

    def copy_to_clipboard(self, text):
        return True

    def send_paste(self, release_first=True):
        return True


@pytest.fixture
def windows_manager(monkeypatch, mock_config):
    """A SystemManager that believes it is on Windows."""
    monkeypatch.setattr(system_module, "IS_WINDOWS", True)
    fake = _FakeWin()
    monkeypatch.setattr(system_module, "system_win", fake, raising=False)
    return system_module.SystemManager(mock_config), fake


def test_windows_remembers_the_window_the_dictation_started_in(windows_manager):
    manager, fake = windows_manager
    fake.foreground = 4321
    manager.save_active_window()
    assert manager._saved_window == 4321


def test_the_transcript_goes_back_to_the_window_it_was_dictated_into(
        windows_manager):
    manager, fake = windows_manager
    fake.foreground = 4321
    manager.save_active_window()

    fake.foreground = 9999              # the user clicked elsewhere meanwhile
    assert manager.type_text("hello there") is True
    assert fake.focused == [4321]
    assert fake.typed == [("hello there", 4321)]


def test_a_paste_also_goes_back_to_the_right_window(windows_manager):
    manager, fake = windows_manager
    fake.foreground = 4321
    manager.save_active_window()

    fake.foreground = 9999
    assert manager.paste_text("hello there") is True
    assert fake.typed == [("hello there", 4321)]


def test_focus_is_left_alone_while_it_is_already_right(windows_manager):
    """This runs between the words of a live dictation, so the ordinary case
    must not touch the foreground window at all."""
    manager, fake = windows_manager
    fake.foreground = 4321
    manager.save_active_window()

    manager.type_text("one")
    manager.type_text("two")
    assert fake.focused == []
    assert [text for text, _ in fake.typed] == ["one", "two"]


def test_nothing_saved_means_nothing_is_yanked(windows_manager):
    """A recording that started with no foreground window at all."""
    manager, fake = windows_manager
    fake.foreground = 0
    manager.save_active_window()
    assert manager._saved_window is None

    fake.foreground = 9999
    manager.type_text("hello")
    assert fake.focused == []


def test_a_window_that_will_not_come_back_does_not_stop_the_typing(
        windows_manager, monkeypatch):
    """Half a transcript somewhere is better than none anywhere."""
    manager, fake = windows_manager
    fake.foreground = 4321
    manager.save_active_window()
    fake.foreground = 9999
    monkeypatch.setattr(fake, "focus_window", lambda hwnd: False)

    assert manager.type_text("hello") is True
    assert fake.typed == [("hello", 9999)]


def test_the_modifiers_we_hold_are_released_through_the_manager(
        windows_manager, monkeypatch):
    """The daemon reaches system_win through here, so the chain must connect."""
    manager, fake = windows_manager
    called = []
    monkeypatch.setattr(fake, "release_injected_modifiers",
                        lambda: called.append(True), raising=False)
    manager.release_stuck_modifiers()
    assert called == [True]


def test_releasing_modifiers_off_windows_does_nothing(monkeypatch, mock_config):
    monkeypatch.setattr(system_module, "IS_WINDOWS", False)
    manager = system_module.SystemManager(mock_config)
    manager.release_stuck_modifiers()        # must not raise or reach Win32


def test_a_failure_to_release_is_not_allowed_to_escape(
        windows_manager, monkeypatch):
    """This runs in the recording thread's cleanup; raising there would lose
    the release of the processing lock behind it."""
    manager, fake = windows_manager

    def explode():
        raise OSError("user32 went away")

    monkeypatch.setattr(fake, "release_injected_modifiers", explode,
                        raising=False)
    manager.release_stuck_modifiers()
