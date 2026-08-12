"""System integration functionality for whisper-flow."""

import os
import shutil
import subprocess
import sys
import tempfile
import time

from .config import Config
from .logging import log

IS_WINDOWS = sys.platform == "win32"
if IS_WINDOWS:  # pragma: no cover - platform dependent
    from . import system_win


class SystemManager:
    """System integration manager for notifications, clipboard, and window operations."""

    def __init__(self, config: Config):
        """Initialize system manager.

        Args:
            config: Configuration object

        """
        self.config = config
        self._wayland = os.environ.get("WAYLAND_DISPLAY", "") != ""
        self._xdg_session = os.environ.get("XDG_SESSION_TYPE", "")
        self._saved_window = None
        self._last_notified: dict[str, float] = {}

    def notify(self, message: str) -> None:
        """Send a desktop notification, unless it is a repeat.

        A failure that recurs - a mic that will not open, a transcription
        server that is down - fires this on every attempt, and on Windows each
        one spawns a PowerShell toast. Identical messages are suppressed for a
        short window so a persistent fault stays one notification.

        Args:
            message: Notification message

        """
        if not self.should_notify(message):
            return

        if IS_WINDOWS:
            system_win.notify("Whisper-Flow", message)
            return
        if shutil.which("notify-send"):
            subprocess.Popen(
                [
                    "notify-send",
                    f"--expire-time={self.config.notification_timeout}",
                    "Whisper-Flow",
                    message,
                ],
            )
        else:
            print(f"[Whisper-Flow] {message}")

    def should_notify(self, message: str) -> bool:
        """Whether this message should be shown, or is a recent repeat.

        Shared with the tray notification path so both obey one rate limit.
        """
        if not getattr(self.config, "notifications_enabled", True):
            return False
        now = time.monotonic()
        window = getattr(self.config, "notification_min_interval", 5.0)
        last = self._last_notified.get(message)
        if last is not None and now - last < window:
            return False
        self._last_notified[message] = now
        if len(self._last_notified) > 64:      # bounded; these are short-lived
            cutoff = now - window
            self._last_notified = {
                k: v for k, v in self._last_notified.items() if v > cutoff
            }
        return True

    MODIFIER_CODES = ["29", "97", "56", "100", "125", "126", "42", "54"]

    def _ydotool_available(self) -> bool:
        """Check if ydotoold is reachable (returns True/False, never auto-starts it)."""
        sock = os.environ.get(
            "YDOTOOL_SOCKET",
            os.path.join(os.environ.get("XDG_RUNTIME_DIR", ""), ".ydotool_socket"),
        )
        return bool(sock) and os.path.exists(sock)

    def _ydotool_paste(self) -> bool:
        """Paste clipboard contents using ydotool with explicit modifier cleanup.

        Uses separate key-down/key-up events to avoid stuck keys in the
        uinput virtual device if ydotool is interrupted mid-operation.
        """
        if not self._ydotool_available():
            return False
        try:
            # Release all modifiers to clear any stuck state
            clear_codes = [f"{c}:0" for c in self.MODIFIER_CODES]
            subprocess.run(["ydotool", "key", *clear_codes], check=False)

            # Ctrl+V with explicit down/up sequence
            result = subprocess.run(
                ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"],
                capture_output=True,
            )
            if result.returncode != 0:
                return False

            # Release Ctrl again (safety net)
            subprocess.run(["ydotool", "key", "29:0", "97:0"], check=False)
            return True
        except Exception:
            return False

    def _wtype_paste(self) -> bool:
        """Paste clipboard contents using wtype (Wayland-native keyboard simulator)."""
        if not shutil.which("wtype"):
            return False
        try:
            result = subprocess.run(
                ["wtype", "-M", "ctrl", "-k", "v", "-m", "ctrl"],
                capture_output=True,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _wtype_type(self, text: str) -> bool:
        """Type text directly using wtype (Wayland-native keyboard simulator)."""
        if not shutil.which("wtype"):
            return False
        try:
            result = subprocess.run(
                ["wtype", text],
                capture_output=True,
            )
            if result.returncode != 0:
                log(f"[PASTE] wtype type failed: {result.stderr.decode(errors='replace')}")
            return result.returncode == 0
        except Exception as e:
            log(f"[PASTE] wtype type error: {e}")
            return False

    def save_active_window(self) -> None:
        """Remember where the dictation was started, to type back into it.

        On Windows this was never recorded at all - the only implementation
        was kdotool's, which does not exist there - so every Windows paste
        went to whatever happened to be focused when the transcript arrived.
        """
        if IS_WINDOWS:
            self._saved_window = system_win.foreground_window() or None
            return
        self._saved_window = self._get_active_window()

    def release_stuck_modifiers(self) -> None:
        """Let go of any modifier this app is holding down.

        Typing releases the push-to-talk modifiers so dictated text is not
        read as shortcuts, then presses back the ones the user is still
        holding. If the user let go during that injection their own key-up
        landed on an already-released key, and the press that followed has
        no release coming - so Windows believes, say, Super is down, and the
        next Shift is Win+Shift.

        Called when a dictation ends, which is the last moment any of it can
        still be wanted.
        """
        if not IS_WINDOWS:
            return
        try:
            system_win.release_injected_modifiers()
        except Exception as e:
            log(f"[PASTE] could not release the modifiers we were holding: {e}")

    def _restore_saved_focus(self) -> None:
        """Bring the window the user dictated into back to the front.

        Only when focus has actually moved elsewhere. Calling it regardless
        would be a no-op in the normal case, but this runs between the words
        of a live dictation and there is no reason to touch the foreground
        window on every committed phrase.
        """
        if not IS_WINDOWS or not self._saved_window:
            return True
        if system_win.foreground_window() == self._saved_window:
            return True
        if system_win.focus_window(self._saved_window):
            return True
        # The Start menu is the one refusal that can be undone. It holds the
        # foreground against SetForegroundWindow for as long as it is open,
        # so every pass of a live dictation found it there and held its words
        # back - a whole utterance spoken into a menu the user did not ask
        # for. Closing it costs one keystroke and gives the dictation its
        # window back, so it is worth trying before reporting the refusal.
        if (system_win.dismiss_start_menu()
                and system_win.focus_window(self._saved_window)):
            log("[PASTE] the Start menu had the foreground; closed it and "
                "carried on where the dictation started")
            return True
        # Name what is in front instead. "Would not come back" says a window
        # refused, not which one refused it, and the answer decides what this
        # is: the overlay taking focus is our own bug, the Start menu is
        # Windows opening on a Super release, and anything else is the user
        # having clicked away mid-sentence.
        log(f"[PASTE] {system_win.describe_foreground()} is in front and the "
            f"window this dictation started in will not come back; "
            f"holding the text rather than typing it there")
        return False

    def _kdotool_available(self) -> bool:
        return shutil.which("kdotool") is not None

    def _get_active_window(self) -> str | None:
        """Get the UUID of the currently active window via kdotool."""
        if not self._kdotool_available():
            return None
        try:
            result = subprocess.run(
                ["kdotool", "getactivewindow"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                log(f"[PASTE] kdotool getactivewindow failed: {result.stderr}")
                return None
            uuid = result.stdout.strip()
            return uuid if uuid else None
        except Exception:
            return None

    def active_window_center(self) -> tuple[int, int] | None:
        """Centre point of the saved (or current) window, in compositor coords.

        Used to put the HUD on the screen the user is actually dictating into;
        a Wayland client cannot work that out for itself.

        On Windows this went through kdotool like everything else here, which
        does not exist there, so it always answered None - and the overlay,
        given nothing to go on, showed on whichever monitor GDK happened to
        list first for the whole session.
        """
        if IS_WINDOWS:
            # The saved window first, then whatever is in front. The saved
            # one is from the last recording and may be closed by now, and a
            # window that is gone has no rectangle - falling back keeps the
            # pill on the right screen instead of on the first one GDK lists.
            return (system_win.window_center(self._saved_window)
                    or system_win.window_center(
                        system_win.foreground_window()))

        window_id = self._saved_window or self._get_active_window()
        if not window_id or not self._kdotool_available():
            return None
        try:
            result = subprocess.run(
                ["kdotool", "getwindowgeometry", window_id],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return None
            pos = size = None
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.startswith("Position:"):
                    pos = line.split(":", 1)[1].strip().split(",")
                elif line.startswith("Geometry:"):
                    size = line.split(":", 1)[1].strip().split("x")
            if not pos or not size:
                return None
            x, y = int(float(pos[0])), int(float(pos[1]))
            w, h = int(float(size[0])), int(float(size[1]))
            return x + w // 2, y + h // 2
        except Exception:
            return None

    def _activate_window(self, window_id: str) -> bool:
        """Activate a window by its UUID via kdotool."""
        if not self._kdotool_available() or not window_id:
            return False
        try:
            result = subprocess.run(
                ["kdotool", "windowactivate", window_id],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                log(f"[PASTE] kdotool activate failed: {result.stderr}")
                return False
            return True
        except Exception:
            return False

    def paste_text(self, text: str) -> bool:
        """Paste text into the current application.

        Args:
            text: Text to paste

        Returns:
            True if successful, False otherwise

        """
        try:
            log(f"[PASTE] Text to paste ({len(text)} chars): {text[:80]}...")

            if IS_WINDOWS:
                sanitized = text.replace("\n", " ")
                # Not refused when the window will not come back: this is the
                # closing transcript and there is no next pass to hold it for.
                self._restore_saved_focus()
                if system_win.type_text(sanitized):
                    return True
                return (system_win.copy_to_clipboard(sanitized)
                        and system_win.send_paste())

            if self._is_wayland():
                target_window = self._saved_window or self._get_active_window()

                if target_window and self._kdotool_available():
                    self._activate_window(target_window)

                sanitized = text.replace('\n', ' ')

                if self._ydotool_type(sanitized):
                    return True

                if self._wtype_type(sanitized):
                    return True

                log("[PASTE] direct typing failed, falling back to the clipboard")
                if not self.copy_to_clipboard(sanitized):
                    return False

                return self._send_paste_keystroke()

            if shutil.which("xdotool"):
                subprocess.run(
                    ["xdotool", "key", "--clearmodifiers", "ctrl+v"],
                    check=False,
                )
                return True

            return False
        except Exception as e:
            log(f"Error pasting text: {e}")
            return False

    def type_text(self, text: str, only_where_it_started: bool = False) -> bool:
        """Type text into the window this dictation was started in.

        `only_where_it_started` refuses rather than typing somewhere else.
        Off by default, because for the closing transcript half a transcript
        somewhere beats none anywhere - it is the last chance those words
        get. Live transcription sets it: those words get another chance on
        every pass and again at the end, so typing them into whatever
        happens to be in front buys nothing and is how a dictation ended up
        in the Start menu's search box.

        Used by live transcription, which appends to what the user is already
        looking at, and for the tail that follows the closing pass. It does
        not touch the clipboard.

        It used to trust the current focus outright, on the grounds that
        stealing it mid-sentence would fight the user. That holds while the
        words keep up with the speaking; it stops holding for the tail, which
        is typed whenever the closing transcription finishes - and on a
        machine where that took twenty seconds, the utterance landed in
        whatever the user had clicked into since. Restoring focus only when it
        has moved keeps the ordinary case untouched.
        """
        if not text:
            return True
        sanitized = text.replace("\n", " ")
        if IS_WINDOWS:
            if not self._restore_saved_focus() and only_where_it_started:
                return False
            return system_win.type_text(sanitized)
        if self._is_wayland():
            if self._ydotool_type(sanitized) or self._wtype_type(sanitized):
                return True
            # Live finalize used to stop here. A multi-sentence tail (~500
            # chars) timed out ydotool's old 15s budget on KDE, wtype is
            # rejected by compositors without virtual-keyboard, and the
            # whole transcript was dropped with nothing on the clipboard.
            log("[PASTE] direct typing failed, falling back to the clipboard")
            if not self.copy_to_clipboard(sanitized):
                return False
            return self._send_paste_keystroke()
        if shutil.which("xdotool"):
            result = subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--", sanitized],
                check=False,
                capture_output=True,
            )
            return result.returncode == 0
        return False

    def _send_paste_keystroke(self) -> bool:
        if self._wtype_paste():
            return True
        return self._ydotool_paste()

    def copy_to_clipboard(self, text: str) -> bool:
        """Copy text to the system clipboard.

        Public because failure reporting depends on it. It was private, and
        the daemon called the public name that did not exist - so every
        attempt to put a failure report on the clipboard raised
        AttributeError into a handler that turned it into "not copied", and
        the feature never worked once. The tests missed it by mocking the
        very attribute whose absence was the bug.

        Args:
            text: Text to copy

        Returns:
            True if successful, False otherwise

        """
        try:
            if IS_WINDOWS:
                return system_win.copy_to_clipboard(text)
            if self._is_wayland() and shutil.which("wl-copy"):
                return self._pipe_to_clipboard(["wl-copy"], text)
            if shutil.which("xclip"):
                return self._pipe_to_clipboard(
                    ["xclip", "-selection", "clipboard"], text)
            if shutil.which("xsel"):
                return self._pipe_to_clipboard(
                    ["xsel", "--clipboard", "--input"], text)
            return False
        except Exception:
            return False

    @staticmethod
    def _pipe_to_clipboard(command: list[str], text: str) -> bool:
        """Hand text to a clipboard helper without waiting on it forever.

        wl-copy and xclip both keep a process alive to serve the selection to
        whoever pastes it. They normally fork so the one launched here still
        exits, which is why the previous unbounded communicate() worked in
        practice - but it is bounded now, because this runs on the recording
        thread while reporting a failure, and that is a poor place to be
        waiting on a helper that has decided to stay in the foreground.

        A helper still running after the write is the successful case, not a
        failure. It must not be killed: for wl-copy and xclip that process
        *is* the clipboard, so killing it discards what was just copied.
        """
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            process.stdin.write(text.encode())
            process.stdin.close()
        except (BrokenPipeError, OSError):
            return False
        try:
            return process.wait(timeout=2) == 0
        except subprocess.TimeoutExpired:
            return True             # alive and holding the selection

    def get_highlighted_text(self) -> str | None:
        """Get currently highlighted/selected text.

        Returns:
            Highlighted text or None if unable to get

        """
        try:
            if self._is_wayland() and shutil.which("wl-paste"):
                result = subprocess.run(
                    ["wl-paste", "--primary"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()

            if shutil.which("xclip"):
                result = subprocess.run(
                    ["xclip", "-selection", "primary", "-o"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()

            if shutil.which("xsel"):
                result = subprocess.run(
                    ["xsel", "--primary", "--output"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()

            return None
        except Exception:
            return None

    def _is_wayland(self) -> bool:
        return self._wayland or self._xdg_session == "wayland"

    def _release_modifiers_windows(self) -> None:
        system_win.release_modifiers()

    def _release_modifiers(self) -> None:
        """Tell the compositor no modifiers are held, before injecting text.

        A backstop. The hotkey listener already tells the compositor the
        push-to-talk keys were released when the combination fires, so this
        normally has nothing to do; it covers paths that inject text without
        a hotkey being held.

        Only releases are sent, never presses, so the worst case is a
        redundant key-up.
        """
        try:
            subprocess.run(
                ["ydotool", "key", *[f"{c}:0" for c in self.MODIFIER_CODES]],
                check=False, capture_output=True, timeout=5,
            )
        except Exception:
            pass

    @staticmethod
    def _ydotool_type_timeout(text: str) -> float:
        """Seconds ydotool is allowed to spend injecting `text`.

        ydotool types keystroke-by-keystroke. A fixed 15s budget was enough
        for a short live chunk and too short for a closing multi-sentence
        tail (~500 chars timed out on KDE). Scale with length, capped so a
        stuck daemon cannot hang the recording thread forever.
        """
        # ~40ms per character plus a few seconds of fixed overhead; floor
        # keeps short strings from racing the helper's own startup.
        return max(15.0, min(120.0, 5.0 + 0.04 * len(text)))

    def _ydotool_type(self, text: str) -> bool:
        self._release_modifiers()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(text)
            tmp = f.name
        try:
            result = subprocess.run(
                ["ydotool", "type", "--file", tmp],
                check=False,
                capture_output=True,
                timeout=self._ydotool_type_timeout(text),
            )
            if result.returncode != 0:
                log(f"[PASTE] ydotool type failed: "
                    f"{result.stderr.decode(errors='replace')}")
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            log(f"[PASTE] ydotool type timed out after "
                f"{self._ydotool_type_timeout(text):.0f}s "
                f"({len(text)} chars)")
            return False
        except Exception as e:
            log(f"[PASTE] ydotool type error: {e}")
            return False
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass
