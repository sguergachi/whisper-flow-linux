"""System integration functionality for whisper-flow."""

import os
import shutil
import subprocess
import tempfile

from .config import Config


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

    def notify(self, message: str) -> None:
        """Send desktop notification.

        Args:
            message: Notification message

        """
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
            import subprocess as _sp

            # Release all modifiers to clear any stuck state
            clear_codes = [f"{c}:0" for c in self.MODIFIER_CODES]
            _sp.run(["ydotool", "key", *clear_codes], check=False)

            # Ctrl+V with explicit down/up sequence
            result = _sp.run(
                ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"],
                capture_output=True,
            )
            if result.returncode != 0:
                return False

            # Release Ctrl again (safety net)
            _sp.run(["ydotool", "key", "29:0", "97:0"], check=False)
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
        import sys as _sys
        _sys.stdout.write(f"[PASTE] _wtype_type called with text={text!r}\n")
        _sys.stdout.flush()
        if not shutil.which("wtype"):
            _sys.stdout.write("[PASTE] wtype not found\n")
            _sys.stdout.flush()
            return False
        try:
            result = subprocess.run(
                ["wtype", text],
                capture_output=True,
            )
            _sys.stdout.write(f"[PASTE] wtype type rc={result.returncode}\n")
            _sys.stdout.flush()
            return result.returncode == 0
        except Exception as e:
            _sys.stdout.write(f"[PASTE] _wtype_type exception: {e}\n")
            _sys.stdout.flush()
            return False
        try:
            result = subprocess.run(
                ["wtype", text],
                capture_output=True,
            )
            return result.returncode == 0
        except Exception:
            return False

    def save_active_window(self) -> None:
        """Save the currently active window UUID for later activation."""
        self._saved_window = self._get_active_window()

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
                import sys as _sys
                _sys.stdout.write(f"[PASTE] kdotool getactivewindow stderr: {result.stderr}\n")
                _sys.stdout.flush()
                return None
            uuid = result.stdout.strip()
            return uuid if uuid else None
        except Exception:
            return None

    def active_window_center(self) -> tuple[int, int] | None:
        """Centre point of the saved (or current) window, in compositor coords.

        Used to put the HUD on the screen the user is actually dictating into;
        a Wayland client cannot work that out for itself.
        """
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
                import sys as _sys
                _sys.stdout.write(f"[PASTE] kdotool activate stderr: {result.stderr}\n")
                _sys.stdout.flush()
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
            import sys as _sys
            _sys.stdout.write(f"[PASTE] Text to paste ({len(text)} chars): {text[:80]}...\n")
            _sys.stdout.flush()

            if self._is_wayland():
                import sys as _sys
                _sys.stdout.write("[PASTE] Wayland path entered\n")
                _sys.stdout.flush()
                target_window = self._saved_window or self._get_active_window()

                if target_window and self._kdotool_available():
                    _sys.stdout.write(f"[PASTE] Activating window: {target_window}\n")
                    _sys.stdout.flush()
                    self._activate_window(target_window)

                sanitized = text.replace('\n', ' ')
                _sys.stdout.write(f"[PASTE] Sanitized text: {sanitized!r}\n")
                _sys.stdout.flush()

                _sys.stdout.write("[PASTE] Trying _ydotool_type\n")
                _sys.stdout.flush()
                if self._ydotool_type(sanitized):
                    _sys.stdout.write("[PASTE] _ydotool_type succeeded\n")
                    _sys.stdout.flush()
                    return True

                _sys.stdout.write("[PASTE] _ydotool_type failed, trying _wtype_type\n")
                _sys.stdout.flush()
                if self._wtype_type(sanitized):
                    _sys.stdout.write("[PASTE] _wtype_type succeeded\n")
                    _sys.stdout.flush()
                    return True

                _sys.stdout.write("[PASTE] Direct type failed, copying to clipboard\n")
                _sys.stdout.flush()
                if not self._copy_to_clipboard(sanitized):
                    _sys.stdout.write("[PASTE] _copy_to_clipboard failed\n")
                    _sys.stdout.flush()
                    return False

                _sys.stdout.write("[PASTE] Clipboard copy succeeded, sending paste keystroke\n")
                _sys.stdout.flush()
                if not self._send_paste_keystroke():
                    _sys.stdout.write("[PASTE] _send_paste_keystroke failed\n")
                    _sys.stdout.flush()
                    return False

                _sys.stdout.write("[PASTE] Paste keystroke succeeded\n")
                _sys.stdout.flush()
                return True

            if shutil.which("xdotool"):
                subprocess.run(
                    ["xdotool", "key", "--clearmodifiers", "ctrl+v"],
                    check=False,
                )
                return True

            return False
        except Exception as e:
            _sys.stdout.write(f"Error pasting text: {e}\n")
            _sys.stdout.flush()
            return False

    def type_text(self, text: str) -> bool:
        """Type text into whatever is focused, right now.

        Used by live transcription, which appends to what the user is already
        looking at. Deliberately does not touch the clipboard and does not
        re-activate a saved window: focus is already correct, and stealing it
        mid-sentence would fight the user.
        """
        if not text:
            return True
        sanitized = text.replace("\n", " ")
        if self._is_wayland():
            return self._ydotool_type(sanitized) or self._wtype_type(sanitized)
        if shutil.which("xdotool"):
            result = subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--", sanitized],
                check=False,
                capture_output=True,
            )
            return result.returncode == 0
        return False

    def _send_paste_keystroke(self) -> bool:
        import sys as _sys
        _sys.stdout.write("[PASTE] _send_paste_keystroke called\n")
        _sys.stdout.flush()
        if self._wtype_paste():
            _sys.stdout.write("[PASTE] _wtype_paste succeeded\n")
            _sys.stdout.flush()
            return True
        if self._ydotool_paste():
            _sys.stdout.write("[PASTE] _ydotool_paste succeeded\n")
            _sys.stdout.flush()
            return True
        _sys.stdout.write("[PASTE] _send_paste_keystroke failed\n")
        _sys.stdout.flush()
        return False

    def _copy_to_clipboard(self, text: str) -> bool:
        """Copy text to system clipboard.

        Args:
            text: Text to copy

        Returns:
            True if successful, False otherwise

        """
        try:
            if self._is_wayland() and shutil.which("wl-copy"):
                p = subprocess.Popen(
                    ["wl-copy"],
                    stdin=subprocess.PIPE,
                )
                p.communicate(text.encode())
                return p.returncode == 0
            if shutil.which("xclip"):
                p = subprocess.Popen(
                    ["xclip", "-selection", "clipboard"],
                    stdin=subprocess.PIPE,
                )
                p.communicate(text.encode())
                return p.returncode == 0
            if shutil.which("xsel"):
                p = subprocess.Popen(
                    ["xsel", "--clipboard", "--input"],
                    stdin=subprocess.PIPE,
                )
                p.communicate(text.encode())
                return p.returncode == 0
            return False
        except Exception:
            return False

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

    def _release_modifiers(self) -> None:
        """Tell the compositor no modifiers are held, before injecting text.

        Live transcription types while the user is still holding the
        push-to-talk combination, so without this every character arrives as
        Super+Alt+<key> - a shortcut, not text. Doing it here is deliberate:
        the alternative is withholding the user's real key events upstream,
        and getting that wrong strands a modifier down system-wide.

        Injecting extra key-up events is safe in a way that withholding
        key-down events is not: the worst case is a redundant release.
        """
        try:
            subprocess.run(
                ["ydotool", "key", *[f"{c}:0" for c in self.MODIFIER_CODES]],
                check=False, capture_output=True, timeout=5,
            )
        except Exception:
            pass

    def _ydotool_type(self, text: str) -> bool:
        import sys as _sys
        _sys.stdout.write(f"[PASTE] _ydotool_type called with text={text!r}\n")
        _sys.stdout.flush()
        self._release_modifiers()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(text)
            tmp = f.name
        try:
            result = subprocess.run(
                ["ydotool", "type", "--file", tmp],
                check=False,
                capture_output=True,
                timeout=15,
            )
            _sys.stdout.write(f"[PASTE] ydotool type rc={result.returncode}\n")
            _sys.stdout.flush()
            if result.returncode != 0:
                _sys.stdout.write(f"[PASTE] ydotool type stderr: {result.stderr.decode() if isinstance(result.stderr, bytes) else result.stderr}\n")
                _sys.stdout.flush()
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            _sys.stdout.write("[PASTE] ydotool type timeout\n")
            _sys.stdout.flush()
            return False
        except Exception as e:
            _sys.stdout.write(f"[PASTE] ydotool type exception: {e}\n")
            _sys.stdout.flush()
            return False
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass
