"""KDE KGlobalAccel backend for global hotkeys on Wayland/KDE Plasma."""

import logging
import threading

log = logging.getLogger(__name__)


class KDEHotkeyBackend:
    """Uses KDE's kglobalaccel D-Bus API for global shortcut registration."""

    # Qt modifier encoding: Qt::MetaModifier=0x10000000, Qt::AltModifier=0x08000000, etc
    # encoded as (modifiers) since no base key for modifier-only combos
    QT_MODS = {
        "shift": 0x02000000,
        "ctrl": 0x04000000,
        "alt": 0x08000000,
        "cmd": 0x10000000,
        "meta": 0x10000000,
        "super": 0x10000000,
        "win": 0x10000000,
    }

    QT_KEYS = {
        "space": 0x20,
    }

    def __init__(self):
        self._shortcuts = {}   # name -> (keys_int, callback_press, callback_release)
        self._bus = None
        self._active = False
        self._listening = False
        self._thread = None

    def register_shortcut(self, name, key_string, callback_press, callback_release=None):
        """Register a global shortcut with KDE."""
        keys_int = self._encode_keys(key_string)
        self._shortcuts[name] = (keys_int, callback_press, callback_release)
        if self._active:
            self._set_kde_shortcut(name, keys_int)

    def _encode_keys(self, key_string):
        """Encode a key string like 'super+alt' into a Qt key sequence int."""
        parts = [p.strip() for p in key_string.lower().split("+")]
        mods = 0
        key = 0
        for p in parts:
            if p in self.QT_MODS:
                mods |= self.QT_MODS[p]
            elif p in self.QT_KEYS:
                key = self.QT_KEYS[p]
        return mods | key

    def _set_kde_shortcut(self, name, keys_int):
        """Set a single shortcut via KDE D-Bus."""
        try:
            import dbus
            bus = dbus.SessionBus()
            kg = bus.get_object("org.kde.kglobalaccel", "/kglobalaccel")
            iface = dbus.Interface(kg, "org.kde.KGlobalAccel")
            action_id = ["whisper_flow", name, f"WhisperFlow {name}"]
            iface.setForeignShortcut(action_id, [keys_int])
        except Exception:
            pass

    def start(self):
        """Start listening for KDE shortcut activations."""
        import dbus
        from dbus.mainloop.glib import DBusGMainLoop

        DBusGMainLoop(set_as_default=True)
        self._bus = dbus.SessionBus()
        self._active = True

        # Register all shortcuts with KDE
        for name, (keys_int, _, _) in self._shortcuts.items():
            self._set_kde_shortcut(name, keys_int)

        # Start listening thread
        self._listening = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def _listen_loop(self):
        """Listen for shortcut activation signals via D-Bus."""
        import dbus
        from gi.repository import GLib

        bus = dbus.SessionBus()

        # Watch for globalShortcutPressed on any component
        def on_shortcut_pressed(component, shortcut, timestamp):
            if component != "whisper_flow":
                return
            info = self._shortcuts.get(shortcut)
            if info:
                _, callback_press, callback_release = info
                if callback_press:
                    callback_press()

        def on_shortcut_released(component, shortcut, timestamp):
            if component != "whisper_flow":
                return
            info = self._shortcuts.get(shortcut)
            if info:
                _, _, callback_release = info
                if callback_release:
                    callback_release()

        # We need to monitor all components since new ones can be created
        # Listen on the kglobalaccel bus for signal matches
        bus.add_signal_receiver(
            on_shortcut_pressed,
            signal_name="globalShortcutPressed",
            dbus_interface="org.kde.kglobalaccel.Component",
        )
        bus.add_signal_receiver(
            on_shortcut_released,
            signal_name="globalShortcutReleased",
            dbus_interface="org.kde.kglobalaccel.Component",
        )

        loop = GLib.MainLoop()
        try:
            loop.run()
        except Exception:
            pass

    def stop(self):
        """Stop listening."""
        self._listening = False
        self._active = False
