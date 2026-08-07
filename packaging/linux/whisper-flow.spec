# PyInstaller spec for the portable Linux AppImage.
#
# One onedir tree, wrapped later as an AppImage. Built on an older Ubuntu
# (CI uses 22.04) so the glibc requirement stays low enough for most current
# desktops — Ubuntu, Fedora, Arch, Mint, Pop!, Debian bookworm+, etc.
#
# GTK reaches us through dlopen + typelibs, so the .so closure and
# girepository data are collected by hand, same idea as the Windows spec.

import glob
import os
import re
import subprocess

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

# Prefixes to search for typelibs / libs (Debian multiarch + Fedora + Arch).
_LIB_PREFIXES = [
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib64",
    "/usr/lib",
    "/usr/local/lib",
]

REQUIRED_TYPELIBS = (
    "Gtk-4.0",
    "Gdk-4.0",
    "Adw-1",
    "Pango-1.0",
    "GLib-2.0",
    "GObject-2.0",
    "Gio-2.0",
    "cairo-1.0",
    "HarfBuzz-0.0",
    "GdkPixbuf-2.0",
    "Graphene-1.0",
)

# Optional but wanted for the Wayland HUD.
OPTIONAL_TYPELIBS = ("Gtk4LayerShell-1.0",)

# Do not ship the host glibc / dynamic linker — those must come from the
# target machine, which is what makes one build run on many distros.
_SKIP_SO = re.compile(
    r"^(ld-linux|ld-linux-x86-64|linux-vdso|libc|libm|libdl|librt|"
    r"libpthread|libresolv|libutil|libnsl|libBrokenLocale|libanl|"
    r"libthread_db|libcidn)\.so"
)


def _find_file(*candidates):
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def _find_typelib(name: str) -> str | None:
    for prefix in (
        "/usr/lib/x86_64-linux-gnu/girepository-1.0",
        "/usr/lib64/girepository-1.0",
        "/usr/lib/girepository-1.0",
        "/usr/local/lib/girepository-1.0",
    ):
        path = os.path.join(prefix, f"{name}.typelib")
        if os.path.isfile(path):
            return path
    return None


def _ldd_libs(binary: str) -> list[str]:
    """Shared libraries a binary needs, absolute paths, host-resolved."""
    try:
        out = subprocess.check_output(["ldd", binary], text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    found = []
    for line in out.splitlines():
        # "libfoo.so.0 => /usr/lib/libfoo.so.0 (0x...)"
        if "=>" not in line:
            continue
        right = line.split("=>", 1)[1].strip().split()[0]
        if right.startswith("/") and os.path.isfile(right):
            found.append(right)
    return found


def _so_closure(roots: list[str]) -> list[str]:
    """Every non-glibc .so reachable from the roots, recursively via ldd."""
    seen: dict[str, str] = {}
    todo = [p for p in roots if p and os.path.isfile(p)]
    while todo:
        path = todo.pop()
        name = os.path.basename(path)
        if name in seen:
            continue
        if _SKIP_SO.match(name):
            continue
        seen[name] = path
        for dep in _ldd_libs(path):
            dep_name = os.path.basename(dep)
            if dep_name not in seen and not _SKIP_SO.match(dep_name):
                todo.append(dep)
    return sorted(seen.values())


def _find_lib(names: list[str]) -> str | None:
    for name in names:
        for prefix in _LIB_PREFIXES:
            path = os.path.join(prefix, name)
            if os.path.isfile(path):
                return path
        # bare name on ldconfig path
        try:
            out = subprocess.check_output(
                ["ldconfig", "-p"], text=True, stderr=subprocess.DEVNULL
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            out = ""
        for line in out.splitlines():
            if f"{name} " in line or line.strip().startswith(name):
                if "=>" in line:
                    path = line.split("=>", 1)[1].strip()
                    if os.path.isfile(path):
                        return path
    return None


def gtk_runtime():
    """Typelibs, schemas, icons, pixbuf loaders, and the .so closure."""
    datas = []
    binaries = []

    for ns in REQUIRED_TYPELIBS:
        path = _find_typelib(ns)
        if not path:
            raise SystemExit(f"required typelib missing: {ns}.typelib")
        datas.append((path, "girepository-1.0"))

    for ns in OPTIONAL_TYPELIBS:
        path = _find_typelib(ns)
        if path:
            datas.append((path, "girepository-1.0"))
        else:
            print(f"warning: optional typelib missing: {ns} (Wayland HUD may degrade)")

    # Compiled schemas — GLib aborts without them when anything touches GSettings.
    schema_candidates = [
        "/usr/share/glib-2.0/schemas/gschemas.compiled",
    ]
    schema = _find_file(*schema_candidates)
    if not schema:
        raise SystemExit("gschemas.compiled not found")
    datas.append((schema, "share/glib-2.0/schemas"))

    # Adwaita symbolic icons the settings window names.
    adwaita_index = "/usr/share/icons/Adwaita/index.theme"
    if os.path.isfile(adwaita_index):
        datas.append((adwaita_index, "share/icons/Adwaita"))
        symbolic = "/usr/share/icons/Adwaita/symbolic"
        if os.path.isdir(symbolic):
            for folder, _subs, names in os.walk(symbolic):
                rel = os.path.relpath(folder, symbolic)
                dest = (
                    "share/icons/Adwaita/symbolic"
                    if rel == "."
                    else f"share/icons/Adwaita/symbolic/{rel}"
                )
                for name in names:
                    if name.endswith(".svg"):
                        datas.append((os.path.join(folder, name), dest))

    # gdk-pixbuf loaders.
    loader_dirs = glob.glob("/usr/lib/*/gdk-pixbuf-2.0/*/loaders") + glob.glob(
        "/usr/lib/gdk-pixbuf-2.0/*/loaders"
    )
    for loader_dir in loader_dirs:
        for so in glob.glob(os.path.join(loader_dir, "*.so*")):
            binaries.append((so, "lib/gdk-pixbuf-2.0/2.10.0/loaders"))

    # Roots of the shared-library closure.
    roots = []
    for names in (
        ["libgtk-4.so.1", "libgtk-4.so"],
        ["libadwaita-1.so.0", "libadwaita-1.so"],
        ["libgobject-2.0.so.0", "libgobject-2.0.so"],
        ["libglib-2.0.so.0", "libglib-2.0.so"],
        ["libgio-2.0.so.0", "libgio-2.0.so"],
        ["libcairo.so.2", "libcairo.so"],
        ["libcairo-gobject.so.2", "libcairo-gobject.so"],
        ["libgirepository-1.0.so.1", "libgirepository-1.0.so",
         "libgirepository-2.0.so.0", "libgirepository-2.0.so"],
        ["libportaudio.so.2", "libportaudio.so"],
        ["libgtk4-layer-shell.so.0", "libgtk4-layer-shell.so"],
    ):
        found = _find_lib(names)
        if found:
            roots.append(found)
        elif "layer-shell" not in names[0]:
            print(f"warning: library not found for {names[0]}")

    # Also pull in the PyGObject C extension's own deps.
    try:
        import gi
        gi_path = os.path.dirname(gi.__file__)
        for so in glob.glob(os.path.join(gi_path, "_gi*.so")) + glob.glob(
            os.path.join(gi_path, "*.so")
        ):
            roots.append(so)
    except Exception as e:
        print(f"warning: could not inspect gi package: {e}")

    for so in _so_closure(roots):
        binaries.append((so, "lib"))

    return datas, binaries


def _cairo_bridge() -> str:
    import importlib.util

    name = "gi._gi_cairo"
    try:
        found = importlib.util.find_spec(name) is not None
    except ImportError:
        found = False
    if not found:
        raise SystemExit(
            f"{name} is missing — install python3-gi and python3-cairo "
            f"(or PyGObject built with cairo). Without it the overlay draws nothing."
        )
    return name


hidden = collect_submodules("pydantic") + [
    _cairo_bridge(),
    "whisper_flow.hotkey_evdev",
    "whisper_flow.system",
    "whisper_flow.hud_app",
    "whisper_flow.settings_gtk",
    "whisper_flow.wayland_blur",
    "pystray._appindicator",
    "pystray._util.gtk",
    "pystray",
    "evdev",
    "pynput",
    "gi",
    "gi.repository.Gtk",
    "gi.repository.Gdk",
    "gi.repository.Adw",
    "gi.repository.GLib",
    "gi.repository.GObject",
    "gi.repository.Gio",
    "gi.repository.Pango",
    "gi.repository.cairo",
]

# Windows-only pieces must not be pulled in.
EXCLUDES = [
    "pywin32",
    "win32api",
    "win32con",
    "win32gui",
    "pythoncom",
    "velopack",
    "whisper_flow.hotkey_win",
    "whisper_flow.system_win",
    "whisper_flow.blur_win",
]

module_files = [
    ("../../src/whisper_flow/wayland_blur.py", "."),
    ("../../src/whisper_flow/hud_anim.py", "."),
]

gtk_datas, gtk_binaries = gtk_runtime()

app = Analysis(
    ["../../src/whisper_flow/__main_linux__.py"],
    pathex=["../../src"],
    binaries=gtk_binaries,
    datas=gtk_datas + module_files,
    hiddenimports=hidden,
    excludes=EXCLUDES,
    cipher=block_cipher,
)

pyz = PYZ(app.pure, app.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    app.scripts,
    [],
    exclude_binaries=True,
    name="whisper-flow",
    console=False,  # tray app; no terminal window on double-click
)

coll = COLLECT(
    exe,
    app.binaries,
    app.zipfiles,
    app.datas,
    strip=False,
    upx=False,
    name="whisper-flow",
)
