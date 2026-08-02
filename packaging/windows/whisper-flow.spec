# PyInstaller spec for the Windows build.
#
# One executable. It runs the tray daemon normally, the overlay when launched
# with --hud (which hud.py does per recording), and the model setup window
# with --setup. Shipping those as separate .exes only raised the question of
# which one to run.
#
# The UI is GTK4 everywhere, so the GTK runtime has to ship too: typelibs,
# schemas, icons and pixbuf loaders, none of which PyInstaller can find on
# its own. They come from an MSYS2 UCRT64 prefix (CI installs it; GTK_PREFIX
# overrides the default location).
#
# The genuinely Linux-only modules must not be pulled in: evdev and pynput
# have no Windows wheels, and a hidden import would force exactly that.

import glob
import os

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

GTK_PREFIX = os.environ.get("GTK_PREFIX", r"C:\msys64\ucrt64")

EXCLUDES = ["evdev", "pynput"]


def gtk_runtime():
    """The pieces of the GTK4 runtime the frozen app cannot run without.

    PyGObject reaches GTK through dlopen at runtime, not through a link-time
    dependency, so PyInstaller cannot discover any of these itself: the DLL
    closure is walked by hand, and the typelibs, schemas, icons and pixbuf
    loaders ship as data.
    """
    if not os.path.isdir(GTK_PREFIX):
        # Building without the runtime produces a binary whose windows all
        # fail at once; say so loudly rather than shipping that.
        raise SystemExit(
            f"GTK_PREFIX {GTK_PREFIX!r} does not exist - install MSYS2 with "
            f"mingw-w64-ucrt-x86_64-gtk4 and libadwaita, or set GTK_PREFIX.")

    datas = []

    def add(pattern, dest, required=False):
        found = glob.glob(os.path.join(GTK_PREFIX, pattern), recursive=True)
        if required and not found:
            raise SystemExit(f"nothing matched {pattern} in {GTK_PREFIX}")
        for path in found:
            datas.append((path, dest))

    # Typelibs: every namespace the UI touches, plus their dependencies as
    # named by `g-ir-inspect` on a stock UCRT64 install.
    for ns in ("Adw-1", "Gtk-4.0", "Gsk-4.0", "Graphene-1.0", "GdkPixbuf-2.0",
               "Pango-1.0", "PangoCairo-1.0", "cairo-1.0", "HarfBuzz-0.0",
               "Gio-2.0", "GLib-2.0", "GObject-2.0"):
        add(rf"lib\girepository-1.0\{ns}.typelib", "girepository-1.0",
            required=True)
    add(r"share\glib-2.0\schemas\*.xml", "share/glib-2.0/schemas", required=True)
    # The symbolic icons the UI names, and the cursor/edit affordances.
    add(r"share\icons\Adwaita\symbolic\**\*.svg", "share/icons/Adwaita/symbolic",
        required=True)
    add(r"share\icons\hicolor\scalable\**\*.svg", "share/icons/hicolor/scalable")
    # gdk-pixbuf loaders, or no icon renders at all. No loaders.cache: it
    # carries absolute build-machine paths, and GDK_PIXBUF_MODULEDIR scans
    # the directory at runtime instead.
    loaders = glob.glob(os.path.join(
        GTK_PREFIX, r"lib\gdk-pixbuf-2.0\2.10.0\loaders\*.dll"))
    if not loaders:
        raise SystemExit(f"no gdk-pixbuf loaders in {GTK_PREFIX}")
    datas += [(path, "lib/gdk-pixbuf-2.0/2.10.0/loaders") for path in loaders]

    binaries = [(dll, ".") for dll in _dll_closure(
        [os.path.join(GTK_PREFIX, "bin", name)
         for name in ("libgtk-4-1.dll", "libadwaita-1-0.dll",
                      "libgirepository-1.0-1.dll", "libgobject-2.0-0.dll",
                      "libglib-2.0-0.dll", "libgio-2.0-0.dll",
                      "libcairo-2.dll", "libcairo-gobject-2.dll")]
        + loaders)]
    return datas, binaries


def _dll_closure(roots):
    """Every MSYS2 DLL the given DLLs import, recursively.

    Only DLLs that live in the prefix are followed; system DLLs (kernel32
    and friends) resolve on the user's machine as they do everywhere else.
    """
    import pefile

    seen = {}
    todo = [path for path in roots if os.path.exists(path)]
    while todo:
        path = todo.pop()
        name = os.path.basename(path).lower()
        if name in seen:
            continue
        seen[name] = path
        try:
            pe = pefile.PE(path, fast_load=True)
            pe.parse_data_directories(directories=[
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
            for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
                dep = entry.dll.decode(errors="replace")
                dep_path = os.path.join(GTK_PREFIX, "bin", dep)
                if os.path.exists(dep_path):
                    todo.append(dep_path)
        except Exception:
            pass
    return sorted(seen.values())


hidden = collect_submodules("pydantic") + [
    "whisper_flow.hotkey_win",
    "whisper_flow.system_win",
    "whisper_flow.blur_win",
    "whisper_flow.hud_app",
    "whisper_flow.setup_gtk",
    "whisper_flow.settings_gtk",
    "pystray._win32",
    # Imported at the point of use rather than at module scope, to keep them
    # off the startup path. PyInstaller only follows imports it can see
    # statically, so anything made lazy has to be named here or it is simply
    # not bundled - and the failure appears only for the user who configures
    # an API key, at the moment they first use it.
    "pystray",
    "openai",
    "velopack",
    # blur_win reaches for these, and ctypes.wintypes is a submodule that
    # importing ctypes does not bring along.
    "ctypes.wintypes",
]

# hud_app loads these two by file path (importing them would drag pystray's
# GTK 3 into the overlay process), so they must exist as plain files.
module_files = [
    ("../../src/whisper_flow/wayland_blur.py", "."),
    ("../../src/whisper_flow/blur_win.py", "."),
]

gtk_datas, gtk_binaries = gtk_runtime()

app = Analysis(
    ["../../src/whisper_flow/__main__win__.py"],
    pathex=["../../src"],
    binaries=gtk_binaries,
    datas=gtk_datas + module_files,
    hiddenimports=hidden,
    excludes=EXCLUDES,
    cipher=block_cipher,
)

pyz = PYZ(app.pure, app.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, app.scripts, [],
    exclude_binaries=True,
    name="whisper-flow",
    console=False,          # tray app; a console window would sit there
    icon=None,
)

COLLECT(
    exe, app.binaries, app.zipfiles, app.datas,
    strip=False, upx=False, name="whisper-flow",
)
