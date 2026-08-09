# PyInstaller spec for the Windows build.
#
# One executable. It runs the tray daemon normally, the overlay when launched
# with --hud (which hud.py does per recording), and the settings window with
# --settings. Shipping those as separate .exes only raised the question of
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

# The namespaces the UI names for itself. Everything they in turn depend on
# comes along with the rest of the prefix; these are asserted because their
# absence means the prefix is not a GTK 4 one at all. Kept in step with
# --selftest, and with the test that reads this list.
REQUIRED_NAMESPACES = ("Gtk-4.0", "Gdk-4.0", "Adw-1", "Pango-1.0",
                       "PangoCairo-1.0", "GLib-2.0", "GObject-2.0",
                       "Gio-2.0")


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

    def add_tree(source, dest, suffixes, required=False):
        """Copy a directory tree, keeping its shape.

        add() flattens - every match lands in the one destination directory.
        That is right for typelibs, which are a flat set, and wrong for an
        icon theme: index.theme names the subdirectories its icons live in,
        symbolic/actions and symbolic/status and the rest, and GTK looks in
        exactly those. Flattened into symbolic/, every lookup missed, and the
        settings window drew the broken-image glyph where its tab icons go.
        """
        root = os.path.join(GTK_PREFIX, source)
        found = 0
        for folder, _subdirs, names in os.walk(root):
            relative = os.path.relpath(folder, root)
            target = dest if relative == "." else f"{dest}/{relative}"
            for name in names:
                if name.endswith(suffixes):
                    datas.append((os.path.join(folder, name),
                                  target.replace("\\", "/")))
                    found += 1
        if required and not found:
            raise SystemExit(f"nothing matched {suffixes} under {root}")

    # Every typelib in the prefix, rather than a hand-picked list.
    #
    # The list was wrong twice. Gdk-4.0 was missing and shipped a build that
    # resolved Gtk and then died on `from gi.repository import Gdk`; the next
    # build died on freetype2-2.0, which nothing imports - HarfBuzz's typelib
    # names it. A typelib's dependencies are not visible from the source that
    # imports it, so this closure cannot be maintained by reading our own
    # code, and each wrong guess costs a full Windows build to discover.
    #
    # They are inert data, a few MB against a 354MB bundle, and one that is
    # never imported is never loaded. Ship them all; assert the ones the UI
    # actually names, so a prefix without GTK 4 still fails here and loudly.
    add(r"lib\girepository-1.0\*.typelib", "girepository-1.0", required=True)
    have = {os.path.basename(path) for path, _ in datas}
    absent = [ns for ns in REQUIRED_NAMESPACES if f"{ns}.typelib" not in have]
    if absent:
        raise SystemExit(
            f"{GTK_PREFIX} has no typelib for: {', '.join(absent)}")
    # gschemas.compiled, not the .gschema.xml files beside it.
    #
    # The XML is source; GLib reads only the compiled binary, and a schema it
    # cannot find is fatal rather than an error it returns - g_settings_new
    # aborts the process. Shipping the XML alone therefore produced a
    # GSETTINGS_SCHEMA_DIR with nothing readable in it, and every window that
    # touched a GSetting died where it stood: the settings window vanished
    # without a trace, and the overlay exited 0xC0000005.
    add(r"share\glib-2.0\schemas\gschemas.compiled", "share/glib-2.0/schemas",
        required=True)
    # The symbolic icons the UI names, and the cursor/edit affordances.
    #
    # index.theme is the theme. Without it GTK does not see a theme here at
    # all - it sees a directory of SVGs it has no reason to look in - and
    # every named icon falls back to the broken-image glyph. The few that
    # still rendered came from the set GTK compiles into libgtk as a
    # GResource, which needs no theme and hid how completely this was broken.
    add(r"share\icons\Adwaita\index.theme", "share/icons/Adwaita", required=True)
    add_tree(r"share\icons\Adwaita\symbolic", "share/icons/Adwaita/symbolic",
             (".svg",), required=True)
    add(r"share\icons\hicolor\index.theme", "share/icons/hicolor")
    add_tree(r"share\icons\hicolor\scalable", "share/icons/hicolor/scalable",
             (".svg",))
    # gdk-pixbuf loaders, or no icon renders at all. No loaders.cache: it
    # carries absolute build-machine paths, and GDK_PIXBUF_MODULEDIR scans
    # the directory at runtime instead.
    loaders = glob.glob(os.path.join(
        GTK_PREFIX, r"lib\gdk-pixbuf-2.0\2.10.0\loaders\*.dll"))
    if not loaders:
        raise SystemExit(f"no gdk-pixbuf loaders in {GTK_PREFIX}")
    datas += [(path, "lib/gdk-pixbuf-2.0/2.10.0/loaders") for path in loaders]

    # Roots of the DLL closure. _dll_closure skips anything that is not
    # there, so a root that gets renamed upstream drops its whole subtree in
    # silence and the build stays green - check them here instead.
    roots = ["libgtk-4-1.dll", "libadwaita-1-0.dll", "libgobject-2.0-0.dll",
             "libglib-2.0-0.dll", "libgio-2.0-0.dll", "libcairo-2.dll",
             "libcairo-gobject-2.dll"]
    missing = [name for name in roots
               if not os.path.exists(os.path.join(GTK_PREFIX, "bin", name))]
    if missing:
        raise SystemExit(
            f"missing from {GTK_PREFIX}\\bin: {', '.join(missing)}")

    # girepository was folded into GLib in 2.80 and versioned with it, so a
    # prefix carries one name or the other depending on its age. Pinning
    # either one breaks on an MSYS2 update; accepting neither is how `import
    # gi` starts failing on the user's machine.
    girepository = [
        name for name in ("libgirepository-2.0-0.dll", "libgirepository-1.0-1.dll")
        if os.path.exists(os.path.join(GTK_PREFIX, "bin", name))]
    if not girepository:
        raise SystemExit(f"no libgirepository in {GTK_PREFIX}\\bin")

    binaries = [(dll, ".") for dll in _dll_closure(
        [os.path.join(GTK_PREFIX, "bin", name)
         for name in roots + girepository] + loaders)]
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


def _cairo_bridge() -> str:
    """The PyGObject extension that hands a cairo_t to pycairo.

    GTK passes the draw callback a cairo_t*, and PyGObject turns it into a
    cairo.Context through a "foreign struct converter" that lives in its own
    extension module. Nothing imports that module by name: pygi-foreign.c
    builds the string "gi._gi_" + namespace at runtime and imports it, which
    is invisible to PyInstaller's import graph. So it was left out, and every
    frame of the overlay raised "Couldn't find foreign struct converter for
    'cairo.Context'" - a window that existed, was styled, and painted nothing.

    Asserted rather than merely named: a hiddenimport that does not resolve is
    a build warning, and this failed silently once already.
    """
    import importlib.util

    name = "gi._gi_cairo"
    try:
        found = importlib.util.find_spec(name) is not None
    except ImportError:
        found = False       # no gi at all, which the typelib checks also catch
    if not found:
        raise SystemExit(
            f"{name} is missing - is python-gobject built against pycairo? "
            f"Without it the overlay draws nothing.")
    return name


hidden = collect_submodules("pydantic") + [
    _cairo_bridge(),
    "whisper_flow.hotkey_win",
    "whisper_flow.system_win",
    "whisper_flow.blur_win",
    "whisper_flow.hud_app",
    "whisper_flow.settings_gtk",
    "pystray._win32",
    # Imported at the point of use rather than at module scope, to keep them
    # off the startup path. PyInstaller only follows imports it can see
    # statically, so anything made lazy has to be named here or it is simply
    # not bundled - and the failure appears only at the moment that lazy
    # import first runs, in front of the user.
    "pystray",
    "velopack",
    # blur_win reaches for these, and ctypes.wintypes is a submodule that
    # importing ctypes does not bring along.
    "ctypes.wintypes",
]

# hud_app loads these by file path (importing them would drag pystray's
# GTK 3 into the overlay process), so they must exist as plain files.
module_files = [
    ("../../src/whisper_flow/wayland_blur.py", "."),
    ("../../src/whisper_flow/blur_win.py", "."),
    ("../../src/whisper_flow/hud_anim.py", "."),
]

def app_icon() -> str:
    """Write the .ico, from the same code that draws the tray icon.

    Generated rather than checked in, so the executable, the installer and
    the tray can never come to show three different marks. It is the tray
    glyph without the halo: that exists so a light icon reads against a
    light panel it is composited onto, and an .ico is composited onto
    nothing.
    """
    import importlib.util

    source = os.path.join(os.path.dirname(os.path.abspath(SPEC)),
                          "..", "..", "src", "whisper_flow", "icon.py")
    spec = importlib.util.spec_from_file_location("whisper_flow_icon", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)         # imports PIL and nothing else
    return module.write_ico(
        os.path.join(os.path.dirname(os.path.abspath(SPEC)),
                     "whisper-flow.ico"))


gtk_datas, gtk_binaries = gtk_runtime()
icon_file = app_icon()

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
    icon=icon_file,
)

COLLECT(
    exe, app.binaries, app.zipfiles, app.datas,
    strip=False, upx=False, name="whisper-flow",
)
