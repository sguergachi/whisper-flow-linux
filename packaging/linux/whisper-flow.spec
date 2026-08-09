# PyInstaller spec for the portable Linux AppImage.
#
# One onedir tree, wrapped later as an AppImage. Built on an older Ubuntu
# (CI uses 22.04) so the glibc requirement stays low enough for most current
# desktops — Ubuntu, Fedora, Arch, Mint, Pop!, Debian bookworm+, etc.
#
# GTK reaches us through dlopen + typelibs, so the .so closure and
# girepository data are collected by hand, same idea as the Windows spec.
#
# The bundle deliberately does NOT ship the GTK/glib stack. Bundling an
# old glib poisons newer distros: the dynamic loader fixes LD_LIBRARY_PATH
# at process start, so a bundled Ubuntu-22.04 glib always wins over the
# host's, and then every host library that depends on a newer glib
# (libjson-glib, libnotify, system binaries like sh and notify-send)
# breaks with "undefined symbol". The host must provide GTK4, libadwaita,
# gtk4-layer-shell and the gi typelibs — the same requirement the venv
# installer (install.sh) already documents. The bundle keeps the parts the
# host cannot reasonably provide: the Python runtime, the app itself, the
# whisper.cpp engine and the model.

import glob
import os
import re
import subprocess

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

# Do not ship the host glibc / dynamic linker — those must come from the
# target machine, which is what makes one build run on many distros.
_SKIP_SO = re.compile(
    r"^(ld-linux|ld-linux-x86-64|linux-vdso|libc|libm|libdl|librt|"
    r"libpthread|libresolv|libutil|libnsl|libBrokenLocale|libanl|"
    r"libthread_db|libcidn)\.so"
)

# System libraries that must come from the host. These are the GTK stack
# PyGObject loads through typelibs, the notification/readline libs that
# system tools (notify-send, sh) load from LD_LIBRARY_PATH, everything the
# GTK closure dragged in, the OpenSSL/curl/gnutls families, the zlib/bzip2/
# lzma/zstd/brotli compression runtimes, and the ALSA+JACK audio stack the
# bundled portaudio links against. Shipping any of them makes the bundle
# shadow the host's version, and when the host's needs a newer one -
# libcurl and libsoup on Arch/Fedora/current Ubuntu need OPENSSL_3.2.0,
# which the bundled Ubuntu-22.04 libssl does not have - the load dies with
# "version ... not found". The bundled Ubuntu libjack also auto-starts a
# JACK server (`jackd -l`) that the host's copy never would, spamming every
# start. Every target this AppImage runs on (glibc of Ubuntu 22.04 or
# newer) ships all of them.
_SKIP_BUNDLE_LIBS = re.compile(
    r"^lib(glib|gobject|gio|gmodule|girepository|gtk-4|gtk-3|adwaita|"
    r"gtk4-layer-shell|gdk-4|gdk-3|gdk_pixbuf|graphene|pango|cairo|"
    r"harfbuzz|fribidi|fontconfig|freetype|graphite2|pixman|epoxy|"
    r"xkbcommon|json-glib|notify|readline|thai|datrie|dbus|appindicator|"
    r"ayatana|wayland|X11|Xau|Xext|Xcursor|Xdamage|Xdmcp|Xfixes|Xinerama|"
    r"Xi|Xrandr|Xrender|xcb|atk|drm|gbm|EGL|GLX|pcre|pcre2|blkid|mount|"
    r"selinux|uuid|lzo2|md|bsd|deflate|jbig|tiff|webp|jpeg|png16|"
    r"ssl|crypto|curl|soup|nghttp2|gnutls|nettle|hogweed|idn2|unistring|"
    r"tasn1|p11-kit|gmp|proxy|bz2|expat|ffi|tinfo|ncurses|zstd|lzma|"
    r"brotli|gcc_s|stdc\+\+|asound|jack|db-5|z\b)"
)

# PyInstaller renames collision-prone libraries to name-<hash>.so and
# rewires every reference to the hashed name, so they can never shadow the
# host's copies and must be kept. The hashed files (Pillow's libjpeg/
# libpng/libtiff/..., numpy/scipy's openblas and gfortran, the renamed
# libxcb/libXau) live under the collection root or pillow.libs/, and the
# plain-SONAME symlinks that the loader follows to them are the ones the
# filter above drops. Without this exception the skip list - which lists
# jpeg/png16/tiff/freetype/harfbuzz and friends by name - also strips those
# hashed files and leaves dangling symlinks behind, so Pillow cannot import
# and the daemon dies on a machine that would otherwise run fine.
_HASHED_LIB = re.compile(r"-[0-9a-f]{8}(?:-[0-9a-f]{8})?\.so(?:\.\d+)*$")


def _is_host_lib(name: str) -> bool:
    if _HASHED_LIB.search(name):
        return False
    return bool(_SKIP_BUNDLE_LIBS.match(name))


def gtk_runtime():
    """Nothing to ship: the host provides GTK, typelibs, schemas and icons.

    Kept as a function so the failure mode is explicit — this build no
    longer bundles any part of the GTK runtime. portaudio stays in the
    bundle: nothing on the host loads it, so it cannot conflict.
    """
    binaries = []
    datas = []
    for names in (["libportaudio.so.2", "libportaudio.so"],):
        for prefix in (
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
            "/usr/lib",
            "/usr/local/lib",
        ):
            path = os.path.join(prefix, names[0])
            if os.path.isfile(path):
                binaries.append((path, "lib"))
                break
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
    "gi.repository.PangoCairo",
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

# PyInstaller's gi hooks collect the gi_typelibs directory, gio modules and
# every shared library the typelibs name — the whole GTK stack again. Strip
# it: the host provides typelibs, schemas, icons and libraries. Shipping
# any of them shadows the host's newer versions and is exactly what broke
# the old fully-bundled build on Arch/Fedora/current Ubuntu.
app.binaries = [
    (dest, src, typ)
    for (dest, src, typ) in app.binaries
    if not _is_host_lib(os.path.basename(src))
    and not _is_host_lib(os.path.basename(dest))
]
app.datas = [
    (dest, src, typ)
    for (dest, src, typ) in app.datas
    if not any(
        dest == prefix or dest.startswith(prefix + "/")
        for prefix in ("girepository-1.0", "gi_typelibs", "gio_modules",
                       "share/glib-2.0", "share/icons", "lib/gdk-pixbuf")
    )
]

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
