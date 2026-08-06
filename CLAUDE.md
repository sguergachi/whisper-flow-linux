# Working in this repository

## Git

**No pull requests. Merge directly to master, always.** Commit the work and
push it to `master`. Do not open a PR, do not ask whether one is wanted, and
do not leave finished work sitting on a branch waiting to be merged.

## Tests

`pytest` runs the whole suite. The GTK tests exercise the real window in a
fresh interpreter, and they skip - silently - wherever GTK4, libadwaita or a
display is missing, so a green run on a machine without them has not tested
the UI at all. On Linux that needs the distro's PyGObject, which lives in
`/usr/lib/python3/dist-packages` and is invisible to any interpreter but the
distro's own:

```sh
sudo apt-get install -y portaudio19-dev libevdev-dev \
    python3-gi gir1.2-gtk-4.0 gir1.2-adw-1 libgtk-4-dev libcairo2-dev xvfb
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e ".[dev]"
PYSTRAY_BACKEND=xorg xvfb-run -a .venv/bin/pytest -q
```

`PYSTRAY_BACKEND=xorg` matters: pystray picks a backend when it is imported,
and left alone it takes GTK 3, which cannot share a process with the GTK 4
the app itself uses.

Two things do not run on an Ubuntu machine and skip by design: the overlay
tests, which need a `Gtk4LayerShell` typelib Ubuntu does not package, and the
Windows-native tests.
