"""What the frozen Windows build must contain.

PyInstaller finds dependencies by reading the source, so it only bundles
imports it can see. Anything deliberately imported at the point of use to
keep it off the startup path is invisible to that analysis and has to be
declared, or the build is missing it - and the failure surfaces on a user's
machine, in the one code path that needed it.
"""

import re
from pathlib import Path

import pytest

SPEC = Path(__file__).resolve().parents[1] / "packaging/windows/whisper-flow.spec"

# Third-party modules the code imports lazily, and where it does it.
LAZY_IMPORTS = {
    "openai": "whisper_flow/completion.py and transcription.py",
    "pystray": "whisper_flow/daemon.py",
    "velopack": "whisper_flow/updater.py",
}


def _hidden_imports() -> str:
    return SPEC.read_text()


@pytest.mark.parametrize("module,where", sorted(LAZY_IMPORTS.items()))
def test_a_lazily_imported_module_is_declared_to_pyinstaller(module, where):
    assert f'"{module}"' in _hidden_imports(), (
        f"{module} is imported lazily in {where}, so PyInstaller cannot see "
        f"it; add it to hiddenimports or the frozen build will not have it"
    )


def test_the_lazy_list_matches_what_the_code_actually_does():
    """Catches a module being made lazy without this list being updated."""
    src = Path(__file__).resolve().parents[1] / "src/whisper_flow"
    found = set()
    for path in src.glob("*.py"):
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped.startswith(("import ", "from ")):
                continue
            if not line.startswith((" ", "\t")):
                continue                    # module scope: PyInstaller sees it
            match = re.match(r"(?:from|import)\s+([a-zA-Z_][\w]*)", stripped)
            if match and match.group(1) in LAZY_IMPORTS:
                found.add(match.group(1))
    assert found, "no lazy imports detected; has the mechanism changed?"
    for module in found:
        assert module in LAZY_IMPORTS
