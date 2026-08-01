"""WhisperFlow - AI-powered voice-to-text with context-aware processing.

Names are resolved on first use rather than at import. Importing this package
eagerly cost about 740ms, most of it the openai SDK and pyaudio, and every
one of those was paid by the overlay process - which is launched on the path
that starts a recording, needs tkinter and nothing else, and is the thing the
user is waiting to see. A frozen build on Windows pays considerably more than
740ms for the same work.

The public names below still import exactly as before; they just do it when
something asks for one.
"""

# Keep in step with pyproject.toml, which is what the build reads: CI pulls
# the version out of it with tomllib to name the installer. This one had
# been left on 0.1.0 through two releases because nothing checks it.
__version__ = "0.3.0"
__author__ = "sapountzis"
__email__ = "sapountzis.andreas@gmail.com"

from importlib import import_module

# public name -> module it lives in
_EXPORTS = {
    "AudioRecorder": ".audio",
    "CompletionService": ".completion",
    "Config": ".config",
    "PromptManager": ".prompts",
    "SystemManager": ".system",
    "TranscriptionService": ".transcription",
    "WhisperFlow": ".app",
    "WhisperFlowDaemon": ".daemon",
    "log": ".logging",
    "set_logging_enabled": ".logging",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """Import the owning module the first time a name is used (PEP 562)."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value          # subsequent lookups skip this entirely
    return value


def __dir__():
    return sorted(list(globals()) + __all__)
