"""The repository icon stays in step with the application icon."""

from pathlib import Path

from PIL import Image

from whisper_flow import icon


def test_project_favicon_is_the_app_icon():
    favicon_path = Path(__file__).resolve().parents[1] / "favicon.png"

    with Image.open(favicon_path) as favicon:
        favicon.load()
        expected = icon.draw_mic(256, icon.APP_COLOR)

        assert favicon.format == "PNG"
        assert favicon.size == (256, 256)
        assert favicon.mode == "RGBA"
        assert favicon.tobytes() == expected.tobytes()
