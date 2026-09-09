"""Settings must be read from the same directory that Save writes to."""

from whisper_flow import config as config_module
from whisper_flow.config import Config, reload_config


def test_explicit_directory_does_not_read_personal_settings(tmp_path, monkeypatch):
    personal = tmp_path / 'personal'
    personal.mkdir()
    (personal / '.env').write_text('MIC_DEVICE_INDEX=12\n')
    monkeypatch.setattr(config_module, 'default_config_dir', lambda: personal)
    assert Config(config_dir=tmp_path / 'other').mic_device_index is None


def test_explicit_directory_loads_its_own_settings(tmp_path):
    (tmp_path / '.env').write_text('MIC_DEVICE_INDEX=7\n')
    assert Config(config_dir=tmp_path).mic_device_index == 7


def test_environment_directory_is_used_on_reload(tmp_path, monkeypatch):
    monkeypatch.setenv('WHISPER_FLOW_CONFIG_DIR', str(tmp_path))
    assert reload_config().mic_device_index is None
    (tmp_path / '.env').write_text('MIC_DEVICE_INDEX=9\n')
    assert reload_config().mic_device_index == 9


def test_explicit_env_file_none_still_disables_file_loading(tmp_path):
    (tmp_path / '.env').write_text('MIC_DEVICE_INDEX=7\n')
    assert Config(config_dir=tmp_path, _env_file=None).mic_device_index is None


def test_reload_keeps_an_explicit_directory(tmp_path):
    config = Config(config_dir=tmp_path)
    (tmp_path / '.env').write_text('MIC_DEVICE_INDEX=6\n')
    assert reload_config(config.config_dir).mic_device_index == 6
