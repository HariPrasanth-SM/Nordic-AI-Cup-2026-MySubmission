import pytest

from solution.config import Config, load_config


def test_base_profile_loads_and_validates():
    config = load_config(profile="dev_8gb")
    assert isinstance(config, Config)
    assert config.profile == "dev_8gb"


def test_profile_overrides_device_and_compute_type():
    dev = load_config(profile="dev_8gb")
    workstation = load_config(profile="workstation_32gb")

    assert dev.asr.compute_type == "int8_float16"
    assert workstation.asr.compute_type == "float16"
    # Both should inherit the same model_size from base.yaml -- profiles only
    # override hardware-driven keys, not algorithm choices.
    assert dev.asr.model_size == workstation.asr.model_size


def test_env_var_selects_profile(monkeypatch):
    monkeypatch.setenv("MEDICAL_APPT_PROFILE", "workstation_32gb")
    config = load_config()
    assert config.profile == "workstation_32gb"


def test_explicit_argument_overrides_env_var(monkeypatch):
    monkeypatch.setenv("MEDICAL_APPT_PROFILE", "workstation_32gb")
    config = load_config(profile="dev_8gb")
    assert config.profile == "dev_8gb"


def test_ad_hoc_overrides_win_over_profile():
    config = load_config(profile="dev_8gb", overrides={"asr": {"model_size": "large-v3-turbo"}})
    assert config.asr.model_size == "large-v3-turbo"
    # untouched sibling keys survive the merge
    assert config.asr.compute_type == "int8_float16"


def test_verifier_kind_env_var_overrides_profile_default(monkeypatch):
    monkeypatch.setenv("MEDICAL_APPT_VERIFIER_KIND", "llm")
    config = load_config(profile="workstation_32gb")
    assert config.verifier.kind == "llm"


def test_explicit_override_wins_over_verifier_kind_env_var(monkeypatch):
    monkeypatch.setenv("MEDICAL_APPT_VERIFIER_KIND", "llm")
    config = load_config(profile="workstation_32gb", overrides={"verifier": {"kind": "similarity"}})
    assert config.verifier.kind == "similarity"


def test_unknown_calibrated_name_raises():
    with pytest.raises(FileNotFoundError):
        load_config(profile="dev_8gb", calibrated_name="does-not-exist")


def test_unknown_top_level_key_is_rejected():
    with pytest.raises(Exception):
        load_config(profile="dev_8gb", overrides={"not_a_real_key": 1})
