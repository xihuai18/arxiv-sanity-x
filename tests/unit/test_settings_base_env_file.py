from __future__ import annotations

from pydantic_settings import BaseSettings


def test_group_model_config_uses_absolute_env_file(tmp_path, monkeypatch):
    from config import settings_base

    repo_env = tmp_path / ".env"
    repo_env.write_text("EXAMPLE_VALUE=repo\n", encoding="utf-8")

    other_cwd = tmp_path / "other"
    other_cwd.mkdir()
    (other_cwd / ".env").write_text("EXAMPLE_VALUE=cwd\n", encoding="utf-8")

    monkeypatch.setattr(settings_base, "ENV_FILE", repo_env)

    class Example(settings_base.SettingsGroup):
        model_config = settings_base.group_model_config("EXAMPLE_")

        value: str = ""

    monkeypatch.chdir(other_cwd)

    assert Example().value == "repo"


def test_root_model_config_uses_absolute_env_file(tmp_path, monkeypatch):
    from config import settings_base

    repo_env = tmp_path / ".env"
    repo_env.write_text("ARXIV_SANITY_VALUE=repo\n", encoding="utf-8")

    other_cwd = tmp_path / "other"
    other_cwd.mkdir()
    (other_cwd / ".env").write_text("ARXIV_SANITY_VALUE=cwd\n", encoding="utf-8")

    monkeypatch.setattr(settings_base, "ENV_FILE", repo_env)

    class Example(BaseSettings):
        model_config = settings_base.root_model_config()

        value: str = ""

    monkeypatch.chdir(other_cwd)

    assert Example().value == "repo"
