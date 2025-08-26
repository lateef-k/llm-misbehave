from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openai_base_url: str = "https://api.openai.com/v1"

    data_dir: Path
    openrouter_api_key: str
    logfire_write_api_key: str
    logfire_read_api_key: str
    openai_api_key: str

    model_config = SettingsConfigDict(env_prefix="REVEAL_", case_sensitive=False)


Settings = AppSettings.model_validate({})  # type: ignore


if Settings.logfire_write_api_key:
    import logfire

    logfire.configure(token=Settings.logfire_write_api_key, scrubbing=False)
    logfire.instrument_openai()
    logfire.instrument_openai_agents()
else:
    print("Not using logfire..")
