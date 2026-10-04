
import os
import threading
import tomllib
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field


def _get_project_root() -> Path:
    # src/mcp_aas/config.py -> parents[2] == MCP_AAS project root
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _get_project_root()


def get_config_path() -> Optional[Path]:
    override = os.environ.get("MCP_AAS_CONFIG_FILE")
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = PROJECT_ROOT / "config" / p
        return p if p.exists() else None

    cfg = PROJECT_ROOT / "config" / "config.toml"
    if cfg.exists():
        return cfg
    example = PROJECT_ROOT / "config" / "config.example.toml"
    if example.exists():
        return example
    return None


class LLMSettings(BaseModel):
    model: str = Field(default="", description="Model / deployment name")
    embedding_model: str = Field(default="text-embedding-3-large-1", description="Embedding model / Azure deployment")
    base_url: str = Field(default="", description="API base URL (Azure endpoint)")
    api_key: str = Field(default="", description="API key")
    max_tokens: int = Field(4096, description="Maximum number of tokens per request")
    max_completion_tokens: int = Field(4096, description="Max tokens for reasoning models")
    max_input_tokens: Optional[int] = Field(None, description="Max input tokens (None = unlimited)")
    temperature: float = Field(1.0, description="Sampling temperature")
    api_type: str = Field(default="", description="azure | openai | ollama")
    api_version: str = Field(default="", description="Azure OpenAI API version")


class AppConfig(BaseModel):
    llm: Dict[str, LLMSettings] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class Config:

    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._config: Optional[AppConfig] = None
                    self._load_initial_config()
                    self._initialized = True

    def _load_raw(self) -> dict:
        path = get_config_path()
        if path is None:
            return {}
        try:
            with path.open("rb") as f:
                return tomllib.load(f)
        except Exception:
            return {}

    def _load_initial_config(self):
        raw = self._load_raw()
        base_llm = raw.get("llm", {}) or {}
        overrides = {k: v for k, v in base_llm.items() if isinstance(v, dict)}

        llm_profiles: Dict[str, dict] = {}
        # A top-level [llm] table (with scalar keys) becomes the "default" profile.
        scalar_keys = {k: v for k, v in base_llm.items() if not isinstance(v, dict)}
        if scalar_keys:
            llm_profiles["default"] = {
                "model": scalar_keys.get("model", ""),
                "embedding_model": scalar_keys.get("embedding_model", "text-embedding-3-large-1"),
                "base_url": scalar_keys.get("base_url", ""),
                "api_key": scalar_keys.get("api_key", ""),
                "max_tokens": scalar_keys.get("max_tokens", 4096),
                "max_completion_tokens": scalar_keys.get("max_completion_tokens", 4096),
                "max_input_tokens": scalar_keys.get("max_input_tokens"),
                "temperature": scalar_keys.get("temperature", 1.0),
                "api_type": scalar_keys.get("api_type", ""),
                "api_version": scalar_keys.get("api_version", ""),
            }
        # Sub-tables like [llm.embedding] / [llm.azure] become named profiles,
        # inheriting any default scalar keys.
        for name, override in overrides.items():
            merged = {**llm_profiles.get("default", {}), **override}
            llm_profiles[name] = merged

        self._config = AppConfig(llm={k: LLMSettings(**v) for k, v in llm_profiles.items()})

    @property
    def llm(self) -> Dict[str, LLMSettings]:
        return self._config.llm


config = Config()
