"""Config loader — YAML (config/default.yaml) with ${ENV:default} interpolation.

Single source of truth for models, thresholds, DPI, paths. Nothing downstream
hardcodes these; everything reads the cached Settings singleton. .env is loaded
first so env vars override YAML defaults.
"""
from __future__ import annotations

import functools
import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

_ENV_TOKEN = re.compile(r"\$\{([A-Z0-9_]+)(?::([^}]*))?\}")


def _interpolate(node):
    """Recursively resolve ${ENV:default} tokens against os.environ."""
    if isinstance(node, dict):
        return {k: _interpolate(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_interpolate(v) for v in node]
    if isinstance(node, str):
        def sub(m):
            return os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else "")
        return _ENV_TOKEN.sub(sub, node)
    return node


# --- typed sections (extra='allow' so YAML can carry fields we haven't modeled yet) ---
class _Section(BaseModel):
    model_config = {"extra": "allow"}


class LLMCfg(_Section):
    provider: str = "openai"
    chat_model: str = "gpt-4o-mini"
    reason_model: str = "gpt-4o"
    embed_model: str = "text-embedding-3-small"
    temperature: float = 0.0


class ScannedCfg(_Section):
    dpi: int = 200
    deskew: bool = True
    threshold: str = "adaptive"
    ocr_engine: str = "paddleocr"
    device: str = "auto"           # auto|cpu|gpu
    cpu_threads: int = 12
    model_tier: str = "default"    # default (accurate) | mobile (fast) | server
    use_textline_orientation: bool = True
    ocr_conf_verify_below: float = 0.80
    vision_verify: bool = True
    vision_max_calls: int = 30


class IngestCfg(_Section):
    scanned: ScannedCfg = ScannedCfg()


class DeltaCfg(_Section):
    text_sim_weight: float = 0.8
    spatial_sim_weight: float = 0.2
    match_threshold: float = 0.5
    modify_min_text_delta: float = 0.05
    note_anchor_sim: float = 0.5   # min text-sim to trust note_number as the same note (else re-match by content)
    move_min_shift: float = 0.02
    geometry_layout_conf: float = 0.4
    large_bucket: int = 400
    use_llm_annotation: bool = True
    annotate_max: int = 40   # cap LLM significance annotations per comparison (cost bound)


class ChatCfg(_Section):
    top_k: int = 8
    vector_store: str = "chroma"
    chroma_dir: str = ".chroma"
    overview_llm: bool = False   # generate doc-overview chunk with an LLM (vs generic field-dump)
    router: dict = {}


class ObsCfg(_Section):
    langfuse_host: str = "http://localhost:3000"
    traces_dir: str = "traces"
    log_level: str = "INFO"


class PathsCfg(_Section):
    artifacts_dir: str = "artifacts"
    samples_dir: str = "data/samples"


class EvalCfg(_Section):
    datasets_dir: str = "eval/datasets"
    results_dir: str = "eval/results"
    judge_model: str = "gpt-4o"


class Settings(BaseModel):
    model_config = {"extra": "allow"}
    llm: LLMCfg = LLMCfg()
    ingest: IngestCfg = IngestCfg()
    delta: DeltaCfg = DeltaCfg()
    chat: ChatCfg = ChatCfg()
    observability: ObsCfg = ObsCfg()
    paths: PathsCfg = PathsCfg()
    eval: EvalCfg = EvalCfg()


@functools.lru_cache(maxsize=1)
def get_settings(config_path: str | None = None) -> Settings:
    """Load + validate the Settings singleton. CONFIG_PATH env overrides the path."""
    load_dotenv()  # populate os.environ from .env before interpolation
    path = Path(config_path or os.environ.get("CONFIG_PATH", "config/default.yaml"))
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    return Settings(**_interpolate(raw))
