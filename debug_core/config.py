from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass(slots=True)
class DebugAppConfig:
    path: Path
    raw: Any
    debug_mode: dict
    runtime: dict
    wan_export: dict
    debug: dict
    refine: dict
    reprompt: dict
    upstream: dict


def load_debug_config(path: str | Path) -> DebugAppConfig:
    path = Path(path).resolve()
    cfg = OmegaConf.load(str(path))
    return DebugAppConfig(
        path=path,
        raw=cfg,
        debug_mode=dict(OmegaConf.to_container(cfg.debug_mode, resolve=True) or {}),
        runtime=dict(OmegaConf.to_container(cfg.runtime, resolve=True) or {}),
        wan_export=dict(OmegaConf.to_container(cfg.wan_export, resolve=True) or {}),
        debug=dict(OmegaConf.to_container(cfg.debug, resolve=True) or {}),
        refine=dict(OmegaConf.to_container(cfg.refine, resolve=True) or {}),
        reprompt=dict(OmegaConf.to_container(cfg.reprompt, resolve=True) or {}),
        upstream=dict(OmegaConf.to_container(getattr(cfg, "upstream", {}), resolve=True) or {}),
    )
