from __future__ import annotations

from pathlib import Path

from characters.response_generation import load_jsonl_rows
from characters.trl_dpo_config import TrlDpoConfig


def extract_traits_from_trl_config(source_config: TrlDpoConfig) -> list[str]:
    paths = [source_config.dataset.train_data_path]
    if source_config.dataset.val_data_path is not None:
        paths.append(source_config.dataset.val_data_path)

    traits: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if path is None or not path.exists():
            continue
        for row in load_jsonl_rows(path):
            trait = str(row.get("trait", "")).strip()
            if not trait or trait in seen:
                continue
            seen.add(trait)
            traits.append(trait)

    if not traits:
        raise ValueError(
            "Could not derive any traits from the source TRL dataset. "
            "Expected at least one non-empty 'trait' field in the train/val JSONL."
        )
    return traits


def resolve_remote_adapter_dir(source_config: TrlDpoConfig, remote_results_root: Path) -> Path:
    return remote_results_root / source_config.name
