"""Locate runtime data in both layouts: installed wheel (fastthaig2p/data,
via the force-include in pyproject.toml) and repo checkout (../data)."""

from pathlib import Path

_PKG = Path(__file__).parent


def data_path(*parts: str) -> Path:
    installed = _PKG / "data" / Path(*parts)
    if installed.exists():
        return installed
    return _PKG.parent / "data" / Path(*parts)
