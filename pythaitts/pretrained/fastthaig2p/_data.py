"""Locate runtime data in both layouts: installed wheel (fastthaig2p/data,
via the force-include in pyproject.toml) and repo checkout (../data).

## License

Copyright 2026 Charin Polpanumas and Amazon Web Services

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from pathlib import Path

_PKG = Path(__file__).parent


def data_path(*parts: str) -> Path:
    installed = _PKG / "data" / Path(*parts)
    if installed.exists():
        return installed
    return _PKG.parent / "data" / Path(*parts)
