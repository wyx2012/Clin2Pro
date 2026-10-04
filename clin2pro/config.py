from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


REQUIRED_DATA_DIRECTORIES = (
    "MIMIC用药前后记录",
    "横截面数据（用于预训练）",
    "两次记录146人",
    "通路数据",
    "药靶",
)


def _contains_data_directories(path: Path, names: Iterable[str]) -> bool:
    return all((path / name).is_dir() for name in names)


def _resolve_data_root(configured_root: str, project_root: Path) -> Path:
    raw = Path(configured_root).expanduser()
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        # Resolve relative paths against the project, independent of shell cwd.
        candidates.extend(
            [
                project_root / raw,
                project_root.parent / raw,
                Path.cwd() / raw,
            ]
        )
    checked = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in checked:
            continue
        checked.append(resolved)
        if _contains_data_directories(resolved, REQUIRED_DATA_DIRECTORIES):
            return resolved
    formatted = "\n".join("  - %s" % path for path in checked)
    required = ", ".join(REQUIRED_DATA_DIRECTORIES)
    raise FileNotFoundError(
        "Could not locate the biomedical data root.\n"
        "Checked:\n%s\n"
        "The data root must directly contain: %s\n"
        "Set data.root in the YAML to a relative path such as '.', './data', "
        "or '..'." % (formatted, required)
    )


def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(project_root)
    config["data"]["root"] = str(
        _resolve_data_root(config["data"].get("root", "."), project_root)
    )
    output = Path(config["output_dir"])
    if not output.is_absolute():
        output = project_root / output
    config["output_dir"] = str(output.resolve())
    return config
