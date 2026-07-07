import json
from pathlib import Path
import pandas as pd
from typing import Any


def load_post_processing_exprs(json_path: str | Path) -> dict[str, dict[str, str]]:
    """
    Load post-processing expressions from a JSON file.

    Parameters
    ----------
    json_path : str | Path
        Path to the JSON file.

    Returns
    -------
    dict[str, dict[str, str]]
        Dictionary like:
        {
          "p_loss": {"expression": "...", "unit": "...", "label": "..."},
          ...
        }

    Raises
    ------
    FileNotFoundError
        If the JSON file does not exist.
    ValueError
        If the JSON structure is invalid.
    """
    path = Path(json_path)
    print(path)

    with path.open("r", encoding="utf-8") as f:
        data: Any = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object/dict.")

    required_keys = {"expression", "unit", "label"}
    for name, entry in data.items():
        if not isinstance(name, str):
            raise ValueError("All top-level keys must be strings.")
        if not isinstance(entry, dict):
            raise ValueError(f"Entry '{name}' must be an object/dict.")
        missing = required_keys - set(entry.keys())
        if missing:
            raise ValueError(f"Entry '{name}' is missing keys: {sorted(missing)}")
        for key in required_keys:
            if not isinstance(entry[key], str):
                raise ValueError(f"Entry '{name}' key '{key}' must be a string.")

    return data


def read_comsol_txt_to_df(filepath: str) -> pd.DataFrame:
    """
    Read a COMSOL-style .txt export into a pandas DataFrame.

    :param filepath: Path to the COMSOL .txt file.
    :return: DataFrame containing the data from the file.
    """
    header_cols = None
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("%"):
                content = line.lstrip("%").strip()
                if content.startswith("x"):
                    # split on 2+ spaces so multi-word final column name stays intact
                    header_cols = (
                        pd.Series(content).str.split(r"\s{2,}", regex=True).iloc[0]
                    )
                    break

    if header_cols is None:
        raise ValueError("Could not find header line starting with '% x'.")

    df = pd.read_csv(
        filepath,
        sep=r"\s+",  # <- modern replacement for delim_whitespace=True
        comment="%",
        header=None,
        names=header_cols,
        engine="python",  # robust with regex separators
    )

    return df