import json
from pathlib import Path
import pandas as pd
from typing import Any


def load_post_processing_exprs(
    json_path: str | Path, print_info: bool = False
) -> dict[str, dict[str, str]]:
    """
    Load post-processing expressions from a JSON file.

    Parameters
    ----------
    json_path : str | Path
        Path to the JSON file.
    print_info : bool, optional
        If True, print info about the resolved file path and a formatted
        preview of the loaded JSON content, by default False.

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
    if print_info:
        print(f"Loading post-processing expressions from: {path.resolve()}")

    with path.open("r", encoding="utf-8") as f:
        data: Any = json.load(f)

    if print_info:
        print("Loaded post-processing expressions:")
        print(json.dumps(data, indent=2, ensure_ascii=False))

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



def read_comsol_dataset(filename):

    dimension = None
    expressions = None
    header = None

    with open(filename) as f:
        for line in f:
            if line.startswith("% Dimension:"):
                dimension = int(line.split(":", 1)[1])
            elif line.startswith("% Expressions:"):
                expressions = int(line.split(":", 1)[1])
            elif line.startswith("% x"):
                header = line[1:].split()

    if dimension is int and expressions is int and header is list:

        coordinates = {
            1: ["x"],
            2: ["x", "y"],
            3: ["x", "y", "z"],
        }.get(dimension)

        if coordinates is None:
            raise ValueError("Dimension must be 1, 2, or 3")

        words = header[dimension:]
        expression_names = []
        start = 0

        for i, word in enumerate(words):
            if word == "@" and i + 1 < len(words):
                expression_names.append(" ".join(words[start:i + 2]))
                start = i + 2

        if len(expression_names) != expressions:
            raise ValueError("Could not parse all expression names")

        df = pd.read_csv(filename, sep=r"\s+", comment="%", header=None)
        df.columns = coordinates + expression_names

        return df
    
    else:
        raise ValueError(f"Could not parse file header for dimension {dimension}, expressions {expressions}, and header {header}")