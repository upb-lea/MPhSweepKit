import pandas as pd


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
