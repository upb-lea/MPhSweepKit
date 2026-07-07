from pathlib import Path
import mph


def set_directory(directory_name: str):
    """
    Set directory for the geometric sweep to the absolute path of the specified directory.

    :param directory_name: Name of the directory to set.
    """
    if not Path(directory_name).exists():
        Path(directory_name).mkdir()
        print(f"Directory '{directory_name}' created.")
    else:
        print(f"Directory '{directory_name}' already exists. Using existing directory.")


def set_batch_directory(parametric_sweep: mph.Node, directory_name: str = "batch_data"):
    """
    Set batch directory for the geometric sweep to the absolute path of the "server_dir" directory

    :param parametric_sweep: The parametric sweep object to set the batch directory for.
    :param directory_name: Name of the batch directory to set.
    """
    set_directory(directory_name)
    absolute_path = Path(directory_name).absolute()
    parametric_sweep.property("batchdir", str(absolute_path))
    print(f"Batch directory for the geometric sweep set to: {absolute_path}")


def set_server_directory(parametric_sweep: mph.Node, directory_name: str = "server_dir"):
    """
    Set server directory for the geometric sweep to the absolute path of the "server_dir" directory

    :param parametric_sweep: The parametric sweep object to set the server directory for.
    :param directory_name: Name of the server directory to set.
    """
    set_directory(directory_name)
    absolute_server_path = Path(directory_name).absolute()
    parametric_sweep.property("serverdir", str(absolute_server_path))
    print(f"Server directory for the geometric sweep set to: {absolute_server_path}")
