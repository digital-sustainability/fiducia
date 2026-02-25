import os
from pathlib import Path


def get_project_root_directory() -> Path:
    """Returns the root directory of the application."""
    return Path(__file__).parent.parent.parent


def relative_project_path(path: os.PathLike) -> Path:
    """
    Returns a Path object that is relative to the app directory. If the provided path is absolute, it returns the path as is.

    :param path: The path to convert.
    :return: A Path object that is relative to the app directory.
    """
    path = Path(path)
    if path.is_absolute():
        return path
    return get_project_root_directory() / path
