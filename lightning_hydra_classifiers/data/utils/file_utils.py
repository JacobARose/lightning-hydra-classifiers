"""


"""

import os
from pathlib import Path
from typing import Union


def ensure_dir_exists(path: Union[str,Path]) -> None:
    """
    Ensures that the directory located at `path` is created if it doesn't already exist. Raises error if path either represents an existing file or does not contain a valid dir path.
    """
#     path = str(path)
    assert not os.path.isfile(path)
    os.makedirs(path, exist_ok=True)
    assert os.path.isdir(path)