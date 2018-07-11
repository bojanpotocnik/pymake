"""
Filename globbing like the python glob module with minor differences:

* glob relative to an arbitrary directory
* include . and ..
* check that link targets exist, not just links
"""

import fnmatch
import os
import re
from pathlib import Path
from typing import Union, List

from . import util

_glob_check = re.compile('[\[*?]')


def has_glob(p: str) -> bool:
    return _glob_check.search(p) is not None


def glob(fs_dir: Union[str, Path], path: Union[str, Path]) -> List[str]:
    """
    Yield paths matching the path glob. Sorts as a bonus. Excludes '.' and '..'
    """

    directory, leaf = os.path.split(path)
    if directory == '':
        return glob_pattern(fs_dir, leaf)

    if has_glob(directory):
        dirs_found = glob(fs_dir, directory)
    else:
        dirs_found = [directory]

    r = []

    for directory in dirs_found:
        fspath = util.normaljoin(fs_dir, directory)
        if not os.path.isdir(fspath):
            continue

        r.extend((util.normaljoin(directory, found) for found in glob_pattern(fspath, leaf)))

    return r


def glob_pattern(directory: Union[str, Path], pattern: str) -> List[str]:
    """
    Return leaf names in the specified directory which match the pattern.
    """

    if not has_glob(pattern):
        if pattern == '':
            if os.path.isdir(directory):
                return ['']
            return []

        if os.path.exists(util.normaljoin(directory, pattern)):
            return [pattern]
        return []

    leaves = os.listdir(directory) + ['.', '..']

    # "hidden" filenames are a bit special
    if not pattern.startswith('.'):
        leaves = [leaf for leaf in leaves
                  if not leaf.startswith('.')]

    leaves = fnmatch.filter(leaves, pattern)
    leaves = [l for l in leaves if os.path.exists(util.normaljoin(directory, l))]

    leaves.sort()
    return leaves
