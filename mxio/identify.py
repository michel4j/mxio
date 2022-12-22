
from typing import BinaryIO, List, Tuple, TypedDict

__all__ = [
    "TagInfo",
    "get_tags"
]


class TagInfo(TypedDict):
    offset: int
    magic: str
    name: str


def get_tags(file: BinaryIO, specs: List[TagInfo]) -> Tuple[str, ...]:
    """Identify a file and return a tuple of tags based on the magic numbers provided in specs.
    Will only match later items if previous items matched. Returns an empty tuple
    if the first item does not match.

    :param file: BinaryIO
    :param specs: List[TagInfo]
    :return:  Tuple[str, ...]
    """
    matches = []

    for item in specs:
        file.seek(item["offset"])
        if file.read(len(item["magic"])) == item["magic"]:
            matches.append(item["name"])
        else:
            break

    return tuple(matches)


