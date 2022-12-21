import re
from os import PathLike
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Sequence, Tuple, TypedDict, ClassVar

import numpy
import magic
from numpy.typing import NDArray, ArrayLike


MAGIC_FILE = Path(__file__).parent.joinpath('data', 'magic')


def get_tags(filename: Union[str, Path]) -> Tuple[str, ...]:
    """Identify the file format using libmagic and return corresponding tags

    :param filename: Path | str
    :return: Tuple[str, ...]
    """

    try:
        m = magic.Magic(magic_file=str(MAGIC_FILE))
        tag_string =  m.from_file(str(filename)).strip()
        return tuple(re.split(r'\s*[,:]\s*', tag_string))
    except magic.MagicException:
        return 'data',


def summarize_sequence(values: ArrayLike) -> List[Tuple[int, int]]:
    """Compress an array of integers into a list of tuple pairs representing contiguous ranges of values.
    For example,  summarize_sequence([1,2,3,4,6,7,8,11]) -> [(1,4),(6,8),(11,11)]

    :param values: ArrayLike
    :return: List[Tuple[int, int]]
    """
    return [
        (sweep[0], sweep[-1])
        for sweep in  numpy.split(values, numpy.where(numpy.diff(values) > 1)[0] + 1)
        if len(sweep)
    ]


def all_subclasses(cls):
    direct = cls.__subclasses__()
    for subclass in direct:
        yield subclass
        for indirect in all_subclasses(subclass):
            yield indirect


class FrameHeader(TypedDict):
    detector: str
    format: str
    pixel_size: Tuple[float, float]
    center: Tuple[float, float]
    size: Tuple[int, int]
    distance: float
    two_theta: float
    exposure: float
    wavelength: float
    start_angle: float
    delta_angle: float
    saturated_value: float
    sensor_thickness: float


@dataclass
class OnDiskInfo:
    name: str
    index: int
    template: str
    frames: ArrayLike


@dataclass
class ImageFrame:
    header: FrameHeader
    data: Union[NDArray, None] = None


class DataSet(ABC):
    name: str
    directory: Path
    template: str
    regex: str
    reference: str
    index: int
    sweeps: List[Tuple[int, int]]
    identifier: str
    frame: Union[ImageFrame, None] = None

    def __init__(self, file_path: Union[PathLike, str]):
        file_path = Path(file_path).absolute()
        self.reference = file_path.name
        self.directory = file_path.parent
        info = self.find_frames(self.directory, self.reference)
        self.index = info.index
        self.template = info.template
        self.name = info.name
        self.sweeps = summarize_sequence(info.frames)
        self.frame = self.get_frame(self.index)

        self.identifier = hashlib.blake2s(
            bytes(self.directory) + self.name.encode('utf-8'), digest_size=16
        ).hexdigest()

    @classmethod
    def find_frames(cls, directory: Path, reference: str) -> OnDiskInfo:
        """
        Find dataset sweeps corresponding to this dataset and update the attributes,
        'name', 'index', 'regex', 'template', and 'sweeps'

        :param directory:  file path
        :param reference:  Reference file name
        """
        pattern = re.compile(
            r'^(?P<root_name>[\w_-]+?)(?P<separator>[._-]?)'
            r'(?P<field>\d{3,12})(?P<extension>(?:\.\D\w+)?)$'
        )
        matched = pattern.match(reference)
        if matched:
            params = matched.groupdict()
            width = len(params['field'])
            name = params['root_name']
            index = int(params['field'])
            template = '{root_name}{separator}{{field}}{extension}'.format(**params)
            frame_pattern = re.compile(
                r'^{root_name}{separator}(\d{{{width}}}){extension}$'.format(width=width, **params)
            )
            frames = numpy.array([
                int(frame_match.group(1)) for file_name in directory.iterdir()
                for frame_match in [frame_pattern.match(file_name.name)]
                if file_name.is_file() and frame_match
            ], dtype=int)
            frames.sort()

        else:
            name = ""
            index = 0
            template = ""
            frames = numpy.array([])
        return OnDiskInfo(name=name, index=index, template=template, frames=frames)

    @classmethod
    def new_from_file(cls, filename: str):

        for format_class in all_subclasses(cls):
            print(format_class.tags)

    @abstractmethod
    def get_frame(self, index:  int) -> ImageFrame:
        ...

    @abstractmethod
    def next_frame(self) -> ImageFrame:
        ...

    @abstractmethod
    def prev_frame(self) -> ImageFrame:
        ...

    @classmethod
    @abstractmethod
    def understands(cls, tags: Sequence[str]) -> bool:
        ...





