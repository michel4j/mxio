import re
import sys
from os import PathLike
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, List, Optional, Tuple, TypedDict, ClassVar, BinaryIO

import numpy
import magic
from numpy.typing import NDArray, ArrayLike
from mxio.identify import get_tags, TagInfo


if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points


class UnknownDataFormat(Exception):
    ...


def summarize_sequence(values: ArrayLike) -> List[Tuple[int, int]]:
    """Compress an array of integers into a list of tuple pairs representing contiguous ranges of values.
    For example,  summarize_sequence([1,2,3,4,6,7,8,11]) -> [(1,4),(6,8),(11,11)]

    :param values: ArrayLike
    :return: List[Tuple[int, int]]
    """
    return [
        (sweep[0], sweep[-1])
        for sweep in numpy.split(values, numpy.where(numpy.diff(values) > 1)[0] + 1)
        if len(sweep)
    ]


@dataclass
class XYPair:
    x: Union[int, float]
    y: Union[int, float]


@dataclass
class DataDescription:
    name: str
    index: int
    template: str
    frames: ArrayLike


@dataclass
class ImageFrame:
    detector: str
    format: str
    filename: str
    pixel_size: XYPair
    center: XYPair
    size: XYPair
    distance: float
    two_theta: float
    exposure: float
    wavelength: float
    start_angle: float
    delta_angle: float
    saturated_value: float = field(repr=False)
    maximum: Optional[float] = field(repr=False, default=0.0)
    minimum: Optional[float] = field(repr=False, default=0.0)
    average: Optional[float] = field(repr=False, default=0.0)
    overloads: Optional[int] = field(repr=False, default=0)
    sensor_thickness: Optional[float] = field(repr=False, default=0.0)

    data: Union[NDArray, None] = field(repr=False, default=None)


class DataSet(ABC):
    name: str
    directory: Path
    template: str
    regex: str
    reference: str
    index: int
    sweeps: List[Tuple[int, int]]
    identifier: str
    tags: Tuple[str, ...]
    magic: ClassVar[List[TagInfo]]
    frame: Union[ImageFrame, None] = None

    def __init__(self, file_path: Union[PathLike, str], tags: Tuple[str, ...] = ()):
        file_path = Path(file_path).absolute()
        self.reference = file_path.name
        self.directory = file_path.parent
        self.tags = tags
        info = self.find_frames(self.directory, self.reference)
        self.index = info.index
        self.template = info.template
        self.name = info.name
        self.sweeps = summarize_sequence(info.frames)
        self.frame = self.get_frame(self.index)

        self.identifier = hashlib.blake2s(
            bytes(self.directory) + self.name.encode('utf-8'), digest_size=16
        ).hexdigest()

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name!r}, identifier={self.identifier!r}, template={self.template!r}, directory={self.directory!r})'

    @classmethod
    def save_frame(cls, file_path: Union[PathLike, str], frame: ImageFrame):
        """Save the image frame to disk.

        :param frame: ImageFrame to save to file
        :param file_path: full path to file
        """
        raise NotImplementedError("This format does not support exporting")

    @classmethod
    def find_frames(cls, directory: Path, reference: str) -> DataDescription:
        """
        Find dataset sweeps corresponding to this dataset and update the attributes,
        'name', 'index', 'regex', 'template', and 'sweeps'

        :param directory:  file path
        :param reference:  Reference file name
        :return: DataDescription object
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
            template = '{root_name}{separator}{{field:>0{width}}}{extension}'.format(width=width, **params)
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
        return DataDescription(name=name, index=index, template=template, frames=frames)

    @classmethod
    def new_from_file(cls, filename: str) -> "DataSet":
        """Create and return an instance of a DataSet subclass providing the appropriate format

        :param filename: str
        :return: DataSet instance
        """

        with open(filename, 'rb') as file:
            format_cls, tags = cls.get_format(file)
            if format_cls is None:
                raise UnknownDataFormat(f"File {filename} not understood by any available file format plugins!")
            else:
                return format_cls(filename, tags=tags)

    @classmethod
    def get_format(cls, file: BinaryIO) -> Tuple[Union[type["DataSet"], None], Tuple[str, ...]]:
        """
        Find the best Datset format class for this dataset and a corresponding tuple of tags

        :param file: BinaryIO
        :return: A concrete dataset class or None and a tuple of tags
        """

        for base_cls in cls.__subclasses__():
            tags = get_tags(file, base_cls.magic)
            if tags:
                for sub_cls in base_cls.__subclasses__():
                    extra_tags = get_tags(file, sub_cls.magic)
                    if extra_tags:
                        return sub_cls, tags + extra_tags
                else:
                    return base_cls, tags
        else:
            return None, ()

    @abstractmethod
    def get_frame(self, index: int) -> Union[ImageFrame, None]:
        ...

    @abstractmethod
    def next_frame(self) -> Union[ImageFrame, None]:
        ...

    @abstractmethod
    def prev_frame(self) -> Union[ImageFrame, None]:
        ...


format_plugins = entry_points(group='mxio.plugins')