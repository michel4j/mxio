import re
import sys
from os import PathLike
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, List, Optional, Tuple, TypedDict, ClassVar, BinaryIO, Sequence

import numpy
from numpy.typing import NDArray, ArrayLike
from mxio.identify import get_tags, TagInfo


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


class HeaderAttrs(TypedDict):
    detector: str
    serial_number: str
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
    cutoff_value: float
    maximum: float
    minimum: float
    average: float
    overloads: int
    sensor_thickness: float


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
    cutoff_value: float = field(repr=False)
    serial_number: str = field(repr=False, default='00-000')
    maximum: Optional[float] = field(repr=False, default=0.0)
    minimum: Optional[float] = field(repr=False, default=0.0)
    average: Optional[float] = field(repr=False, default=0.0)
    overloads: Optional[int] = field(repr=False, default=0)
    sensor_thickness: Optional[float] = field(repr=False, default=0.0)
    data: Union[NDArray, None] = field(repr=False, default=None)


class DataSetAttrs(TypedDict):
    name: str
    directory: Union[Path, None]
    template: str
    index: int
    series: Sequence[Tuple[int, int]]
    identifier: str


class DataSet(ABC):
    name: str
    directory: Path
    series: ArrayLike
    identifier: str
    template: str
    reference: str
    index: int
    tags: Tuple[str, ...]
    magic: ClassVar[List[TagInfo]]
    frame: Union[ImageFrame, None]

    def __init__(
            self, file_path: Union[PathLike, str, None] = None,
            tags: Tuple[str, ...] = (),
            attrs: Union[DataSetAttrs, None] = None
    ):
        self.name = ""
        self.directory = Path()
        self.series = numpy.array([])
        self.template = ""
        self.reference = ""
        self.index = 0
        self.tags = tags

        if file_path is not None:
            file_path = Path(file_path).absolute()
            self.reference = file_path.name
            self.directory = file_path.parent
            self.setup()
        elif attrs is not None:
            self.name = attrs['name']
            self.index = attrs['index']
            self.template = attrs['template']
            self.series = attrs.get('series', numpy.array([]))

        self.identifier = hashlib.blake2s(
            bytes(self.directory) + self.name.encode('utf-8'), digest_size=16
        ).hexdigest()

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(name={self.name!r}, '
            f'identifier={self.identifier!r}, '
            f'template={self.template!r}, '
            f'directory={self.directory!r})'
        )

    def setup(self):
        """
        Find dataset sweeps corresponding to this dataset on disk and update the attributes,
        'name', 'index', 'regex', 'template', 'frame' and 'sweeps'
        """

        pattern = re.compile(
            r'^(?P<name>[\w_-]+?)(?P<separator>[._-]?)'
            r'(?P<field>\d{3,12})(?P<extension>(?:\.\D\w+)?)$'
        )
        matched = pattern.match(self.reference)
        if matched:
            params = matched.groupdict()
            width = len(params['field'])
            name = params['name']
            index = int(params['field'])
            template = '{name}{separator}{{field:>0{width}}}{extension}'.format(width=width, **params)
            frame_pattern = re.compile(
                r'^{name}{separator}(\d{{{width}}}){extension}$'.format(width=width, **params)
            )
            frames = numpy.array([
                int(frame_match.group(1)) for file_name in self.directory.iterdir()
                for frame_match in [frame_pattern.match(file_name.name)]
                if file_name.is_file() and frame_match
            ], dtype=int)
            frames.sort()

        else:
            name = ""
            index = 0
            template = ""
            frames = numpy.array([])

        self.name = name
        self.index = index
        self.template = template
        self.series = frames
        self.frame = self.get_frame(self.index)

    def get_frame(self, index: int) -> Union[ImageFrame, None]:
        if index in self.series:
            file_name = self.directory.joinpath(self.template.format(field=index))
            header, data = self.read_file(file_name)

            # calculate statistics if missing in header
            if any(key not in header for key in ("average", "minimum", "maximum", "overloads")):
                w, h = numpy.array(data.shape) // 2
                stats_data = data[:h, :w]
                mask = stats_data > 0
                header.update({
                    "maximum": stats_data[mask].max(),
                    "average": stats_data[mask].mean(),
                    "minimum": stats_data[mask].min(),
                    "overloads": 4 * (stats_data[mask] >= header['cutoff_value']).sum()
                })
            frame = ImageFrame(**header, data=data)
            self.frame = frame
            self.index = index
            return self.frame

    @classmethod
    def save_frame(cls, file_path: Union[PathLike, str], frame: ImageFrame):
        """Save the image frame to disk.

        :param frame: ImageFrame to save to file
        :param file_path: full path to file
        """
        raise RuntimeWarning("This format does not support exporting")

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

    def next_frame(self) -> Union[ImageFrame, None]:
        """
        Load and return the next Frame in the dataset. Also updates the current frame and index to this
        frame.

        :return: ImageFrame | None if there is no next frame in the sequence
        """
        return self.get_frame(self.index + 1)

    def prev_frame(self) -> Union[ImageFrame, None]:
        """
        Load and return the previous Frame in the dataset. Also updates the current frame and index to this
        frame.

        :return: ImageFrame | None if there is no previous frame in the sequence
        """
        return self.get_frame(self.index - 1)

    def set_frame(self, frame: ImageFrame, index: int):
        """
        Set the current frame and frame index directly from a frame instance.

        :param frame: ImageFrame
        :param index: int
        """
        self.index = index
        self.frame = frame

    def get_sweeps(self) -> Sequence[Tuple[int, int]]:
        """
        Compress an the frame series which is a sequence of ints into a sequence of tuple pairs representing
        contiguous ranges of values.
        For example,  summarize_sequence([1,2,3,4,6,7,8,11]) -> [(1,4),(6,8),(11,11)]

         :return: Sequence[Tuple[int, int]]
         """

        return tuple(
            (sweep[0], sweep[-1])
            for sweep in numpy.split(numpy.array(self.series), numpy.where(numpy.diff(self.series) > 1)[0] + 1)
            if len(sweep)
        )

    @abstractmethod
    def read_file(self, filename: Union[str, Path]) -> Tuple[HeaderAttrs, NDArray]:
        """
        Read
        :param filename: file to read
        :return: Tuple[dict, NDArray]
        """
        ...

