import sys
import re
from pathlib import Path
import os.path
import re
import sys
from os import PathLike
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Union, List, Optional, Tuple, TypedDict, ClassVar, BinaryIO, Sequence
import numpy
from numpy.typing import NDArray, ArrayLike


if sys.version_info < (3, 10):
    from importlib_metadata import entry_points, version, PackageNotFoundError
else:
    from importlib.metadata import entry_points, version, PackageNotFoundError


try:
    __version__ = version("mxio")
except PackageNotFoundError:
    __version__ = "unknown version"


__all__ = [
    'UnknownDataFormat',
    'read_image',
    'read_header',
    'XYPair',
    'HeaderAttrs',
    'ImageFrame',
    'DataSetAttrs',
    'DataSet',
]

class UnknownDataFormat(Exception):
    ...


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
    sigma: float
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
    sigma: Optional[float] = field(repr=False, default=1.0)
    overloads: Optional[int] = field(repr=False, default=0)
    sensor_thickness: Optional[float] = field(repr=False, default=0.0)
    data: Union[NDArray, None] = field(repr=False, default=None)


class DataSetAttrs(TypedDict):
    name: str
    directory: Union[Path, None]
    template: str
    index: int
    series: ArrayLike
    identifier: str


class DataSet(ABC):
    name: str
    directory: Path
    series: ArrayLike
    identifier: str
    template: str
    reference: str
    glob: str
    index: int
    size: int
    tags: Tuple[str, ...]
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
        self.size = 0
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

        params = find_sweep(self.directory / self.reference)
        pattern = params['pattern']
        if pattern:
            frames = numpy.array([
                int(frame_match.group(1)) for file_path in self.directory.iterdir()
                for frame_match in [pattern.match(file_path.name)]
                if file_path.is_file() and frame_match
            ], dtype=int)
            frames.sort()
        else:
            frames = numpy.array([])

        self.name = params['name']
        self.glob = params['glob']
        self.index = params['index']
        self.template = params['template']
        self.series = frames
        self.size = len(frames)

        header, data = self.read_file(self.directory / self.reference)
        self.set_frame(header, data, self.index)

    @classmethod
    def save_frame(cls, file_path: Union[PathLike, str], frame: ImageFrame):
        """Save the image frame to disk.

        :param frame: ImageFrame to save to file
        :param file_path: full path to file
        """
        raise RuntimeWarning("This format does not support exporting")

    @classmethod
    def new_from_file(cls, filename: Union[str, Path]) -> "DataSet":
        """Create and return an instance of a DataSet subclass providing the appropriate format

        :param filename: str
        :return: DataSet instance
        """

        test_path = Path(filename)
        test_path = test_path.parent if re.match(r'^\d+$', test_path.name) else test_path

        with open(test_path, 'rb') as file:
            name, extension = os.path.splitext(test_path)
            format_cls, tags = cls.get_format(file, extension)
            if format_cls is None:
                raise UnknownDataFormat(f"File {test_path} not understood by any available file format plugins!")
            else:
                return format_cls(filename, tags=tags)

    @classmethod
    def get_format(cls, file: BinaryIO, extension: str) -> Tuple[Union[type["DataSet"], None], Tuple[str, ...]]:
        """
        Find the best Dataset format class for this dataset and the corresponding tuple of tags
        The best format is the deepest subclass which returns tags for the file.

        :param extension: File extension
        :param file: BinaryIO
        :return: A concrete dataset class or None and a tuple of tags
        """

        for plugin in entry_points(group='mxio.plugins'):
            plugin.load()

        for base_cls in cls.__subclasses__():
            tags = base_cls.identify(file, extension)
            file.seek(0)
            if tags:
                for sub_cls in base_cls.__subclasses__():
                    extra_tags = sub_cls.identify(file, extension)
                    file.seek(0)
                    if extra_tags and extra_tags != tags:   # ignore subclasses which do nothing new
                        return sub_cls, tags + extra_tags
                else:
                    return base_cls, tags
        else:
            return None, ()

    def get_frame(self, index: int) -> Union[ImageFrame, None]:
        """
        Fetch and return a frame given an index

        :param index: Frame number
        :return: ImageFrame or None if no frame of the given number exists
        """

        if index in self.series:
            file_name = self.directory.joinpath(self.template.format(field=index))
            header, data = self.read_file(file_name)
            self.set_frame(header, data, index)
            return self.frame

    def next_frame(self) -> Union[ImageFrame, None]:
        """
        Load and return the next Frame in the dataset. Also updates the current frame and index to this
        frame.

        :return: ImageFrame | None if there is no next frame in the sequence
        """
        next_pos = numpy.searchsorted(self.series, self.index) + 1
        if next_pos < len(self.series):
            return self.get_frame(self.series[next_pos])

    def prev_frame(self) -> Union[ImageFrame, None]:
        """
        Load and return the previous Frame in the dataset. Also updates the current frame and index to this
        frame.

        :return: ImageFrame | None if there is no previous frame in the sequence
        """

        prev_pos = numpy.searchsorted(self.series, self.index) - 1
        if prev_pos >= 0:
            return self.get_frame(self.series[prev_pos])

    def set_frame(self, header: HeaderAttrs, data: NDArray, index: int = 1):
        """
        Set the current frame and frame index directly from a frame instance.

        :param header: dictionary of header meta data
        :param data: Image array
        :param index: int Frame index, defaults ot 1 if not provided
        """

        # calculate statistics if missing in header
        if any(key not in header for key in ("average", "minimum", "maximum", "overloads", "sigma")):
            w, h = numpy.array(data.shape) // 2
            quadrant_data = data[:h, :w]
            mask = (quadrant_data > 0)
            stats_data = quadrant_data[mask] if mask.sum() > 0 else quadrant_data

            average = stats_data.mean()
            maximum = stats_data.max()
            minimum = stats_data.min()
            residuals = (stats_data - average).ravel()
            sigma = numpy.sqrt(numpy.dot(residuals, residuals)/stats_data.size)

            header.update({
                "maximum": maximum,
                "average": average,
                "minimum": minimum,
                "sigma": sigma,
                "overloads": 4*(data >= header['cutoff_value']).sum()
            })

        frame = ImageFrame(**header, data=data)
        self.frame = frame
        self.index = index

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

    @classmethod
    @abstractmethod
    def identify(cls, file: BinaryIO, extension: str) -> Tuple[str, ...]:
        """Identify a file and return a tuple of tags describing the file.
        Should return an empty tuple if the file is not recognized.

        :param file: BinaryIO
        :param extension: File name extension
        :return:  Tuple[str, ...]
        """
        ...

    @abstractmethod
    def read_file(self, filename: Union[str, Path]) -> Tuple[HeaderAttrs, NDArray]:
        """
        Read
        :param filename: file to read
        :return: Tuple[dict, NDArray]
        """
        ...


def read_image(path: str) -> Union[DataSet, None]:
    """
    Determine the file type open the image using the correct image IO
    back-end, and return an image frame
    :param path: str
    """

    dset = DataSet.new_from_file(path)
    return dset


def read_header(path) -> HeaderAttrs:
    """
    Determine the file type open the image using the correct image IO
    back-end, and return an image frame
    :param path: str
    """

    dset = DataSet.new_from_file(path)
    return asdict(dset.frame)


def find_sweep(path: Path, name=r'[\w_-]+?') -> dict:
    """
    Find the disk representation of a dataset from a single frame in a series
    :param path: full path to one image
    :param name: optional raw regex string representing the root name of the dataset
    :return: dictionary containing the "name", "index", "template", "glob" and compiled regex "pattern"
    """

    pattern = re.compile(
        rf'^(?P<name>{name})'
        r'(?P<separator>[._-]?)'
        r'(?P<field>\d{3,12})'
        r'(?P<extension>(?:\.\D\w+)?)$'
    )
    directory = path.parent
    file_name = path.name

    matched = pattern.match(file_name)
    if matched:
        params = matched.groupdict()
        width = len(params['field'])
        index = int(params['field'])
        template = '{name}{separator}{{field:>0{width}}}{extension}'.format(width=width, **params)
        frame_pattern = re.compile(
            r'^{name}{separator}(\d{{{width}}}){extension}$'.format(width=width, **params)
        )
        glob = '{name}{separator}{wildcard}{extension}'.format(width=width, wildcard='?' * width, **params)
        name = params['name']

        # unusual frame names without field separator
        if width > 6:
            names = [
                file_path.name for file_path in directory.iterdir()
                if file_path.is_file() and frame_pattern.match(file_path.name)
            ]
            common_name = os.path.commonprefix(names)
            return find_sweep(path, name=common_name)
        else:
            return {
                'name': name,
                'index': index,
                'template': template,
                'glob': glob,
                'pattern': frame_pattern
            }
    return {
        'name': file_name,
        'index': 0,
        'template': file_name,
        'glob': file_name,
        'pattern': ''
    }