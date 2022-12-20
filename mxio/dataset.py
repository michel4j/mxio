import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Sequence, Tuple, TypedDict, ClassVar, BinaryIO

import numpy
from numpy.typing import NDArray, ArrayLike


def summarize_sequence(values: ArrayLike) -> List[Tuple[int, int]]:
    return [
        (sweep[0], sweep[-1])
        for sweep in  numpy.split(values, numpy.where(numpy.diff(values) > 1)[0] + 1)
        if len(sweep)
    ]


class FrameHeader(TypedDict):
    detector: str
    format: str
    pitch: Tuple[float, float]
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
class ImageFrame:
    header: FrameHeader
    data: Union[NDArray, None] = None


class DataSet(ABC):
    name : str
    directory: Path
    template: str
    regex: str
    reference: Path
    index: int
    sweeps: List[Tuple[int, int]]
    identifier: str
    frame: Union[ImageFrame, None] = None
    description: ClassVar[str] = ""

    def __init__(self, file_path: Union[Path, str]):
        file_path = Path(file_path).absolute()

        self.reference = file_path.name
        self.directory = file_path.parent

        pattern = re.compile(
            r'^(?P<root_name>[\w_-]+?)(?P<separator>(?:[._-])?)'
            r'(?P<field>\d{3,12})(?P<extension>(?:\.\D\w+)?)$'
        )
        matched = pattern.match(self.reference)
        if matched:
            params = matched.groupdict()
            width = len(params['field'])
            self.name = params['root_name']
            self.index = int(params['field'])
            self.regex = '^{root_name}{separator}(\d{{{width}}}){extension}$'.format(width=width, **params)
            self.template = '{root_name}{separator}{{field}}{extension}'.format(**params)
            frame_pattern = re.compile(self.regex)
            frames = numpy.array([
                int(frame_match.group(1)) for file_name in self.directory.iterdir()
                for frame_match in [frame_pattern.match(file_name.name)]
                if file_name.is_file() and frame_match
            ], dtype=int)
            frames.sort()
            self.sweeps = summarize_sequence(frames)
        else:
            raise ValueError('Invalid dataset file')

        self.frame = self.get_frame(self.index)

    @abstractmethod
    def understands(self, tags: Sequence[str]) -> bool:
        ...


    @abstractmethod
    def get_frame(self, index:  int) -> ImageFrame:
        ...

    @abstractmethod
    def next_frame(self) -> ImageFrame:
        ...

    @abstractmethod
    def prev_frame(self) -> ImageFrame:
        ...






