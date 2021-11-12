class UnknownImageFormat(Exception):
    pass


class FormatNotAvailable(Exception):
    pass


class ImageIOError(Exception):
    pass


__all__ = [UnknownImageFormat, FormatNotAvailable, ImageIOError]