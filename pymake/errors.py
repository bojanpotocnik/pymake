from typing import TYPE_CHECKING

if TYPE_CHECKING:  # To prevent cyclic and unused imports in runtime.
    from . import parserdata


class MakeError(Exception):
    def __init__(self, message: str, loc: 'parserdata.Location' = None):
        self.msg: str = message
        self.loc: 'parserdata.Location' = loc

    def __str__(self):
        location_str = ''
        if self.loc is not None:
            location_str = "%s:" % (self.loc,)

        return "%s%s" % (location_str, self.msg)


class MakeSyntaxError(MakeError):
    pass


class DataError(MakeError):
    pass


class ResolutionError(DataError):
    """
    Raised when dependency resolution fails, either due to recursion or to missing
    prerequisites.This is separately catchable so that implicit rule search can try things
    without having to commit.
    """
    pass


class PythonError(Exception):
    def __init__(self, message, exitcode):
        Exception.__init__(self)
        self.message = message
        self.exitcode = exitcode

    def __str__(self):
        return self.message
