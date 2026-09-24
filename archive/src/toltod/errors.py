"""Exceptions raised by TolTOD ingestion."""


class TolTODReadError(RuntimeError):
    """Base exception for errors encountered while reading a TOD file."""


class SchemaError(TolTODReadError):
    """The input file does not satisfy the configured TOD schema."""


class ReaderClosedError(TolTODReadError):
    """An operation requiring the NetCDF file was attempted after closing it."""


class UnknownNetworkError(TolTODReadError, KeyError):
    """The requested ``apt_nw`` value is not present in the file."""


class RemediationError(RuntimeError):
    """A remediation plan cannot be safely applied to the supplied signal."""
