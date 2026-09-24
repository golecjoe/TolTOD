"""NetCDF ingestion for TolTEC TOD files."""

from .reader import NetworkTOD, TolTECFile, open_tod

__all__ = ["NetworkTOD", "TolTECFile", "open_tod"]

