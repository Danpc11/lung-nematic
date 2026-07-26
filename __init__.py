"""Tools for nematic analysis of lung histology images."""

from importlib.metadata import PackageNotFoundError, version

from .batch import analyze_folder
from .config import AnalysisConfig, load_config
from .pipeline import analyze_image

try:
    __version__ = version("lung-nematic")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0+local"

__all__ = [
    "AnalysisConfig",
    "__version__",
    "analyze_folder",
    "analyze_image",
    "load_config",
]
