try:
    from importlib.metadata import PackageNotFoundError, version
except ModuleNotFoundError:
    # Try backported to PY<38 `importlib_metadata`.
    from importlib_metadata import PackageNotFoundError, version


try:
    __version__ = version("etflow")
except PackageNotFoundError:
    # package is not installed
    __version__ = "dev"
