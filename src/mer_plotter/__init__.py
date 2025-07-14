import gzip
import logging
from os import PathLike
from functools import lru_cache
from pathlib import Path
from shutil import copyfileobj
from shutil import move
from tempfile import gettempdir
from urllib.parse import urlparse
from urllib.request import urlopen
from warnings import catch_warnings
from warnings import filterwarnings
from warnings import warn

import cartopy.crs as ccrs
import dask.array as da
import numpy as np
import xarray as xr
from ipywidgets import Dropdown
from ipywidgets import SelectionSlider
from ipywidgets import interact
from matplotlib.pyplot import Figure


STATIC_FILES_URL = urlparse(
    "https://medeaf.ogs.it/internal-validation/gbolzon/MER/Domain_static_data"
)
DOMAIN_NAMES = ("nad", "sad", "ion", "tyr", "lig", "sar", "sic", "got", "gsn")
TEMP_DIR = Path(gettempdir()) / "mer_meshmasks"
_INTERNAL_DIR: Path | None = None


LOGGER = logging.getLogger(__name__)


def set_meshmask_dir(path: PathLike | str):
    """
    Sets the directory where we can find the meshmask files.

    Usually, the meshmask files are downloaded from the MER website. However,
    if you are on a computational node that does not have access to the
    internet, you can download the files manually and call this function to
    specify that the files must be read from a local directory.
    """
    global _INTERNAL_DIR
    _INTERNAL_DIR = Path(path)


def get_meshmask_dir() -> Path | None:
    """
    Return the current value of the meshmask directory. If this function
    returns `None`, the meshmask files will be downloaded from the MER website.
    """
    return _INTERNAL_DIR


def _download_meshmask_file(domain_name: str, output_file: Path):
    """Downloads and decompresses a meshmask file for a given domain.

    Args:
        domain_name: Name of the domain to download the meshmask file for.
        output_file: Path where the decompressed meshmask file will be saved.

    Raises:
        ValueError: If the HTTP request fails with a non-200 status code.
        OSError: If there are issues with file operations.
    """
    # Construct the file URL
    meshmask_file_url = \
        f"{STATIC_FILES_URL.geturl()}/{domain_name.upper()}/meshmask.nc.gz"

    # Open the URL and get the compressed file
    LOGGER.debug("Connecting to %s...", meshmask_file_url)
    with urlopen(meshmask_file_url) as response:
        LOGGER.debug("Connection status = %s", response.status)
        if response.status != 200:
            raise ValueError(
                f"Failed to download meshmask file: HTTP {response.status}"
            )

        temp_file = output_file.with_suffix(".tmp")
        LOGGER.debug("Saving temporary meshmask file to %s...", temp_file)

        # Decompress the data on-the-fly and write to the output file
        with gzip.GzipFile(fileobj=response, mode="r") as compressed_file:
            with temp_file.open('wb') as uncompressed_file:
                # noinspection PyTypeChecker
                copyfileobj(compressed_file, uncompressed_file)

    LOGGER.debug("Moving temporary meshmask file to %s...", output_file)
    move(temp_file, output_file)
    LOGGER.debug("Meshmask file downloaded and saved to %s", output_file)


def _mask_values(mask: da.Array, ds: xr.Dataset):
    """
    Put np.nan on the cells where the mask is False (i.e., on the land)

    The files produced by the MER model have 0 values on the land, so we need
    to mask them. This is exactly what this function does. It sets the values
    that are on the land to np.nan.

    This function modifies the input dataset in place. To do that, it replaces
    the "data" of each variable that must be masked with a masked version of
    the original data. (
    """
    for var_name in ds.variables:
        # We do not mask the coordinates of the dataset (latitude, longitude,
        # depth and time)
        if var_name in ds.coords:
            LOGGER.debug(
                "Skipping masking of variable %s because it is a"
                "coordinate",
                var_name
            )
            continue

        # If a variable is 3D, it must have the following shape:
        var_dims_3d = (
            ("time", "depth", "latitude", "longitude"),
            ("depth", "latitude", "longitude")
        )
        # Instead, this is what we expect from a 2D variable
        var_dims_2d = (
            ("time", "latitude", "longitude"),
            ("latitude", "longitude")
        )

        # If the variable is 3D, we mask using all the values of the mask
        if ds[var_name].dims in var_dims_3d:
            LOGGER.debug(
                "Variable %s has 3D dimensions; masking...", var_name
            )
            var_data = ds[var_name].data
            ds[var_name].data = da.where(mask, var_data, np.nan)
            continue

        # If the variable is 2D, we use only the first layer of the mask
        if ds[var_name].dims in var_dims_2d:
            LOGGER.debug(
                "Variable %s has 2D dimensions; masking...", var_name
            )
            var_data = ds[var_name].data
            ds[var_name].data = da.where(mask[0, :, :], var_data, np.nan)
            continue

        LOGGER.debug(
            "Variable %s has unsupported dimensions %s; skipping",
            var_name,
            ds[var_name].dims
        )
        continue


def get_meshmask_file(domain_name: str):
    """Gets the meshmask file for a specified domain, downloading it if not
    already present.

    Downloads and caches the meshmask file for the specified domain if it
    doesn't exist in the temporary directory. If the file already exists,
    returns the path to the cached file.

    If you call the function `set_meshmask_dir`, the meshmask files will be
    read from the specified directory instead of the MER website.

    Args:
        domain_name: A string representing the name of the domain. Must be one
            of: 'nad', 'sad', 'ion', 'tyr', 'lig', 'sar', 'sic', 'got', 'gsn'.

    Returns:
        The path to the meshmask file for the specified domain.

    Raises:
        ValueError: If the domain name is not valid or if the meshmask file
            download fails.
        OSError: If there are issues with file operations during download or
            caching.
    """
    domain_name = domain_name.lower()
    if domain_name not in DOMAIN_NAMES:
        raise ValueError(
            f"Invalid domain name: {domain_name}; the only allowed values "
            f"are {', '.join(DOMAIN_NAMES)}."
        )

    TEMP_DIR.mkdir(exist_ok=True)

    meshmask_file = TEMP_DIR / f"{domain_name}.nc"

    internal_dir = get_meshmask_dir()
    if internal_dir is not None:
        internal_file = internal_dir / f"{domain_name.lower()}.nc"
        if internal_file.exists():
            LOGGER.debug(
                "Meshmask for domain %s found in internal directory at "
                "the following path: %s",
                domain_name,
                internal_file
            )
            return internal_file
        else:
            LOGGER.warning(
                "Meshmask directory has been set to %s but file %s does not "
                "exist; this code will try to download the file from the MER "
                "webpage",
                internal_dir,
                internal_file
            )

    if not meshmask_file.exists():
        try:
            LOGGER.debug(
                'Meshmask file for domain "%s" not found; downloading...',
                domain_name
            )
            _download_meshmask_file(domain_name, meshmask_file)
        except Exception as e:
            raise type(e)(
                f"Unable to download meshmask file for domain {domain_name}"
            ) from e

    return meshmask_file


@lru_cache
def get_domains() -> dict[str, xr.Dataset]:
    """Gets all MER domain meshmask datasets.

    Loads and caches meshmask datasets for all MER domains. Each domain's
    meshmask file is downloaded if not already present locally.

    Returns:
        Dictionary mapping domain names to their respective meshmask datasets.
        Keys are domain names ('nad', 'sad', etc.), and values are the
        corresponding xarray Datasets.

    Note:
        This function is cached using @lru_cache to avoid reloading datasets 
        unnecessarily.
    """
    output = {}
    for d in DOMAIN_NAMES:
        output[d] = xr.open_dataset(get_meshmask_file(d), chunks={})
    return output


def open_mer_file(data_path: PathLike | str, mask_values=True):
    """Opens and processes a MER Zarr file.

    This function opens a MER file, processes its contents by renaming
    coordinates, identifying the correct domain, transposing variables if
    needed, and optionally masking land values.

    Args:
        data_path: Path to the MER file to open. Can be either a string or a
            PathLike object.
        mask_values: If True, replaces values on land cells with NaN using the
            domain's mask. Defaults to True.

    Returns:
        The processed dataset containing the MER data with standardized
        coordinate names, properly ordered dimensions, and optionally masked
        land values.

    Warns:
        UserWarning: If no matching domain is found for the file, or if the
            depth dimension cannot be identified.
    """
    data_path = Path(data_path)
    LOGGER.debug("Opening MER file %s...", data_path)
    with catch_warnings():
        filterwarnings(
            "ignore",
            message="Failed to open Zarr store with consolidated metadata",
            category=RuntimeWarning
        )
        ds = xr.open_zarr(data_path)

    # We rename the coordinates to be more readable. We still have to fix the
    # depth that is more complicated
    ds = ds.rename(
        {"X": "longitude", "Y": "latitude", "T": "time"},
    )

    # The dimension for the depth has different names among the different
    # domains. We look for a dimension that starts with "Zm" and rename it as
    # depth
    depth_candidates = [
        d for d in ds.dims if str(d).startswith("Zm")
    ]
    if len(depth_candidates) != 1:
        warn(
            f"Unable to find a depth dimension in file {data_path}; the file "
            "will not be furtherly processed"
        )
        return ds
    ds = ds.rename_dims({depth_candidates[0]: "depth"})

    lat_dim = ds.sizes["latitude"]
    lon_dim = ds.sizes["longitude"]
    # Now we compare the dimension of this file with the dimensions of the
    # different domains to check if this file is coherent with one of the
    # domains we know
    for domain_name, domain_mask in get_domains().items():
        LOGGER.debug("Checking if file is coherent with domain %s...", domain_name)
        if lat_dim != domain_mask.sizes["latitude"]:
            LOGGER.debug(
                "Current lat dim is %s, while domain %s has %s points on "
                "the latitude axis; this is not the right domain",
                lat_dim,
                domain_name,
                domain_mask.sizes["latitude"]
            )
            continue
        if lon_dim != domain_mask.sizes["longitude"]:
            LOGGER.debug(
                "Current lon dim is %s, while domain %s has %s points on "
                "the longitude axis; this is not the right domain",
                lon_dim,
                domain_name,
                domain_mask.sizes["longitude"]
            )
            continue

        LOGGER.debug(
            "File %s is coherent with domain %s", data_path, domain_name
        )

        # We copy the values of the coordinates of the domain into the
        # coordinates of the data file
        ds.latitude.data[:] = domain_mask.latitude[:]
        ds.longitude.data[:] = domain_mask.longitude[:]
        ds = ds.assign_coords({"depth": domain_mask.depth})

        # We also save the name of the domain inside an attribute of the
        # dataset
        ds.attrs.update({"mer_domain": domain_name})
        break
    else:
        warn(
            f'No domain found for the file "{data_path}"; the file will not'
            f'be furtherly processed')
        return ds

    # transpose variables that have dimensions scrambled (for example, conc03)
    for var_name in ds.variables:
        # We ignore the coordinates
        if var_name in ds.coords:
            continue

        # We do not transpose the variables that do not use
        # "standard" dimensions. Usually, this is only "wo" that is not
        # centered in the middle of the cells
        LOGGER.debug("Checking dimensions of variable %s...", var_name)
        skip_transpose = False
        for dim_name in ds[var_name].dims:
            valid_dims = ("latitude", "longitude", "time", "depth")
            if dim_name not in valid_dims:
                LOGGER.debug(
                    "Variable %s has unexpected dimension %s and it will not "
                    "be transposed",
                    var_name,
                    dim_name
                )
                skip_transpose = True
                continue
        if skip_transpose:
            continue
        LOGGER.debug(
            "Variable %s has standard dimensions; checking if they are "
            "ordered correctly...",
            var_name
        )
        expected_dimensions = []
        if "time" in ds[var_name].dims:
            expected_dimensions.append("time")
        if "depth" in ds[var_name].dims:
            expected_dimensions.append("depth")
        expected_dimensions.extend(["latitude", "longitude"])

        LOGGER.debug(
            "We expect the dimensions of variable %s to have the "
            "following order: %s",
            var_name,
            expected_dimensions
        )

        if tuple(ds[var_name].dims) != tuple(expected_dimensions):
            LOGGER.debug(
                "The order of dimensions of variable %s is not correct (%s),"
                " transposing...",
            )
            ds[var_name] = ds[var_name].transpose(*expected_dimensions)
        else:
            LOGGER.debug(
                "The order of dimensions of variable %s is the expected "
                "one (%s) and it will not be transposed",
                var_name,
                tuple(ds[var_name].dims)
            )

    LOGGER.debug("All the variables have been transposed if needed")
    if mask_values:
        LOGGER.debug("Masking values on the land...")
        mask = domain_mask.tmask > 0
        _mask_values(mask, ds)

    LOGGER.debug("Mer file %s opened successfully", data_path)
    return ds


class MerMap:
    """A class for creating and managing interactive maps of MER data.

    This class provides functionality to create maps of oceanographic data
    from the MER model, with support for interactive visualization
    of multidimensional data inside Jupyter notebooks.
    """

    def __init__(self, data_var, figure:Figure, vmin=None, vmax=None, cmap="viridis"):
        """Initialize a MerMap instance.

        Args:
            data_var: xarray DataArray containing the data to be plotted.
            figure: matplotlib Figure object to plot on.
            vmin: Optional minimum value for the colorbar.
                If None, will use data minimum.
            vmax: Optional maximum value for the colorbar.
                If None, will use data maximum.
            cmap: String name of the colormap to use. Defaults to "viridis".

        Raises:
            ValueError: If the input data_var doesn't have both latitude
                and longitude dimensions.
        """
        if "latitude" not in data_var.dims:
            raise ValueError(
                "The input data array must have a latitude dimension"
            )
        if "longitude" not in data_var.dims:
            raise ValueError(
                "The input data array must have a longitude dimension"
            )
        self._data_var = data_var

        self._vmin = vmin
        self._vmax = vmax
        self._cmap = cmap
        self._figure = figure
        self._axes = None

        self._plot_dims = tuple(
            v for v in data_var.dims if v not in ("latitude", "longitude")
        )
        if len(self._plot_dims) > 2:
            raise ValueError(
                "The input data array must have at most two dimensions "
                "other than latitude and longitude"
            )

    def plot(self, **kwargs):
        """Plot the data on the map.

        Creates a map plot of the data using cartopy's PlateCarree projection.
        If the data has multiple dimensions beyond latitude and longitude,
        specific values for these dimensions must be provided as keyword
        arguments.

        Args:
            **kwargs: Dimension name-value pairs specifying which slice of
                the data to plot.

        Raises:
            ValueError: If any non-spatial dimension has more than one element
                after selection.
        """
        self._figure.clf()
        self._axes = self._figure.add_subplot(
            1, 1, 1, projection=ccrs.PlateCarree()
        )
        if len(kwargs) > 0:
            subset = self._data_var.sel(**kwargs)
        else:
            subset = self._data_var

        for dim in subset.dims:
            if dim in ("latitude", "longitude"):
                continue
            if subset.sizes[dim] > 1:
                raise ValueError(
                    f"The dataset that must be plot has {subset.sizes[dim]} "
                    f"elements on the dimension {dim}. Please, choose the "
                    f"frame you want to plot by specifying {dim}=XXX in the "
                    f"plot function"
                )

        plot_kwargs = {"cmap": self._cmap}
        if self._vmin is not None:
            plot_kwargs["vmin"] = self._vmin
        if self._vmax is not None:
            plot_kwargs["vmax"] = self._vmax

        subset.plot(x="longitude", y="latitude", ax=self._axes, **plot_kwargs)

    def interact(self):
        """Create an interactive widget for exploring the data.

        Creates interactive widgets (dropdowns or sliders) for each
        dimension of the data beyond latitude and longitude, allowing
        interactive exploration of the dataset.

        Returns:
            ipywidgets interactive object.

        Raises:
            ValueError: If the data has no dimensions other than latitude and
                longitude, or if it has more than two additional dimensions.
        """
        if len(self._plot_dims) == 0:
            raise ValueError(
                "The interactive function cannot be used with a DataArray  "
                "that has no dimensions other than latitude and longitude"
            )
        if len(self._plot_dims) > 2:
            raise ValueError(
                "The interactive function cannot be used with a DataArray  "
                "that has more than two dimensions other than latitude and "
                "longitude"
            )

        control_widgets = {}
        for dim in self._plot_dims:
            if self._data_var.sizes[dim] == 1:
                continue
            coord_values = self._data_var.coords[dim].values
            if coord_values.dtype.kind in {'U', 'S', 'O'}:
                control_widgets[dim] = Dropdown(
                    options=list(coord_values),
                    description=dim
                )
            else:
                control_widgets[dim] = SelectionSlider(
                    options=list(coord_values),
                    description=dim
                )
        return interact(self.plot, **control_widgets)
