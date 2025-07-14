# MER Data Visualization and Processing Library

This library provides tools for managing, processing, and visualizing
oceanographic data produced by the MER project. It simplifies the handling of
MER data by offering functionality for opening, masking, and visualizing
datasets interactively with Python.

---

## Features

### Core Functions
1. **Data Handling**:

   These are tools that help load the datasets produced by the MER simulations.
   Use this function if you want to access to the numerical values of the data.
   - `open_mer_file(data_path: PathLike | str, mask_values=True)`:
     Opens a MER Zarr dataset, standardizes coordinate and dimensions,
     identifies its domain, and optionally masks land values.

2. **Interactive Visualization**

   These are tools designed to visualize the results.
   - `MerMap`: A class for interactive mapping and visualization of MER data.
     Features include:
     - `plot()`: Creates a static map projection.
     - `interact()`: Creates interactive controls for exploring multidimensional data.

3. **Meshmask Management**
 
   Opening MER files requires some knowledge about the geographical position
   of the cells and the position of land and sea. This information is stored
   inside the `meshmask` files. Usually, you do not need to care about these
   files. The function `open_mer_file` automatically retrieves the appropriate
   meshmask file for the domain the file you are opening belongs to. When a
   file is opened, the code automatically downloads the required meshmask from
   the Internet. If your machine does not have access to an external
   connection, a convenient workaround is to put all the meshmask files for the
   different domains in the same directory (each meshmask must have the same
   name and a ".nc" extension). Then it is possible to call the
   `set_meshmask_dir` function passing the path of the directory. This will
   force the code to read the meshmask files from the directory instead of
   downloading them from the Internet.
   - `set_meshmask_dir(path: PathLike | str)`: Sets a custom directory for
     reading meshmask files, useful for computational nodes without internet
     access.
   - `get_meshmask_dir() -> Path | None`: Retrieves the current directory for
     meshmask files. If `None`, data is fetched online.
   - `get_meshmask_file(domain_name: str)`: Retrieves the meshmask `.nc` file
     for a specified domain, downloading it if not already available.
   - `get_domains() -> dict`: Returns a dictionary of MER domain meshmask
     datasets as `xarray.Dataset`.
---

## Installation

To install the library, navigate to the root directory of the project
(where `pyproject.toml` is located) and run:

```bash
pip install .
```

---

## How the Meshmask Retrieval System Works

### Description of Mechanism
Meshmasks are files defining the spatial characteristics of the MER model
domains (land-sea masks). Here's how the retrieval system operates:

1. **Domain Validation**: The function `get_meshmask_file(domain_name)`
   ensures the provided `domain_name` matches one of the predefined MER 
   domains: Valid values are:
   `nad`, `sad`, `ion`, `tyr`, `lig`, `sar`, `sic`, `got`, `gsn`.

2. **Caching**: Meshmask files are cached locally in a temporary directory
   (`TEMP_DIR`) to prevent repeated downloads.

3. **Custom Directories**: If meshmask files are already available locally,
   `set_meshmask_dir` can be used to set a custom directory.
   `get_meshmask_file` will prioritize this directory over downloading files.

4. **Downloading and Decompressing**: If no local file is found,
   the `_download_meshmask_file` function fetches and decompresses the
   required meshmask file from the default MER static data URL.

5. **Dataset Loading**: Once a meshmask file is prepared locally,
   `xarray` opens it as a dataset ready for processing or visualization.

---

## Example Usage

### Loading a MER File
The following snippet shows how to open a MER zarr file.
```python
from mer_plotter import open_mer_file
# Open a MER Zarr data file, masking land cells
dataset = open_mer_file("path/to/mer_data.zarr")
# Access the dataset's standardized dimensions and attributes
print(dataset)
``` 

### Interactive Visualization
```python
import matplotlib.pyplot as plt
from mer_plotter import MerMap
fig = plt.figure()
# Create an interactive map for a variable
map_plot = MerMap(data_var=dataset['variable_name'], figure=fig)
map_plot.interact()
``` 

### Setting and Using a Custom Meshmask Directory
```python
from mer_library import set_meshmask_dir, get_meshmask_file
# Set a custom directory for meshmask files
set_meshmask_dir("/path/to/meshmask")
# Retrieve a meshmask file for a domain
meshmask_path = get_meshmask_file("ion")
print(f"Meshmask file path: {meshmask_path}")
``` 

---

## Dependencies

- `cartopy` for map projections
- `dask` for chunked data processing
- `xarray` for handling datasets
- `ipywidgets` for interactive visualization
- `matplotlib` for plotting

---
