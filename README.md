# hsview

View and annotate hyperspectral image cubes, built on [napari](https://napari.org).

- Load cubes: ENVI (`.hdr` + data), `.npy`, HDF5, TIFF, and pickled dicts (`ref` cube, optional `wavelengths`, optional `lbl` label image)
- Show a single band (slider) or a sum/mean of bands (`10-20, 35` or `650-700nm`)
- RGB composites where each channel is a band sum, with a true-colour preset when wavelengths are known
- Annotate with points, rectangles and polygons; save/load as JSON
- Spectra panel: point spectra, mean ±1σ inside each region, live spectrum under the cursor, and mean spectrum per class of a label overlay
- Overlays: segmentations (integer images become napari Labels layers) or continuous maps (shown with a colormap at 50% opacity)
- Export all annotation spectra to CSV

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

## Run

```bash
hsview                                  # empty viewer, use "Open cube…"
hsview cube.hdr --overlay seg.npy       # cube plus a segmentation
hsview data.pkl --band-axis 0           # cube stored as (bands, rows, cols)
python -m hsview.demo demo_cube.pkl     # make a synthetic cube to try it
```

From Python / Jupyter:

```python
import hsview
viewer, widget = hsview.launch("data.pkl", block=False)
widget.show_rgb(["620-670nm", "530-570nm", "450-490nm"])
widget.add_overlay(my_segmentation, name="kmeans")
```

## Usage notes

- Cube layout is `(rows, cols, bands)`. For `.npy`/HDF5/TIFF/pickle a band-first array is detected when the first axis is clearly the smallest; override with `--band-axis`.
- Band specs: plain numbers are band indices, numbers with `nm` are wavelengths (nearest bands are used).
- Annotation colours match their spectrum lines. Select mode + Delete removes annotations.
- Missing wavelengths fall back to band index on the spectrum axis.
- Pickle files can run code when loaded, so only open ones you trust.

## Adding a cube format

Write a function that returns `hsview.io.Cube(data, wavelengths, units, name, overlays={...})` and register its extensions in `hsview.io.READERS`.

## Tests

```bash
pip install -e ".[test]"
pytest
```
