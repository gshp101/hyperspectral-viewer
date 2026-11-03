# Segmentation and saved cube files

hsview writes plain Python data so results can be used without hsview installed.

## Segmentation files (*Segment → Save segmentation…*)

Two containers hold the same content. Pick by the file suffix:

| Suffix | Read with | Notes |
|---|---|---|
| `.pkl` | `pickle.load` | A dict of numpy arrays, lists, dicts and strings. |
| `.npz` | `np.load` | No pickle needed (`allow_pickle=False` works). Non-array entries are in `metadata_json`. |

### Entries

| Key | Type | Meaning |
|---|---|---|
| `labels` | `int32 (rows, cols)` | Region of every pixel. `0` means not assigned. |
| `label_values` | `int32 (n,)` | The label values in use, in order. |
| `label_names` | `list[str] (n,)` | Name of each value: `Cluster 1`… for k-means, the annotation class (`Red`, `Green`…) when supervised. |
| `label_colors` | `float (n, 3)` | RGB 0–1 used for the overlay. |
| `mean_spectra` | `float (n, bands)` | Mean spectrum of each region, in the units of the cube that was segmented. |
| `pixel_counts` | `int64 (n,)` | Pixels in each region. |
| `wavelengths` | `float (bands,)` or `None` | Missing from `.npz` when the cube had none. |
| `method` | `str` | `kmeans`, `nearest_mean`, `lda` or `knn`. |
| `supervised` | `bool` | |
| `params` | `dict` | Every option used (number of regions, principal components, `intensity` (`use`, `ignore` or `ignore_offset`), smoothing, seed; for supervised runs also the training pixel count per class). |
| `cube` | `dict` | `name`, `path`, `shape` and `processing`, the analyses applied to the cube before segmenting (for example SVD background removal with its `k`). |
| `created` | `str` | UTC time, ISO 8601. |
| `class_names` | `list[str]` | The 7 annotation class names; annotation `class` indexes this list. |
| `annotations` | `list[dict]` or `None` | Supervised only: the training annotations, each `{"kind": "point" \| "rect" \| "polygon", "class": int, "class_name": str, "xy": [[x, y], …]}`. `x` is the column and `y` the row. |
| `training_labels` | `int32 (rows, cols)` or `None` | Supervised only: `class + 1` under every annotation, `0` elsewhere. These are exactly the training pixels. |
| `annotations_legacy` | `dict` | `.pkl` only, supervised only: the same annotations in the original viewer's `points`/`rects`/`polygons` layout. |

For supervised segmentations the label value is the annotation class index + 1, so `1` is always Red, `2` Green, `3` Blue, `4` Yellow, `5` Magenta, `6` Turquoise and `7` White, whichever classes were annotated.

In `.npz` files the annotations are also stored as arrays: `annotation_kind` (`str (m,)`), `annotation_class` (`int32 (m,)`), and all vertices concatenated in `annotation_xy` (`float (total, 2)`), where annotation `i` is `annotation_xy[annotation_offsets[i]:annotation_offsets[i + 1]]`.

### Loading

```python
import pickle
import numpy as np

with open("scene_segmentation.pkl", "rb") as f:
    seg = pickle.load(f)

labels = seg["labels"]                      # (rows, cols)
names = dict(zip(seg["label_values"], seg["label_names"]))
for ann in seg["annotations"] or []:        # None for unsupervised runs
    print(ann["kind"], ann["class_name"], ann["xy"])
```

```python
import json
import numpy as np

z = np.load("scene_segmentation.npz")
labels = z["labels"]
meta = json.loads(str(z["metadata_json"]))  # method, params, label_names, annotations, cube, ...
names = dict(zip(z["label_values"], meta["label_names"]))
train = z["training_labels"] if "training_labels" in z.files else None
```

With hsview installed, `hsview.analysis.read_segmentation(path)` returns the same dict for either suffix.

Training spectra for a model of your own:

```python
cube = pickle.load(open("scene.pkl", "rb"))["ref"]   # (rows, cols, bands)
mask = seg["training_labels"] > 0
X, y = cube[mask], seg["training_labels"][mask] - 1  # y is the annotation class index
```

### Using a saved segmentation in hsview

- *Segment → Open segmentation…* (or *Add overlay…*) shows it again in its own colours.
- *File → Load annotations…* accepts a supervised segmentation file and loads its training annotations. Saving them afterwards asks for a new file, so the segmentation file is never overwritten.
- Any other label overlay (for example a `.npy` from elsewhere) can also be saved with *Save segmentation…*; its region means are computed from the current cube.

## Cube files (*File → Save cube as…*)

The cube shown, including any analysis such as background removal, is written with these entries:

| Key | Meaning |
|---|---|
| `ref` | The cube, `(rows, cols, bands)`. |
| `wavelengths` | `(bands,)`, or `None`/absent when unknown. |
| `lbl` | The reference image, when there is one. |
| `processing` | List of the analyses applied, e.g. `[{"name": "svd_background", "k": 2, "add_mean": false, "variance_removed": 0.93}]`. |

| Suffix | Container |
|---|---|
| `.pkl` | Pickled dict (the camera tools' layout). |
| `.npz` | `np.load`; `processing` is a JSON string: `json.loads(str(z["processing"]))`. |
| `.h5` | One HDF5 dataset per entry; `processing` is a JSON string in the file attribute `processing`. |

All three open again in hsview, which keeps the processing history.
