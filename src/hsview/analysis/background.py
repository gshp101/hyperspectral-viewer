"""Background removal by projecting out the largest SVD components.

Treat the cube as a matrix ``X`` with one row per pixel and one column per
band. After removing the mean spectrum ``mu``, the right singular vectors of
``X - mu`` are the principal spectral directions. The ``k`` largest are taken
to be background, and every spectrum is replaced by its residual::

    x  ->  (x - mu) - V_k V_k^T (x - mu)

The singular vectors are found from the bands x bands scatter matrix
``(X - mu)^T (X - mu) = V S^2 V^T``, which gives the same ``V`` and ``S`` as a
full SVD of the pixels x bands matrix but needs memory for only one block of
pixels at a time, so it also works on memory-mapped cubes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io import Cube

CHUNK_PIXELS = 1 << 18  # pixels processed per block


def _blocks(cube: Cube):
    """Yield (row_slice, pixels x bands float64 block) over the whole cube."""
    rows, cols, bands = cube.shape
    step = max(1, CHUNK_PIXELS // max(cols, 1))
    for r0 in range(0, rows, step):
        rs = slice(r0, min(rows, r0 + step))
        yield rs, np.asarray(cube.data[rs], dtype=np.float64).reshape(-1, bands)


@dataclass
class SVDBackground:
    """The fitted model: mean spectrum, singular values and right singular vectors."""

    mean: np.ndarray  # (bands,)
    singular_values: np.ndarray  # (bands,), largest first
    components: np.ndarray  # (bands, bands); column i is the i-th right singular vector

    @property
    def variance_fraction(self) -> np.ndarray:
        """Fraction of the (mean-removed) variance carried by each component."""
        energy = self.singular_values ** 2
        total = energy.sum()
        return energy / total if total > 0 else np.zeros_like(energy)

    def project_out(self, spectra: np.ndarray, k: int, add_mean: bool = False) -> np.ndarray:
        """Remove the mean and the first ``k`` components from spectra of shape (..., bands)."""
        x = np.asarray(spectra, dtype=np.float64) - self.mean
        if k > 0:
            v = self.components[:, :k]
            x = x - (x @ v) @ v.T
        if add_mean:
            x = x + self.mean
        return x


def fit_svd_background(cube: Cube) -> SVDBackground:
    """Mean spectrum and SVD of the mean-removed cube (pixels x bands)."""
    rows, cols, bands = cube.shape
    n = rows * cols
    if n == 0:
        raise ValueError("The cube has no pixels.")
    total = np.zeros(bands)
    for _, x in _blocks(cube):
        total += x.sum(axis=0)
    mean = total / n
    scatter = np.zeros((bands, bands))
    for _, x in _blocks(cube):
        x = x - mean
        scatter += x.T @ x
    evals, evecs = np.linalg.eigh(scatter)
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], 0, None)
    evecs = evecs[:, order]
    # Fix the sign of each vector so results are reproducible (largest entry positive).
    signs = np.sign(evecs[np.argmax(np.abs(evecs), axis=0), np.arange(bands)])
    signs[signs == 0] = 1
    return SVDBackground(mean, np.sqrt(evals), evecs * signs)


def remove_svd_background(cube: Cube, k: int, add_mean: bool = False,
                          model: SVDBackground | None = None) -> Cube:
    """A new float32 cube with the mean and the ``k`` largest SVD components projected out."""
    if model is None:
        model = fit_svd_background(cube)
    bands = cube.n_bands
    k = int(k)
    if not 0 <= k <= bands:
        raise ValueError(f"k must be between 0 and {bands}.")
    out = np.empty(cube.shape, dtype=np.float32)
    for rs, x in _blocks(cube):
        out[rs] = model.project_out(x, k, add_mean).reshape(-1, cube.shape[1], bands)
    removed = float(model.variance_fraction[:k].sum())
    metadata = dict(cube.metadata)
    metadata["analysis"] = list(metadata.get("analysis", [])) + [
        {"name": "svd_background", "k": k, "add_mean": add_mean, "variance_removed": removed}]
    return Cube(out, cube.wavelengths, cube.wavelength_units, name=f"{cube.name} [bg −{k}]",
                metadata=metadata, reference=cube.reference, band_names=cube.band_names, path=cube.path)
