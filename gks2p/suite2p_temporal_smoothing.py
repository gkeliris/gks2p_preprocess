
"""
suite2p_temporal_smoothing.py

Temporal smoothing utilities for Suite2p .bin movies (headerless, row-major).
Smoothing is applied along the *time/frame* axis, leaving spatial pixels intact.

Supported methods
-----------------
- block averaging / temporal binning (reduces frames by factor x)
- gaussian smoothing (sigma in frames; optional downsample)
- EMA (exponential moving average; optional downsample)
- median filter (odd window length; optional downsample)
- Savitzky–Golay smoothing (window length + polyorder; optional downsample)

Design
------
- Large files handled with numpy.memmap and chunked processing where needed
- SciPy is optional (used if available)
- Safe dtype handling: float32 output by default; integer outputs clipped to range
- Optional in-place replace: rename input to *.orig and replace with smoothed result

Author: ChatGPT (GPT-5 Pro)
"""

from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Optional dependencies
try:
    from scipy.ndimage import gaussian_filter1d  # type: ignore
    _HAVE_SCIPY_ND = True
except Exception:
    _HAVE_SCIPY_ND = False

try:
    from scipy.signal import savgol_filter  # type: ignore
    _HAVE_SCIPY_SIG = True
except Exception:
    _HAVE_SCIPY_SIG = False


# --------------------------- Data description ---------------------------------

@dataclass
class BinSpec:
    """
    Description of a Suite2p binary.
    """
    path: str
    Lx: int
    Ly: int
    # suite2p usually splits channels; keep for rare cases, default=1
    nchannels: int = 1
    dtype: str | np.dtype = "int16"

    def n_pixels(self) -> int:
        return int(self.Lx) * int(self.Ly) * int(self.nchannels)

    def n_frames(self) -> int:
        """
        Infer the number of frames from the file size.
        Raises if the file size is not an exact multiple.
        """
        itemsize = np.dtype(self.dtype).itemsize
        file_bytes = os.path.getsize(self.path)
        denom = self.n_pixels() * itemsize
        if denom == 0:
            raise ValueError("Invalid dimensions; zero pixels.")
        if file_bytes % denom != 0:
            raise ValueError(
                f"File size {file_bytes} is not an exact multiple of "
                f"pixels*itemsize ({self.n_pixels()} * {itemsize} = {denom}). "
                "Check Lx, Ly, nchannels, and dtype."
            )
        return file_bytes // denom

    def memmap(self, mode: str = "r") -> np.memmap:
        """
        Return a memmap reshaped to (n_frames, n_pixels).
        """
        n_frames = self.n_frames()
        mm = np.memmap(self.path, dtype=self.dtype, mode=mode)
        return mm.reshape(n_frames, self.n_pixels())


# --------------------------- Helpers ------------------------------------------

def _default_out_path(in_path: str, tag: str) -> str:
    base, ext = os.path.splitext(in_path)
    ext = ext if ext else ".bin"
    return f"{base}.{tag}{ext}"

def _ensure_parent_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _make_out_memmap(out_path: str, shape: Tuple[int, int], dtype: str | np.dtype):
    _ensure_parent_dir(out_path)
    return np.memmap(out_path, dtype=dtype, mode="w+", shape=shape)

def _astype_clip(arr: np.ndarray, out_dtype: np.dtype) -> np.ndarray:
    """
    Convert float array to integer dtype with clipping if needed.
    If out_dtype is a float, just astype without clipping.
    """
    if np.issubdtype(out_dtype, np.integer):
        info = np.iinfo(out_dtype)
        return np.clip(arr, info.min, info.max).astype(out_dtype, copy=False)
    return arr.astype(out_dtype, copy=False)

def _gaussian_kernel1d(sigma: float, truncate: float = 3.0) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return k.astype(np.float32, copy=False)

def _pad_fetch_bounds(mm: np.memmap, start: int, end: int, pad_left: int, pad_right: int,
                      mode: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fetch left and right 'context' needed for convolution at [start:end),
    handling global boundaries with 'reflect' or 'edge' padding.
    Returns (left_ctx, right_ctx), each shape (pad_left/right, n_pixels).
    """
    n_frames = mm.shape[0]

    # Left context
    if pad_left <= 0:
        left_ctx = mm[0:0]
    elif start - pad_left >= 0:
        left_ctx = mm[start - pad_left:start]
    else:
        need = pad_left - start
        available = mm[0:start]
        if mode == "reflect":
            extra = mm[0:need][::-1] if need > 0 else mm[0:0]
        elif mode == "edge":
            extra = np.repeat(mm[0:1], repeats=need, axis=0)
        else:
            raise ValueError("mode must be 'reflect' or 'edge'")
        left_ctx = np.vstack([extra, available])

    # Right context
    if pad_right <= 0:
        right_ctx = mm[0:0]
    elif end + pad_right <= n_frames:
        right_ctx = mm[end:end + pad_right]
    else:
        need = end + pad_right - n_frames
        available = mm[end:n_frames]
        if mode == "reflect":
            extra = mm[n_frames - need:n_frames][::-1] if need > 0 else mm[0:0]
        elif mode == "edge":
            extra = np.repeat(mm[n_frames - 1:n_frames], repeats=need, axis=0)
        else:
            raise ValueError("mode must be 'reflect' or 'edge'")
        right_ctx = np.vstack([available, extra])

    # Ensure exact sizes
    if left_ctx.shape[0] != pad_left:
        if left_ctx.shape[0] > pad_left:
            left_ctx = left_ctx[-pad_left:]
        else:  # rare: pad more
            if mode == "edge":
                left_ctx = np.vstack([np.repeat(mm[0:1], repeats=pad_left-left_ctx.shape[0], axis=0), left_ctx])
            else:
                left_ctx = np.vstack([left_ctx, left_ctx[::-1]])[:pad_left]
    if right_ctx.shape[0] != pad_right:
        if right_ctx.shape[0] > pad_right:
            right_ctx = right_ctx[:pad_right]
        else:
            if mode == "edge":
                right_ctx = np.vstack([right_ctx, np.repeat(mm[-1:], repeats=pad_right-right_ctx.shape[0], axis=0)])
            else:
                right_ctx = np.vstack([right_ctx, right_ctx[::-1]])[:pad_right]

    return left_ctx, right_ctx

def _unique_backup_path(path: str, suffix: str = ".orig") -> str:
    """Return a non-clobbering backup path like file.bin.orig, file.bin.orig2, ..."""
    base = path + suffix
    if not os.path.exists(base):
        return base
    i = 2
    while True:
        candidate = f"{base}{i}"
        if not os.path.exists(candidate):
            return candidate
        i += 1

def _finalize_output(original_path: str, temp_out_path: str, inplace: bool) -> Tuple[str, Optional[str]]:
    """
    If inplace=True, rename original -> .orig*, and move temp output to original path.
    Returns (final_out_path, backup_path or None).
    """
    if not inplace:
        return temp_out_path, None

    backup = _unique_backup_path(original_path, ".orig")
    # Use os.replace for atomic-ish moves on the same filesystem
    os.replace(original_path, backup)
    os.replace(temp_out_path, original_path)
    return original_path, backup

def _prepare_out_path(spec: BinSpec, tag: str, inplace: bool, out_path: Optional[str]) -> Tuple[str, str]:
    """
    Decide on a temporary output path and a tag. When inplace=True and out_path is None,
    write to a temporary sidecar that will be moved into place at the end.
    """
    if out_path is not None:
        return out_path, tag
    if inplace:
        # write to a sidecar .tmp so we can atomically swap later
        return _default_out_path(spec.path, f"{tag}.tmp"), tag
    else:
        return _default_out_path(spec.path, tag), tag


# --------------------------- Downsample helpers --------------------------------

def _n_out_decimate(T: int, ds: int) -> int:
    # indices 0, ds, 2ds, ..., <= T-1
    return (T - 1) // ds + 1

def _downsample_write_decimate(filtered_chunk: np.ndarray, start_idx: int, ds: int,
                               mm_out: np.memmap, out_pos: int) -> int:
    """
    Write only frames whose absolute index t satisfies t % ds == 0.
    Returns updated out_pos.
    """
    Tchunk = filtered_chunk.shape[0]
    # First absolute index in this chunk that is kept
    # t = start_idx + k; want (start_idx + k) % ds == 0 => k % ds == (-start_idx) % ds
    offset = (-start_idx) % ds
    sel = np.arange(offset, Tchunk, ds, dtype=int)
    n = len(sel)
    if n > 0:
        mm_out[out_pos:out_pos + n] = filtered_chunk[sel]
        out_pos += n
    return out_pos

class _BlockAccumulator:
    """
    For 'block' downsample mode: average every ds consecutive frames from a stream.
    Keeps a running sum and outputs when count==ds.
    """
    def __init__(self, n_pixels: int, ds: int, out_mm: np.memmap, out_pos: int, dtype: np.dtype):
        self.sum = np.zeros(n_pixels, dtype=np.float32)
        self.count = 0
        self.ds = ds
        self.out_mm = out_mm
        self.out_pos = out_pos
        self.dtype = dtype

    def ingest(self, frames: np.ndarray):
        # frames: (Tchunk, P)
        for row in frames:
            self.sum += row
            self.count += 1
            if self.count == self.ds:
                avg = (self.sum / float(self.ds))
                self.out_mm[self.out_pos] = _astype_clip(avg, self.dtype)
                self.out_pos += 1
                self.sum.fill(0.0)
                self.count = 0

    def flush_remainder(self, keep_remainder: bool):
        if keep_remainder and self.count > 0:
            avg = (self.sum / float(self.count))
            self.out_mm[self.out_pos] = _astype_clip(avg, self.dtype)
            self.out_pos += 1
            self.sum.fill(0.0)
            self.count = 0
        return self.out_pos


# --------------------------- Methods ------------------------------------------

def block_average(
    spec: BinSpec,
    x: int,
    out_path: Optional[str] = None,
    out_dtype: str | np.dtype = "float32",
    keep_remainder: bool = False,
    verbose: bool = True,
    inplace: bool = False,
) -> Tuple[str, Optional[str]]:
    """
    Average every x consecutive frames (temporal binning). Output has floor(T/x)
    frames (or ceil if keep_remainder=True).

    Returns (final_out_path, backup_path_or_None).
    """
    if x < 2:
        raise ValueError("x must be >= 2 for block averaging.")

    mm_in = spec.memmap("r")
    T, P = mm_in.shape
    n_out = T // x + (1 if keep_remainder and (T % x) else 0)

    temp_out, _ = _prepare_out_path(spec, f"binavg{x}", inplace, out_path)
    mm_out = _make_out_memmap(temp_out, shape=(n_out, P), dtype=out_dtype)

    out_i = 0
    for start in range(0, T, x):
        end = min(start + x, T)
        if end - start < x and not keep_remainder:
            break
        chunk = mm_in[start:end].astype(np.float32, copy=False)
        avg = chunk.mean(axis=0, dtype=np.float64)
        mm_out[out_i] = _astype_clip(avg.astype(np.float32), np.dtype(out_dtype))
        out_i += 1
        if verbose and (out_i % 50 == 0 or end == T):
            print(f"[block_average] wrote {out_i}/{n_out} bins")

    mm_out.flush()
    final_out, backup = _finalize_output(spec.path, temp_out, inplace)
    return final_out, backup


def gaussian_smooth(
    spec: BinSpec,
    sigma_frames: float,
    out_path: Optional[str] = None,
    out_dtype: str | np.dtype = "float32",
    truncate: float = 3.0,
    chunk_frames: int = 64,
    mode: str = "reflect",
    verbose: bool = True,
    inplace: bool = False,
    # Downsample options
    downsample_factor: Optional[int] = None,
    downsample_mode: str = "decimate",  # {"decimate", "block"}
    keep_remainder: bool = False,
) -> Tuple[str, Optional[str]]:
    """
    Temporal Gaussian smoothing with std-dev = sigma_frames (frames).
    - Preserves frame count unless `downsample_factor` is set.
    - If downsample_factor is provided:
        * "decimate": keep every ds-th smoothed frame
        * "block": average blocks of ds smoothed frames (like temporal binning)

    Returns (final_out_path, backup_path_or_None).
    """
    if sigma_frames <= 0:
        raise ValueError("sigma_frames must be > 0")
    if chunk_frames < 8:
        raise ValueError("chunk_frames should be at least 8 frames")
    if downsample_factor is not None and downsample_factor < 2:
        raise ValueError("downsample_factor must be >= 2")

    mm_in = spec.memmap("r")
    T, P = mm_in.shape

    # Decide output length
    if downsample_factor is None:
        n_out = T
        tag = f"gauss{sigma_frames:g}"
    else:
        if downsample_mode == "decimate":
            n_out = _n_out_decimate(T, downsample_factor)
        elif downsample_mode == "block":
            n_out = T // downsample_factor + (1 if keep_remainder and (T % downsample_factor) else 0)
        else:
            raise ValueError("downsample_mode must be 'decimate' or 'block'")
        tag = f"gauss{sigma_frames:g}_ds{downsample_factor}{'b' if downsample_mode=='block' else ''}"

    temp_out, _ = _prepare_out_path(spec, tag, inplace, out_path)
    mm_out = _make_out_memmap(temp_out, shape=(n_out, P), dtype=out_dtype)

    if _HAVE_SCIPY_ND:
        radius = int(truncate * sigma_frames + 0.5)
        pad = radius
        pos = 0
        out_pos = 0
        acc = _BlockAccumulator(P, downsample_factor or 1, mm_out, out_pos, np.dtype(out_dtype)) if (downsample_factor and downsample_mode == "block") else None

        while pos < T:
            end = min(pos + chunk_frames, T)
            left_ctx, right_ctx = _pad_fetch_bounds(mm_in, pos, end, pad, pad, mode)
            block = np.vstack([left_ctx, mm_in[pos:end], right_ctx]).astype(np.float32, copy=False)
            block_f = gaussian_filter1d(block, sigma=sigma_frames, axis=0, mode='reflect' if mode=='reflect' else 'nearest', truncate=truncate)
            filtered = block_f[pad:pad + (end - pos)]

            if downsample_factor is None:
                mm_out[pos:end] = _astype_clip(filtered, np.dtype(out_dtype))
            else:
                if downsample_mode == "decimate":
                    out_pos = _downsample_write_decimate(filtered, pos, downsample_factor, mm_out, out_pos)
                else:  # block
                    acc.ingest(filtered)

            if verbose and (end == T or (pos // chunk_frames) % 10 == 0):
                done = end if downsample_factor is None else min(n_out, out_pos)
                print(f"[gaussian_smooth] processed frames {end}/{T}")

            pos = end

        if acc is not None:
            out_pos = acc.flush_remainder(keep_remainder)

    else:
        # Pure NumPy fallback: manual convolution along time with overlap
        k = _gaussian_kernel1d(sigma_frames, truncate=truncate)  # float32 normalized
        K = len(k)
        pad = K // 2

        pos = 0
        out_pos = 0
        acc = _BlockAccumulator(P, downsample_factor or 1, mm_out, out_pos, np.dtype(out_dtype)) if (downsample_factor and downsample_mode == "block") else None

        while pos < T:
            end = min(pos + chunk_frames, T)
            left_ctx, right_ctx = _pad_fetch_bounds(mm_in, pos, end, pad, pad, mode)
            block = np.vstack([left_ctx, mm_in[pos:end], right_ctx]).astype(np.float32, copy=False)  # (chunk+2*pad, P)

            windows = sliding_window_view(block, window_shape=(K,), axis=0)  # (chunk+2*pad-K+1, K, P)
            windows_center = windows[0:(end - pos)]  # (chunk, K, P)
            filtered = np.tensordot(windows_center, k, axes=([1], [0]))  # float32 (chunk, P)

            if downsample_factor is None:
                mm_out[pos:end] = _astype_clip(filtered, np.dtype(out_dtype))
            else:
                if downsample_mode == "decimate":
                    out_pos = _downsample_write_decimate(filtered, pos, downsample_factor, mm_out, out_pos)
                else:
                    acc.ingest(filtered)

            if verbose and (end == T or (pos // chunk_frames) % 10 == 0):
                print(f"[gaussian_smooth] processed frames {end}/{T}")
            pos = end

        if acc is not None:
            out_pos = acc.flush_remainder(keep_remainder)

    mm_out.flush()
    final_out, backup = _finalize_output(spec.path, temp_out, inplace)
    return final_out, backup


def ema_smooth(
    spec: BinSpec,
    tau_frames: float,
    out_path: Optional[str] = None,
    out_dtype: str | np.dtype = "float32",
    init: str = "copy_first",
    verbose: bool = True,
    inplace: bool = False,
    # Downsample options
    downsample_factor: Optional[int] = None,
    downsample_mode: str = "decimate",  # {"decimate", "block"}
    keep_remainder: bool = False,
) -> Tuple[str, Optional[str]]:
    """
    Exponential Moving Average (IIR) smoothing along time.
    y[t] = alpha*x[t] + (1-alpha)*y[t-1], where alpha = 1 - exp(-1/tau_frames).

    - Preserves frame count unless downsample_factor is set.
    - "decimate": write every ds-th smoothed frame
    - "block": average blocks of ds smoothed frames

    Returns (final_out_path, backup_path_or_None).
    """
    if tau_frames <= 0:
        raise ValueError("tau_frames must be > 0")
    alpha = 1.0 - math.exp(-1.0 / float(tau_frames))

    mm_in = spec.memmap("r")
    T, P = mm_in.shape

    if init == "copy_first":
        y_prev = mm_in[0].astype(np.float32, copy=False)
    elif init == "zeros":
        y_prev = np.zeros(P, dtype=np.float32)
    else:
        raise ValueError("init must be 'copy_first' or 'zeros'")

    # Decide output length
    if downsample_factor is None:
        n_out = T
        tag = f"ema{tau_frames:g}"
    else:
        if downsample_mode == "decimate":
            n_out = _n_out_decimate(T, downsample_factor)
        elif downsample_mode == "block":
            n_out = T // downsample_factor + (1 if keep_remainder and (T % downsample_factor) else 0)
        else:
            raise ValueError("downsample_mode must be 'decimate' or 'block'")
        tag = f"ema{tau_frames:g}_ds{downsample_factor}{'b' if downsample_mode=='block' else ''}"

    temp_out, _ = _prepare_out_path(spec, tag, inplace, out_path)
    mm_out = _make_out_memmap(temp_out, shape=(n_out, P), dtype=out_dtype)

    out_pos = 0
    acc = _BlockAccumulator(P, downsample_factor or 1, mm_out, out_pos, np.dtype(out_dtype)) if (downsample_factor and downsample_mode == "block") else None

    # First frame
    if downsample_factor is None:
        mm_out[0] = _astype_clip(y_prev, np.dtype(out_dtype))
    else:
        if downsample_mode == "decimate":
            # write t=0
            mm_out[out_pos] = _astype_clip(y_prev, np.dtype(out_dtype))
            out_pos += 1
        else:
            acc.ingest(y_prev[None, :])

    # Iterate remaining frames
    for t in range(1, T):
        x = mm_in[t].astype(np.float32, copy=False)
        y_prev = alpha * x + (1.0 - alpha) * y_prev

        if downsample_factor is None:
            mm_out[t] = _astype_clip(y_prev, np.dtype(out_dtype))
        else:
            if downsample_mode == "decimate":
                if t % downsample_factor == 0:
                    mm_out[out_pos] = _astype_clip(y_prev, np.dtype(out_dtype))
                    out_pos += 1
            else:
                acc.ingest(y_prev[None, :])

        if verbose and (t % 2000 == 0 or t == T-1):
            print(f"[ema_smooth] processed {t+1}/{T} frames")

    if acc is not None:
        out_pos = acc.flush_remainder(keep_remainder)

    mm_out.flush()
    final_out, backup = _finalize_output(spec.path, temp_out, inplace)
    return final_out, backup


def median_smooth(
    spec: BinSpec,
    window_frames: int,
    out_path: Optional[str] = None,
    out_dtype: str | np.dtype = "float32",
    chunk_frames: int = 64,
    mode: str = "edge",
    verbose: bool = True,
    inplace: bool = False,
    downsample_factor: Optional[int] = None,
    downsample_mode: str = "decimate",
    keep_remainder: bool = False,
) -> Tuple[str, Optional[str]]:
    """
    Temporal median filter with odd window size.
    - Preserves frame count unless downsample_factor is set.
    - Chunked with overlap; medians computed via NumPy.

    Returns (final_out_path, backup_path_or_None).
    """
    if window_frames < 1 or window_frames % 2 == 0:
        raise ValueError("window_frames must be an odd integer >= 1")
    if downsample_factor is not None and downsample_factor < 2:
        raise ValueError("downsample_factor must be >= 2")

    mm_in = spec.memmap("r")
    T, P = mm_in.shape
    pad = window_frames // 2

    # Decide output length
    if downsample_factor is None:
        n_out = T
        tag = f"median{window_frames}"
    else:
        if downsample_mode == "decimate":
            n_out = _n_out_decimate(T, downsample_factor)
        elif downsample_mode == "block":
            n_out = T // downsample_factor + (1 if keep_remainder and (T % downsample_factor) else 0)
        else:
            raise ValueError("downsample_mode must be 'decimate' or 'block'")
        tag = f"median{window_frames}_ds{downsample_factor}{'b' if downsample_mode=='block' else ''}"

    temp_out, _ = _prepare_out_path(spec, tag, inplace, out_path)
    mm_out = _make_out_memmap(temp_out, shape=(n_out, P), dtype=out_dtype)

    pos = 0
    out_pos = 0
    acc = _BlockAccumulator(P, downsample_factor or 1, mm_out, out_pos, np.dtype(out_dtype)) if (downsample_factor and downsample_mode == "block") else None

    while pos < T:
        end = min(pos + chunk_frames, T)
        left_ctx, right_ctx = _pad_fetch_bounds(mm_in, pos, end, pad, pad, mode)
        block = np.vstack([left_ctx, mm_in[pos:end], right_ctx]).astype(np.float32, copy=False)
        windows = sliding_window_view(block, window_shape=(window_frames,), axis=0)  # (chunk+2*pad - w +1, w, P)
        windows_center = windows[0:(end - pos)]  # (chunk, w, P)
        filtered = np.median(windows_center, axis=1)  # (chunk, P) but be defensive

        # Defensive check: ensure filtered has shape (end-pos, P). In some
        # rare edge cases with very small inputs and padding the computation
        # above may return unexpected trailing dimension sizes; coerce to the
        # expected shape by trimming or repeating the last column.
        expected_cols = P
        if filtered.ndim == 1:
            # single-column result -> expand to 2D
            filtered = filtered[:, None]
        if filtered.shape[1] != expected_cols:
            # If fewer columns, pad by repeating last column; if more, trim.
            cols = filtered.shape[1]
            if cols < expected_cols:
                out_chunk = np.zeros((filtered.shape[0], expected_cols), dtype=filtered.dtype)
                out_chunk[:, :cols] = filtered
                # repeat last column for remaining pixels
                last_col = filtered[:, -1:]
                out_chunk[:, cols:] = np.repeat(last_col, expected_cols - cols, axis=1)
                filtered = out_chunk
            else:
                filtered = filtered[:, :expected_cols]

        if downsample_factor is None:
            mm_out[pos:end] = _astype_clip(filtered, np.dtype(out_dtype))
        else:
            if downsample_mode == "decimate":
                out_pos = _downsample_write_decimate(filtered, pos, downsample_factor, mm_out, out_pos)
            else:
                acc.ingest(filtered)

        if verbose and (end == T or (pos // chunk_frames) % 10 == 0):
            print(f"[median_smooth] processed frames {end}/{T}")
        pos = end

    if acc is not None:
        out_pos = acc.flush_remainder(keep_remainder)

    mm_out.flush()
    final_out, backup = _finalize_output(spec.path, temp_out, inplace)
    return final_out, backup


# --------- Savitzky–Golay (NumPy fallback) ------------------------------------

def _savgol_coeffs_centered(window_length: int, polyorder: int) -> np.ndarray:
    """
    Compute Savitzky–Golay convolution coefficients (deriv=0) for a centered window.
    Equivalent to scipy.signal.savgol_coeffs(..., deriv=0, delta=1, use='conv').
    """
    if window_length % 2 == 0 or window_length < 1:
        raise ValueError("window_length must be a positive odd integer")
    if polyorder >= window_length:
        raise ValueError("polyorder must be less than window_length")
    m = window_length // 2
    # Build Vandermonde matrix A[i,k] = (i-m)^k for i in 0..window_length-1
    x = np.arange(-m, m + 1, dtype=np.float64)
    A = np.vstack([x**k for k in range(polyorder + 1)]).T  # shape (window_length, polyorder+1)
    # Pseudoinverse: (A^T A)^{-1} A^T
    ATA = A.T @ A
    pinv = np.linalg.pinv(ATA) @ A.T  # shape (polyorder+1, window_length)
    coeffs = pinv[0]  # row corresponding to evaluation at x=0 (power 0)
    return coeffs.astype(np.float32, copy=False)

def savgol_smooth(
    spec: BinSpec,
    window_length: int,
    polyorder: int,
    out_path: Optional[str] = None,
    out_dtype: str | np.dtype = "float32",
    chunk_frames: int = 64,
    mode: str = "edge",
    verbose: bool = True,
    inplace: bool = False,
    downsample_factor: Optional[int] = None,
    downsample_mode: str = "decimate",
    keep_remainder: bool = False,
) -> Tuple[str, Optional[str]]:
    """
    Savitzky–Golay polynomial smoothing (centered window).
    - Preserves frame count unless downsample_factor is set.
    - Uses SciPy if available, else a NumPy-based convolution with precomputed coeffs.

    Returns (final_out_path, backup_path_or_None).
    """
    if window_length < 1 or window_length % 2 == 0:
        raise ValueError("window_length must be a positive odd integer")
    if polyorder >= window_length:
        raise ValueError("polyorder must be less than window_length")
    if downsample_factor is not None and downsample_factor < 2:
        raise ValueError("downsample_factor must be >= 2")

    mm_in = spec.memmap("r")
    T, P = mm_in.shape
    pad = window_length // 2

    # Decide output length
    if downsample_factor is None:
        n_out = T
        tag = f"savgol{window_length}p{polyorder}"
    else:
        if downsample_mode == "decimate":
            n_out = _n_out_decimate(T, downsample_factor)
        elif downsample_mode == "block":
            n_out = T // downsample_factor + (1 if keep_remainder and (T % downsample_factor) else 0)
        else:
            raise ValueError("downsample_mode must be 'decimate' or 'block'")
        tag = f"savgol{window_length}p{polyorder}_ds{downsample_factor}{'b' if downsample_mode=='block' else ''}"

    temp_out, _ = _prepare_out_path(spec, tag, inplace, out_path)
    mm_out = _make_out_memmap(temp_out, shape=(n_out, P), dtype=out_dtype)

    pos = 0
    out_pos = 0
    acc = _BlockAccumulator(P, downsample_factor or 1, mm_out, out_pos, np.dtype(out_dtype)) if (downsample_factor and downsample_mode == "block") else None

    if _HAVE_SCIPY_SIG:
        while pos < T:
            end = min(pos + chunk_frames, T)
            left_ctx, right_ctx = _pad_fetch_bounds(mm_in, pos, end, pad, pad, mode)
            block = np.vstack([left_ctx, mm_in[pos:end], right_ctx]).astype(np.float32, copy=False)
            block_f = savgol_filter(block, window_length=window_length, polyorder=polyorder, axis=0, mode='interp' if mode=='reflect' else 'nearest')
            filtered = block_f[pad:pad + (end - pos)]

            if downsample_factor is None:
                mm_out[pos:end] = _astype_clip(filtered, np.dtype(out_dtype))
            else:
                if downsample_mode == "decimate":
                    out_pos = _downsample_write_decimate(filtered, pos, downsample_factor, mm_out, out_pos)
                else:
                    acc.ingest(filtered)

            if verbose and (end == T or (pos // chunk_frames) % 10 == 0):
                print(f"[savgol_smooth] processed frames {end}/{T}")
            pos = end
    else:
        coeffs = _savgol_coeffs_centered(window_length, polyorder)
        K = len(coeffs)
        while pos < T:
            end = min(pos + chunk_frames, T)
            left_ctx, right_ctx = _pad_fetch_bounds(mm_in, pos, end, pad, pad, mode)
            block = np.vstack([left_ctx, mm_in[pos:end], right_ctx]).astype(np.float32, copy=False)
            windows = sliding_window_view(block, window_shape=(K,), axis=0)
            windows_center = windows[0:(end - pos)]  # (chunk, K, P)
            filtered = np.tensordot(windows_center, coeffs, axes=([1], [0]))  # (chunk, P)

            if downsample_factor is None:
                mm_out[pos:end] = _astype_clip(filtered, np.dtype(out_dtype))
            else:
                if downsample_mode == "decimate":
                    out_pos = _downsample_write_decimate(filtered, pos, downsample_factor, mm_out, out_pos)
                else:
                    acc.ingest(filtered)

            if verbose and (end == T or (pos // chunk_frames) % 10 == 0):
                print(f"[savgol_smooth] processed frames {end}/{T}")
            pos = end

    if acc is not None:
        out_pos = acc.flush_remainder(keep_remainder)

    mm_out.flush()
    final_out, backup = _finalize_output(spec.path, temp_out, inplace)
    return final_out, backup


# --------------------------- High-level wrapper --------------------------------

def smooth_suite2p_bin(
    bin_path: str,
    Lx: int,
    Ly: int,
    nchannels: int = 1,
    dtype: str | np.dtype = "int16",
    method: str = "block",
    # method-specific
    x: Optional[int] = None,
    sigma_frames: Optional[float] = None,
    tau_frames: Optional[float] = None,
    window_frames: Optional[int] = None,
    sg_window: Optional[int] = None,
    sg_polyorder: Optional[int] = None,
    # generic
    out_path: Optional[str] = None,
    out_dtype: str | np.dtype = "float32",
    inplace: bool = False,
    # downsample
    downsample_factor: Optional[int] = None,
    downsample_mode: str = "decimate",
    keep_remainder: bool = False,
    # misc
    **kwargs,
) -> Tuple[str, Optional[str]]:
    """
    Convenience wrapper. Pick a method and its parameter(s).

    method:
      - "block": requires x (int >=2)
      - "gaussian": requires sigma_frames (float >0)
      - "ema": requires tau_frames (float >0)
      - "median": requires window_frames (odd int >=1)
      - "savgol": requires sg_window (odd int >=1) and sg_polyorder (< sg_window)

    If inplace=True, the input file is renamed to *.orig* and the output is moved
    to the original path.

    Returns (final_out_path, backup_path_or_None).
    """
    spec = BinSpec(path=bin_path, Lx=Lx, Ly=Ly, nchannels=nchannels, dtype=dtype)

    if method == "block":
        if x is None or x < 2:
            raise ValueError("For method='block', provide x >= 2.")
        return block_average(spec, x=x, out_path=out_path, out_dtype=out_dtype, keep_remainder=keep_remainder, verbose=kwargs.get("verbose", True), inplace=inplace)

    elif method == "gaussian":
        if sigma_frames is None or sigma_frames <= 0:
            raise ValueError("For method='gaussian', provide sigma_frames > 0.")
        return gaussian_smooth(spec, sigma_frames=sigma_frames, out_path=out_path, out_dtype=out_dtype,
                               truncate=kwargs.get("truncate", 3.0),
                               chunk_frames=kwargs.get("chunk_frames", 64),
                               mode=kwargs.get("mode", "reflect"),
                               verbose=kwargs.get("verbose", True),
                               inplace=inplace,
                               downsample_factor=downsample_factor,
                               downsample_mode=downsample_mode,
                               keep_remainder=keep_remainder)

    elif method == "ema":
        if tau_frames is None or tau_frames <= 0:
            raise ValueError("For method='ema', provide tau_frames > 0.")
        return ema_smooth(spec, tau_frames=tau_frames, out_path=out_path, out_dtype=out_dtype,
                          init=kwargs.get("init", "copy_first"),
                          verbose=kwargs.get("verbose", True),
                          inplace=inplace,
                          downsample_factor=downsample_factor,
                          downsample_mode=downsample_mode,
                          keep_remainder=keep_remainder)

    elif method == "median":
        if window_frames is None or window_frames < 1 or window_frames % 2 == 0:
            raise ValueError("For method='median', provide odd window_frames >= 1.")
        return median_smooth(spec, window_frames=window_frames, out_path=out_path, out_dtype=out_dtype,
                             chunk_frames=kwargs.get("chunk_frames", 64),
                             mode=kwargs.get("mode", "edge"),
                             verbose=kwargs.get("verbose", True),
                             inplace=inplace,
                             downsample_factor=downsample_factor,
                             downsample_mode=downsample_mode,
                             keep_remainder=keep_remainder)

    elif method == "savgol":
        if sg_window is None or sg_polyorder is None:
            raise ValueError("For method='savgol', provide sg_window and sg_polyorder.")
        return savgol_smooth(spec, window_length=sg_window, polyorder=sg_polyorder, out_path=out_path, out_dtype=out_dtype,
                             chunk_frames=kwargs.get("chunk_frames", 64),
                             mode=kwargs.get("mode", "edge"),
                             verbose=kwargs.get("verbose", True),
                             inplace=inplace,
                             downsample_factor=downsample_factor,
                             downsample_mode=downsample_mode,
                             keep_remainder=keep_remainder)

    else:
        raise ValueError("Unknown method. Choose from {'block','gaussian','ema','median','savgol'}.")


# --------------------------- ops.npy updater -----------------------------------

def update_ops_nframes(ops_path: str, new_nframes: int, original_nframes: int | None = None,
                       scale_sampling: bool = True, backup_path: str | None = None) -> None:
    """
    Update ops.npy after smoothing that changes the number of frames.

    - Updates common frame-count keys ("nframes", "nFrames", "nFramesTot").
    - If a sampling-rate key is present ("fs", "sampling_frequency", "fps"),
      it is scaled by ratio = new_nframes / original_nframes to preserve total
      recording duration. If `original_nframes` is not provided, the function
      will attempt to read an existing nframes value from ops to compute the
      ratio; if none exists, sampling-rate keys are left unchanged.

    Note: this mutates the ops file in-place.
    """
    ops = np.load(ops_path, allow_pickle=True).item()

    # Determine old frame count
    old_n = None
    if original_nframes is not None:
        old_n = int(original_nframes)
    else:
        for k in ("nframes", "nFrames", "nFramesTot"):
            if k in ops:
                try:
                    old_n = int(ops[k])
                    break
                except Exception:
                    old_n = None

    # Update frame count keys
    # Capture previous values for logging
    sampling_keys = ("fs", "sampling_frequency", "fps")
    prev_sampling = {k: ops[k] for k in sampling_keys if k in ops}
    prev_frames = None
    for k in ("nframes", "nFrames", "nFramesTot"):
        if k in ops:
            try:
                prev_frames = int(ops[k])
                break
            except Exception:
                prev_frames = None

    # Update frame count keys
    for k in ("nframes", "nFrames", "nFramesTot"):
        if k in ops:
            ops[k] = int(new_nframes)

    # If we know old_n and scaling is enabled, scale sampling rate fields to preserve total duration
    new_sampling = {}
    if scale_sampling and old_n is not None and old_n > 0:
        ratio = float(new_nframes) / float(old_n)
        for s_k in sampling_keys:
            if s_k in ops:
                try:
                    old_val = float(ops[s_k])
                    new_val = old_val * ratio
                    ops[s_k] = new_val
                    new_sampling[s_k] = new_val
                except Exception:
                    # ignore non-numeric sampling fields
                    pass

    # Append history entry recording previous and new values
    try:
        from datetime import datetime
        entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'action': 'update_ops_nframes',
            'old_nframes': prev_frames,
            'new_nframes': int(new_nframes),
            'old_sampling': prev_sampling,
            'new_sampling': new_sampling,
            'scale_sampling': bool(scale_sampling),
            'backup_path': backup_path,
        }
        if 'ops_history' in ops and isinstance(ops['ops_history'], list):
            ops['ops_history'].append(entry)
        else:
            ops['ops_history'] = [entry]
    except Exception:
        # non-critical: don't fail the update if logging fails
        pass

    np.save(ops_path, ops, allow_pickle=True)


def print_ops_history(ops_path: str) -> None:
    """Pretty-print the ops_history entries (if any) from an ops.npy file."""
    ops = np.load(ops_path, allow_pickle=True).item()
    hist = ops.get('ops_history', [])
    if not hist:
        print(f"No ops_history found in {ops_path}")
        return
    for i, e in enumerate(hist):
        print(f"[{i}] {e.get('timestamp')} action={e.get('action')} old_n={e.get('old_nframes')} new_n={e.get('new_nframes')} scale_sampling={e.get('scale_sampling')} backup={e.get('backup_path')}")


# --------------------------- CLI ----------------------------------------------

def main():
    import argparse

    p = argparse.ArgumentParser(description="Temporal smoothing for Suite2p binaries")
    p.add_argument("bin_path", help="Path to Suite2p .bin file")
    p.add_argument("--Lx", type=int, required=True)
    p.add_argument("--Ly", type=int, required=True)
    p.add_argument("--nchannels", type=int, default=1, help="Usually 1 (Suite2p splits channels)")
    p.add_argument("--dtype", type=str, default="int16", help="Input dtype (e.g., int16, uint16, float32)")

    p.add_argument("--method", choices=["block", "gaussian", "ema", "median", "savgol"], default="block")
    p.add_argument("--x", type=int, help="Block size for method=block")
    p.add_argument("--sigma_frames", type=float, help="Sigma (frames) for method=gaussian")
    p.add_argument("--tau_frames", type=float, help="Tau (frames) for method=ema")
    p.add_argument("--window_frames", type=int, help="Odd window length for method=median")
    p.add_argument("--sg_window", type=int, help="Odd window length for method=savgol")
    p.add_argument("--sg_polyorder", type=int, help="Polyorder for method=savgol")

    p.add_argument("--out", type=str, default=None, help="Output .bin path (ignored for --inplace)")
    p.add_argument("--out_dtype", type=str, default="float32")

    p.add_argument("--keep_remainder", action="store_true", help="For 'block' downsample or block_average")
    p.add_argument("--truncate", type=float, default=3.0, help="Gaussian truncate (radius = truncate*sigma)")
    p.add_argument("--chunk_frames", type=int, default=64, help="Frames per chunk for chunked methods")
    p.add_argument("--mode", choices=["reflect", "edge"], default="reflect", help="Boundary handling for Gaussian; median/savgol use 'edge' by default")
    p.add_argument("--init", choices=["copy_first", "zeros"], default="copy_first", help="EMA initialization")

    p.add_argument("--downsample_factor", type=int, default=None, help="Optional factor for gaussian/ema/median/savgol")
    p.add_argument("--downsample_mode", choices=["decimate", "block"], default="decimate", help="Downsample mode")
    p.add_argument("--inplace", action="store_true", help="Rename input to .orig and replace with output")
    p.add_argument("--ops", type=str, default=None, help="Path to ops.npy to update nframes")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-scale-sampling", action="store_true", help="Do not scale sampling rate (fs) when updating nframes")
    p.add_argument("--ops-backup", type=str, default=None, help="Path to the backup file created when doing inplace swaps (logged into ops_history)")

    args = p.parse_args()

    final_path, backup = smooth_suite2p_bin(
        bin_path=args.bin_path,
        Lx=args.Lx, Ly=args.Ly, nchannels=args.nchannels, dtype=args.dtype,
        method=args.method,
        x=args.x, sigma_frames=args.sigma_frames, tau_frames=args.tau_frames,
        window_frames=args.window_frames, sg_window=args.sg_window, sg_polyorder=args.sg_polyorder,
        out_path=args.out, out_dtype=args.out_dtype,
        inplace=args.inplace,
        downsample_factor=args.downsample_factor, downsample_mode=args.downsample_mode,
        keep_remainder=args.keep_remainder,
        truncate=args.truncate, chunk_frames=args.chunk_frames,
        mode=args.mode, verbose=args.verbose, init=args.init
    )

    print("Wrote:", final_path)
    if backup is not None:
        print("Backup of original:", backup)

    if args.ops is not None:
        # Compute new nframes from final_path
        spec_new = BinSpec(path=final_path, Lx=args.Lx, Ly=args.Ly, nchannels=args.nchannels, dtype=args.out_dtype)
        new_T = spec_new.n_frames()
    update_ops_nframes(args.ops, new_T, original_nframes=None, scale_sampling=(not args.no_scale_sampling), backup_path=args.ops_backup)
    print(f"Updated {args.ops} with nframes={new_T} (scale_sampling={not args.no_scale_sampling})")


if __name__ == "__main__":
    main()
