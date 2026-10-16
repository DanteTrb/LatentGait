"""Documented S1/S1a equations; unspecified numerical choices are required arguments.

Input axes are physical V/ML/AP at 100 Hz, with verified stride boundaries (one
full gait cycle, not alternating left/right contacts). BTS pelvis angles and
spatiotemporal metrics must be imported from authorised vendor exports.
"""

import numpy as np
from scipy import signal
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform


def vector(x):
    x = np.asarray(x, float)
    if x.ndim != 1 or not np.isfinite(x).all() or len(x) < 3:
        raise ValueError("A finite one-dimensional signal is required")
    return x


def trim_boundaries(x, stride_boundaries):
    x = vector(x)
    b = np.asarray(stride_boundaries)
    if b.ndim != 1 or not np.issubdtype(b.dtype, np.integer) or not np.all(np.diff(b) > 0):
        raise ValueError("Increasing integer full-stride boundaries required")
    if b[0] < 0 or b[-1] > len(x):
        raise ValueError("Stride boundary outside signal")
    kept = b[2:-2]  # remove exactly the first and last two complete strides
    if len(kept) - 1 < 20:
        raise ValueError("Fewer than 20 retained strides after trimming")
    return kept


def event_candidates(vertical, fs, *, fir_numtaps, fir_window):
    """FIR 3.2 Hz + gaus1 scale-9 CWT for external event QC.

    FIR length/window and boundary handling are not given by the paper. This
    helper returns candidates and the wavelet signal, not certified gait events.
    Users must verify full-stride boundaries before calling feature extraction.
    """
    import pywt

    x = signal.detrend(vector(vertical))
    if fs != 100 or fir_numtaps < 3:
        raise ValueError("Paper acquisition rate is 100 Hz; supply FIR design")
    taps = signal.firwin(fir_numtaps, 3.2, fs=fs, window=fir_window)
    filtered = signal.filtfilt(taps, [1.0], x)
    wavelet = pywt.cwt(filtered, [9], "gaus1", sampling_period=1 / fs)[0][0]
    crossings = np.flatnonzero(np.diff(np.signbit(filtered)))
    candidates = []
    for a, b in zip(crossings[:-1], crossings[1:]):
        part = filtered[a + 1 : b + 1]
        if len(part) and part.max() > 0:
            candidates.append(a + 1 + int(np.argmax(part)))
    return np.asarray(candidates, int), wavelet


def harmonic_features(x, fs, stride_boundaries, axis):
    """Mean stride HR and iHR: 20 harmonic amplitudes/powers, axis-specific parity."""
    if axis.lower() not in ("v", "ml", "ap") or fs != 100:
        raise ValueError("Select V/ML/AP at the manuscript's 100-Hz sampling rate")
    x = vector(x)
    b = trim_boundaries(x, stride_boundaries)
    filtered = signal.sosfiltfilt(signal.butter(4, 20, fs=fs, output="sos"), x - x.mean())
    hrs, ihrs = [], []
    intrinsic = np.arange(20) % 2 == (0 if axis.lower() == "ml" else 1)
    for start, end in zip(b[:-1], b[1:]):
        stride = filtered[start:end]
        if len(stride) <= 40:
            raise ValueError("Stride sampling cannot resolve the first 20 harmonics")
        phase = 2 * np.pi * np.arange(len(stride)) / len(stride)
        h = np.arange(1, 21)[:, None]
        cosine = 2 / len(stride) * (np.cos(h * phase) @ stride)
        sine = 2 / len(stride) * (np.sin(h * phase) @ stride)
        power = cosine**2 + sine**2
        amplitude = np.sqrt(power)
        denominator = amplitude[~intrinsic].sum()
        if denominator <= 1e-12 or power.sum() <= 1e-12:
            raise ValueError("Undefined harmonic ratio in degenerate stride")
        hrs.append(amplitude[intrinsic].sum() / denominator)
        ihrs.append(100 * power[intrinsic].sum() / power.sum())
    return float(np.mean(hrs)), float(np.mean(ihrs))


def resample_rbf(x, *, kernel, smoothing, neighbors, epsilon):
    """2,000-point RBF resampling. All unspecified RBF settings are explicit."""
    x = vector(x)
    if smoothing < 0 or neighbors is not None and neighbors < 2:
        raise ValueError("Invalid RBF settings")
    t = np.linspace(0, 1, len(x))[:, None]
    interpolator = RBFInterpolator(t, x, kernel=kernel, smoothing=smoothing, neighbors=neighbors, epsilon=epsilon)
    return np.asarray(interpolator(np.linspace(0, 1, 2000)[:, None]))


def ami_delay(x, *, bins):
    """First interior AMI minimum among delays 5..19 for a declared histogram bin count."""
    x = vector(x)
    if not 4 <= bins <= 19:
        raise ValueError("AMI histogram bins must lie in documented range 4..19")
    values = []
    edges = np.linspace(x.min(), x.max(), bins + 1)
    if edges[0] == edges[-1]:
        raise ValueError("Constant signal")
    for lag in range(5, 20):
        counts = np.histogram2d(x[:-lag], x[lag:], bins=(edges, edges))[0]
        p = counts / counts.sum()
        independent = p.sum(1)[:, None] * p.sum(0)[None, :]
        valid = p > 0
        values.append(np.sum(p[valid] * np.log(p[valid] / independent[valid])))
    minima = [j for j in range(1, 14) if values[j] < values[j - 1] and values[j] <= values[j + 1]]
    if not minima:
        raise ValueError("No first interior AMI minimum in documented lag range; protocol decision required")
    return 5 + minima[0]


def embedding(x, dimension, delay):
    x = vector(x)
    n = len(x) - (dimension - 1) * delay
    if dimension < 1 or delay < 1 or n < 2:
        raise ValueError("Insufficient signal for embedding")
    return np.column_stack([x[j * delay : j * delay + n] for j in range(dimension)])


def fnn_dimension(x, delay, *, rtol, atol, theiler):
    """Smallest dimension 3..10 with <=1% false neighbours for declared tolerances."""
    x = vector(x)
    if not 15 <= rtol <= 19 or not 2 <= atol <= 4 or theiler < 0 or x.std() <= 0:
        raise ValueError("Invalid FNN parameters")
    for m in range(3, 11):
        full = embedding(x, m + 1, delay)
        base = full[:, :-1]
        distances, indices = cKDTree(base).query(base, k=min(len(base), 2 * theiler + 3))
        false, total = 0, 0
        for i, (d, ix) in enumerate(zip(distances, indices)):
            permitted = np.flatnonzero(abs(ix - i) > theiler)
            if not len(permitted):
                continue
            j = permitted[0]
            extra = abs(full[i, -1] - full[ix[j], -1])
            false += extra > rtol * max(d[j], 1e-15) or np.hypot(d[j], extra) / x.std() > atol
            total += 1
        if total and false / total <= 0.01:
            return m
    raise ValueError("FNN criterion not met in dimensions 3..10; no silent fallback")


def percent_determinism(x, dimension, delay, *, radius_fraction, lmin, theiler):
    """100 sum(l P(l), l>=lmin)/sum(l P(l)); upper triangle, Euclidean distance."""
    if not 0.05 <= radius_fraction <= 0.20 or lmin < 2 or theiler < 0:
        raise ValueError("Declare documented radius and diagonal-line/Theiler settings")
    distances = pdist(embedding(x, dimension, delay), metric="euclidean")
    maximum = distances.max()
    if maximum <= 0:
        raise ValueError("Degenerate recurrence phase space")
    recurrence = squareform(distances) <= radius_fraction * maximum
    total, deterministic = 0, 0
    for offset in range(theiler + 1, len(recurrence)):
        line = np.diag(recurrence, offset)
        edges = np.diff(np.r_[False, line, False].astype(int))
        lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
        total += lengths.sum()
        deterministic += lengths[lengths >= lmin].sum()
    if total == 0:
        raise ValueError("Empty recurrence plot at selected radius")
    return float(100 * deterministic / total)


def sample_entropy(x, *, m=2, tolerance=0.1):
    """Chebyshev matches, self-matches excluded; equal template support for m/m+1."""
    x = vector(x)
    templates = np.lib.stride_tricks.sliding_window_view(x, m + 1)
    if tolerance <= 0 or len(templates) < 2:
        raise ValueError("Invalid entropy tolerance or series length")
    b = len(cKDTree(templates[:, :m]).query_pairs(tolerance, p=np.inf))
    a = len(cKDTree(templates).query_pairs(tolerance, p=np.inf))
    if not b or not a:
        raise ValueError("Sample entropy undefined: no m or m+1 template matches")
    return float(-np.log(a / b))


def multiscale_entropy(x):
    """Normalize the 2,000-point signal; coarse-grain at scale 4; m=2, lag=1, r=.1."""
    x = vector(x)
    if len(x) != 2000 or x.std() <= 1e-12:
        raise ValueError("MSE requires a nonconstant, resampled 2,000-point signal")
    z = (x - x.mean()) / x.std(ddof=0)
    coarse = z.reshape(-1, 4).mean(axis=1)
    # r is based on the normalized pre-coarse-grained signal SD, not recomputed at scale 4.
    return sample_entropy(coarse, m=2, tolerance=0.1)


def extract_axis(x, boundaries, axis, *, rbf, rqa):
    """Feature extraction from a QC-verified axis and full-stride boundaries."""
    kept = trim_boundaries(x, boundaries)
    hr, ihr = harmonic_features(x, 100, boundaries, axis)
    segment = resample_rbf(np.asarray(x)[kept[0] : kept[-1]], **rbf)
    delay = ami_delay(segment, bins=rqa["ami_bins"])
    dimension = fnn_dimension(segment, delay, rtol=rqa["rtol"], atol=rqa["atol"], theiler=rqa["theiler"])
    det = percent_determinism(
        segment, dimension, delay, radius_fraction=rqa["radius_fraction"], lmin=rqa["lmin"], theiler=rqa["theiler"]
    )
    start = (len(kept) - 1 - 20) // 2
    entropy_segment = resample_rbf(np.asarray(x)[kept[start] : kept[start + 20]], **rbf)
    return dict(hr=hr, ihr=ihr, det=det, mse=multiscale_entropy(entropy_segment), delay=delay, dimension=dimension)
