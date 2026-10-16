"""Feature-family implementations for authorised tri-axial trunk acceleration.

Pelvis orientation and BTS spatiotemporal metrics are vendor exports: this module does
not attempt to reverse-engineer proprietary BTS algorithms.  Map validated exports in
the input configuration and retain vendor/version metadata externally.
"""
from __future__ import annotations
import numpy as np
from scipy import signal


def steady_state_segment(x: np.ndarray, contacts: np.ndarray, *, min_strides: int = 20) -> np.ndarray:
    """Return the segment excluding first/last two strides after record-level QC."""
    if len(contacts) - 1 < min_strides:
        raise ValueError("Trial has fewer than 20 valid consecutive strides")
    kept = contacts[2:-2]
    return np.asarray(x)[kept[0]:kept[-1]]


def harmonic_ratio(x: np.ndarray, fs: float, stride_seconds: float, axis: str, improved: bool = False) -> float:
    """HR/iHR from first 20 DFT harmonics; protocol parameters are explicit."""
    b, a = signal.butter(4, 20, btype="low", fs=fs)
    x = signal.filtfilt(b, a, np.asarray(x, float) - np.mean(x))
    t = np.arange(len(x)) / fs
    f0 = 1.0 / stride_seconds
    amp = np.array([np.hypot(np.sum(x*np.cos(2*np.pi*h*f0*t)), np.sum(x*np.sin(2*np.pi*h*f0*t))) for h in range(1, 21)])
    if improved:
        # iHR harmonic classification requires the documented gait-cycle phase convention.
        # Implement only after the protocol's intrinsic/extrinsic mapping is supplied.
        raise NotImplementedError("Configure the validated iHR phase/harmonic mapping before use.")
    even = amp[1::2].sum(); odd = amp[0::2].sum()
    return float((even / odd) if axis.upper() in {"AP", "V"} else (odd / even))


def feature_protocol_todos() -> dict[str, str]:
    return {"iHR": "supply intrinsic/extrinsic harmonic mapping and phase convention", "determinism": "supply selected AMI/FNN/radius values within documented ranges", "MSE": "validate sample-entropy implementation against the original custom routine"}
