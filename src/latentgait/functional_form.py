"""Restricted cubic spline and descriptive derivative geometry (Supplementary S5)."""

import numpy as np


def rcs_basis(x, knots):
    """Harrell-type linear + K-2 nonlinear terms; linear tails beyond outer knots."""
    x, knots = np.asarray(x, float), np.asarray(knots, float)
    if knots.shape != (4,) or not np.all(np.diff(knots) > 0):
        raise ValueError("Four distinct increasing knots are required")
    a, b = knots[-2:]
    plus = lambda u: np.maximum(u, 0) ** 3
    terms = [x]
    for t in knots[:-2]:
        terms.append(
            (plus(x - t) - plus(x - a) * (b - t) / (b - a) + plus(x - b) * (a - t) / (b - a)) / (b - knots[0]) ** 2
        )
    return np.stack(terms, axis=-1)


def rcs_tensor(x, knots, center, scale):
    import pytensor.tensor as pt

    a, b = knots[-2:]
    plus = lambda u: pt.maximum(u, 0) ** 3
    terms = [x]
    for t in knots[:-2]:
        terms.append(
            (plus(x - t) - plus(x - a) * (b - t) / (b - a) + plus(x - b) * (a - t) / (b - a)) / (b - knots[0]) ** 2
        )
    return (pt.stack(terms, axis=-1) - center) / scale


def derivative_region(grid, probability, fraction=0.8):
    """Use derivative of posterior mean; retain disconnected sets and boundary flags."""
    grid, p = np.asarray(grid), np.asarray(probability)
    if grid.ndim != 1 or len(grid) < 3 or not np.all(np.diff(grid) > 0) or not 0 < fraction < 1:
        raise ValueError("Invalid evaluation grid or region fraction")
    d = np.gradient(p, grid, axis=-1, edge_order=2)
    mean_d = d.mean(axis=0) if d.ndim == 2 else d
    peak = mean_d.max()
    if peak <= 1e-12:
        return dict(peak_location=None, segments=[], boundary=False, derivative=mean_d)
    ix = np.flatnonzero(mean_d >= fraction * peak)
    groups = np.split(ix, np.where(np.diff(ix) > 1)[0] + 1)
    return dict(
        peak_location=float(grid[np.argmax(mean_d)]),
        segments=[[float(grid[g[0]]), float(grid[g[-1]])] for g in groups if len(g)],
        boundary=bool(ix[0] == 0 or ix[-1] == len(grid) - 1),
        derivative=mean_d,
    )


def common_region(segment_sets):
    """Intersect full connected-set representations without filling gaps."""
    sets = list(segment_sets)
    if not sets:
        return []
    result = sets[0]
    for other in sets[1:]:
        result = [[max(a, c), min(b, d)] for a, b in result for c, d in other if max(a, c) <= min(b, d)]
    return result
