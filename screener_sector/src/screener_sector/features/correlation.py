"""Return correlation, benchmark residualization, and hierarchical clustering.

Correlating price levels is meaningless when every name is trending: the
result is near 1.0 for unrelated stocks. Everything here works on log returns.
Residualizing against the sector benchmark then answers the sharper question:
which names move together for reasons beyond shared sector beta?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def log_returns(panel: pd.DataFrame) -> pd.DataFrame:
    return np.log(panel.astype(float)).diff().iloc[1:]


def correlation_matrix(returns: pd.DataFrame, min_overlap: int = 30) -> pd.DataFrame:
    corr = returns.corr(min_periods=min_overlap)
    return corr.reindex(index=returns.columns, columns=returns.columns)


def residualize(returns: pd.DataFrame, factor: pd.Series) -> pd.DataFrame:
    """Strip the benchmark factor from each column via OLS, keeping residuals."""
    aligned_factor = factor.reindex(returns.index)
    out: dict[str, pd.Series] = {}
    for column in returns.columns:
        pair = pd.concat([returns[column], aligned_factor], axis=1).dropna()
        if len(pair) < 30 or pair.iloc[:, 1].std() == 0:
            out[column] = returns[column]
            continue
        y = pair.iloc[:, 0].to_numpy(dtype=float)
        x = pair.iloc[:, 1].to_numpy(dtype=float)
        beta, alpha = np.polyfit(x, y, 1)
        residual = pd.Series(y - (beta * x + alpha), index=pair.index)
        out[column] = residual.reindex(returns.index)
    return pd.DataFrame(out, index=returns.index)


def correlation_distance(corr: pd.DataFrame) -> np.ndarray:
    """Condensed distance vector from a correlation matrix.

    d = sqrt(2 * (1 - rho)): 0 for perfectly correlated, 2 for perfectly
    anticorrelated. NaN correlations become maximum distance so that names
    without enough overlap never merge into a cluster.
    """
    filled = corr.to_numpy(dtype=float).copy()
    filled[np.isnan(filled)] = -1.0
    np.fill_diagonal(filled, 1.0)
    distance = np.sqrt(2.0 * (1.0 - np.clip(filled, -1.0, 1.0)))
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    return squareform(distance, checks=False)


@dataclass(frozen=True)
class Cluster:
    label: int
    members: tuple[str, ...]
    mean_correlation: float


@dataclass(frozen=True)
class ClusterResult:
    clusters: tuple[Cluster, ...]
    assignments: pd.Series
    raw_corr: pd.DataFrame
    residual_corr: pd.DataFrame


def _mean_pairwise(corr: pd.DataFrame, members: list[str]) -> float:
    block = corr.loc[members, members].to_numpy(dtype=float)
    upper = block[np.triu_indices(len(members), 1)]
    upper = upper[~np.isnan(upper)]
    return float(upper.mean()) if upper.size else float("nan")


def cluster_universe(
    panel: pd.DataFrame,
    benchmark: pd.Series | None,
    threshold: float,
    min_size: int,
    window: int,
) -> ClusterResult:
    returns = log_returns(panel).tail(window)
    raw_corr = correlation_matrix(returns)

    if benchmark is not None:
        factor = log_returns(benchmark.to_frame("bm")).tail(window)["bm"]
        residual_returns = residualize(returns, factor)
    else:
        residual_returns = returns
    residual_corr = correlation_matrix(residual_returns)

    tickers = list(returns.columns)
    assignments = pd.Series(-1, index=tickers, dtype=int)

    if len(tickers) < min_size:
        return ClusterResult((), assignments, raw_corr, residual_corr)

    distance = correlation_distance(residual_corr)
    tree = linkage(distance, method="average")
    cut = np.sqrt(2.0 * (1.0 - threshold))
    labels = fcluster(tree, t=cut, criterion="distance")

    clusters: list[Cluster] = []
    next_label = 0
    for raw_label in sorted(set(labels)):
        members = [t for t, lab in zip(tickers, labels) if lab == raw_label]
        if len(members) < min_size:
            continue
        mean_corr = _mean_pairwise(residual_corr, members)
        if not np.isfinite(mean_corr) or mean_corr < threshold:
            continue
        clusters.append(Cluster(next_label, tuple(members), mean_corr))
        assignments.loc[members] = next_label
        next_label += 1

    return ClusterResult(tuple(clusters), assignments, raw_corr, residual_corr)
