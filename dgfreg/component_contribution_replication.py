"""Functions that replicate component contribution's model training.

See https://gitlab.com/equilibrator/component-contribution/-/blob/develop/src/component_contribution/trainer.py for the target logic

"""
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from component_contribution.linalg import LINALG
from tqdm import tqdm


@dataclass
class ComponentContributionFit:
    """Result of fitting component contribution."""

    S: pd.DataFrame
    G: pd.DataFrame
    measurements: pd.DataFrame
    mu_dgf: pd.Series
    cov_dgf: pd.DataFrame


@dataclass
class ComponentContributionReplication:
    """Result of fitting component contribution and doing cross-validation."""

    fit: ComponentContributionFit
    fits_cv: List[ComponentContributionFit]
    splits: List[List[List[int]]]


def replicate_component_contribution(
    S_in: pd.DataFrame,
    G_in: pd.DataFrame,
    measurements_in: pd.DataFrame,
    splits: List[List[List[int]]],
) -> ComponentContributionReplication:
    """Do the main analysis."""
    S = S_in.copy()
    G = G_in.copy()
    measurements = measurements_in.copy()
    S.columns = map(int, S.columns)
    S.index = map(int, S.index)  # type: ignore
    G.index = map(int, G.index)  # type: ignore
    measurements.index = map(int, measurements.index)  # type: ignore
    fit = fit_component_contribution(S, G, measurements)
    fits_cv = []
    for ix_train, _ in tqdm(splits):
        S_train = S.iloc[:, ix_train]
        measurements_train = measurements.iloc[ix_train]
        fits_cv.append(
            fit_component_contribution(S_train, G, measurements_train)
        )
    return ComponentContributionReplication(fit, fits_cv, splits)


def fit_component_contribution(
    S_in: pd.DataFrame, G_in: pd.DataFrame, measurements: pd.DataFrame
) -> ComponentContributionFit:
    """Copy the component contribution fitting logic."""
    b = measurements["y"].values
    S = S_in.values
    G = G_in.values
    GS = G.T @ S

    assert (G_in.index == S_in.index).all()
    assert (measurements.index == S_in.columns).all()

    # Linear regression for the reactant layer (aka RC)
    inv_S, r_rc, P_R_rc, P_N_rc = LINALG.invert_project(S)

    # Linear regression for the group layer (aka GC)
    inv_GS, r_gc, P_R_gc, P_N_gc = LINALG.invert_project(GS)

    # calculate the group contributions
    dG0_gc = inv_GS.T @ b

    # Calculate the contributions in the stoichiometric space
    dG0_rc = inv_S.T @ b
    dG0_cc = P_R_rc @ dG0_rc + P_N_rc @ G @ dG0_gc

    # Calculate the residual error (unweighted squared error divided
    # by N - rank)
    e_rc = S.T @ dG0_rc - b
    MSE_rc = (e_rc.T @ e_rc) / (S.shape[1] - r_rc)

    e_gc = GS.T @ dG0_gc - b
    MSE_gc = (e_gc.T @ e_gc) / (S.shape[1] - r_gc)

    # Calculate the MSE of GC residuals for all reactions in ker(G).
    # This will help later to give an estimate of the uncertainty for such
    # reactions, which otherwise would have a 0 uncertainty in the GC
    # method.
    kerG_inds = list(np.where(np.all(GS == 0, 0))[0].flat)

    e_kerG = e_gc[kerG_inds]
    MSE_kerG = (e_kerG.T @ e_kerG) / len(kerG_inds)

    # Calculate the uncertainty covariance matrices
    inv_SWS, _, _, _ = LINALG.invert_project(S @ S.T)
    inv_GSWGS, _, _, _ = LINALG.invert_project(GS @ GS.T)

    V_rc = P_R_rc @ inv_SWS @ P_R_rc
    V_gc = P_N_rc @ G @ inv_GSWGS @ G.T @ P_N_rc
    V_inf = P_N_rc @ G @ P_N_gc @ G.T @ P_N_rc

    mu_dgf = pd.Series(dG0_cc, index=S_in.index)
    cov_dgf = pd.DataFrame(
        MSE_rc * V_rc + MSE_gc * V_gc + MSE_kerG * V_inf,
        index=S_in.index,
        columns=S_in.index,
    )
    return ComponentContributionFit(S_in, G_in, measurements, mu_dgf, cov_dgf)

