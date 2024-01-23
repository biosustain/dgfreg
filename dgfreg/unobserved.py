"""Functions generating samples for unobserved reactions and compounds.

The new compounds must have known group composition (and the new reactions must
only create and destroy such compounds).

"""

import numpy as np
import pandas as pd
import xarray as xr
from arviz import InferenceData


def get_compound_samples(
    idata: InferenceData, compounds: list[str], G: pd.DataFrame
) -> xr.DataArray:
    """Get formation energies for a list of compounds."""
    for cpd_id in compounds:
        assert cpd_id in G.index, f"{cpd_id} not in group composition matrix"
    dgfg_samples = idata.posterior["dgfG"]  # type: ignore
    for grp_id in G.columns:
        assert (
            grp_id in dgfg_samples.coords["group_id"]
        ), f"{grp_id} not in InferenceData coord 'group_id'."
    Gx = xr.DataArray(
        G.values, coords={"compound_id": G.index, "group_id": G.columns}
    )
    gc_deviation_random = np.random.normal(
        0,
        scale=np.repeat(
            idata.posterior["tauC"].values[:, :, np.newaxis],
            len(compounds),
            axis=2,
        ),
    )
    gc_deviation = (
        idata.posterior["qC"]
        .reindex(compound_id=compounds)
        .fillna(gc_deviation_random)
    )
    arr = Gx.loc[compounds] @ dgfg_samples.sel(group_id=G.columns)
    return arr.transpose("chain", "draw", "compound_id") + gc_deviation


def get_reaction_samples(
    idata: InferenceData, S: pd.DataFrame, G: pd.DataFrame
) -> xr.DataArray:
    dgfc_samples = get_compound_samples(idata, S.index.tolist(), G)
    Sx = xr.DataArray(
        S.values, coords={"compound_id": S.index, "reaction_id": S.columns}
    )
    return Sx.T @ dgfc_samples
