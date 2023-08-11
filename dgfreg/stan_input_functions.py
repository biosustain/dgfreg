"""Functions for generating input to Stan from prepared data."""


from typing import Dict, List

import numpy as np
import pandas as pd
from dgfreg.data_preparation import PreparedData


def get_stan_input(prepped: PreparedData) -> Dict:
    """General function for creating a Stan input."""
    S, G = (
        getattr(prepped, attr).set_index(["compound_id", colsname])
        ["stoichiometric_coefficient"]
        .unstack()
        .fillna(0.0)
        for attr, colsname in zip(["S", "G"], ["reaction_id", "group_id"])
    )
    y = (
        prepped.measurements
        .groupby("reaction_id")
        .agg({"y": ["mean", "count"]})
        ["y"]
    )
    return {
        "NR": len(prepped.reactions),
        "NC": len(S),
        "NG": len(G.columns),
        "y": y["mean"],
        "nobs": y["count"],
        "N_train": len(y),
        "N_test": len(y),
        "ix_train": np.arange(len(y)) + 1,
        "ix_test": np.arange(len(y)) + 1,
        "S": S.values,
        "G": G.values,
    }


def get_stan_input_custom_holdback(prepped: PreparedData) -> Dict:
    """Create a Stan input with a custom train/test split."""
    excluded_compounds = [23, 70, 152]  # Acetyl CoA, PEP and G6P
    excluded_reactions = (
        prepped.S
        .set_index(["compound_id", "reaction_id"])
        ["stoichiometric_coefficient"]
        .unstack()
        .loc[excluded_compounds]
        .replace(0, np.nan)
        .stack()
        .reset_index()
        ["reaction_id"]
        .unique()
    )
    full = get_stan_input(prepped)
    ix_test = prepped.reactions.loc[
        lambda df: df["reaction_id"].isin(excluded_reactions)
    ].index + 1
    ix_train = [i for i in full["ix_train"] if i not in ix_test]
    return full | {
        "N_train": len(ix_train),
        "ix_train": ix_train,
    }

