import os
from pathlib import Path
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from component_contribution.trainer import Trainer
from component_contribution.training_data import FullTrainingDataFactory
from equilibrator_api import ComponentContribution
from equilibrator_cache import create_compound_cache_from_zenodo
from equilibrator_cache.models.compound import Compound

CC_URLS = {
    "tecr": "https://zenodo.org/record/3978440/files/TECRDB.csv?download=1",
    "group_definitions": "https://zenodo.org/record/4010930/files/"
    "group_definitions.csv?download=1",
}
CC_DIR = Path(__file__).parent.parent / "data" / "raw"
S_FILE = CC_DIR / "S.csv"
G_FILE = CC_DIR / "G.csv"
COV_FILE = CC_DIR / "dgf_cov.csv"
RXN_FILE = CC_DIR / "reactions.csv"
CPD_FILE = CC_DIR / "compounds.csv"
GRP_FILE = CC_DIR / "group_summary.csv"
TECR_FILE = CC_DIR / "tecr.csv"
GROUP_DEF_FILE = CC_DIR / "group_definitions.csv"


def get_multivariate_dgf_distribution(
    compounds: List[Compound], cc: ComponentContribution
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Get the formation energy distribution for a list of compounds."""
    ix = [c.id for c in compounds]
    mu, sfin, sinf = zip(*[cc.standard_dg_formation(c) for c in compounds])
    cov = (
        np.array(sfin) @ np.array(sfin).T
#        + 1e6 * np.array(sinf) @ np.array(sinf).T
    )
    return mu, pd.DataFrame(cov, index=ix, columns=ix)


def main():
    """Run the script."""
    print("Fetching component contribution data...")
    cc = ComponentContribution()
    ccache = create_compound_cache_from_zenodo()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        td = FullTrainingDataFactory(ccache=ccache).make()
    tecr_df = pd.read_csv(CC_URLS["tecr"])
    group_definition_df = pd.read_csv(CC_URLS["group_definitions"])
    print("Putting component contribution data in tables...")
    group_df = td.group_summary
    # Add the equilibrator coefficients for each group to the group df
    n_groups = cc.predictor.params.dimensions.at["Ng", "number"]
    group_energies = cc.predictor.params.dG0_gc[:n_groups]
    group_df["dgg"] = group_energies
    G = Trainer.group_incidence_matrix(td)
    S = td.stoichiometric_matrix
    compounds = list(S.index)
    S.index = S.index.map(lambda c: c.id)
    S.index.name = "compound_cc_id"
    S.columns.name = "reaction_cc_id"
    G.index = G.index.map(lambda c: c.id)
    G.index.name = "compound_cc_id"
    G.columns = [g.id if isinstance(g, Compound) else g for g in G.columns]
    G.columns.name = "group_id"
    mu, cov = get_multivariate_dgf_distribution(compounds, cc)
    reaction_df = replace_pint_columns(td.reaction_df)
    reaction_df = reaction_df.drop(
        [
            "reference",
            "reaction",
            "temperature(kelvin)",
            "ionic_strength(molar)",
            "p_h",
            "p_mg",
            "balance",
            "weight",
        ],
        axis=1,
    )
    reaction_df["is_formation"] = (~S.abs().eq(0)).sum(axis=0).eq(1)
    compound_df = pd.DataFrame(
        {
            "inchi_key": [c.inchi_key for c in compounds],
            "smiles": [c.smiles for c in compounds],
            "cc_id": [c.id for c in compounds],
            "common_name": [c.get_common_name() for c in compounds],
            "dgf_cc": mu,
            "mass": [c.mass for c in compounds],
        }
    )
    print("Saving tables...")
    S.to_csv(S_FILE, index=True)
    G.to_csv(G_FILE, index=True)
    cov.to_csv(COV_FILE, index=True)
    reaction_df.to_csv(RXN_FILE, index=True)
    compound_df.to_csv(CPD_FILE, index=False)
    group_df.to_csv(GRP_FILE, index=True)
    tecr_df.to_csv(TECR_FILE, index=False)
    group_definition_df.to_csv(GROUP_DEF_FILE, index=False)


def replace_pint_columns(df):
    """
    Replace pint entries in a dataframe with plain magnitudes.
    """
    formatted_df = df.copy()
    new_cols = []
    for i in range(len(formatted_df.columns)):
        is_pint = True
        colname = formatted_df.columns[i]
        try:
            # Check that the units of all entries are the same
            units = [str(entry.units) for entry in formatted_df.iloc[:, i]]
            assert all(
                [units[0] == u for u in units]
            ), "All units in the column should be the same"
            # Dimensionless units can be ignored
            if units[0] != "dimensionless":
                colname += f"({units[0]})"
        except AttributeError:
            is_pint = False
        new_cols.append(colname)
        # Now replace the column's values with their magnitudes
        if is_pint:
            formatted_df.iloc[:, i] = [
                entry.magnitude for entry in formatted_df.iloc[:, i]
            ]
    formatted_df.columns = new_cols
    return formatted_df


if __name__ == "__main__":
    main()
