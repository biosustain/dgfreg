"""Provides functions prepare_data_x.

These functions should take in a dataframe of measurements and return a
PreparedData object.

"""
import json
import os

from os.path import join as pjoin

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series
from pydantic import BaseModel
from dgfreg import util

PREPARED_FILES = [
    "name.txt",
    "coords.json",
    "measurements.csv",
    "S.csv",
    "G.csv",
    "compounds.csv",
    "reactions.csv",
]
N_CV_FOLDS = 10
HERE = os.path.dirname(__file__)
DATA_DIR = pjoin(HERE, "..", "data")
RAW_DIR = pjoin(DATA_DIR, "raw")
PREPARED_DIR = pjoin(DATA_DIR, "prepared")
RAW_DATA_FILES = {
    "e_coli_ccm_metabolites": pjoin(RAW_DIR, "e_coli_ccm_metabolites.csv"),
    "e_coli_ccm_reactions": pjoin(RAW_DIR, "e_coli_ccm_reactions.csv"),
    "equilibrator_group_decomposition": pjoin(RAW_DIR, "G.csv"),
    "equilibrator_stoichiometries": pjoin(RAW_DIR, "S.csv"),
    "equilibrator_compounds": pjoin(RAW_DIR, "compounds.csv"),
    "equilibrator_dgf_cov": pjoin(RAW_DIR, "dgf_cov.csv"),
    "equilibrator_reactions": pjoin(RAW_DIR, "reactions.csv"),
    "equilibrator_tecr": pjoin(RAW_DIR, "tecr.csv"),
    "equilibrator_group_definitions": pjoin(RAW_DIR, "group_definitions.csv"),
    "equilibrator_group_summary": pjoin(RAW_DIR, "group_summary.csv"),
}



class MeasurementsDF(pa.SchemaModel):
    """A PreparedData should have a measurements dataframe like this.

    Other columns are also allowed!
    """

    y: Series[float]


class StoichiometricMatrix(pa.SchemaModel):
    """Map of compound id/reaction id pairs to stoichiometric coefficients.

    To save space the matrix is stored in long form, with zero entries ignored.

    """

    compound_id: Series[int]
    reaction_id: Series[int]
    stoichiometric_coefficient: Series[float] = pa.Field(ne=0)


class GroupDecompositionMatrix(pa.SchemaModel):
    """Map of compound id/group id to stoichiometric coefficient.

    Stored in long form, all entries should be greater than zero.

    """

    compound_id: Series[int]
    group_id: Series[str]
    stoichiometric_coefficient: Series[float] = pa.Field(gt=0)


class CompoundDF(pa.SchemaModel):
    """Information about compounds."""
    compound_id: Series[int]
    is_e_coli_ccm: Series[bool]


class ReactionDF(pa.SchemaModel):
    """Information about compounds."""
    reaction_id: Series[int]
    is_e_coli_ccm: Series[bool]


class PreparedData(BaseModel, arbitrary_types_allowed=True):
    """What prepared data looks like in this analysis."""

    name: str
    coords: util.CoordDict
    measurements: DataFrame[MeasurementsDF]
    reactions: DataFrame[ReactionDF]
    S: DataFrame[StoichiometricMatrix]
    G: DataFrame[GroupDecompositionMatrix]
    compounds: DataFrame[CompoundDF]


def prepare_data_equilibrator(**raw_data) -> PreparedData:
    ccm_ecs = raw_data["e_coli_ccm_reactions"]["ec-code"].unique()
    ccm_inchi_keys = (
        raw_data["e_coli_ccm_metabolites"]["inchi_key"].unique()
    )
    y_cols = [
        "measurement_id",
        "reaction_id",
        "y",
        "p_h",
        "temperature",
        "ionic_strength",
        "p_mg",
        "reference",
        "method",
        "eval",
    ]
    reaction_cols = [
        "reaction_id",
        "EC",
        "is_formation",
        "is_e_coli_ccm",
        "description",
        "reaction",
    ]
    G = (
        raw_data["equilibrator_group_decomposition"]
        .rename(columns={"compound_cc_id": "compound_id"})
        .mask(lambda df: df == 0)
        .set_index("compound_id")
        .rename_axis("group_id", axis=1)
        .stack()
        .rename("stoichiometric_coefficient")
        .reset_index()
        .assign(
            group_id=lambda df: df["group_id"].astype("string"),
            compound_id=lambda df: df["compound_id"].astype(int),
        )
    )
    S_full = (
        raw_data["equilibrator_stoichiometries"]
        .rename(columns={"compound_cc_id": "compound_id"})
        .set_index("compound_id")
        .T
        .drop_duplicates()
        .T
    )
    S_full = (
        S_full
        .set_axis(range(1, len(S_full.columns) + 1), axis=1)
        .rename_axis("reaction_id", axis=1)
    )
    measurement_to_reaction = (
        raw_data["equilibrator_stoichiometries"]
        .set_index("compound_cc_id")
        .T
        .join(
            S_full.T.reset_index().set_index([c for c in S_full.index])
            ["reaction_id"], 
            on=[c for c in S_full.index]
        )
        ["reaction_id"]
    )
    measurement_to_reaction.index = map(int, measurement_to_reaction.index)
    S = (
        S_full
        .mask(lambda df: df == 0)
        .rename_axis("reaction_id", axis=1)
        .stack()
        .rename("stoichiometric_coefficient")
        .reset_index()
        .assign(
            compound_id=lambda df: df["compound_id"].astype(int),
            reaction_id=lambda df: df["reaction_id"].astype(int)
        )
    )
    reactions = (
        raw_data["equilibrator_reactions"]
        .rename(columns={"Unnamed: 0": "measurement_id"})
        .join(
            raw_data["equilibrator_tecr"], 
            on="measurement_id", rsuffix="_tecr"
        )
        .join(measurement_to_reaction, on="measurement_id")
        .assign(is_e_coli_ccm=lambda df: df["EC"].isin(ccm_ecs))
        .groupby("reaction_id")
        .first()
        .reset_index()
        [reaction_cols]
    )
    measurements = (
        raw_data["equilibrator_reactions"]
        .rename(
            columns={
                "Unnamed: 0": "measurement_id", 
                "standard_dg(kilojoule / mole)": "y"
            }
        )
        .join(
            raw_data["equilibrator_tecr"], 
            on="measurement_id", rsuffix="_tecr"
        )
        .join(measurement_to_reaction, on="measurement_id")
        [y_cols]
    )
    compounds = (
        raw_data["equilibrator_compounds"]
        .rename(columns={"cc_id": "compound_id"})
        .assign(
            is_e_coli_ccm=lambda df: df["inchi_key"].isin(ccm_inchi_keys),
            compound_id=lambda df: df["compound_id"].astype(int),
        )
    )
    coords = {
        "compound_id": compounds["compound_id"].astype(str).tolist(),
        "group_id": (
            G.set_index(["compound_id", "group_id"])
            ["stoichiometric_coefficient"]
            .unstack()
            .columns
            .astype(str)
            .tolist()
        ),
        "measurement_id": measurements["measurement_id"].astype(str).tolist(),
        "reaction_id": reactions["reaction_id"].astype(str).tolist(),
    }
    return PreparedData(
        name="equilibrator",
        coords=coords,
        measurements=measurements,
        reactions=reactions,
        S=S,
        G=G,
        compounds=compounds,
    )


def prepare_data():
    """Run main function."""
    print("Reading raw data...")
    raw_data = {
        k: pd.read_csv(v, index_col=None) for k, v in RAW_DATA_FILES.items()
    }
    data_preparation_functions_to_run = [prepare_data_equilibrator]
    print("Preparing data...")
    for dpf in data_preparation_functions_to_run:
        print(f"Running data preparation function {dpf.__name__}...")
        prepared_data = dpf(**raw_data)
        output_dir = os.path.join(PREPARED_DIR, prepared_data.name)
        print(f"\twriting files to {output_dir}")
        if not os.path.exists(PREPARED_DIR):
            os.mkdir(PREPARED_DIR)
        write_prepared_data(prepared_data, output_dir)


def load_prepared_data(directory: str) -> PreparedData:
    """Load prepared data from files in directory."""
    with open(os.path.join(directory, "coords.json"), "r") as f:
        coords = json.load(f)
    with open(os.path.join(directory, "name.txt"), "r") as f:
        name = f.read()
    dfs = {
        "".join(f.split(".")[:-1]): pd.read_csv(os.path.join(directory, f))
        for f in PREPARED_FILES if f.endswith(".csv")
    }
    return PreparedData(
        name=name,
        coords=coords,
        measurements=DataFrame[MeasurementsDF](dfs["measurements"]),
        reactions=DataFrame[ReactionDF](dfs["reactions"]),
        S=DataFrame[StoichiometricMatrix](dfs["S"]),
        G=DataFrame[GroupDecompositionMatrix](dfs["G"]),
        compounds=DataFrame[CompoundDF](dfs["compounds"]),
    )


def write_prepared_data(prepped: PreparedData, directory):
    """Write prepared data files to a directory."""
    if not os.path.exists(directory):
        os.mkdir(directory)
    with open(os.path.join(directory, "coords.json"), "w") as f:
        json.dump(prepped.coords, f)
    with open(os.path.join(directory, "name.txt"), "w") as f:
        f.write(prepped.name)
    for attr in ["measurements", "S", "G", "compounds", "reactions"]:
        getattr(prepped, attr).to_csv(
            os.path.join(directory, attr + ".csv"), index=False
        )


