import pytest
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class TestData:
    G: pd.DataFrame
    S: pd.DataFrame
    dgr: pd.Series
    dgf: pd.Series
    dgg: pd.Series
    dgf_rand_scale: float
    dgr_rand_scale: float

@pytest.fixture
def basic_network():
    """ A small network test case with 4 groups, 4 mets and 5 rxns"""
    rng = np.random.default_rng(seed=42)
    G_val = np.array([[1,2,11,1], [2,7,0,4], [1,5,2,2], [3,3,2,7]])
    S_val = np.array([[-1, 1, 0, 0], 
                  [-1, 0, 1, 0],
                  [0, -1, 1, 0],
                  [-1, -1, 0, 1],
                  [0, 1, 1, -1],
                  [-1, 0, -1, 1]]).T
                  # Repeat for more observations
    rxn_names = ["R"+str(i) for i in range(S_val.shape[1])]
    met_names = ["M"+str(i) for i in range(S_val.shape[0])]
    grp_names = ["G"+str(i) for i in range(G_val.shape[1])]
    G = pd.DataFrame(G_val, index=met_names, columns=grp_names)
    S = pd.DataFrame(S_val, index=met_names, columns=rxn_names)

    #TODO: REMOVE
    S = pd.DataFrame(rng.normal(size=S.shape), index=met_names, columns=rxn_names)

    dgg = pd.Series([1, 12, 6, 3], index=grp_names)
    # We assume all formation energy random effects are independent draws of a 0-centred normal distribution (we can't expect estimation to be good with a sample size of 3)
    dgf_rand_scale = 0.1
    dgf_rand = pd.Series(rng.normal(0, dgf_rand_scale, size=len(met_names)), index=met_names)
    # Now we calculate dgf and dgr measurements
    dgf = G @ dgg + dgf_rand
    # The reactions also have an error term
    dgr_rand_scale = 1
    dgr_rand = pd.Series(rng.normal(0, dgr_rand_scale, size=len(rxn_names)), index=rxn_names) 
    dgr = S.T @ dgf + dgr_rand
    return TestData(G, S, dgr, dgf, dgg, dgf_rand_scale, dgr_rand_scale)



# Small network where some values are floating as well
