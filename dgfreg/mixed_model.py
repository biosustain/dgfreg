from dataclasses import dataclass
import statsmodels.api as sm
import pandas as pd

@dataclass
class MixedModelFit:
    """Result of fitting component contribution."""
    S: pd.DataFrame
    G: pd.DataFrame
    measurements: pd.DataFrame
    mu_dgf: pd.Series
    cov_dgf: pd.DataFrame
    # TODO: We need dgr eps


    
    def pred_dgr():
        """ Predict the dgr from the prediction"""
        mu_dgr = S.T @ mu_dgf
        # TODO: Test that this is correct
        cov_dgr = S.T @ cov_dgf @ S


def fit_mixed_model(
    S_in: pd.DataFrame, G_in: pd.DataFrame, measurements: pd.DataFrame
) -> MixedModelFit:
    """ Fit a mixed model with normal gc as fixed effects with random effects for compounds"""
    b = measurements["y"]
    S = S_in.values 
    G = G_in.values
    SG = S.T @ G
    compound_names = G_in.index
    mixed_model = sm.MixedLM(b, SG, compound_names, exog_re=S.T)

    breakpoint()

    result = mixed_model.fit()

    #TODO: Look for scale param
    breakpoint()
    #TODO: Return scale param as well

    fixed_dgf = G @ result.fe_params
    final_dgf = fixed_dgf + result.random_effects
    # Transform the fixed effects covariance matrix with the group matrix to get the covariance on the means
    fe_cov = results.cov_params(r_matrix=G)
    final_cov = fe_cov + results.cov_re
    return MixedModelFit(S_in, G_in, measurements, final_dgf, final_cov)


