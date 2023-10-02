import pytest
import pandas as pd
import dgfreg.mixed_model as mixed_model


def test_fit_mixed_model(basic_network):
    """ Basic test that the model can recover parameters of a mixed model"""
    bn = basic_network
    measurements = pd.DataFrame()
    measurements["y"] = bn.dgr
    fit = mixed_model.fit_mixed_model(bn.S, bn.G, measurements)
    
