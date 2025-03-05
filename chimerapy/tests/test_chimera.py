import os
from pathlib import Path
import pytest
import numpy as np
from sunpy.map import Map
from chimerapy.chimera import generate_candidate_mask
from examples.paper_figures import mask_map

m171 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0171.fits")
m193 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0193.fits")
m211 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0211.fits")

def test_generate_candidate_mask():

    result_mask = generate_candidate_mask(m171, m193, m211)

    expected_shape = m171.data.shape
    assert result_mask.shape == expected_shape, "Mask shape does not match expected shape."
    
    expected_mask = mask_map.data.astype(bool)
    np.testing.assert_allclose(result_mask, expected_mask, rtol=1)