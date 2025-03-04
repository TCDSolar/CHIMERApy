import os
from pathlib import Path
import pytest
import numpy as np
from sunpy.map import Map
from chimerapy.chimera import generate_candidate_mask
from examples.paper_figures import mask_map

aia171= Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0171.fits")
aia193= Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0193.fits")
aia211= Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0211.fits")

def test_generate_candidate_mask():

    result_mask = generate_candidate_mask(aia171, aia193, aia211)

    expected_shape = aia171.data.shape
    assert result_mask.shape == expected_shape, "Mask shape does not match expected shape."
    
    expected_mask = mask_map.data.astype(bool)
    np.testing.assert_allclose(result_mask, expected_mask, atol=1)