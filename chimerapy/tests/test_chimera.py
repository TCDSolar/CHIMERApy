import os
from pathlib import Path
import pytest
from parfive import Downloader
import numpy as np
from sunpy.map import Map
from chimerapy.chimera import generate_candidate_mask
from examples.paper_figures import mask_map

aia171= Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/09/22/H1200/AIA20160922_1200_0171.fits")
aia193= Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/09/22/H1200/AIA20160922_1200_0193.fits")
aia211= Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/09/22/H1200/AIA20160922_1200_0211.fits")

def test_generate_candidate_mask():

    result_mask = generate_candidate_mask(aia171, aia193, aia211)

    expected_shape = aia171.data.shape
    assert result_mask.shape == expected_shape, "Mask shape does not match expected shape."
    
    expected_mask = mask_map
    np.testing.assert_allclose(result_mask, expected_mask, atol=0.1)
