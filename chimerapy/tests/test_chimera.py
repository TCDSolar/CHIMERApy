import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sunpy.map import Map

from chimerapy.chimera import generate_candidate_mask, get_area_map
from examples.paper_figures import mask_map


@pytest.fixture(scope="module")
def m171():
    return Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0171.fits")


@pytest.fixture(scope="module")
def m193():
    return Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0193.fits")


@pytest.fixture(scope="module")
def m211():
    return Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0211.fits")


def test_generate_candidate_mask(m171, m193, m211):
    result_mask = generate_candidate_mask(m171, m193, m211)

    expected_shape = m171.data.shape
    assert result_mask.shape == expected_shape, "Mask shape does not match expected shape."

    expected_mask = mask_map.data.astype(bool)
    np.testing.assert_allclose(result_mask, expected_mask)


def test_get_area_map(m171):
    area_map = get_area_map(m171)
    total_area = area_map.sum()
    assert_almost_equal(m171.rsun_meters**2 * np.pi * 2, total_area)
