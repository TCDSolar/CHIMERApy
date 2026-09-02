from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose

from sunpy.map import Map, all_coordinates_from_map

from chimerapy.chimera import calculate_area_map, filter_ch, generate_candidate_mask

DATA_DIR = Path(__file__).parent / "data"


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
    # Frozen baseline of generate_candidate_mask for the 2016-10-31 synoptic maps.
    # Regenerate with:
    # import numpy as np; from sunpy.map import Map
    # from chimerapy.chimera import generate_candidate_mask as g
    # m=[Map(u) for u in ('.../AIA20161031_0232_0171.fits', ...0193..., ...0211...)]
    # np.savez_compressed('chimerapy/tests/data/candidate_mask_2016-10-31.npz',
    #                     mask=np.asarray(g(*m)).astype(bool))
    expected_mask = np.load(DATA_DIR / "candidate_mask_2016-10-31.npz")["mask"]

    result_mask = generate_candidate_mask(m171, m193, m211)

    assert result_mask.shape == m171.data.shape, "Mask shape does not match expected shape."
    np.testing.assert_array_equal(np.asarray(result_mask).astype(bool), expected_mask)


def test_calculate_area_map(m171):
    area_map, disk_mask = calculate_area_map(m171)
    total_area = area_map.sum()
    hemi_sphere_area = 2 * np.pi * m171.rsun_meters**2
    assert_allclose(total_area, hemi_sphere_area, rtol=5e-4)  # 0.05% seems pretty ok?


@pytest.mark.parametrize("theta", ([5, 10, 15, 45, 75, 90] * u.deg))
def test_filter_by_area_size(m171, theta):
    hpc_coords = all_coordinates_from_map(m171)
    hgs_coords = hpc_coords.transform_to("heliographic_stonyhurst")
    ref = m171.reference_coordinate
    radial_angle = hgs_coords.separation(ref)
    data = np.zeros_like(m171.data)
    data[radial_angle <= theta] = 1
    m171.data[:, :] = data
    label_mask, regions = filter_ch(data, m171, min_area=0 * u.m**2)
    expected_area = 2 * np.pi * (1 - np.cos(theta)) * m171.rsun_meters**2
    assert_allclose(regions[0].surface_area, expected_area, rtol=0.01)


@pytest.mark.parametrize("pos", ([5, 10, 15, 45, 75, 90] * u.deg))
def test_filter_by_area_position(m171, pos):
    rtol = 0.01
    theta = 10 * u.deg
    hpc_coords = all_coordinates_from_map(m171)
    hgs_coords = hpc_coords.transform_to("heliographic_stonyhurst")
    ref = m171.reference_coordinate
    center = ref.transform_to("heliographic_stonyhurst").spherical_offsets_by(0 * u.deg, pos)
    radial_angle = hgs_coords.separation(center)
    data = np.zeros_like(m171.data)
    data[radial_angle <= theta] = 1
    m171.data[:, :] = data
    label_mask, regions = filter_ch(data, m171, min_area=0 * u.m**2)
    expected_area = 2 * np.pi * (1 - np.cos(theta)) * m171.rsun_meters**2
    if u.allclose(pos, 90 * u.deg):
        expected_area *= 0.5  # half behind the limb
        rtol = 0.10  # more error as area per pixel is huge
    assert_quantity_allclose(regions[0].surface_area, expected_area, rtol=rtol)
