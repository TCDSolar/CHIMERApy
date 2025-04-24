import numpy as np
import pytest
from numpy.testing import assert_allclose
from sunpy.map import Map, all_coordinates_from_map

import astropy.units as u

from chimerapy.chimera import calculate_area_map, filter_by_area, generate_candidate_mask


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
    from examples.paper_figures import mask_map

    result_mask = generate_candidate_mask(m171, m193, m211)

    expected_shape = m171.data.shape
    assert result_mask.shape == expected_shape, "Mask shape does not match expected shape."

    expected_mask = mask_map.data.astype(bool)
    np.testing.assert_allclose(result_mask, expected_mask)


def test_calculate_area_map(
    m171,
):
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
    label_mask, regions = filter_by_area(data, m171, min_area=0 * u.m**2)
    expected_area = 2 * np.pi * (1 - np.cos(theta)) * m171.rsun_meters**2
    assert_allclose(regions[0].surface_area, expected_area, rtol=0.01)


@pytest.mark.parametrize("pos", ([5, 10, 15, 45, 75, 90] * u.deg))
def test_filter_by_area_position(m171, pos):
    rtol = 0.01
    theta = 10 * u.deg
    hpc_coords = all_coordinates_from_map(m171)
    hgs_coords = hpc_coords.transform_to("heliographic_stonyhurst")
    ref = m171.reference_coordinate
    center = ref.transform_to("heliographic_stonyhurst").spherical_offsets_by(pos, 0 * u.deg)
    radial_angle = hgs_coords.separation(center)
    data = np.zeros_like(m171.data)
    data[radial_angle <= theta] = 1
    m171.data[:, :] = data
    label_mask, regions = filter_by_area(data, m171, min_area=0 * u.m**2)
    expected_area = 2 * np.pi * (1 - np.cos(theta)) * m171.rsun_meters**2
    if u.allclose(pos, 90 * u.deg):
        expected_area *= 0.5  # half behind the limb
        rtol = 0.10  # more error as area per pixel is huge
    assert_allclose(regions[0].surface_area, expected_area, rtol=rtol)
