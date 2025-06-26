import os
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from parfive import Downloader
from sunpy.map import Map, all_coordinates_from_map

import astropy.units as u
from astropy.io import fits
from astropy.tests.helper import assert_quantity_allclose

from chimerapy.chimera_original import chimera, chimera_legacy

INPUT_FILES = {
    "aia171": "https://solarmonitor.org/data/2016/09/22/fits/saia/saia_00171_fd_20160922_103010.fts.gz",
    "aia193": "https://solarmonitor.org/data/2016/09/22/fits/saia/saia_00193_fd_20160922_103041.fts.gz",
    "aia211": "https://solarmonitor.org/data/2016/09/22/fits/saia/saia_00211_fd_20160922_103046.fts.gz",
    "hmi_mag": "https://solarmonitor.org/data/2016/09/22/fits/shmi/shmi_maglc_fd_20160922_094640.fts.gz",
}


def map_fix_cunit(url):
    hdul = fits.open(url)
    hdul[0].header["CUNIT1"] = "arcsec"
    hdul[0].header["CUNIT2"] = "arcsec"
    return Map((hdul[0].data, hdul[0].header))


@pytest.fixture()
def p171():
    return Downloader().simple_download(
        ["https://solarmonitor.org/data/2016/09/22/fits/saia/saia_00171_fd_20160922_103010.fts.gz"],
        path=Path.home() / "sunpy" / "data",
    )


@pytest.fixture()
def p193():
    return Downloader().simple_download(
        ["https://solarmonitor.org/data/2016/09/22/fits/saia/saia_00193_fd_20160922_103041.fts.gz"],
        path=Path.home() / "sunpy" / "data",
    )


@pytest.fixture()
def p211():
    return Downloader().simple_download(
        ["https://solarmonitor.org/data/2016/09/22/fits/saia/saia_00211_fd_20160922_103046.fts.gz"],
        path=Path.home() / "sunpy" / "data",
    )


@pytest.fixture()
def pmag():
    return Downloader().simple_download(
        ["https://solarmonitor.org/data/2016/09/22/fits/shmi/shmi_maglc_fd_20160922_094640.fts.gz"],
        path=Path.home() / "sunpy" / "data",
    )


@pytest.fixture()
def m171(p171):
    return map_fix_cunit(p171[0])


def test_chimera(tmp_path, p171, p193, p211, pmag):
    os.chdir(tmp_path)
    chimera_legacy(p171, p193, p211, pmag)

    test_summary_file = Path(__file__).parent / "test_ch_summary.txt"
    with test_summary_file.open("r") as f:
        test_summary_text = f.read()

    ch_summary_file = tmp_path / "ch_summary.txt"
    with ch_summary_file.open("r") as f:
        ch_summary_text = f.read()

    assert ch_summary_text == test_summary_text


@pytest.mark.parametrize(
    ("pos", "rtol"),
    (((0, 0), 0.05), ((0, 15), 0.05), ((0, 30), 0.05), ((0, 45), 0.05), ((0, 60), 0.1), ((0, 75), 0.35)),
    ids=lambda x: str(x),
)
@patch("chimerapy.chimera_original.cv2")
def test_chimera_orig_area(mock_contours, tmp_path, p171, p193, p211, pmag, m171, pos, rtol):
    dummy_map = m171.resample([4096, 4096] * u.pix)

    theta = 15 * u.deg
    hpc_coords = all_coordinates_from_map(dummy_map)
    hgs_coords = hpc_coords.transform_to("heliographic_stonyhurst")
    ref = dummy_map.reference_coordinate
    center = ref.transform_to("heliographic_stonyhurst").spherical_offsets_by(pos[0] * u.deg, pos[1] * u.deg)
    radial_angle = hgs_coords.separation(center)
    data = np.zeros_like(dummy_map.data)
    data[radial_angle <= theta] = 1

    # Otherwise old code doesn't work because of nans due to off disk pixels
    circ = np.zeros_like(data)
    sep = hpc_coords.separation(dummy_map.reference_coordinate)
    circ[sep <= dummy_map.rsun_obs - 10 * u.pix * dummy_map.scale[0]] = 1
    circ[sep >= dummy_map.rsun_obs + 40 * u.pix * dummy_map.scale[0]] = 1
    data = data * circ

    out = cv2.findContours(data.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mock_contours.findContours.return_value = (data, *out)
    res = chimera(p171, p193, p211, pmag)
    ch_area = float(res[7][14][2]) * u.Mm**2

    cone_area = 2 * np.pi * dummy_map.rsun_meters**2 * (1 - np.cos(theta))

    assert_quantity_allclose(ch_area, cone_area, rtol=rtol)
