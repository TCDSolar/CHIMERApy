import pytest
from sunpy.map import Map

from astropy.utils.data import download_file

from chimerapy.chimera import chimera
from chimerapy.chimera_original import chimera as chimera_original


@pytest.fixture()
def p171():
    return [
        download_file(
            "https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0171.fits"
        )
    ]


@pytest.fixture()
def p193():
    return [
        download_file(
            "https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0193.fits"
        )
    ]


@pytest.fixture()
def p211():
    return [
        download_file(
            "https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0211.fits"
        )
    ]


@pytest.fixture()
def pmag():
    return [
        download_file(
            "https://solarmonitor.org/data/2016/09/22/fits/shmi/shmi_maglc_fd_20160922_094640.fts.gz"
        )
    ]


@pytest.fixture()
def m171(p171):
    return Map(p171[0])


@pytest.fixture()
def m193(p193):
    return Map(p193[0])


@pytest.fixture()
def m211(p211):
    return Map(p211[0])


def test_compare(p171, p193, p211, pmag, m171, m193, m211):
    candidates, ch_maks, ch_props = chimera(m171, m193, m211)  # noqa F841
    circ, data, datb, datc, dattoarc, hedb, iarr, props, rs, slate, center, xgrid, ygrid = chimera_original(
        p171, p193, p211, pmag
    )  # noqa F841

    ch_props.pprint_all()
    print(props)

    # plt.imshow(original[6].reshape(1024, 4, 1024, 4).sum(axis=(1, 3)))
    # plt.contour(current[1])
    # plt.imshow(m193.data, vmin=50, vmax=500)
