import os
import warnings

import mahotas
import numpy as np
from astropy.io import fits
from astropy.modeling.models import Gaussian2D

from chimerapy.chimera import *
from chimerapy.chimera import (Bounds, Xeb, Xnb, Xsb, Xwb, Yeb, Ynb, Ysb, Ywb,
                               ang, arcar, arccent, area, cent, centlat,
                               centlon, chpts, cont, coords, csys, data, datb,
                               datc, datm, dist, eastl, extent, filter, hg,
                               ins_prop, mB, mBneg, mBpos, npix, pos,
                               remove_neg, rescale_aia, rescale_hmi,
                               set_contour, sort, threshold, truarcar, trummar,
                               trupixar, westl, width, xpos, ypos)

INPUT_FILES = {
    "aia171": "https://solarmonitor.org/data/2016/09/22/fits/saia/saia_00171_fd_20160922_103010.fts.gz",
    "aia193": "https://solarmonitor.org/data/2016/09/22/fits/saia/saia_00193_fd_20160922_103041.fts.gz",
    "aia211": "https://solarmonitor.org/data/2016/09/22/fits/saia/saia_00211_fd_20160922_103046.fts.gz",
    "hmi_mag": "https://solarmonitor.org/data/2016/09/22/fits/shmi/shmi_maglc_fd_20160922_094640.fts.gz",
}


def img_present():
    assert im171 != [] or im193 != [] or im211 != [] or imhmi != [], "Not all required files present"


def rest_rescale():
    global data, datb, datc, datm
    data = rescale_aia(im171, 1024, 4096)
    datb = rescale_aia(im193, 1024, 4096)
    datc = rescale_aia(im211, 1024, 4096)
    datm = rescale_hmi(imhmi, 1024, 4096)
    assert (
        len(data) == len(datb) == len(datc) == len(datm) == 4096
    ), "Array size does not match desired array size"


heda = fits.getheader(im171[0], 0)
hedb = fits.getheader(im193[0], 0)
hedc = fits.getheader(im211[0], 0)
hedm = fits.getheader(imhmi[0], 0)


def test_filter():
    initial_values = {
        'hedb["ctyple1"]': hedb["ctype1"],
        'hedb["ctype2"]': hedb["ctype2"],
        'heda["cdelt1"]': heda["cdelt1"],
        'heda["cdelt2"]': heda["cdelt2"],
        'heda["crpix1"]': heda["crpix1"],
        'heda["crpix2"]': heda["crpix2"],
        'hedb["cdelt1"]': hedb["cdelt1"],
        'hedb["cdelt2"]': hedb["cdelt2"],
        'hedb["crpix1"]': hedb["crpix1"],
        'hedb["crpix2"]': hedb["crpix2"],
        'hedc["cdelt1"]': hedc["cdelt1"],
        'hedc["cdelt2"]': hedc["cdelt2"],
        'hedc["crpix1"]': hedc["crpix1"],
        'hedc["crpix2"]': hedc["crpix2"],
        "datm": datm,
    }
    filter(im171, im193, im211, imhmi)
    final_values = {
        'hedb["ctyple1"]': hedb["ctype1"],
        'hedb["ctype2"]': hedb["ctype2"],
        'heda["cdelt1"]': heda["cdelt1"],
        'heda["cdelt2"]': heda["cdelt2"],
        'heda["crpix1"]': heda["crpix1"],
        'heda["crpix2"]': heda["crpix2"],
        'hedb["cdelt1"]': hedb["cdelt1"],
        'hedb["cdelt2"]': hedb["cdelt2"],
        'hedb["crpix1"]': hedb["crpix1"],
        'hedb["crpix2"]': hedb["crpix2"],
        'hedc["cdelt1"]': hedc["cdelt1"],
        'hedc["cdelt2"]': hedc["cdelt2"],
        'hedc["crpix1"]': hedc["crpix1"],
        'hedc["crpix2"]': hedc["crpix2"],
        "datm": datm,
    }
    if initial_values == final_values:
        raise Warning("No filtering occured - ensure filtering conditions were not met")


def test_neg():
    remove_neg(im171, im193, im211)
    for num in im171:
        assert num >= 0, "Array still contains negative number"
    for num in im193:
        assert num >= 0, "Array still contains negative number"
    for num in im211:
        assert num >= 0, "Array still contains negative number"


def tremove_neg():
    assert len(data[data < 0]) == 0, "Data contains negative values"
    assert len(datb[datb < 0]) == 0, "Data contains negative values"
    assert len(datc[datc < 0]) == 0, "Data contains negative values"


def s_rs():
    global s, rs
    s = np.shape(data)
    rs = heda["rsun"]
    assert s == (4096, 4096), "Incorrect data shape"
    if rs < 970 or rs > 975:
        warnings.warn("Solar radius may be inconsistant with accepted value (~973)")


ident = 1
iarr = np.zeros((s[0], s[1]), dtype=np.byte)
bmcool = np.zeros((s[0], s[1]), dtype=np.float32)
offarr, slate = np.array(iarr), np.array(iarr)
cand, bmmix, bmhot = np.array(bmcool), np.array(bmcool), np.array(bmcool)
circ = np.zeros((s[0], s[1]), dtype=int)

r = (s[1] / 2.0) - 450
xgrid, ygrid = np.meshgrid(np.arange(s[0]), np.arange(s[1]))
center = [int(s[1] / 2.0), int(s[1] / 2.0)]
w = np.where((xgrid - center[0]) ** 2 + (ygrid - center[1]) ** 2 > r**2)
y, x = np.mgrid[0:4096, 0:4096]
garr = Gaussian2D(1, s[0] / 2, s[1] / 2, 2000 / 2.3548, 2000 / 2.3548)(x, y)
garr[w] = 1.0

props = np.zeros((26, 30), dtype="<U16")
props[:, 0] = (
    "ID",
    "XCEN",
    "YCEN",
    "CENTROID",
    "X_EB",
    "Y_EB",
    "X_WB",
    "Y_WB",
    "X_NB",
    "Y_NB",
    "X_SB",
    "Y_SB",
    "WIDTH",
    "WIDTH°",
    "AREA",
    "AREA%",
    "<B>",
    "<B+>",
    "<B->",
    "BMAX",
    "BMIN",
    "TOT_B+",
    "TOT_B-",
    "<PHI>",
    "<PHI+>",
    "<PHI->",
)
props[:, 1] = (
    "num",
    '"',
    '"',
    "H°",
    '"',
    '"',
    '"',
    '"',
    '"',
    '"',
    '"',
    '"',
    "H°",
    "°",
    "Mm^2",
    "%",
    "G",
    "G",
    "G",
    "G",
    "G",
    "G",
    "G",
    "Mx",
    "Mx",
    "Mx",
)

assert len(props) == 26, "Incorrect property array size"


def test_bounds():
    t0b = Bounds(0.8, 2.7, 255)
    t1b = Bounds(1.4, 3.0, 255)
    t2b = Bounds(1.2, 3.9, 255)
    assert t0b.upper > t0b.lower, "Upper bound must be greater than lower bound"
    assert t1b.upper > t0b.lower, "Upper bound must be greater than lower bound"
    assert t2b.upper > t0b.lower, "Upper bound must be greater than lower bound"


with np.errstate(divide="ignore"):
    t0 = np.log10(datc)
    t1 = np.log10(datb)
    t2 = np.log10(data)


def test_threshold():
    init = {"t0": t0, "t1": t1, "t2": t2}
    assert init["t0"] != threshold(t0), "Data was not bounded by threshold values"
    assert init["t1"] != threshold(t0), "Data was not bounded by threshold values"
    assert init["t2"] != threshold(t0), "Data was not bounded by threshold values"


t0b = Bounds(0.8, 2.7, 255)
t1b = Bounds(1.4, 3.0, 255)
t2b = Bounds(1.2, 3.9, 255)


def test_contour():
    init = {"t0": t0, "t1": t1, "t2": t2}
    assert init["t0"] != set_contour(t0), "Data was not bounded by threshold values"
    assert init["t1"] != set_contour(t0), "Data was not bounded by threshold values"
    assert init["t2"] != set_contour(t0), "Data was not bounded by threshold values"


def has_dupl(arr):
    seen = set()
    for num in arr:
        if num in seen:
            return True
        seen.add(num)
    return False


def test_dupl():
    global sizes, reord, tmp, cont
    sort()
    assert not has_dupl(reord), "Sorted list should contain no duplicates"
    assert not has_dupl(tmp), "Sorted list should contain no duplicates"
    assert not has_dupl(cont), "Sorted list should contain no duplicates"


for i in range(len(cont)):
    x = np.append(x, len(cont[i]))


def test_extent():
    i_maxxlat = None
    i_maxxlon = None
    i_maxylat = None
    i_maxylon = None
    i_minxlon = None
    i_minylat = None
    i_minylon = None
    i_minxlat = None
    extent(i, ypos, xpos, hg, cont)
    global maxxlat, maxxlon, maxylat, maxylon, minxlon, minylat, minylon, minxlat
    assert i_maxxlat != maxxlat, "maxxlat not created successfully"
    assert i_maxxlon != maxxlon, "maxxlon not created successfully"
    assert i_maxylat != maxylat, "maxylat not created successfully"
    assert i_maxylon != maxylon, "maxylon not created successfully"
    assert i_minxlon != minxlon, "minxlon not created successfully"
    assert i_minylat != minylat, "minylat not created successfully"
    assert i_minylon != minylon, "minylon not created successfully"
    assert i_minxlat != minxlat, "minxlat not created successfully"


def test_coords():
    i_Ywb = None
    i_Xwb = None
    i_Yeb = None
    i_Xeb = None
    i_Ynb = None
    i_Xnb = None
    i_Ysb = None
    i_Xsb = None
    coords(i, csys, cont)
    assert i_Ywb != Ywb, "Ywb not created successfully"
    assert i_Xwb != Xwb, "Xwb not created successfully"
    assert i_Yeb != Yeb, "Yeb not created successfully"
    assert i_Xeb != Xeb, "Xeb not created successfully"
    assert i_Ynb != Ynb, "Ynb not created successfully"
    assert i_Xnb != Xnb, "Xnb not created successfully"
    assert i_Ysb != Ysb, "Ysb not created successfully"
    assert i_Xsb != Xsb, "Xsb not created successfully"


def test_props():
    ins_prop(
        datm,
        rs,
        ident,
        props,
        arcar,
        arccent,
        pos,
        npix,
        trummar,
        centlat,
        centlon,
        mB,
        mBpos,
        mBneg,
        Ywb,
        Xwb,
        Yeb,
        Xeb,
        Ynb,
        Xnb,
        Ysb,
        Xsb,
        width,
        eastl,
        westl,
    )
    assert np.any(props is None) is not False, "Property array should not contain empty entries"
    assert np.all(props is None) is not False, "Property array should not be empty"


def test_loop_variables():
    for i in range(len(cont)):
        assert len(cont[i]) > 100, "Contour length should be greater than 100"


def test_fill_polygon():
    cand = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])

    iarr = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])

    cont = np.array([[[[1, 2], [1, 3], [2, 3], [2, 2]]]])

    slate = np.array(iarr)

    polygon_vertices = np.array(list(zip(cont[0][:, 0, 1], cont[0][:, 0, 0])))

    if (
        cand[
            np.max(cont[0][:, 0, 0]) + 1,
            cont[0][np.where(cont[0][:, 0, 0] == np.max(cont[0][:, 0, 0]))[0][0], 0, 1],
        ]
        > 0
    ) and (
        iarr[
            np.max(cont[0][:, 0, 0]) + 1,
            cont[0][np.where(cont[0][:, 0, 0] == np.max(cont[0][:, 0, 0]))[0][0], 0, 1],
        ]
        > 0
    ):
        mahotas.polygon.fill_polygon(polygon_vertices, slate)

        print("After filling polygon:")
        print(slate)
        iarr[np.where(slate == 1)] = 0
        slate[:] = 0

    assert np.array_equal(
        slate,
        np.array([[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
    ), "Test failed: slate array does not match expected result"

    assert np.array_equal(
        iarr,
        np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
    ), "Test failed: iarr array does not match expected result"


def test_limb():
    arccent = csys.all_pix2world(cent[0], cent[1], 0)
    if (((arccent[0] ** 2) + (arccent[1] ** 2)) > (rs**2)) or (
        np.sum(np.array(csys.all_pix2world(cont[i][0, 0, 0], cont[i][0, 0, 1], 0)) ** 2) > (rs**2)
    ):
        mahotas.polygon.fill_polygon(np.array(list(zip(cont[i][:, 0, 1], cont[i][:, 0, 0]))), offarr)
        assert np.sum(offarr) > 0, "Offarr was not modified successfully"
    else:
        mahotas.polygon.fill_polygon(np.array(list(zip(cont[i][:, 0, 1], cont[i][:, 0, 0]))), slate)
        slate[:] = 0
        assert np.sum(slate) > 0, "Slate was not modified successfully"


def test_magpol():
    assert npix[0][np.where(npix[0] == 0)] != 0, "Npix[0] should not be equal to zero at its zeros"
    assert npix[1] != 0, "Npix[1] should not be equal to 0"
    npixtest = [
        np.array([2, -1, 0, 3, -2, 1]),
        np.array([1, -1, 0, 2, -2, 1]),
    ]
    wh1_expected = np.where(npixtest[1] > 0)
    wh1_actual = np.where(npixtest[1] > 0)

    wh2_expected = np.where(npixtest[1] < 0)
    wh2_actual = np.where(npixtest[1] < 0)

    assert np.array_equal(
        wh1_actual, wh1_expected
    ), f"Test failed for wh1: Expected {wh1_expected}, got {wh1_actual}"
    assert np.array_equal(
        wh2_actual, wh2_expected
    ), f"Test failed for wh1: Expected {wh2_expected}, got {wh2_actual}"


assert area is not None, "Area variable not created successfully"
assert arcar is not None, "Arcar variable not created successfully"
assert chpts is not None, "Chpts variable not created successfully"
assert cent is not None, "Cent variable not created successfully"
assert arccent is not None, "Arccent variable not created successfully"
assert ypos is not None, "Ypos variable not created successfully"
assert xpos is not None, "Xpos variable not created suffessfully"
assert dist is not None, "Dist variable not created successfully"
assert ang is not None, "Ang variable not created successfully"
assert trupixar is not None, "Trupixar variable not created successfully"
assert truarcar is not None, "Truarcar variable not created successfully"
assert trummar is not None, "Trummar variable not created successfully"
assert mB is not None, "mB variable not created successfully"
assert mBpos is not None, "mBpos variable not created successfully"
assert mBneg is not None, "mBneg variable not created successfully"
assert width is not None, "width variable not created successfully"
assert eastl is not None, "eastl variable not created successfully"
assert westl is not None, "westl variable not created successfully"
assert centlat is not None, "centlat variable not created successfully"
assert centlon is not None, "centlon variable not created successfully"

assert os.path.exists("ch_summary.txt"), "Summary file not saved correctly"
assert os.path.exists("tricolor.png"), "Tricolor image not saved correctly"
assert os.path.exists("CH_mask_" + hedb["DATE"] + ".png")
