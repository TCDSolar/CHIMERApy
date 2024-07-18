"""Package for Coronal Hole Identification Algorithm"""

import astropy.units as u
import cv2
import mahotas
import matplotlib.pyplot as plt
import numpy as np
import sunpy
import sunpy.map
from astropy import wcs
from astropy.modeling.models import Gaussian2D
from astropy.visualization import astropy_mpl_style
from skimage.util import img_as_ubyte
from sunpy.coordinates import (HeliographicStonyhurst,
                               propagate_with_solar_surface)
from sunpy.map import Map, all_coordinates_from_map

INPUT_FILES = {
    "aia171": "http://jsoc.stanford.edu/data/aia/synoptic/2016/09/22/H1000/AIA20160922_1030_0171.fits",
    "aia193": "http://jsoc.stanford.edu/data/aia/synoptic/2016/09/22/H1000/AIA20160922_1030_0193.fits",
    "aia211": "http://jsoc.stanford.edu/data/aia/synoptic/2016/09/22/H1000/AIA20160922_1030_0211.fits",
    "hmi_mag": "http://jsoc.stanford.edu/data/hmi/fits/2016/09/22/hmi.M_720s.20160922_010000_TAI.fits",
}

im171 = Map(INPUT_FILES["aia171"])
im193 = Map(INPUT_FILES["aia193"])
im211 = Map(INPUT_FILES["aia211"])
imhmi = Map(INPUT_FILES["hmi_mag"])


def reproject_diff_rot(target_wcs: wcs.wcs.WCS, input_map: sunpy.map.Map):
    """
    Rescale the input aia image dimensions.

    Parameters
    ----------
    proj_to: 'sunpy.map.Map'
    input_map: 'sunpy.map.Map

    Returns
    -------
    array: 'np.array'

    """
    with propagate_with_solar_surface():
        amap = input_map.reproject_to(target_wcs.wcs)
        new_x_scale = amap.scale[0].to(u.arcsec / u.pixel).value
        new_y_scale = amap.scale[1].to(u.arcsec / u.pixel).value
        amap.meta["cdelt1"] = new_x_scale
        amap.meta["cdelt2"] = new_y_scale
        amap.meta["cunit1"] = "arcsec"
        amap.meta["cunit2"] = "arcsec"
        return amap


im193 = reproject_diff_rot(im171, im193)
im211 = reproject_diff_rot(im171, im211)
imhmi = reproject_diff_rot(im171, imhmi)


def filter(map1: np.array, map2: np.array, map3: np.array):
    """
    Removes negative values from each map by setting each equal to zero

    Parameters
    ----------
    map1: 'sunpy.map.Map'
    map2: 'sunpy.map.Map'
    map3: 'sunpy.map.Map'

    Returns
    -------
    map1: 'sunpy.map.Map'
    map2: 'sunpy.map.Map'
    map3: 'sunpy.map.Map'

    """
    map1.data[np.where(map1.data <= 0)] = 0
    map2.data[np.where(map2.data <= 0)] = 0
    map3.data[np.where(map3.data <= 0)] = 0

    return map1, map2, map3


im171, im193, im211 = filter(im171, im193, im211)

def shape(map1: sunpy.map.Map, map2: sunpy.map.Map, map3: sunpy.map.Map):
    """
    defines the shape of the arrays as "s" and "rs" as the solar radius
    
    Parameters
    ----------
    map1: 'sunpy.map.Map'
    map2: 'sunpy.map.Map'
    map3: 'sunpy.map.Map'

    Returns
    -------
    s: 'tuple'
    rs: 'astropy.units.quantity.Quantity'
    rs_pixels: 'astropy.units.quantity.Quantity'
    
    """
    
    im171, im193, im211 = filter(map1, map2, map3)
    # defines the shape of the arrays as "s" and "rs" as the solar radius
    s = np.shape(im171.data)
    rs = im171.rsun_obs
    print(rs)
    rs_pixels = im171.rsun_obs / im171.scale[0]
    return s, rs, rs_pixels

s, rs, rs_pixels = shape(im171, im193, im211)


def pix_arc(amap: sunpy.map.Map):
    """
    Defines conversion values between pixels and arcsec

    Parameters
    ----------
    amap: 'sunpy.map.Map'

    Returns

    """
    dattoarc = amap.scale[0].value
    s = amap.dimensions
    conver = (s.x/2) * amap.scale[0].value / amap.meta['cdelt1'], (s.y / 2)
    convermul = dattoarc / amap.meta["cdelt1"]
    return dattoarc, conver, convermul


dattoarc, conver, convermul = pix_arc(im171)

print(conver)


def to_helio(amap: sunpy.map.Map):
    """
    Converts maps to the Heliographic Stonyhurst coordinate system

    Parameters
    ----------
    amap: 'sunpy.map.Map'

    Returns
    -------
    hpc: 'astropy.coordinates.sky_coordinate.SkyCoord'
    hg: 'astropy.coordinates.sky_coordinate.SkyCoord'
    csys: 'astropy.wcs.wcs.WCS'


    """
    hpc = all_coordinates_from_map(amap)
    hg = hpc.transform_to(sunpy.coordinates.frames.HeliographicStonyhurst)
    csys = wcs.WCS(dict(amap.meta))
    return hpc, hg, csys


hpc, hg, csys = to_helio(im171)

"""Setting up arrays to be used in later processing"""
ident = 1
iarr = np.zeros((s[0], s[1]), dtype=np.byte)
bmcool = np.zeros((s[0], s[1]), dtype=np.float32)
offarr, slate = np.array(iarr), np.array(iarr)
cand, bmmix, bmhot = np.array(bmcool), np.array(bmcool), np.array(bmcool)
circ = np.zeros((s[0], s[1]), dtype=int)

"""creation of a 2d gaussian for magnetic cut offs"""
r = (s[1] / 2.0) - 450
xgrid, ygrid = np.meshgrid(np.arange(s[0]), np.arange(s[1]))
center = [int(s[1] / 2.0), int(s[1] / 2.0)]
w = np.where((xgrid - center[0]) ** 2 + (ygrid - center[1]) ** 2 > r**2)
y, x = np.mgrid[0:1024, 0:1024]
width = (2000 * u.arcsec)
garr = Gaussian2D(
    1,
    im171.reference_pixel.x.to_value(u.pix),
    im171.reference_pixel.y.to_value(u.pix),
    width / im171.scale[0],
    width / im171.scale[1],
)(x, y)
garr[w] = 1.0

"""creates sub-arrays of props to isolate column of index 0 and column of index 1"""
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

"""define threshold values in log space"""


def log_dat(map1: sunpy.map.Map, map2: sunpy.map.Map, map3: sunpy.map.Map):
    """
    Takes the log base-10 of all sunpy map data

    Parameters
    ----------
    map1: 'sunpy.map.Map'
    map2: 'sunpy.map.Map'
    map3: 'sunpy.map.Map'

    Returns
    -------
    t0: 'np.array'
    t1: 'np.array'
    t2: 'np.array'
    """
    with np.errstate(divide="ignore"):
        t0 = np.log10(map1.data)
        t1 = np.log10(map2.data)
        t2 = np.log10(map3.data)
        return t0, t1, t2


t0, t1, t2 = log_dat(im171, im193, im211)


class Bounds:
    """Class to change and define array boundaries and slopes"""

    def __init__(self, upper, lower, slope):
        self.upper = upper
        self.lower = lower
        self.slope = slope

    def new_u(self, new_upper):
        self.upper = new_upper

    def new_l(self, new_lower):
        self.lower = new_lower

    def new_s(self, new_slope):
        self.slope = new_slope


t0b = Bounds(0.8, 2.7, 255)
t1b = Bounds(1.4, 3.0, 255)
t2b = Bounds(1.2, 3.9, 255)


# set to also take in boundaries
def set_contour(t0: np.array, t1: np.array, t2: np.array):
    """
    Threshold arrays based on desired boundaries and sets contours.

    Parameters
    ----------
    t0: 'np.array'
    t1: 'np.array'
    t2: 'np.array''

    Returns
    -------
    t0: 'np.array'
    t1: 'np.array'
    t2: 'np.array'

    """
    if t0 is not None and t1 is not None and t2 is not None:
        # set the threshold and contours for t0
        t0[np.where(t0 < t0b.upper)] = t0b.upper
        t0[np.where(t0 > t0b.lower)] = t0b.lower
        t0 = np.array(((t0 - t0b.upper) / (t0b.lower - t0b.upper)) * t0b.slope, dtype=np.float32)
        # set the threshold and contours for t1
        t1[np.where(t1 < t1b.upper)] = t1b.upper
        t1[np.where(t1 > t1b.lower)] = t2b.lower
        t1 = np.array(((t1 - t1b.upper) / (t1b.lower - t1b.upper)) * t1b.slope, dtype=np.float32)
        # set the threshold and contours for t2
        t2[np.where(t2 < t2b.upper)] = t2b.upper
        t2[np.where(t2 > t2b.lower)] = t2b.lower
        t2 = np.array(((t2 - t2b.upper) / (t2b.lower - t2b.upper)) * t2b.slope, dtype=np.float32)
    else:
        print("Must input valid logarithmic arrays")
    return t0, t1, t2


t0, t1, t2 = set_contour(t0, t1, t2)


def create_mask(
    tm1: np.array, tm2: np.array, tm3: np.array, map1: sunpy.map.Map, map2: sunpy.map.Map, map3: sunpy.map.Map
):
    """
    Creates 3 segmented bitmasks

    Parameters
    -------
    tm1: 'np.array'
    tm2: 'np.array'
    tm3: 'np.array'
    map1: 'sunpy.map.Map'
    map2: 'sunpy.map.Map'
    map3: 'sunpy.map.Map'

    Returns
    -------
    bmmix: 'np.array'
    bmhot: 'np.array'
    bmcool: 'np.array'

    """
    with np.errstate(divide="ignore", invalid="ignore"):
        bmmix[np.where(tm3 / tm1 >= ((np.mean(map1.data) * 0.6357) / (np.mean(map3.data))))] = 1
        bmhot[np.where(tm1 + tm2 < (0.7 * (np.mean(map2.data) + np.mean(map3.data))))] = 1
        bmcool[np.where(tm3 / tm2 >= ((np.mean(map2.data) * 1.5102) / (np.mean(map2.data))))] = 1
    return bmmix, bmhot, bmcool


bmmix, bmhot, bmcool = create_mask(t0, t1, t2, im171, im193, im211)

# conjunction of 3 bitmasks
cand = bmcool * bmmix * bmhot


def misid(can: np.array, cir: np.array, xgir: np.array, ygir: np.array, thresh_rad: int):
    """
    Removes off-detector mis-identification

    Parameters
    ----------
    can: 'np.array'
    cir: 'np.array'
    xgir: 'np.array'
    ygir: 'np.array'

    Returns
    -------
    'np.array'

    """
    # make r a function argument, give name and unit
    r = thresh_rad
    w = np.where((xgir - center[0]) ** 2 + (ygir - center[1]) ** 2 <= thresh_rad**2)
    cir[w] = 1.0
    cand = can * cir
    return r, w, cir, cand


r, w, cir, cand = misid(cand, circ, xgrid, ygrid, (s[1] / 2.0) - 100)


def on_off(cir: np.array, can: np.array):
    """
    Seperates on-disk and off-limb coronal holes

    Parameters
    ----------
    cir: 'np.array'
    can: 'np.array'

    Returns
    -------
    'np.array'

    """
    cir[:] = 0
    r = (rs.value / dattoarc) - 10
    inside = np.where((xgrid - center[0]) ** 2 + (ygrid - center[1]) ** 2 <= r**2)
    cir[inside] = 1.0
    r = (rs.value / dattoarc) + 40
    outside = np.where((xgrid - center[0]) ** 2 + (ygrid - center[1]) ** 2 >= r**2)
    cir[outside] = 1.0
    can = can * cir
    return can


cand = on_off(circ, cand)


def contour_data(cand: np.array):
    """
    Contours the identified datapoints

    Parameters
    ----------
    cand: 'np.array'

    Returns
    -------
    cand: 'np.array'
    cont: 'tuple'
    heir: 'np.array'

    """
    cand = np.array(cand, dtype=np.uint8)
    cont, heir = cv2.findContours(cand, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cand, cont, heir


cand, cont, heir = contour_data(cand)


def sort(cont: tuple):
    """
    Sorts the contours by size

    Parameters
    ----------
    cont: 'tuple'

    Returns
    -------
    reord: 'list'
    tmp: 'list'
    cont: 'list'
    sizes: 'list'

    """
    sizes = []
    for i in range(len(cont)):
        sizes = np.append(sizes, len(cont[i]))
    reord = sizes.ravel().argsort()[::-1]
    tmp = list(cont)
    for i in range(len(cont)):
        tmp[i] = cont[reord[i]]
    cont = list(tmp)
    return cont, sizes, reord, tmp


cont, sizes, reord, tmp = sort(cont)


# =====cycles through contours=========


def extent(amap: sunpy.map.Map, cont: tuple, xpos: int, ypos: int):
    """
    Finds coronal hole extent in latitude and longitude

    Parameters
    ----------
    amap: 'sunpy.map.Map'
    cont: 'tuple'
    xpos: 'int'
    ypos: 'int'

    Returns
    -------
    maxxlon: 'astropy.coordinates.angles.core.Longitude'
    minxlon: 'astropy.coordinates.angles.core.Longitude'
    centlat: 'astropy.coordinates.angles.core.Latitude'
    centlon: 'astropy.coordinates.angles.core.Longitude'

    """

    coord_hpc = amap.world2pix(cont)
    maxlat = coord_hpc.transform_to(HeliographicStonyhurst).lat.max()
    maxlon = coord_hpc.transform_to(HeliographicStonyhurst).lon.max()
    minlat = coord_hpc.transform_to(HeliographicStonyhurst).lat.min()
    minlon = coord_hpc.transform_to(HeliographicStonyhurst).lat.min()

    # =====CH centroid in lat/lon=======

    centlat = hg.lat[int(ypos), int(xpos)]
    centlon = hg.lon[int(ypos), int(xpos)]
    return maxlat, maxlon, minlat, minlon, centlat, centlon


def coords(i, csys, cont):
    """
    Finds coordinates of CH boundaries in world coordinates

    Parameters
    ----------
    i: 'int'
    csys: 'astropy.wcs.wcs.WCS'
    cont: 'list'

    Returns
    -------
    Ywb: 'np.array'
    Xwb: 'np.array'
    Yeb: 'np.array'
    Xeb: 'np.array'
    Ynb: 'np.array'
    Xnb: 'np.array'
    Ysb: 'np.array'
    Xsb: 'np.array'
    """
    global Ywb, Xwb, Yeb, Xeb, Ynb, Xnb, Ysb, Xsb
    Ywb, Xwb = csys.all_pix2world(
        cont[i][np.where(cont[i][:, 0, 0] == np.max(cont[i][:, 0, 0]))[0][0], 0, 1],
        np.max(cont[i][:, 0, 0]),
        0,
    )
    Yeb, Xeb = csys.all_pix2world(
        cont[i][np.where(cont[i][:, 0, 0] == np.min(cont[i][:, 0, 0]))[0][0], 0, 1],
        np.min(cont[i][:, 0, 0]),
        0,
    )
    Ynb, Xnb = csys.all_pix2world(
        np.max(cont[i][:, 0, 1]),
        cont[i][np.where(cont[i][:, 0, 1] == np.max(cont[i][:, 0, 1]))[0][0], 0, 0],
        0,
    )
    Ysb, Xsb = csys.all_pix2world(
        np.min(cont[i][:, 0, 1]),
        cont[i][np.where(cont[i][:, 0, 1] == np.min(cont[i][:, 0, 1]))[0][0], 0, 0],
        0,
    )

    return Ywb, Xwb, Yeb, Xeb, Ynb, Xnb, Ysb, Xsb


def ins_prop(
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
):
    """
    Insertion of CH properties into property array

    Parameters
    ----------
    datm: 'np.array'
    rs: 'float'
    ident: 'int'
    props: 'np.array'
    arcar: 'np.float64'
    arccent: 'list'
    pos: 'np.array'
    npix: 'list'
    trummar: 'np.float64'
    centlat: 'str'
    centlon: 'str'
    mB: 'np.float64'
    mBpos: 'np.float64'
    mBneg: 'np.float64'

    Returns
    -------
    props[0, ident + 1]: 'str'
    props[1, ident + 1]: 'str'
    props[2, ident + 1]: 'str'
    props[3, ident + 1]: 'str'
    props[4, ident + 1]: 'str'
    props[5, ident + 1]: 'str'
    props[6, ident + 1]: 'str'
    props[7, ident + 1]: 'str'
    props[8, ident + 1]: 'str'
    props[9, ident + 1]: 'str'
    props[10, ident + 1]: 'str'
    props[11, ident + 1]: 'str'
    props[12, ident + 1]: 'str'
    props[13, ident + 1]: 'str'
    props[14, ident + 1]: 'str'
    props[15, ident + 1]: 'str'
    props[16, ident + 1]: 'str'
    props[17, ident + 1]: 'str'
    props[18, ident + 1]: 'str'
    props[19, ident + 1]: 'str'
    props[20, ident + 1]: 'str'
    tbpos: 'np.float64'
    props[21, ident + 1]: 'str'
    tbneg: 'np.float64'
    props[22, ident + 1]: 'str'
    props[23, ident + 1]: 'str'
    props[24, ident + 1]: 'str'
    props[25, ident + 1]: 'str'

    """
    props[0, ident + 1] = str(ident)
    props[1, ident + 1] = str(np.round(arccent[0]))
    props[2, ident + 1] = str(np.round(arccent[1]))
    props[3, ident + 1] = str(centlon + centlat)
    props[4, ident + 1] = str(np.round(Xeb))
    props[5, ident + 1] = str(np.round(Yeb))
    props[6, ident + 1] = str(np.round(Xwb))
    props[7, ident + 1] = str(np.round(Ywb))
    props[8, ident + 1] = str(np.round(Xnb))
    props[9, ident + 1] = str(np.round(Ynb))
    props[10, ident + 1] = str(np.round(Xsb))
    props[11, ident + 1] = str(np.round(Ysb))
    props[12, ident + 1] = str(eastl + "-" + westl)
    props[13, ident + 1] = str(width)
    props[14, ident + 1] = f"{trummar/1e+12:.1e}"
    props[15, ident + 1] = str(np.round((arcar * 100 / (np.pi * (rs**2))), 1))
    props[16, ident + 1] = str(np.round(mB, 1))
    props[17, ident + 1] = str(np.round(mBpos, 1))
    props[18, ident + 1] = str(np.round(mBneg, 1))
    props[19, ident + 1] = str(np.round(np.max(npix[1]), 1))
    props[20, ident + 1] = str(np.round(np.min(npix[1]), 1))
    tbpos = np.sum(datm[pos[:, 0], pos[:, 1]][np.where(datm[pos[:, 0], pos[:, 1]] > 0)])
    props[21, ident + 1] = f"{tbpos:.1e}"
    tbneg = np.sum(datm[pos[:, 0], pos[:, 1]][np.where(datm[pos[:, 0], pos[:, 1]] < 0)])
    props[22, ident + 1] = f"{tbneg:.1e}"
    props[23, ident + 1] = f"{mB*trummar*1e+16:.1e}"
    props[24, ident + 1] = f"{mBpos*trummar*1e+16:.1e}"
    props[25, ident + 1] = f"{mBneg*trummar*1e+16:.1e}"


"""Cycles through contours"""

for i in range(len(cont)):
    x = np.append(x, len(cont[i]))

    """only takes values of minimum surface length and calculates area"""

    if len(cont[i]) <= 100:
        continue
    area = 0.5 * np.abs(
        np.dot(cont[i][:, 0, 0], np.roll(cont[i][:, 0, 1], 1))
        - np.dot(cont[i][:, 0, 1], np.roll(cont[i][:, 0, 0], 1))
    )
    arcar = area * (dattoarc**2)
    if arcar > 1000:
        """finds centroid"""

        chpts = len(cont[i])
        cent = [np.mean(cont[i][:, 0, 0]), np.mean(cont[i][:, 0, 1])]

        """remove quiet sun regions encompassed by coronal holes"""
        if (
            cand[
                np.max(cont[i][:, 0, 0]) + 1,
                cont[i][np.where(cont[i][:, 0, 0] == np.max(cont[i][:, 0, 0]))[0][0], 0, 1],
            ]
            > 0
        ) and (
            iarr[
                np.max(cont[i][:, 0, 0]) + 1,
                cont[i][np.where(cont[i][:, 0, 0] == np.max(cont[i][:, 0, 0]))[0][0], 0, 1],
            ]
            > 0
        ):
            mahotas.polygon.fill_polygon(np.array(list(zip(cont[i][:, 0, 1], cont[i][:, 0, 0]))), slate)
            print(slate)
            iarr[np.where(slate == 1)] = 0
            slate[:] = 0

        else:
            """Create a simple centre point if coronal hole region is not quiet"""

            arccent = csys.all_pix2world(cent[0], cent[1], 0)

            """classifies off limb CH regions"""

            if (((arccent[0] ** 2) + (arccent[1] ** 2)) > (rs**2)) or (
                np.sum(np.array(csys.all_pix2world(cont[i][0, 0, 0], cont[i][0, 0, 1], 0)) ** 2) > (rs**2)
            ):
                mahotas.polygon.fill_polygon(np.array(list(zip(cont[i][:, 0, 1], cont[i][:, 0, 0]))), offarr)
            else:
                """classifies on disk coronal holes"""

                mahotas.polygon.fill_polygon(np.array(list(zip(cont[i][:, 0, 1], cont[i][:, 0, 0]))), slate)
                poslin = np.where(slate == 1)
                slate[:] = 0
                print(poslin)

                """create an array for magnetic polarity"""

                pos = np.zeros((len(poslin[0]), 2), dtype=np.uint)
                pos[:, 0] = np.array((poslin[0] - (s[0] / 2)) * convermul + (s[1] / 2), dtype=np.uint)
                pos[:, 1] = np.array((poslin[1] - (s[0] / 2)) * convermul + (s[1] / 2), dtype=np.uint)
                npix = list(
                    np.histogram(
                        datm[pos[:, 0], pos[:, 1]],
                        bins=np.arange(
                            np.round(np.min(datm[pos[:, 0], pos[:, 1]])) - 0.5,
                            np.round(np.max(datm[pos[:, 0], pos[:, 1]])) + 0.6,
                            1,
                        ),
                    )
                )
                npix[0][np.where(npix[0] == 0)] = 1
                npix[1] = npix[1][:-1] + 0.5

                wh1 = np.where(npix[1] > 0)
                wh2 = np.where(npix[1] < 0)

                """Filters magnetic cutoff values by area"""

                if (
                    np.absolute((np.sum(npix[0][wh1]) - np.sum(npix[0][wh2])) / np.sqrt(np.sum(npix[0])))
                    <= 10
                    and arcar < 9000
                ):
                    continue
                if (
                    np.absolute(np.mean(datm[pos[:, 0], pos[:, 1]])) < garr[int(cent[0]), int(cent[1])]
                    and arcar < 40000
                ):
                    continue
                iarr[poslin] = ident

                """create an accurate center point"""

                ypos = np.sum((poslin[0]) * np.absolute(hg.lat[poslin])) / np.sum(np.absolute(hg.lat[poslin]))
                xpos = np.sum((poslin[1]) * np.absolute(hg.lon[poslin])) / np.sum(np.absolute(hg.lon[poslin]))

                arccent = csys.all_pix2world(xpos, ypos, 0)

                """calculate average angle coronal hole is subjected to"""

                dist = np.sqrt((arccent[0] ** 2) + (arccent[1] ** 2))
                ang = np.arcsin(dist / rs)

                """calculate area of CH with minimal projection effects"""

                trupixar = abs(area / np.cos(ang))
                truarcar = trupixar * (dattoarc**2)
                trummar = truarcar * ((6.96e08 / rs) ** 2)

                """find CH extent in lattitude and longitude"""

                maxxlon, minxlon, centlat, centlon = extent(i, ypos, xpos, hg, cont)

                """caluclate the mean magnetic field"""

                mB = np.mean(datm[pos[:, 0], pos[:, 1]])
                mBpos = np.sum(npix[0][wh1] * npix[1][wh1]) / np.sum(npix[0][wh1])
                mBneg = np.sum(npix[0][wh2] * npix[1][wh2]) / np.sum(npix[0][wh2])

                """finds coordinates of CH boundaries"""

                Ywb, Xwb, Yeb, Xeb, Ynb, Xnb, Ysb, Xsb = coords(i, csys, cont)

                width = round(maxxlon.value) - round(minxlon.value)

                if minxlon.value >= 0.0:
                    eastl = "W" + str(int(np.round(minxlon.value)))
                else:
                    eastl = "E" + str(np.absolute(int(np.round(minxlon.value))))
                if maxxlon.value >= 0.0:
                    westl = "W" + str(int(np.round(maxxlon.value)))
                else:
                    westl = "E" + str(np.absolute(int(np.round(maxxlon.value))))

                if centlat >= 0.0:
                    centlat = "N" + str(int(np.round(centlat.value)))
                else:
                    centlat = "S" + str(np.absolute(int(np.round(centlat.value))))
                if centlon >= 0.0:
                    centlon = "W" + str(int(np.round(centlon.value)))
                else:
                    centlon = "E" + str(np.absolute(int(np.round(centlon.value))))

                """insertions of CH properties into property array"""

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
                """sets up code for next possible coronal hole"""

                ident = ident + 1

"""sets ident back to max value of iarr"""

ident = ident - 1

"""stores all CH properties in a text file"""
np.savetxt("ch_summary.txt", props, fmt="%s")


def rescale01(arr, cmin=None, cmax=None, a=0, b=1):
    """
    Rescales array

    Parameters
    ----------
    arr: 'np.arr'
    cmin: 'np.float'
    cmax: 'np.float'
    a: 'int'
    b: 'int'

    Returns
    -------
    np.array

    """
    if cmin or cmax:
        arr = np.clip(arr, cmin, cmax)
    return (b - a) * ((arr - np.min(arr)) / (np.max(arr) - np.min(arr))) + a


def plot_tricolor():
    """
    Plots a tricolor mask of image data

    Returns
    -------
    plot: 'matplotlib.image.AxesImage'

    """

    tricolorarray = np.zeros((1024, 1024, 3))

    data_a = img_as_ubyte(rescale01(np.log10(data), cmin=1.2, cmax=3.9))
    data_b = img_as_ubyte(rescale01(np.log10(datb), cmin=1.4, cmax=3.0))
    data_c = img_as_ubyte(rescale01(np.log10(datc), cmin=0.8, cmax=2.7))

    tricolorarray[..., 0] = data_c / np.max(data_c)
    tricolorarray[..., 1] = data_b / np.max(data_b)
    tricolorarray[..., 2] = data_a / np.max(data_a)

    fig, ax = plt.subplots(figsize=(10, 10))

    plt.imshow(tricolorarray, origin="lower")
    plt.contour(xgrid, ygrid, slate, colors="white", linewidths=0.5)
    plt.savefig("tricolor.png")
    plt.close()


def plot_mask(slate=slate):
    """
    Plots the contour mask

    Parameters
    ----------
    slate: 'np.array'

    Returns
    -------
    plot: 'matplotlib.image.AxesImage'

    """

    chs = np.where(iarr > 0)
    slate[chs] = 1
    slate = np.array(slate, dtype=np.uint8)
    cont, heir = cv2.findContours(slate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    circ[:] = 0
    r = rs / dattoarc
    w = np.where((xgrid - center[0]) ** 2 + (ygrid - center[1]) ** 2 <= r**2)
    circ[w] = 1.0

    plt.figure(figsize=(10, 10))
    plt.xlim(143, 4014)
    plt.ylim(143, 4014)
    plt.scatter(chs[1], chs[0], marker="s", s=0.0205, c="black", cmap="viridis", edgecolor="none", alpha=0.2)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis("off")
    plt.contour(xgrid, ygrid, slate, colors="black", linewidths=0.5)
    plt.contour(xgrid, ygrid, circ, colors="black", linewidths=1.0)

    plt.savefig("CH_mask_" + hedb["DATE"] + ".png", transparent=True)


plot_tricolor()
plot_mask()

if __name__ == "__main__":
    import_functions(
        INPUT_FILES["aia171"], INPUT_FILES["aia193"], INPUT_FILES["aia211"], INPUT_FILES["hmi_mag"]
    )

'''
Document detailing process summary and all functions/variables:

https://docs.google.com/document/d/1V5LkZq_AAHdTrGsnCl2hoYhm_fvyjfuzODiEHbt4ebo/edit?usp=sharing

'''
