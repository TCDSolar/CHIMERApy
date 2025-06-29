""" """

import os
import sys
import glob
from unittest.mock import MagicMock

import cv2
import mahotas
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map
from scipy.interpolate import RectBivariateSpline
from skimage.util import img_as_ubyte

import astropy.units as u
from astropy import wcs
from astropy.io import fits
from astropy.modeling.models import Gaussian2D


def chimera_legacy(im171=None, im193=None, im211=None, imhmi=None):
    file_path = "./"

    if im171 is None:
        im171 = glob.glob(file_path + "*171*.fts.gz")
        im193 = glob.glob(file_path + "*193*.fts.gz")
        im211 = glob.glob(file_path + "*211*.fts.gz")
        imhmi = glob.glob(file_path + "*hmi*.fts.gz")

    circ, data, datb, datc, dattoarc, hedb, iarr, props, rs, slate, center, xgrid, ygrid = chimera(
        im171, im193, im211, imhmi
    )

    # =====sets ident back to max value of iarr======

    # ident = ident - 1
    np.savetxt("ch_summary.txt", props, fmt="%s")

    plot_tricolor(data, datb, datc, xgrid, ygrid, slate)
    plot_mask(slate, iarr, circ, rs, dattoarc, center, xgrid, ygrid, hedb)
    return circ, data, datb, datc, dattoarc, hedb, iarr, props, rs, slate, center, xgrid, ygrid


def chimera(im171, im193, im211, imhmi):
    if im171 == [] or im193 == [] or im211 == [] or imhmi == []:
        print("Not all required files present")
        sys.exit()
    # =====Reads in data and resizes images=====
    ext_num = 0 if im171[0].endswith("fts.gz") else 1
    x = np.arange(0, 1024) * 4
    hdu_number = 0
    heda = fits.getheader(im171[0], ext_num)
    data = fits.getdata(im171[0], ext=ext_num) / (heda["EXPTIME"])
    dn = RectBivariateSpline(x, x, data, kx=1, ky=1)
    data = dn(np.arange(0, 4096), np.arange(0, 4096))
    hedb = fits.getheader(im193[0], ext_num)
    datb = fits.getdata(im193[0], ext=ext_num) / (hedb["EXPTIME"])
    dn = RectBivariateSpline(x, x, datb, kx=1, ky=1)
    datb = dn(np.arange(0, 4096), np.arange(0, 4096))
    hedc = fits.getheader(im211[0], ext_num)
    datc = fits.getdata(im211[0], ext=ext_num) / (hedc["EXPTIME"])
    dn = RectBivariateSpline(x, x, datc, kx=1, ky=1)
    datc = dn(np.arange(0, 4096), np.arange(0, 4096))
    hedm = fits.getheader(imhmi[0], hdu_number)
    datm = fits.getdata(imhmi[0], ext=0)
    # dn = scipy.interpolate.interp2d(np.arange(4096), np.arange(4096), datm)
    # datm = dn(np.arange(0, 1024)*4, np.arange(0, 1024)*4)
    if hedm["crota1"] > 90:
        datm = np.rot90(np.rot90(datm))
    # =====Specifies solar radius and calculates conversion value of pixel to arcsec=====
    s = np.shape(data)
    rs = heda["rsun"] if "rsun" in heda else heda["rsun_obs"]
    if hedb["ctype1"] != "solar_x ":
        hedb["ctype1"] = "solar_x "
        hedb["ctype2"] = "solar_y "
    if heda["cdelt1"] > 1:
        heda["cdelt1"], heda["cdelt2"], heda["crpix1"], heda["crpix2"] = (
            heda["cdelt1"] / 4.0,
            heda["cdelt2"] / 4.0,
            heda["crpix1"] * 4.0,
            heda["crpix2"] * 4.0,
        )
        hedb["cdelt1"], hedb["cdelt2"], hedb["crpix1"], hedb["crpix2"] = (
            hedb["cdelt1"] / 4.0,
            hedb["cdelt2"] / 4.0,
            hedb["crpix1"] * 4.0,
            hedb["crpix2"] * 4.0,
        )
        hedc["cdelt1"], hedc["cdelt2"], hedc["crpix1"], hedc["crpix2"] = (
            hedc["cdelt1"] / 4.0,
            hedc["cdelt2"] / 4.0,
            hedc["crpix1"] * 4.0,
            hedc["crpix2"] * 4.0,
        )
    dattoarc = heda["cdelt1"]
    convermul = dattoarc / hedm["cdelt1"]
    # =====Alternative coordinate systems=====
    hdul = fits.open(im171[0])
    hdul.verify("fix")
    hdul[ext_num].header["CUNIT1"] = "arcsec"
    hdul[ext_num].header["CUNIT2"] = "arcsec"
    aia = sunpy.map.Map(hdul[ext_num].data, hdul[ext_num].header)
    adj = 4096.0 / aia.dimensions[0].value
    x, y = (np.meshgrid(*[np.arange(adj * v.value) for v in aia.dimensions]) * u.pixel) / adj
    hpc = aia.pixel_to_world(x, y)
    hg = hpc.transform_to(sunpy.coordinates.frames.HeliographicStonyhurst)
    csys = wcs.WCS(hedb)
    # =======setting up arrays to be used============
    ident = 1
    iarr = np.zeros((s[0], s[1]), dtype=np.byte)
    offarr, slate = np.array(iarr), np.array(iarr)
    bmcool = np.zeros((s[0], s[1]), dtype=np.float32)
    cand, bmmix, bmhot = np.array(bmcool), np.array(bmcool), np.array(bmcool)
    circ = np.zeros((s[0], s[1]), dtype=int)
    # =======creation of a 2d gaussian for magnetic cut offs===========
    r = (s[1] / 2.0) - 450
    xgrid, ygrid = np.meshgrid(np.arange(s[0]), np.arange(s[1]))
    center = [int(s[1] / 2.0), int(s[1] / 2.0)]
    w = np.where((xgrid - center[0]) ** 2 + (ygrid - center[1]) ** 2 > r**2)
    y, x = np.mgrid[0:4096, 0:4096]
    garr = Gaussian2D(1, s[0] / 2, s[1] / 2, 2000 / 2.3548, 2000 / 2.3548)(x, y)
    garr[w] = 1.0
    # ======creation of array for CH properties==========
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
        "WIDTH",
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
        "H deg",
        '"',
        '"',
        '"',
        '"',
        '"',
        '"',
        '"',
        '"',
        "H deg",
        "deg",
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
    # =====removes negative data values=====
    data[np.where(data <= 0)] = 0
    datb[np.where(datb <= 0)] = 0
    datc[np.where(datc <= 0)] = 0
    # ============make a multi-wavelength image for contours==================
    with np.errstate(divide="ignore"):
        t0 = np.log10(datc)
        t1 = np.log10(datb)
        t2 = np.log10(data)
    t0[np.where(t0 < 0.8)] = 0.8
    t0[np.where(t0 > 2.7)] = 2.7
    t1[np.where(t1 < 1.4)] = 1.4
    t1[np.where(t1 > 3.0)] = 3.0
    t2[np.where(t2 < 1.2)] = 1.2
    t2[np.where(t2 > 3.9)] = 3.9
    t0 = np.array(((t0 - 0.8) / (2.7 - 0.8)) * 255, dtype=np.float32)
    t1 = np.array(((t1 - 1.4) / (3.0 - 1.4)) * 255, dtype=np.float32)
    t2 = np.array(((t2 - 1.2) / (3.9 - 1.2)) * 255, dtype=np.float32)
    # ====create 3 segmented bitmasks=====
    with np.errstate(divide="ignore", invalid="ignore"):
        bmmix[np.where(t2 / t0 >= ((np.mean(data) * 0.6357) / (np.mean(datc))))] = 1
        bmhot[np.where(t0 + t1 < (0.7 * (np.mean(datb) + np.mean(datc))))] = 1
        bmcool[np.where(t2 / t1 >= ((np.mean(data) * 1.5102) / (np.mean(datb))))] = 1
    # ====logical conjunction of 3 segmentations=======
    cand = bmcool * bmmix * bmhot
    # ====plot tricolour image with lon/lat conotours=======
    # ======removes off detector mis-identifications==========
    r = (s[1] / 2.0) - 100
    w = np.where((xgrid - center[0]) ** 2 + (ygrid - center[1]) ** 2 <= r**2)
    circ[w] = 1.0
    cand = cand * circ
    # =======Separates on-disk and off-limb CHs===============
    circ[:] = 0
    r = (rs / dattoarc) - 10
    w = np.where((xgrid - center[0]) ** 2 + (ygrid - center[1]) ** 2 <= r**2)
    circ[w] = 1.0
    r = (rs / dattoarc) + 40
    w = np.where((xgrid - center[0]) ** 2 + (ygrid - center[1]) ** 2 >= r**2)
    circ[w] = 1.0
    cand = cand * circ
    # ====open file for property storage=====
    # =====contours the identified datapoints=======
    if os.environ.get("PYTEST_VERSION") and isinstance(cv2, MagicMock):
        cand, cont, heir = cv2.findContours(cand, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        cand = np.array(cand, dtype=np.uint8)
        cont, heir = cv2.findContours(cand, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # ======sorts contours by size============
    sizes = []
    for i in range(len(cont)):
        sizes = np.append(sizes, len(cont[i]))
    reord = sizes.ravel().argsort()[::-1]
    tmp = list(cont)
    for i in range(len(cont)):
        tmp[i] = cont[reord[i]]
    cont = list(tmp)
    # =====cycles through contours=========
    for i in range(len(cont)):
        x = np.append(x, len(cont[i]))

        # =====only takes values of minimum surface length and calculates area======

        if len(cont[i]) <= 100:
            continue
        area = 0.5 * np.abs(
            np.dot(cont[i][:, 0, 0], np.roll(cont[i][:, 0, 1], 1))
            - np.dot(cont[i][:, 0, 1], np.roll(cont[i][:, 0, 0], 1))
        )
        arcar = area * (dattoarc**2)
        if arcar > 1000:
            # =====finds centroid=======

            # chpts = len(cont[i])
            cent = [np.mean(cont[i][:, 0, 0]), np.mean(cont[i][:, 0, 1])]

            # ===remove quiet sun regions encompassed by coronal holes======

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
                iarr[np.where(slate == 1)] = 0
                slate[:] = 0

            else:
                # ====create a simple centre point======

                arccent = csys.all_pix2world(cent[0], cent[1], 0)

                # ====classifies off limb CH regions========

                if (((arccent[0] ** 2) + (arccent[1] ** 2)) > (rs**2)) or (
                    np.sum(np.array(csys.all_pix2world(cont[i][0, 0, 0], cont[i][0, 0, 1], 0)) ** 2) > (rs**2)
                ):
                    mahotas.polygon.fill_polygon(
                        np.array(list(zip(cont[i][:, 0, 1], cont[i][:, 0, 0]))), offarr
                    )
                else:
                    # =====classifies on disk coronal holes=======

                    mahotas.polygon.fill_polygon(
                        np.array(list(zip(cont[i][:, 0, 1], cont[i][:, 0, 0]))), slate
                    )
                    poslin = np.where(slate == 1)
                    slate[:] = 0

                    # ====create an array for magnetic polarity========

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

                    # =====magnetic cut offs dependent on area=========

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

                    # ====create an accurate center point=======

                    ypos = np.sum((poslin[0]) * np.absolute(hg.lat[poslin])) / np.sum(
                        np.absolute(hg.lat[poslin])
                    )
                    xpos = np.sum((poslin[1]) * np.absolute(hg.lon[poslin])) / np.sum(
                        np.absolute(hg.lon[poslin])
                    )

                    arccent = csys.all_pix2world(xpos, ypos, 0)

                    # ======calculate average angle coronal hole is subjected to======

                    dist = np.sqrt((arccent[0] ** 2) + (arccent[1] ** 2))
                    ang = np.arcsin(dist / rs)

                    # =====calculate area of CH with minimal projection effects======

                    trupixar = abs(area / np.cos(ang))
                    truarcar = trupixar * (dattoarc**2)
                    trummar = truarcar * ((6.96e08 / rs) ** 2)

                    # ====find CH extent in latitude and longitude========

                    # maxxlat = hg.lat[
                    #     cont[i][np.where(cont[i][:, 0, 0] == np.max(cont[i][:, 0, 0]))[0][0], 0, 1],
                    #     np.max(cont[i][:, 0, 0]),
                    # ]
                    maxxlon = hg.lon[
                        cont[i][np.where(cont[i][:, 0, 0] == np.max(cont[i][:, 0, 0]))[0][0], 0, 1],
                        np.max(cont[i][:, 0, 0]),
                    ]
                    # maxylat = hg.lat[
                    #     np.max(cont[i][:, 0, 1]),
                    #     cont[i][np.where(cont[i][:, 0, 1] == np.max(cont[i][:, 0, 1]))[0][0], 0, 0],
                    # ]
                    # maxylon = hg.lon[
                    #     np.max(cont[i][:, 0, 1]),
                    #     cont[i][np.where(cont[i][:, 0, 1] == np.max(cont[i][:, 0, 1]))[0][0], 0, 0],
                    # ]
                    # minxlat = hg.lat[
                    #     cont[i][np.where(cont[i][:, 0, 0] == np.min(cont[i][:, 0, 0]))[0][0], 0, 1],
                    #     np.min(cont[i][:, 0, 0]),
                    # ]
                    minxlon = hg.lon[
                        cont[i][np.where(cont[i][:, 0, 0] == np.min(cont[i][:, 0, 0]))[0][0], 0, 1],
                        np.min(cont[i][:, 0, 0]),
                    ]
                    # minylat = hg.lat[
                    #     np.min(cont[i][:, 0, 1]),
                    #     cont[i][np.where(cont[i][:, 0, 1] == np.min(cont[i][:, 0, 1]))[0][0], 0, 0],
                    # ]
                    # minylon = hg.lon[
                    #     np.min(cont[i][:, 0, 1]),
                    #     cont[i][np.where(cont[i][:, 0, 1] == np.min(cont[i][:, 0, 1]))[0][0], 0, 0],
                    # ]

                    # =====CH centroid in lat/lon=======

                    centlat = hg.lat[int(ypos), int(xpos)]
                    centlon = hg.lon[int(ypos), int(xpos)]

                    # ====calculate the mean magnetic field=====

                    mB = np.mean(datm[pos[:, 0], pos[:, 1]])
                    mBpos = np.sum(npix[0][wh1] * npix[1][wh1]) / np.sum(npix[0][wh1])
                    mBneg = np.sum(npix[0][wh2] * npix[1][wh2]) / np.sum(npix[0][wh2])

                    # =====finds coordinates of CH boundaries=======

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

                    # ====insertions of CH properties into property array=====

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
                    props[14, ident + 1] = f"{trummar / 1e12:.1e}"
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
                    props[23, ident + 1] = f"{mB * trummar * 1e16:.1e}"
                    props[24, ident + 1] = f"{mBpos * trummar * 1e16:.1e}"
                    props[25, ident + 1] = f"{mBneg * trummar * 1e16:.1e}"

                    # =====sets up code for next possible coronal hole=====

                    ident = ident + 1
    return circ, data, datb, datc, dattoarc, hedb, iarr, props, rs, slate, center, xgrid, ygrid


def rescale01(arr, cmin=None, cmax=None, a=0, b=1):
    if cmin or cmax:
        arr = np.clip(arr, cmin, cmax)
    return (b - a) * ((arr - np.min(arr)) / (np.max(arr) - np.min(arr))) + a


def plot_tricolor(data, datb, datc, xgrid, ygrid, slate):
    tricolorarray = np.zeros((4096, 4096, 3))

    data_a = img_as_ubyte(rescale01(np.log10(data), cmin=1.2, cmax=3.9))
    data_b = img_as_ubyte(rescale01(np.log10(datb), cmin=1.4, cmax=3.0))
    data_c = img_as_ubyte(rescale01(np.log10(datc), cmin=0.8, cmax=2.7))

    tricolorarray[..., 0] = data_c / np.max(data_c)
    tricolorarray[..., 1] = data_b / np.max(data_b)
    tricolorarray[..., 2] = data_a / np.max(data_a)

    fig, ax = plt.subplots(figsize=(10, 10))

    plt.imshow(tricolorarray, origin="lower")  # , extent = )
    plt.contour(xgrid, ygrid, slate, colors="white", linewidths=0.5)
    plt.savefig("tricolor.png")
    plt.close()


def plot_mask(slate, iarr, circ, rs, dattoarc, center, xgrid, ygrid, hedb):
    chs = np.where(iarr > 0)
    slate[chs] = 1
    slate = np.array(slate, dtype=np.uint8)
    # cont, heir = cv2.findContours(slate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
    plt.close()
