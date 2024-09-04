import copy

import astropy.units as u
import numpy as np
import sunpy
import sunpy.map
from astropy.units import UnitsError
from astropy.visualization import make_lupton_rgb
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from sunpy.map import Map, all_coordinates_from_map

m171 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0171.fits")
m193 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0193.fits")
m211 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0211.fits")


def segmenting_plots(scale_cold: float, scale_warm: float, scale_hot: float):
    m171 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0171.fits")
    m193 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0193.fits")
    m211 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0211.fits")

    m171_orig = copy.deepcopy(m171)
    m193_orig = copy.deepcopy(m193)
    m211_orig = copy.deepcopy(m211)

    # Since the data are taken at similar times neglect any coordinate changes so just use 171 maps coordinates
    coords = all_coordinates_from_map(m171)
    disk_mask = (coords.Tx**2 + coords.Ty**2) ** 0.5 < m171.rsun_obs

    m171 = m171 / m171.exposure_time
    m193 = m193 / m193.exposure_time
    m211 = m211 / m211.exposure_time

    xx = np.linspace(0, 300, 500)

    fig, axes = plt.subplot_mosaic(
        [["cool_hist"], ["warm_hist"], ["hot_hist"]],
        layout="constrained",
        figsize=(3, 5),
    )

    # 171 v 193
    cool_counts, *cool_bins = axes["cool_hist"].hist2d(
        m171.data[disk_mask].flatten(),
        m193.data[disk_mask].flatten(),
        bins=150,
        range=[[0, 300], [0, 300]],
        norm=LogNorm(),
        density=True,
    )
    # Finding the indices of nonzero counts
    non_zero_cool = np.where(cool_counts > 0)

    # Finding the corresponding minimum y-index
    min_y_index = np.min(non_zero_cool[1])

    # Map the index to the actual y-value
    min_y_cool = cool_bins[1][min_y_index]
    axes["cool_hist"].set_facecolor("k")
    axes["cool_hist"].plot(xx, (xx**scale_cold) + min_y_cool, "w")

    # 171 v 211
    warm_counts, *warm_bins = axes["warm_hist"].hist2d(
        m171.data[disk_mask].flatten(),
        m211.data[disk_mask].flatten(),
        bins=250,
        range=[[0, 300], [0, 300]],
        norm=LogNorm(),
        density=True,
    )
    # Finding the indices of nonzero counts
    non_zero_warm = np.where(warm_counts > 0)

    # Finding the corresponding minimum y-index
    min_y_index = np.min(non_zero_warm[1])

    # Map the index to the actual y-value
    min_y_warm = warm_bins[1][min_y_index]
    axes["warm_hist"].set_ylim(0, 100)
    axes["warm_hist"].set_facecolor("k")
    axes["warm_hist"].plot(xx, (xx**scale_warm) + min_y_warm, "w")

    # 193 v 311
    hot_counts, *hot_bins = axes["hot_hist"].hist2d(
        m193.data[disk_mask].flatten(),
        m211.data[disk_mask].flatten(),
        bins=250,
        range=[[0, 300], [0, 300]],
        norm=LogNorm(),
        density=True,
    )
    # Finding the indices of nonzero counts
    non_zero_hot = np.where(hot_counts > 0)

    # Finding the corresponding minimum y-index
    min_y_index = np.min(non_zero_hot[1])

    # Map the index to the actual y-value
    min_y_hot = hot_bins[1][min_y_index]

    axes["hot_hist"].set_ylim(0, 100)
    axes["hot_hist"].set_facecolor("k")
    axes["hot_hist"].plot(xx, (xx**scale_hot) + min_y_hot, "w")

    plt.show()


segmenting_plots(0.7, 0.6, 0.7)


def clip_scale(map1: sunpy.map, clip1: float, clip2: float, clip3: float, scale: float):
    map_clipped = np.clip(np.log10(map1.data), clip1, clip3)
    map_clipped_scaled = ((map_clipped - clip1) / clip2) * scale
    return map_clipped_scaled


cs_171 = clip_scale(m171, 1.2, 2.7, 3.9, 255)
cs_193 = clip_scale(m193, 1.4, 1.6, 3.0, 255)
cs_211 = clip_scale(m211, 0.8, 1.9, 2.7, 255)


def create_mask(scale1: float, scale2: float, scale3: float):
    mask1 = (cs_171 / cs_211) >= (np.mean(m171.data) * scale1) / np.mean(m211.data)
    mask2 = (cs_211 + cs_193) < (scale2 * (np.mean(m193.data) + np.mean(m211.data)))
    mask3 = (cs_171 / cs_193) >= ((np.mean(m171.data) * scale3) / np.mean(m193.data))
    return mask1, mask2, mask3


mask211_171, mask211_193, mask171_193 = create_mask(0.6357, 0.7, 1.5102)

tri_color_img = make_lupton_rgb(cs_171, cs_193, cs_211, Q=10, stretch=50)
comb_mask = mask211_171 * mask211_193 * mask171_193


fig, axes = plt.subplot_mosaic([["tri", "comb_mask"]], layout="constrained", figsize=(6, 3))

axes["tri"].imshow(tri_color_img, origin="lower")
axes["comb_mask"].imshow(comb_mask, origin="lower")

plt.show()


def create_contours(mask1: np.ndarray, mask2: np.ndarray, mask3: np.ndarray):
    mask_map = Map(((mask1 * mask2 * mask3).astype(int), m171.meta))
    try:
        contours = mask_map.contour(0.5 / u.s)
    except UnitsError:
        contours = mask_map.contour(50 * u.percent)

    contours = sorted(contours, key=lambda x: x.size, reverse=True)

    fig, axes = plt.subplot_mosaic(
        [["seg"]], layout="constrained", figsize=(6, 3), subplot_kw={"projection": m171}
    )
    m171.plot(axes=axes["seg"])
    axes["seg"].imshow(tri_color_img)

    # For the moment just plot to top 5 contours based on "size" for contour
    for contour in contours[:6]:
        axes["seg"].plot_coord(contour, color="w", linewidth=0.5)
    plt.show()


create_contours(mask211_171, mask211_193, mask171_193)
